"""Build panoramic inverse-dynamics samples from an existing EBS JSONL.

Each eligible EBS row observes trajectory state ``t`` and predicts the next
four actions.  The paired inverse-dynamics row keeps the same instruction and
full observation history, but predicts the four actions executed immediately
before state ``t``.  Rows before step four are retained only as forward EBS
samples because a complete past-four target does not exist.

The output is written atomically.  When a mixed output is requested, forward
and paired inverse-dynamics rows are emitted together in one pass so the large
EBS file is not scanned twice.  No sidecar summary file is created.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, BinaryIO, Dict, Optional, Tuple

from tqdm import tqdm


ACTION_HORIZON = 4
ACTION_WORDS = {"stop", "forward", "left", "right"}
FORWARD_DYNAMICS_TASK = "fds"
INVERSE_DYNAMICS_TASK = "ids"


def _require_non_empty_string(row: Dict[str, Any], field: str, context: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: {field} must be a non-empty string")
    return value


def _validate_action_list(
    actions: Any,
    field: str,
    context: str,
    expected_length: Optional[int] = None,
) -> Tuple[str, ...]:
    if not isinstance(actions, list):
        raise ValueError(f"{context}: {field} must be a list")
    if expected_length is not None and len(actions) != expected_length:
        raise ValueError(
            f"{context}: {field} must contain exactly {expected_length} actions, "
            f"got {len(actions)}"
        )
    invalid = [action for action in actions if action not in ACTION_WORDS]
    if invalid:
        raise ValueError(f"{context}: {field} contains invalid actions {invalid}")
    return tuple(actions)


def validate_ebs_row(row: Any, context: str) -> int:
    if not isinstance(row, dict):
        raise ValueError(f"{context}: row must be a JSON object")

    task_type = row.get("task_type")
    if task_type != FORWARD_DYNAMICS_TASK:
        raise ValueError(
            f"{context}: expected task_type={FORWARD_DYNAMICS_TASK!r}, "
            f"got {task_type!r}"
        )

    _require_non_empty_string(row, "instruction", context)
    _require_non_empty_string(row, "dataset", context)
    if row.get("episode_id") is None:
        raise ValueError(f"{context}: episode_id is required")

    current_step = row.get("step_index")
    if isinstance(current_step, bool) or not isinstance(current_step, int):
        raise ValueError(f"{context}: step_index must be an integer")
    if current_step < 0:
        raise ValueError(f"{context}: step_index must be non-negative")

    images = row.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError(f"{context}: images must be a non-empty list")
    if any(not isinstance(image, str) or not image for image in images):
        raise ValueError(f"{context}: images must contain non-empty paths")
    if len(images) != current_step + 1:
        raise ValueError(
            f"{context}: images must contain every observation through step_index "
            f"({current_step + 1} expected, got {len(images)})"
        )

    future_actions = _validate_action_list(
        row.get("action_sequence"),
        "action_sequence",
        context,
        expected_length=ACTION_HORIZON,
    )
    if "stop" in future_actions:
        first_stop = future_actions.index("stop")
        if any(action != "stop" for action in future_actions[first_stop:]):
            raise ValueError(f"{context}: actions after the first stop must also be stop")

    history_actions = _validate_action_list(
        row.get("history_actions"),
        "history_actions",
        context,
    )
    if len(history_actions) != current_step:
        raise ValueError(
            f"{context}: history_actions length must equal step_index "
            f"({current_step} expected, got {len(history_actions)})"
        )
    if "stop" in history_actions:
        raise ValueError(f"{context}: history_actions cannot contain terminal stop")

    return current_step


def build_inverse_dynamics_row(row: Dict[str, Any], current_step: int) -> Dict[str, Any]:
    if current_step < ACTION_HORIZON:
        raise ValueError(
            f"Cannot build past-{ACTION_HORIZON} target at step {current_step}"
        )

    history_actions = row["history_actions"]
    past_actions = list(history_actions[-ACTION_HORIZON:])
    target_start_step = current_step - ACTION_HORIZON

    inverse_row: Dict[str, Any] = {
        "instruction": row["instruction"],
        "action_sequence": past_actions,
        "images": list(row["images"]),
        "episode_id": row["episode_id"],
        "dataset": row["dataset"],
        "task_type": INVERSE_DYNAMICS_TASK,
        "current_step": current_step,
        "target_start_step": target_start_step,
        "target_end_step": current_step - 1,
        "real_action_count": ACTION_HORIZON,
    }
    if row.get("trajectory_id") is not None:
        inverse_row["trajectory_id"] = row["trajectory_id"]
    return inverse_row


def _open_atomic_output(path: Path) -> Tuple[BinaryIO, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path_string = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    return os.fdopen(descriptor, "wb"), Path(temporary_path_string)


def _write_jsonl_row(handle: BinaryIO, row: Dict[str, Any]) -> int:
    encoded = (
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    handle.write(encoded)
    return len(encoded)


def prepare_inverse_dynamics_training_data(
    input_path: Path,
    ids_output_path: Path,
    mixed_output_path: Optional[Path] = None,
    overwrite: bool = False,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    input_path = input_path.resolve()
    ids_output_path = ids_output_path.resolve()
    mixed_output_path = mixed_output_path.resolve() if mixed_output_path else None

    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_paths = [ids_output_path]
    if mixed_output_path is not None:
        output_paths.append(mixed_output_path)
    if len(set(output_paths)) != len(output_paths):
        raise ValueError("IDS and mixed output paths must be distinct")
    if input_path in output_paths:
        raise ValueError("Output paths must differ from the EBS input path")
    for output_path in output_paths:
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists; pass --overwrite to replace it: {output_path}"
            )
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive when provided")

    ids_handle, ids_temporary_path = _open_atomic_output(ids_output_path)
    mixed_handle: Optional[BinaryIO] = None
    mixed_temporary_path: Optional[Path] = None
    if mixed_output_path is not None:
        mixed_handle, mixed_temporary_path = _open_atomic_output(mixed_output_path)

    forward_rows = 0
    ids_rows = 0
    skipped_short_history = 0
    ids_bytes = 0
    mixed_bytes = 0
    dataset_forward_counts: Counter[str] = Counter()
    dataset_ids_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()

    try:
        with input_path.open("rb") as input_handle, tqdm(
            total=input_path.stat().st_size,
            desc="build panoramic IDS",
            unit="B",
            unit_scale=True,
            dynamic_ncols=True,
        ) as progress:
            for line_number, raw_line in enumerate(input_handle, start=1):
                progress.update(len(raw_line))
                if not raw_line.strip():
                    continue
                if max_rows is not None and forward_rows >= max_rows:
                    break

                context = f"{input_path}:{line_number}"
                try:
                    row = json.loads(raw_line)
                except (json.JSONDecodeError, UnicodeDecodeError) as error:
                    raise ValueError(f"{context}: invalid JSON: {error}") from error

                current_step = validate_ebs_row(row, context)
                forward_rows += 1
                dataset_name = str(row["dataset"])
                dataset_forward_counts[dataset_name] += 1

                if mixed_handle is not None:
                    normalized_raw_line = raw_line.rstrip(b"\r\n") + b"\n"
                    mixed_handle.write(normalized_raw_line)
                    mixed_bytes += len(normalized_raw_line)

                if current_step < ACTION_HORIZON:
                    skipped_short_history += 1
                    continue

                inverse_row = build_inverse_dynamics_row(row, current_step)
                encoded_size = _write_jsonl_row(ids_handle, inverse_row)
                ids_bytes += encoded_size
                ids_rows += 1
                dataset_ids_counts[dataset_name] += 1
                action_counts.update(inverse_row["action_sequence"])

                if mixed_handle is not None:
                    mixed_bytes += _write_jsonl_row(mixed_handle, inverse_row)

        ids_handle.flush()
        os.fsync(ids_handle.fileno())
        ids_handle.close()
        ids_handle = None
        if mixed_handle is not None:
            mixed_handle.flush()
            os.fsync(mixed_handle.fileno())
            mixed_handle.close()
            mixed_handle = None

        os.chmod(ids_temporary_path, 0o644)
        os.replace(ids_temporary_path, ids_output_path)
        if mixed_output_path is not None and mixed_temporary_path is not None:
            os.chmod(mixed_temporary_path, 0o644)
            os.replace(mixed_temporary_path, mixed_output_path)
    finally:
        if ids_handle is not None:
            ids_handle.close()
        if mixed_handle is not None:
            mixed_handle.close()
        if ids_temporary_path.exists():
            ids_temporary_path.unlink()
        if mixed_temporary_path is not None and mixed_temporary_path.exists():
            mixed_temporary_path.unlink()

    summary: Dict[str, Any] = {
        "input_path": str(input_path),
        "ids_output_path": str(ids_output_path),
        "mixed_output_path": str(mixed_output_path) if mixed_output_path else None,
        "forward_rows": forward_rows,
        "ids_rows": ids_rows,
        "skipped_short_history": skipped_short_history,
        "ids_to_forward_ratio": ids_rows / forward_rows if forward_rows else 0.0,
        "dataset_forward_counts": dict(sorted(dataset_forward_counts.items())),
        "dataset_ids_counts": dict(sorted(dataset_ids_counts.items())),
        "ids_action_counts": dict(sorted(action_counts.items())),
        "ids_output_bytes": ids_bytes,
        "mixed_output_bytes": mixed_bytes if mixed_output_path else None,
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--ids_output_path", type=Path, required=True)
    parser.add_argument("--mixed_output_path", type=Path)
    parser.add_argument("--max_rows", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_inverse_dynamics_training_data(
        input_path=args.input_path,
        ids_output_path=args.ids_output_path,
        mixed_output_path=args.mixed_output_path,
        overwrite=args.overwrite,
        max_rows=args.max_rows,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
