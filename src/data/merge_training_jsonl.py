"""Atomically concatenate validated training JSONL files.

The VLN dataloader shuffles line offsets, so preserving each source's order in
the merged file is both deterministic and sufficient.  This script validates
the compact fields consumed by the loader while copying and creates no sidecar
files.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Sequence

from tqdm import tqdm


ACTION_WORDS = {"stop", "forward", "left", "right"}
ACTION_HORIZON = 4


def validate_training_row(row: Any, path: Path, line_number: int) -> Dict[str, Any]:
    context = f"{path}:{line_number}"
    if not isinstance(row, dict):
        raise ValueError(f"{context}: row must be a JSON object")
    instruction = row.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"{context}: instruction must be a non-empty string")
    actions = row.get("action_sequence")
    if not isinstance(actions, list) or len(actions) != ACTION_HORIZON:
        raise ValueError(
            f"{context}: action_sequence must contain exactly {ACTION_HORIZON} actions"
        )
    invalid_actions = [action for action in actions if action not in ACTION_WORDS]
    if invalid_actions:
        raise ValueError(f"{context}: invalid actions {invalid_actions}")
    if "stop" in actions and actions[-1] != "stop":
        raise ValueError(f"{context}: stop must terminate the action chunk")
    images = row.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError(f"{context}: images must be a non-empty list")
    if any(not isinstance(image, str) or not image for image in images):
        raise ValueError(f"{context}: images must contain non-empty paths")
    return row


def merge_training_jsonl(
    input_paths: Sequence[Path],
    output_path: Path,
    overwrite: bool,
) -> Dict[str, Any]:
    if len(input_paths) < 2:
        raise ValueError("At least two input JSONL files are required")
    input_paths = [path.resolve() for path in input_paths]
    if len(set(input_paths)) != len(input_paths):
        raise ValueError("Input JSONL paths must be distinct")
    for input_path in input_paths:
        if not input_path.is_file():
            raise FileNotFoundError(input_path)

    output_path = output_path.resolve()
    if output_path in input_paths:
        raise ValueError("Output path must differ from every input path")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Mixed training JSONL already exists; pass --overwrite: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path_string = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    temporary_path = Path(temporary_path_string)
    source_rows: Dict[str, int] = {}
    dataset_counts: Counter[str] = Counter()
    total_rows = 0
    try:
        with os.fdopen(descriptor, "wb") as output_handle:
            for input_path in input_paths:
                row_count = 0
                with input_path.open("rb") as input_handle, tqdm(
                    total=input_path.stat().st_size,
                    desc=f"merge {input_path.name}",
                    unit="B",
                    unit_scale=True,
                    dynamic_ncols=True,
                ) as progress:
                    for line_number, raw_line in enumerate(input_handle, start=1):
                        progress.update(len(raw_line))
                        if not raw_line.strip():
                            continue
                        try:
                            row = json.loads(raw_line)
                        except (json.JSONDecodeError, UnicodeDecodeError) as error:
                            raise ValueError(
                                f"invalid JSON at {input_path}:{line_number}: {error}"
                            ) from error
                        validate_training_row(row, input_path, line_number)
                        output_handle.write(raw_line.rstrip(b"\r\n"))
                        output_handle.write(b"\n")
                        row_count += 1
                        total_rows += 1
                        dataset_counts[str(row.get("dataset", "<missing>"))] += 1
                source_rows[str(input_path)] = row_count
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.chmod(temporary_path, 0o644)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return {
        "output_path": str(output_path),
        "total_rows": total_rows,
        "source_rows": source_rows,
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "output_bytes": output_path.stat().st_size,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_paths", type=Path, nargs="+", required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = merge_training_jsonl(
        input_paths=args.input_paths,
        output_path=args.output_path,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
