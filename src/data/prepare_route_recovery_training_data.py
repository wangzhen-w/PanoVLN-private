"""Build compact four-action recovery SFT data from continuous trajectories.

The raw collector records the clean prefix, injected deviation, and breadcrumb
return as one causal panorama sequence.  This script never supervises injected
deviation actions.  It emits every executable four-action chunk at the model's
four-action replanning cadence, including all intermediate states of an
in-place U-turn.  An optional clean anchor counterexample can be enabled for a
later ablation.  The generated recovery-only rows use the same compact schema
as the existing EBS navigation data, but this script never reads, filters,
copies, or merges EBS data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from tqdm import tqdm


DEFAULT_INPUT_ROOT = Path("/workspace/z_PanoVLN")
DEFAULT_DATASET_NAME = "r2r_rxr_recovery"
ACTION_HORIZON = 4
STOP_ACTION_ID = 0
FORWARD_ACTION_ID = 1
LEFT_ACTION_ID = 2
RIGHT_ACTION_ID = 3
SUPPORTED_ACTION_IDS = {
    STOP_ACTION_ID,
    FORWARD_ACTION_ID,
    LEFT_ACTION_ID,
    RIGHT_ACTION_ID,
}
ACTION_WORDS = {
    STOP_ACTION_ID: "stop",
    FORWARD_ACTION_ID: "forward",
    LEFT_ACTION_ID: "left",
    RIGHT_ACTION_ID: "right",
}
DEFAULT_HISTORY_WINDOW_FRAMES = 100
DEFAULT_SEED = 42


class RecoveryPreparationError(ValueError):
    pass


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise RecoveryPreparationError(
                    f"invalid JSON at {path}:{line_number}: {error}"
                ) from error


def write_jsonl_row(handle, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
    handle.write("\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def physical_route_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(row["source_dataset"]),
        str(row["scene_id"]),
        str(row["source_trajectory_id"]),
    )


def validate_action_ids(raw_actions: Any, context: str) -> List[int]:
    if not isinstance(raw_actions, list):
        raise RecoveryPreparationError(f"{context}: actions must be a list")
    actions = [int(action) for action in raw_actions]
    invalid = sorted(set(actions) - SUPPORTED_ACTION_IDS)
    if invalid:
        raise RecoveryPreparationError(
            f"{context}: unsupported action ids {invalid}"
        )
    return actions


def resolve_image_paths(
    row: Dict[str, Any],
    input_root: Path,
    dataset_name: str,
    expected_frame_count: int,
    validate_image_files: bool,
) -> List[str]:
    trajectory_id = str(row.get("trajectory_id", row.get("episode_id", "")))
    if not trajectory_id:
        raise RecoveryPreparationError("recovery row has no trajectory_id")
    relative_directory = Path("images") / dataset_name / trajectory_id
    absolute_directory = input_root / relative_directory
    images = [
        (relative_directory / f"frame_{frame_index}.jpg").as_posix()
        for frame_index in range(expected_frame_count)
    ]
    if validate_image_files:
        if not absolute_directory.is_dir():
            raise RecoveryPreparationError(
                f"missing recovery image directory: {absolute_directory}"
            )
        for relative_image in images:
            image_path = input_root / relative_image
            if not image_path.is_file():
                raise RecoveryPreparationError(f"missing recovery image: {image_path}")
        actual_frame_count = sum(
            1 for _ in absolute_directory.glob("frame_*.jpg")
        )
        if actual_frame_count != expected_frame_count:
            raise RecoveryPreparationError(
                f"{trajectory_id}: expected {expected_frame_count} frames, found "
                f"{actual_frame_count}"
            )
    return images


def validate_raw_trajectory(
    row: Dict[str, Any],
    input_root: Path,
    dataset_name: str,
    validate_image_files: bool,
) -> Tuple[List[int], List[str]]:
    context = str(row.get("episode_id", "<unknown>"))
    if int(row.get("schema_version", 0)) != 2:
        raise RecoveryPreparationError(
            f"{context}: expected continuous recovery schema_version=2"
        )
    if int(row.get("trajectory_algorithm_version", 0)) != 3:
        raise RecoveryPreparationError(
            f"{context}: expected trajectory_algorithm_version=3"
        )
    if float(row.get("standard_waypoint_radius", -1.0)) != 0.3:
        raise RecoveryPreparationError(
            f"{context}: expected 0.3m standard waypoint radius"
        )
    if row.get("standard_action_source") != "reference_path_shortest_path_follower":
        raise RecoveryPreparationError(
            f"{context}: unexpected standard action source"
        )
    actions = validate_action_ids(row.get("actions"), context)
    if STOP_ACTION_ID in actions:
        raise RecoveryPreparationError(
            f"{context}: continuous recovery trajectory must not contain stop"
        )
    standard_actions = validate_action_ids(row.get("standard_actions"), context)
    if not standard_actions or standard_actions[-1] != STOP_ACTION_ID:
        raise RecoveryPreparationError(
            f"{context}: standard actions must end in stop"
        )
    images = resolve_image_paths(
        row,
        input_root=input_root,
        dataset_name=dataset_name,
        expected_frame_count=len(actions) + 1,
        validate_image_files=validate_image_files,
    )
    if len(images) != len(actions) + 1:
        raise RecoveryPreparationError(
            f"{context}: expected actions+1 images, got "
            f"actions={len(actions)} images={len(images)}"
        )

    standard_prefix_end = int(row.get("standard_prefix_end", -1))
    deviation_start = int(row.get("deviation_start", -1))
    recovery_start = int(row.get("recovery_start", -1))
    recovery_end = int(row.get("recovery_end", -1))
    if standard_prefix_end != deviation_start:
        raise RecoveryPreparationError(
            f"{context}: standard_prefix_end must equal deviation_start"
        )
    if not (
        0 <= deviation_start < recovery_start < recovery_end == len(actions)
    ):
        raise RecoveryPreparationError(
            f"{context}: invalid segment boundaries "
            f"deviation_start={deviation_start}, recovery_start={recovery_start}, "
            f"recovery_end={recovery_end}, actions={len(actions)}"
        )
    if int(row.get("standard_prefix_action_count", -1)) != deviation_start:
        raise RecoveryPreparationError(
            f"{context}: inconsistent standard-prefix action count"
        )
    if actions[:deviation_start] != standard_actions[:deviation_start]:
        raise RecoveryPreparationError(
            f"{context}: clean prefix differs from standard actions"
        )
    if STOP_ACTION_ID in actions[recovery_start:recovery_end]:
        raise RecoveryPreparationError(
            f"{context}: recovery interval must not contain stop"
        )
    positions = row.get("action_positions")
    rotations = row.get("action_rotations")
    if not isinstance(positions, list) or len(positions) != len(actions) + 1:
        raise RecoveryPreparationError(
            f"{context}: action_positions must contain actions+1 entries"
        )
    if not isinstance(rotations, list) or len(rotations) != len(actions) + 1:
        raise RecoveryPreparationError(
            f"{context}: action_rotations must contain actions+1 entries"
        )
    return actions, images


def build_recovery_chunk_starts(
    recovery_start: int,
    recovery_end: int,
    action_horizon: int = ACTION_HORIZON,
) -> List[int]:
    recovery_length = recovery_end - recovery_start
    if recovery_length < action_horizon:
        raise RecoveryPreparationError(
            f"recovery has {recovery_length} actions, fewer than horizon "
            f"{action_horizon}"
        )
    last_start = recovery_end - action_horizon
    starts = list(range(recovery_start, last_start + 1, action_horizon))
    if last_start not in starts:
        starts.append(last_start)
    return sorted(set(starts))


def select_causal_history(
    images: Sequence[str],
    current_frame_index: int,
    history_window_frames: int,
) -> Tuple[List[str], int]:
    if not 0 <= current_frame_index < len(images):
        raise RecoveryPreparationError(
            f"current frame {current_frame_index} outside {len(images)} images"
        )
    history_start = max(0, current_frame_index - history_window_frames + 1)
    return list(images[history_start:current_frame_index + 1]), history_start


def action_words(action_ids: Sequence[int]) -> List[str]:
    return [ACTION_WORDS[int(action)] for action in action_ids]


def base_sample_fields(
    row: Dict[str, Any],
    images: Sequence[str],
    action_ids: Sequence[int],
    start_step: int,
    history_start_frame: int,
    sample_type: str,
) -> Dict[str, Any]:
    return {
        "instruction": str(row["instruction"]),
        "action_sequence": action_words(action_ids),
        "images": list(images),
        "episode_id": str(row["episode_id"]),
        "dataset": f"route_recovery_{row['source_dataset']}",
        "step_index": int(start_step),
        "end_step": int(start_step + len(action_ids) - 1),
        "real_action_count": len(action_ids),
        "trajectory_id": str(row.get("trajectory_id", row["episode_id"])),
        "source_trajectory_id": str(row["source_trajectory_id"]),
        "source_episode_id": str(row["source_episode_id"]),
        "sample_type": sample_type,
        "history_start_frame": int(history_start_frame),
    }


def build_clean_anchor_sample(
    row: Dict[str, Any],
    images: Sequence[str],
    history_window_frames: int,
) -> Dict[str, Any]:
    context = str(row["episode_id"])
    standard_actions = validate_action_ids(row.get("standard_actions"), context)
    anchor_step = int(row["standard_prefix_end"])
    action_ids = standard_actions[anchor_step:anchor_step + ACTION_HORIZON]
    real_action_count = len(action_ids)
    if len(action_ids) < ACTION_HORIZON:
        if not action_ids or action_ids[-1] != STOP_ACTION_ID:
            raise RecoveryPreparationError(
                f"{context}: clean anchor has fewer than four non-terminal actions"
            )
        action_ids.extend(
            [STOP_ACTION_ID] * (ACTION_HORIZON - len(action_ids))
        )
    history, history_start = select_causal_history(
        images,
        current_frame_index=anchor_step,
        history_window_frames=history_window_frames,
    )
    sample = base_sample_fields(
        row,
        images=history,
        action_ids=action_ids,
        start_step=anchor_step,
        history_start_frame=history_start,
        sample_type="clean_anchor",
    )
    sample["real_action_count"] = real_action_count
    return sample


def build_trajectory_samples(
    row: Dict[str, Any],
    input_root: Path,
    dataset_name: str,
    history_window_frames: int,
    include_clean_anchor: bool,
    validate_image_files: bool,
) -> List[Dict[str, Any]]:
    actions, images = validate_raw_trajectory(
        row,
        input_root=input_root,
        dataset_name=dataset_name,
        validate_image_files=validate_image_files,
    )
    recovery_start = int(row["recovery_start"])
    recovery_end = int(row["recovery_end"])
    starts = build_recovery_chunk_starts(
        recovery_start=recovery_start,
        recovery_end=recovery_end,
    )

    samples = []
    if include_clean_anchor:
        samples.append(
            build_clean_anchor_sample(
                row,
                images=images,
                history_window_frames=history_window_frames,
            )
        )
    for start_step in starts:
        action_ids = actions[start_step:start_step + ACTION_HORIZON]
        if len(action_ids) != ACTION_HORIZON:
            raise RecoveryPreparationError(
                f"{row['episode_id']}: incomplete recovery action chunk at {start_step}"
            )
        history, history_start = select_causal_history(
            images,
            current_frame_index=start_step,
            history_window_frames=history_window_frames,
        )
        sample = base_sample_fields(
            row,
            images=history,
            action_ids=action_ids,
            start_step=start_step,
            history_start_frame=history_start,
            sample_type="recovery",
        )
        sample["recovery_step_index"] = start_step - recovery_start
        samples.append(sample)
    return samples


def prepare_training_data(
    input_root: Path,
    dataset_name: str,
    output_path: Path,
    history_window_frames: int,
    include_clean_anchor: bool,
    validate_image_files: bool,
    max_trajectories: int | None,
    overwrite: bool,
    seed: int,
) -> Dict[str, Any]:
    if history_window_frames <= 0:
        raise ValueError(
            f"history_window_frames must be positive, got {history_window_frames}"
        )
    if max_trajectories is not None and max_trajectories <= 0:
        raise ValueError(
            f"max_trajectories must be positive when set, got {max_trajectories}"
        )
    input_root = input_root.resolve()
    if not input_root.is_dir():
        raise NotADirectoryError(input_root)
    if dataset_name in {"", ".", ".."} or Path(dataset_name).name != dataset_name:
        raise ValueError(f"invalid dataset_name: {dataset_name!r}")
    input_path = input_root / "sub_dataset" / f"{dataset_name}.jsonl"
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    progress_path = Path(f"{input_path}.inprogress")
    if progress_path.exists():
        raise RecoveryPreparationError(
            f"recovery collection is still in progress: {progress_path}"
        )
    collection_summary_path = input_path.with_suffix(".summary.json")
    if not collection_summary_path.is_file():
        raise FileNotFoundError(
            f"recovery collection has no final summary: {collection_summary_path}"
        )
    with collection_summary_path.open("r", encoding="utf-8") as handle:
        collection_summary = json.load(handle)
    if collection_summary.get("complete") is not True:
        raise RecoveryPreparationError("recovery collection is not marked complete")
    if collection_summary.get("manifest_sha256") != file_sha256(input_path):
        raise RecoveryPreparationError("recovery manifest hash does not match summary")
    route_outcomes_path = input_path.with_suffix(".route_outcomes.jsonl")
    if not route_outcomes_path.is_file():
        raise FileNotFoundError(
            f"recovery collection has no route ledger: {route_outcomes_path}"
        )
    if collection_summary.get("route_outcomes_sha256") != file_sha256(
        route_outcomes_path
    ):
        raise RecoveryPreparationError("recovery route-ledger hash does not match summary")
    output_path = output_path.resolve()
    if not overwrite and output_path.exists():
        raise FileExistsError(
            "training output already exists; pass --overwrite to replace it: "
            f"{output_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path_string = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        text=True,
    )
    temporary_path = Path(temporary_path_string)
    route_keys: set[Tuple[str, str, str]] = set()
    sample_type_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()
    trajectory_count = 0
    recovery_sample_count = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output_handle:
            expected_trajectories = int(collection_summary.get("samples", 0)) or None
            if max_trajectories is not None and expected_trajectories is not None:
                expected_trajectories = min(expected_trajectories, max_trajectories)
            progress = tqdm(
                total=expected_trajectories,
                desc="recovery training data",
                unit="trajectory",
                dynamic_ncols=True,
            )
            for row in read_jsonl(input_path):
                route_key = physical_route_key(row)
                if route_key in route_keys:
                    raise RecoveryPreparationError(
                        f"duplicate physical recovery route: {route_key}"
                    )
                route_keys.add(route_key)
                samples = build_trajectory_samples(
                    row,
                    input_root=input_root,
                    dataset_name=dataset_name,
                    history_window_frames=history_window_frames,
                    include_clean_anchor=include_clean_anchor,
                    validate_image_files=validate_image_files,
                )
                trajectory_count += 1
                progress.update(1)
                for sample in samples:
                    write_jsonl_row(output_handle, sample)
                    sample_type_counts[sample["sample_type"]] += 1
                    dataset_counts[sample["dataset"]] += 1
                    action_counts.update(sample["action_sequence"])
                    if sample["sample_type"] == "recovery":
                        recovery_sample_count += 1
                if (
                    max_trajectories is not None
                    and trajectory_count >= max_trajectories
                ):
                    break
            progress.close()
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.chmod(temporary_path, 0o644)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    summary = {
        "input_root": str(input_root),
        "dataset_name": dataset_name,
        "input_jsonl": str(input_path),
        "output_path": str(output_path),
        "physical_trajectories": trajectory_count,
        "total_samples": sum(sample_type_counts.values()),
        "recovery_samples": recovery_sample_count,
        "sample_type_counts": dict(sorted(sample_type_counts.items())),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "target_action_counts": dict(sorted(action_counts.items())),
        "history_window_frames": history_window_frames,
        "include_clean_anchor": include_clean_anchor,
        "validate_image_files": validate_image_files,
        "seed": int(seed),
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument(
        "--history_window_frames",
        type=int,
        default=DEFAULT_HISTORY_WINDOW_FRAMES,
        help="Store only the causal frame window that the VLN loader can sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed recorded with the deterministic conversion statistics.",
    )
    parser.add_argument(
        "--include_clean_anchor",
        action="store_true",
        help="Also emit one counterfactual clean action chunk at each branch anchor.",
    )
    parser.add_argument(
        "--validate_image_files",
        action="store_true",
        help="Stat every referenced raw image while building the compact JSONL.",
    )
    parser.add_argument(
        "--max_trajectories",
        type=int,
        default=None,
        help="Optional pilot cap across all input manifests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing training JSONL and summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_training_data(
        input_root=args.input_root,
        dataset_name=args.dataset_name,
        output_path=args.output_path,
        history_window_frames=args.history_window_frames,
        include_clean_anchor=args.include_clean_anchor,
        validate_image_files=args.validate_image_files,
        max_trajectories=args.max_trajectories,
        overwrite=args.overwrite,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
