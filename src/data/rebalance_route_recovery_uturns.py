"""Balance exact recovery U-turns with native Habitat RGB observations.

The recovery collector's shortest-path follower almost always resolves an
exact 180-degree tie by turning left.  For a deterministic subset of physical
routes, this post-process replaces the complete initial ``left x 12`` recovery
prefix with ``right x 12``.  Only the eleven intermediate panoramas differ:
after the twelfth 15-degree turn, position and heading are identical again.

Images are rendered natively in Habitat from the saved recovery-start pose.
The source JPEGs are never geometrically warped.  Per-worker changes remain
recoverable under a hidden progress directory until final manifest validation
succeeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.generate_route_recovery_data import (  # noqa: E402
    DATASET_SPECS,
    LEFT_ACTION,
    RIGHT_ACTION,
    agent_state,
    atomic_write_json,
    atomic_write_jsonl,
    build_env_config,
    file_sha256,
    position_array,
    read_jsonl,
    rotation_list,
    save_rgb,
    silence_external_output,
)

with silence_external_output():
    import habitat
    from habitat_sim.utils.common import quat_from_coeffs


DEFAULT_INPUT_ROOT = Path("/workspace/z_PanoVLN")
DEFAULT_DATASET_NAME = "r2r_rxr_recovery"
DEFAULT_SEED = 42
DEFAULT_IMAGE_WIDTH = 1280
DEFAULT_IMAGE_HEIGHT = 640
EXACT_UTURN_STEPS = 12
INTERMEDIATE_UTURN_STEPS = range(1, EXACT_UTURN_STEPS)
# Habitat stores poses in float32; repeated pure rotations can introduce
# micrometer-scale position jitter even though no translation action occurs.
POSITION_TOLERANCE_METERS = 1e-5
QUATERNION_TOLERANCE = 1e-5
ENDPOINT_IMAGE_MAE_TOLERANCE = 0.5
REBALANCE_VERSION = 1


def physical_route_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(row["source_dataset"]),
        str(row["scene_id"]),
        str(row["source_trajectory_id"]),
    )


def select_right_uturn(row: Dict[str, Any], seed: int) -> bool:
    """Match the established deterministic physical-route hash selection."""

    route_key = "|".join(physical_route_key(row))
    digest = hashlib.sha256(f"{int(seed)}|{route_key}".encode("utf-8")).digest()
    return digest[0] % 2 == 1


def exact_initial_turn_pattern(row: Dict[str, Any]) -> Tuple[int, ...]:
    recovery_start = int(row["recovery_start"])
    return tuple(
        int(action)
        for action in row["actions"][
            recovery_start:recovery_start + EXACT_UTURN_STEPS
        ]
    )


def is_convertible_left_uturn(row: Dict[str, Any], seed: int) -> bool:
    return (
        int(row.get("ambiguous_initial_turn_prefix", 0)) == EXACT_UTURN_STEPS
        and exact_initial_turn_pattern(row) == (LEFT_ACTION,) * EXACT_UTURN_STEPS
        and select_right_uturn(row, seed=seed)
    )


def collection_paths(input_root: Path, dataset_name: str) -> Dict[str, Path]:
    sub_dataset_root = input_root / "sub_dataset"
    return {
        "manifest": sub_dataset_root / f"{dataset_name}.jsonl",
        "summary": sub_dataset_root / f"{dataset_name}.summary.json",
        "images": input_root / "images" / dataset_name,
        "progress": sub_dataset_root / f".{dataset_name}.uturn_rebalance.inprogress",
    }


def worker_plan_path(progress_root: Path, worker_index: int) -> Path:
    return progress_root / "assignments" / f"worker_{worker_index:03d}.jsonl"


def completion_path(progress_root: Path, episode_id: str) -> Path:
    safe_id = str(episode_id)
    if Path(safe_id).name != safe_id:
        raise ValueError(f"unsafe episode id: {episode_id!r}")
    return progress_root / "completed" / f"{safe_id}.json"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def assignment_digest(assignments: Dict[int, Sequence[Dict[str, Any]]]) -> str:
    payload = {
        str(worker_index): [str(row["episode_id"]) for row in rows]
        for worker_index, rows in sorted(assignments.items())
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def balance_scene_assignments(
    selected_rows: Sequence[Dict[str, Any]],
    num_workers: int,
) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        grouped[(str(row["source_dataset"]), str(row["scene_id"]))].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: str(row["episode_id"]))

    assignments = {worker_index: [] for worker_index in range(num_workers)}
    loads = [0] * num_workers
    ordered_groups = sorted(
        grouped.items(),
        key=lambda item: (-len(item[1]), item[0][0], item[0][1]),
    )
    for _, rows in ordered_groups:
        worker_index = min(range(num_workers), key=lambda index: (loads[index], index))
        assignments[worker_index].extend(rows)
        loads[worker_index] += len(rows)
    for rows in assignments.values():
        rows.sort(
            key=lambda row: (
                str(row["source_dataset"]),
                str(row["scene_id"]),
                str(row["episode_id"]),
            )
        )
    return assignments


def uturn_counts(rows: Iterable[Dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        prefix_length = int(row.get("ambiguous_initial_turn_prefix", 0))
        counts[f"prefix_{prefix_length}"] += 1
        if prefix_length != EXACT_UTURN_STEPS:
            continue
        pattern = exact_initial_turn_pattern(row)
        if pattern == (LEFT_ACTION,) * EXACT_UTURN_STEPS:
            counts["exact_left"] += 1
        elif pattern == (RIGHT_ACTION,) * EXACT_UTURN_STEPS:
            counts["exact_right"] += 1
        else:
            counts["exact_other"] += 1
    return counts


def initialize(args: argparse.Namespace, paths: Dict[str, Path]) -> Dict[str, Any]:
    for label in ("manifest", "summary", "images"):
        if not paths[label].exists():
            raise FileNotFoundError(paths[label])
    existing_summary = load_json(paths["summary"])
    existing_rebalance = existing_summary.get("uturn_rebalance", {})
    if existing_rebalance.get("complete") is True:
        raise ValueError("recovery U-turn rebalance is already complete")
    if paths["progress"].exists():
        if not args.resume:
            raise FileExistsError(
                f"unfinished U-turn rebalance exists: {paths['progress']}"
            )
        return load_json(paths["progress"] / "configuration.json")

    rows = read_jsonl(paths["manifest"])
    episode_ids = [str(row["episode_id"]) for row in rows]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("recovery manifest contains duplicate episode ids")
    selected_rows = [
        row for row in rows if is_convertible_left_uturn(row, seed=args.seed)
    ]
    if not selected_rows:
        raise ValueError("no exact left U-turns selected for conversion")
    assignments = balance_scene_assignments(selected_rows, args.num_workers)
    source_manifest_sha256 = file_sha256(paths["manifest"])
    configuration = {
        "rebalance_version": REBALANCE_VERSION,
        "source_manifest_sha256": source_manifest_sha256,
        "dataset_name": args.dataset_name,
        "seed": int(args.seed),
        "num_workers": int(args.num_workers),
        "image_size": [int(args.image_width), int(args.image_height)],
        "selected_trajectories": len(selected_rows),
        "source_counts": dict(sorted(uturn_counts(rows).items())),
        "selection_rule": "sha256(seed|dataset|scene|physical_route)[0] % 2 == 1",
        "assignment_sha256": assignment_digest(assignments),
        "worker_loads": [len(assignments[index]) for index in range(args.num_workers)],
    }

    progress_root = paths["progress"]
    progress_root.mkdir(parents=True)
    try:
        for directory_name in ("completed", "backups", "staging", "assignments"):
            (progress_root / directory_name).mkdir(parents=True, exist_ok=True)
        shutil.copy2(paths["manifest"], progress_root / "source_manifest.jsonl")
        shutil.copy2(paths["summary"], progress_root / "source_summary.json")
        for worker_index, worker_rows in assignments.items():
            atomic_write_jsonl(
                worker_plan_path(progress_root, worker_index), worker_rows
            )
        atomic_write_json(progress_root / "configuration.json", configuration)
    except Exception:
        shutil.rmtree(progress_root, ignore_errors=True)
        raise
    print(json.dumps({"status": "initialized", **configuration}, ensure_ascii=False))
    return configuration


def require_configuration(
    args: argparse.Namespace,
    paths: Dict[str, Path],
) -> Dict[str, Any]:
    configuration = load_json(paths["progress"] / "configuration.json")
    expected = {
        "dataset_name": args.dataset_name,
        "seed": int(args.seed),
        "num_workers": int(args.num_workers),
        "image_size": [int(args.image_width), int(args.image_height)],
    }
    for key, value in expected.items():
        if configuration.get(key) != value:
            raise ValueError(
                f"rebalance configuration mismatch for {key}: "
                f"{configuration.get(key)!r} != {value!r}"
            )
    source_manifest = paths["progress"] / "source_manifest.jsonl"
    if file_sha256(source_manifest) != configuration["source_manifest_sha256"]:
        raise ValueError("saved source recovery manifest hash mismatch")
    return configuration


def quaternion_sign_invariant_error(
    actual: Sequence[float], expected: Sequence[float]
) -> float:
    actual_array = np.asarray(actual, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    return float(
        min(
            np.max(np.abs(actual_array - expected_array)),
            np.max(np.abs(actual_array + expected_array)),
        )
    )


def image_mae(left_path: Path, right_path: Path) -> float:
    with Image.open(left_path) as left_image, Image.open(right_path) as right_image:
        left = np.asarray(left_image.convert("RGB"), dtype=np.int16)
        right = np.asarray(right_image.convert("RGB"), dtype=np.int16)
    if left.shape != right.shape:
        raise ValueError(f"image shape mismatch: {left.shape} != {right.shape}")
    return float(np.mean(np.abs(left - right)))


def restore_incomplete_images(
    row: Dict[str, Any],
    image_directory: Path,
    backup_directory: Path,
) -> None:
    recovery_start = int(row["recovery_start"])
    for step_offset in INTERMEDIATE_UTURN_STEPS:
        backup = backup_directory / f"frame_{recovery_start + step_offset}.jpg"
        destination = image_directory / backup.name
        if backup.is_file():
            shutil.copy2(backup, destination)


def render_balanced_trajectory(
    env,
    episode,
    row: Dict[str, Any],
    paths: Dict[str, Path],
    image_size: Tuple[int, int],
) -> Dict[str, Any]:
    episode_id = str(row["episode_id"])
    recovery_start = int(row["recovery_start"])
    image_directory = paths["images"] / str(row["trajectory_id"])
    if not image_directory.is_dir():
        raise FileNotFoundError(image_directory)
    progress_root = paths["progress"]
    backup_directory = progress_root / "backups" / episode_id
    staging_directory = progress_root / "staging" / episode_id
    completed_path = completion_path(progress_root, episode_id)
    if completed_path.is_file():
        return load_json(completed_path)

    backup_directory.mkdir(parents=True, exist_ok=True)
    for step_offset in INTERMEDIATE_UTURN_STEPS:
        source = image_directory / f"frame_{recovery_start + step_offset}.jpg"
        backup = backup_directory / source.name
        if not source.is_file():
            raise FileNotFoundError(source)
        if not backup.exists():
            shutil.copy2(source, backup)
    restore_incomplete_images(row, image_directory, backup_directory)
    if staging_directory.exists():
        shutil.rmtree(staging_directory)
    staging_directory.mkdir(parents=True)

    env.current_episode = episode
    env.reset()
    start_position = np.asarray(
        row["action_positions"][recovery_start], dtype=np.float32
    )
    start_rotation = quat_from_coeffs(
        np.asarray(row["action_rotations"][recovery_start], dtype=np.float64)
    )
    env.sim.set_agent_state(
        position=start_position,
        rotation=start_rotation,
        reset_sensors=True,
    )
    rendered_rotations: List[List[float]] = []
    max_position_drift = 0.0
    endpoint_frame = staging_directory / "endpoint_step12.jpg"
    for step_offset in range(1, EXACT_UTURN_STEPS + 1):
        observation = env.step(RIGHT_ACTION)
        position, rotation = agent_state(env)
        position_drift = float(np.linalg.norm(position - start_position))
        max_position_drift = max(max_position_drift, position_drift)
        if position_drift > POSITION_TOLERANCE_METERS:
            raise ValueError(
                f"{episode_id}: right U-turn translated by {position_drift:.8f}m"
            )
        rendered_rotations.append(rotation)
        if step_offset < EXACT_UTURN_STEPS:
            destination = staging_directory / (
                f"frame_{recovery_start + step_offset}.jpg"
            )
        else:
            destination = endpoint_frame
        save_rgb(observation, destination, image_size=image_size)

    expected_endpoint_position = np.asarray(
        row["action_positions"][recovery_start + EXACT_UTURN_STEPS],
        dtype=np.float64,
    )
    endpoint_position, endpoint_rotation = agent_state(env)
    endpoint_position_error = float(
        np.linalg.norm(endpoint_position - expected_endpoint_position)
    )
    endpoint_rotation_error = quaternion_sign_invariant_error(
        endpoint_rotation,
        row["action_rotations"][recovery_start + EXACT_UTURN_STEPS],
    )
    if endpoint_position_error > POSITION_TOLERANCE_METERS:
        raise ValueError(
            f"{episode_id}: U-turn endpoint position error "
            f"{endpoint_position_error:.8f}m"
        )
    if endpoint_rotation_error > QUATERNION_TOLERANCE:
        raise ValueError(
            f"{episode_id}: U-turn endpoint rotation error "
            f"{endpoint_rotation_error:.8f}"
        )
    original_endpoint_frame = image_directory / (
        f"frame_{recovery_start + EXACT_UTURN_STEPS}.jpg"
    )
    endpoint_mae = image_mae(endpoint_frame, original_endpoint_frame)
    if endpoint_mae > ENDPOINT_IMAGE_MAE_TOLERANCE:
        raise ValueError(
            f"{episode_id}: right/left 180-degree endpoint image MAE "
            f"{endpoint_mae:.4f} exceeds {ENDPOINT_IMAGE_MAE_TOLERANCE}"
        )

    patched_row = dict(row)
    patched_actions = [int(action) for action in row["actions"]]
    patched_actions[
        recovery_start:recovery_start + EXACT_UTURN_STEPS
    ] = [RIGHT_ACTION] * EXACT_UTURN_STEPS
    patched_row["actions"] = patched_actions
    patched_rotations = [list(rotation) for rotation in row["action_rotations"]]
    for step_offset in INTERMEDIATE_UTURN_STEPS:
        patched_rotations[recovery_start + step_offset] = rendered_rotations[
            step_offset - 1
        ]
    patched_row["action_rotations"] = patched_rotations

    new_image_hashes = {}
    for step_offset in INTERMEDIATE_UTURN_STEPS:
        staged = staging_directory / f"frame_{recovery_start + step_offset}.jpg"
        destination = image_directory / staged.name
        os.replace(staged, destination)
        new_image_hashes[destination.name] = file_sha256(destination)
    completion = {
        "episode_id": episode_id,
        "row": patched_row,
        "new_image_sha256": new_image_hashes,
        "metrics": {
            "max_position_drift_m": max_position_drift,
            "endpoint_position_error_m": endpoint_position_error,
            "endpoint_rotation_error": endpoint_rotation_error,
            "endpoint_image_mae": endpoint_mae,
        },
    }
    atomic_write_json(completed_path, completion)
    shutil.rmtree(staging_directory, ignore_errors=True)
    return completion


def run_worker(args: argparse.Namespace, paths: Dict[str, Path]) -> None:
    require_configuration(args, paths)
    plan_path = worker_plan_path(paths["progress"], args.worker_index)
    rows = read_jsonl(plan_path)
    if not rows:
        print(json.dumps({"status": "worker_complete", "converted": 0}))
        return

    converted = 0
    for dataset_name in DATASET_SPECS:
        dataset_rows = [
            row for row in rows if str(row["source_dataset"]) == dataset_name
        ]
        if not dataset_rows:
            continue
        config = build_env_config(
            dataset_name,
            args.gpu_id,
            args.image_width,
            args.image_height,
        )
        with silence_external_output():
            dataset = habitat.datasets.make_dataset(
                id_dataset=config.habitat.dataset.type,
                config=config.habitat.dataset,
            )
        episode_by_id = {
            str(episode.episode_id): episode for episode in dataset.episodes
        }
        missing = sorted(
            {
                str(row["canonical_episode_id"])
                for row in dataset_rows
                if str(row["canonical_episode_id"]) not in episode_by_id
            }
        )
        if missing:
            raise ValueError(
                f"{dataset_name}: missing Habitat episodes {missing[:10]}"
            )
        dataset.episodes = [
            episode_by_id[str(row["canonical_episode_id"])]
            for row in dataset_rows
        ]
        env = None
        try:
            with silence_external_output():
                env = habitat.Env(config=config.habitat, dataset=dataset)
            progress = tqdm(
                dataset_rows,
                desc=f"worker {args.worker_index:02d} {dataset_name}",
                dynamic_ncols=True,
            )
            for row in progress:
                if not is_convertible_left_uturn(row, seed=args.seed):
                    raise ValueError(
                        f"planned row is no longer a convertible U-turn: "
                        f"{row['episode_id']}"
                    )
                render_balanced_trajectory(
                    env,
                    episode_by_id[str(row["canonical_episode_id"])],
                    row,
                    paths=paths,
                    image_size=(args.image_width, args.image_height),
                )
                converted += 1
                progress.set_postfix(converted=converted)
        finally:
            if env is not None:
                env.close()
    print(
        json.dumps(
            {
                "status": "worker_complete",
                "worker_index": args.worker_index,
                "converted": converted,
            }
        )
    )


def verify_completion_images(
    completion: Dict[str, Any], paths: Dict[str, Path]
) -> None:
    row = completion["row"]
    image_directory = paths["images"] / str(row["trajectory_id"])
    for filename, expected_hash in completion["new_image_sha256"].items():
        image_path = image_directory / filename
        if file_sha256(image_path) != expected_hash:
            raise ValueError(f"completed image hash mismatch: {image_path}")


def image_tree_stats(image_root: Path) -> Tuple[int, int]:
    image_count = 0
    image_bytes = 0
    for image_path in image_root.glob("*/frame_*.jpg"):
        image_count += 1
        image_bytes += image_path.stat().st_size
    return image_count, image_bytes


def finalize(args: argparse.Namespace, paths: Dict[str, Path]) -> Dict[str, Any]:
    configuration = require_configuration(args, paths)
    progress_root = paths["progress"]
    source_rows = read_jsonl(progress_root / "source_manifest.jsonl")
    selected_ids = {
        str(row["episode_id"])
        for row in source_rows
        if is_convertible_left_uturn(row, seed=args.seed)
    }
    if len(selected_ids) != int(configuration["selected_trajectories"]):
        raise ValueError("selected trajectory count changed since initialization")

    completions = {}
    for episode_id in tqdm(
        sorted(selected_ids), desc="validating rendered U-turns", dynamic_ncols=True
    ):
        completed_path = completion_path(progress_root, episode_id)
        if not completed_path.is_file():
            raise FileNotFoundError(f"missing worker completion: {completed_path}")
        completion = load_json(completed_path)
        verify_completion_images(completion, paths)
        completions[episode_id] = completion

    final_rows = []
    for row in source_rows:
        episode_id = str(row["episode_id"])
        if episode_id in completions:
            final_rows.append(completions[episode_id]["row"])
        else:
            final_rows.append(row)
    final_counts = uturn_counts(final_rows)
    if final_counts.get("exact_other", 0):
        raise ValueError("final manifest contains a malformed exact U-turn prefix")
    if final_counts["exact_right"] != (
        configuration["source_counts"].get("exact_right", 0) + len(selected_ids)
    ):
        raise ValueError("final exact-right U-turn count is inconsistent")

    atomic_write_jsonl(paths["manifest"], final_rows)
    summary = load_json(progress_root / "source_summary.json")
    image_count, image_bytes = image_tree_stats(paths["images"])
    summary["image_count"] = image_count
    summary["image_bytes"] = image_bytes
    summary["manifest_sha256"] = file_sha256(paths["manifest"])
    summary["uturn_rebalance"] = {
        "complete": True,
        "version": REBALANCE_VERSION,
        "seed": int(args.seed),
        "method": "native_habitat_rerender_from_saved_recovery_start_pose",
        "selection_rule": configuration["selection_rule"],
        "converted_trajectories": len(selected_ids),
        "rerendered_intermediate_frames": len(selected_ids)
        * len(INTERMEDIATE_UTURN_STEPS),
        "source_manifest_sha256": configuration["source_manifest_sha256"],
        "final_manifest_sha256": summary["manifest_sha256"],
        "source_counts": configuration["source_counts"],
        "final_counts": dict(sorted(final_counts.items())),
    }
    atomic_write_json(paths["summary"], summary)
    result = verify_final(args, paths)
    shutil.rmtree(progress_root)
    print(json.dumps({"status": "finalized", **result}, ensure_ascii=False))
    return result


def verify_final(args: argparse.Namespace, paths: Dict[str, Path]) -> Dict[str, Any]:
    summary = load_json(paths["summary"])
    rebalance = summary.get("uturn_rebalance", {})
    if rebalance.get("complete") is not True:
        raise ValueError("final summary does not record a complete U-turn rebalance")
    if int(rebalance.get("seed", -1)) != int(args.seed):
        raise ValueError("final U-turn rebalance seed mismatch")
    manifest_hash = file_sha256(paths["manifest"])
    if summary.get("manifest_sha256") != manifest_hash:
        raise ValueError("final recovery manifest hash mismatch")
    if rebalance.get("final_manifest_sha256") != manifest_hash:
        raise ValueError("final U-turn rebalance manifest hash mismatch")

    rows = read_jsonl(paths["manifest"])
    counts = uturn_counts(rows)
    expected_counts = rebalance.get("final_counts", {})
    if dict(sorted(counts.items())) != expected_counts:
        raise ValueError("final U-turn distribution differs from summary")
    selected_right = 0
    for row in rows:
        if (
            int(row.get("ambiguous_initial_turn_prefix", 0))
            == EXACT_UTURN_STEPS
            and exact_initial_turn_pattern(row)
            == (RIGHT_ACTION,) * EXACT_UTURN_STEPS
            and select_right_uturn(row, args.seed)
        ):
            selected_right += 1
    if selected_right < int(rebalance["converted_trajectories"]):
        raise ValueError("too few deterministically selected right U-turns")
    if int(summary.get("samples", -1)) != len(rows):
        raise ValueError("final recovery sample count mismatch")
    result = {
        "trajectories": len(rows),
        "converted_trajectories": int(rebalance["converted_trajectories"]),
        "exact_left": counts["exact_left"],
        "exact_right": counts["exact_right"],
        "prefix_9": counts["prefix_9"],
        "prefix_10": counts["prefix_10"],
        "prefix_11": counts["prefix_11"],
        "manifest_sha256": manifest_hash,
    }
    print(json.dumps({"status": "verified", **result}, ensure_ascii=False))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--image_width", type=int, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument("--image_height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--worker_index", type=int, default=0)
    parser.add_argument("--initialize_only", action="store_true")
    parser.add_argument("--finalize_only", action="store_true")
    parser.add_argument("--verify_only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_workers <= 0:
        raise ValueError("num_workers must be positive")
    if not 0 <= args.worker_index < args.num_workers:
        raise ValueError("worker_index is outside num_workers")
    if args.dataset_name in {"", ".", ".."} or Path(args.dataset_name).name != args.dataset_name:
        raise ValueError(f"unsafe dataset name: {args.dataset_name!r}")
    if args.image_width <= 0 or args.image_height <= 0:
        raise ValueError("image dimensions must be positive")
    mode_count = sum((args.initialize_only, args.finalize_only, args.verify_only))
    if mode_count > 1:
        raise ValueError("initialize, finalize and verify modes are mutually exclusive")
    input_root = args.input_root.resolve()
    paths = collection_paths(input_root, args.dataset_name)
    if args.verify_only:
        verify_final(args, paths)
    elif args.initialize_only:
        initialize(args, paths)
    elif args.finalize_only:
        finalize(args, paths)
    else:
        run_worker(args, paths)


if __name__ == "__main__":
    main()
