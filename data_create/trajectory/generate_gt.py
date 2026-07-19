#!/usr/bin/env python3
"""Generate VLN-CE expert actions for trajectory datasets.

This module is the generic, ScaleVLN-independent GT entry point used by the
self-collected HM3D data pipeline.  Episodes are grouped by scene so Habitat
loads each scene once.  A JSONL journal is the resumable source of truth; the
formal ``json.gz`` file is only published after every selected episode has a
record and is replaced atomically.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from data_create.trajectory.scene_paths import local_scene_id

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional presentation dependency
    tqdm = None


DEFAULT_SCENE_ROOT = "/workspace/data1/dataset/general_VLN_data/HM3D"
DEFAULT_CONFIG_PATH = "/workspace/code/VLN/data_create/config/hm3d_vln.yaml"
DEFAULT_REPO_ROOT = "/workspace/code/VLN"
STOP_ACTION = 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Collected trajectory JSON.GZ.")
    parser.add_argument("--output", required=True, help="Expert GT JSON.GZ output.")
    parser.add_argument(
        "--journal",
        default=None,
        help="Resume journal; defaults to OUTPUT.journal.jsonl.",
    )
    parser.add_argument(
        "--scene-root",
        default=DEFAULT_SCENE_ROOT,
        help="The same split-aware HM3D root used during trajectory collection.",
    )
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--repo-root", default=DEFAULT_REPO_ROOT)
    parser.add_argument("--goal-radius", type=float, default=0.3)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument(
        "--gpu-device-ids",
        default=None,
        help=(
            "Comma-separated Habitat GPU IDs. Multiple process slots enable the "
            "scene-sharded GT coordinator."
        ),
    )
    parser.add_argument(
        "--processes-per-gpu",
        type=int,
        default=1,
        help="Independent scene-batched GT processes assigned to each GPU.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from a valid JSONL journal, or accept an already-complete output.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove this module's existing GT output/journal and start again.",
    )
    parser.add_argument(
        "--keep-jsonl",
        action="store_true",
        help="Retain the resume journal after successful finalization.",
    )
    parser.add_argument(
        "--skip-finalize",
        action="store_true",
        help="Generate/append journal records without publishing the formal json.gz.",
    )
    parser.add_argument(
        "--sort-by-scene",
        action="store_true",
        help="Sort scenes and trajectories deterministically (scene batching is always on).",
    )
    parser.add_argument(
        "--minimal-observations",
        action="store_true",
        help="Disable render and task sensors while generating actions.",
    )
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--gt-log-every", type=int, default=200)
    parser.add_argument(
        "--allow-partial-finalize",
        action="store_true",
        help=(
            "Publish only the selected slice/shard. Normally slices/shards must use "
            "--skip-finalize so an incomplete GT file cannot look production-ready."
        ),
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not Path(args.scene_root).is_dir():
        raise FileNotFoundError(f"Scene root does not exist: {args.scene_root}")
    dataset_path = Path(args.dataset).resolve()
    output_path = Path(args.output).resolve()
    journal_path = (
        Path(args.journal).resolve()
        if args.journal
        else Path(str(output_path) + ".journal.jsonl")
    )
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")
    if len({dataset_path, output_path, journal_path}) != 3:
        raise ValueError("Dataset, GT output, and journal paths must be distinct")
    if args.world_size < 1:
        raise ValueError("--world-size must be >= 1")
    if not 0 <= args.rank < args.world_size:
        raise ValueError("--rank must satisfy 0 <= rank < world-size")
    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0")
    if args.max_episodes is not None and args.max_episodes < 1:
        raise ValueError("--max-episodes must be >= 1")
    if args.goal_radius <= 0:
        raise ValueError("--goal-radius must be > 0")
    if args.processes_per_gpu < 1:
        raise ValueError("--processes-per-gpu must be >= 1")
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    is_partial_selection = (
        args.start_index != 0
        or args.max_episodes is not None
        or args.rank != 0
        or args.world_size != 1
    )
    if (
        is_partial_selection
        and not args.skip_finalize
        and not args.allow_partial_finalize
    ):
        raise ValueError(
            "A slice/shard cannot publish a formal GT file by default. Use "
            "--skip-finalize, or explicitly pass --allow-partial-finalize."
        )


def import_habitat_runtime(repo_root: str):
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import habitat  # noqa: WPS433
    from habitat.config import read_write  # noqa: WPS433
    from habitat.config.default import get_config  # noqa: WPS433
    from habitat.tasks.nav.shortest_path_follower import (  # noqa: WPS433
        ShortestPathFollower,
    )

    import habitat_extensions.measures  # noqa: F401,WPS433
    import habitat_extensions.task  # noqa: F401,WPS433

    return habitat, get_config, read_write, ShortestPathFollower


def slice_and_shard_episodes(
    episodes: Sequence[Dict[str, Any]],
    start_index: int,
    max_episodes: Optional[int],
    rank: int,
    world_size: int,
) -> List[Dict[str, Any]]:
    selected = list(episodes[start_index:])
    if max_episodes is not None:
        selected = selected[:max_episodes]
    return selected[rank::world_size]


def group_episodes_by_scene(
    episodes: Sequence[Dict[str, Any]],
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for episode in episodes:
        grouped.setdefault(str(episode["scene_id"]), []).append(episode)
    return list(grouped.items())


def positions_equal(
    first: Sequence[float],
    second: Sequence[float],
    tolerance: float = 1e-5,
) -> bool:
    return len(first) == len(second) == 3 and all(
        abs(float(left) - float(right)) <= tolerance
        for left, right in zip(first, second)
    )


def episode_id_of(episode: Any) -> str:
    if isinstance(episode, dict):
        return str(episode["episode_id"])
    return str(episode.episode_id)


def validate_unique_episode_ids(episodes: Sequence[Dict[str, Any]]) -> List[str]:
    ids = [episode_id_of(episode) for episode in episodes]
    seen: Set[str] = set()
    duplicates: Set[str] = set()
    for episode_id in ids:
        if episode_id in seen:
            duplicates.add(episode_id)
        seen.add(episode_id)
    if duplicates:
        examples = sorted(duplicates)[:10]
        raise ValueError(f"Dataset contains duplicate episode_id values: {examples}")
    return ids


def read_journal(path: Path, repair_trailing_partial: bool) -> Dict[str, Dict[str, Any]]:
    """Load unique records and optionally remove one crash-truncated final line."""

    if not path.exists():
        return {}
    raw_lines = path.read_bytes().splitlines(keepends=True)
    records: Dict[str, Dict[str, Any]] = {}
    valid_bytes = 0
    for index, raw_line in enumerate(raw_lines):
        if not raw_line.strip():
            valid_bytes += len(raw_line)
            continue
        try:
            row = json.loads(raw_line.decode("utf-8"))
            episode_id = str(row["episode_id"])
            if episode_id in records:
                raise ValueError(f"Duplicate episode_id={episode_id} in {path}")
            if not isinstance(row.get("data"), dict):
                raise ValueError(f"Missing data object for episode_id={episode_id} in {path}")
            records[episode_id] = row
            valid_bytes += len(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            is_last = index == len(raw_lines) - 1
            if not repair_trailing_partial or not is_last:
                raise ValueError(f"Invalid JSONL journal at line {index + 1}: {path}") from exc
            with path.open("r+b") as handle:
                handle.truncate(valid_bytes)
            break
    return records


def append_journal_record(handle, row: Dict[str, Any]) -> None:
    # One write plus flush keeps a completed record visible to a subsequent
    # --resume invocation. read_journal repairs a crash-truncated final write.
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def atomic_write_json_gz(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def finalize_gt_records(
    journal_records: Dict[str, Dict[str, Any]],
    expected_episode_ids: Sequence[str],
    output_path: Path,
) -> None:
    missing = [
        episode_id
        for episode_id in expected_episode_ids
        if episode_id not in journal_records
    ]
    if missing:
        raise RuntimeError(
            f"Refusing to finalize incomplete GT: missing {len(missing)} episodes, "
            f"examples={missing[:10]}"
        )
    # Dataset order, rather than journal/scene order, makes output deterministic.
    payload = {
        episode_id: journal_records[episode_id]["data"]
        for episode_id in expected_episode_ids
    }
    atomic_write_json_gz(output_path, payload)


def load_final_gt(path: Path) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"GT output must be a JSON object: {path}")
    return {str(key): value for key, value in payload.items()}


def generate_episode_gt(
    env,
    episode,
    follower_cls,
    goal_radius: float,
) -> Dict[str, Any]:
    env.current_episode = episode
    env.reset()
    follower = follower_cls(
        env.sim,
        goal_radius=goal_radius,
        return_one_hot=False,
        stop_on_error=True,
    )

    actions: List[int] = []
    locations: List[List[float]] = [
        [float(value) for value in env.sim.get_agent_state().position.tolist()]
    ]
    reference_path = [[float(value) for value in point] for point in episode.reference_path]
    if not reference_path:
        raise ValueError(f"episode_id={episode.episode_id} has an empty reference_path")
    waypoint_index = 1 if len(reference_path) > 1 else 0

    while not env.episode_over and waypoint_index < len(reference_path):
        next_action = follower.get_next_action(reference_path[waypoint_index])
        if next_action is None:
            raise RuntimeError(
                f"ShortestPathFollower failed for episode_id={episode.episode_id}, "
                f"waypoint_index={waypoint_index}"
            )

        while int(next_action) == STOP_ACTION:
            waypoint_index += 1
            if waypoint_index >= len(reference_path):
                break
            next_action = follower.get_next_action(reference_path[waypoint_index])
            if next_action is None:
                raise RuntimeError(
                    f"ShortestPathFollower failed for episode_id={episode.episode_id}, "
                    f"waypoint_index={waypoint_index}"
                )

        if waypoint_index >= len(reference_path):
            break

        env.step(next_action)
        actions.append(int(next_action))
        position = [
            float(value) for value in env.sim.get_agent_state().position.tolist()
        ]
        if not positions_equal(position, locations[-1]):
            locations.append(position)

    if waypoint_index < len(reference_path):
        raise RuntimeError(
            f"Episode ended before all reference waypoints were reached: "
            f"episode_id={episode.episode_id}, reached={waypoint_index}/{len(reference_path)}"
        )

    return {
        "locations": locations,
        "actions": actions,
        "forward_steps": len(locations) - 1,
    }


def scene_batch_path(work_directory: Path, scene_id: str) -> Path:
    digest = hashlib.sha256(scene_id.encode("utf-8")).hexdigest()[:12]
    directory = work_directory / ".gt_scene_batches"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{digest}.{os.getpid()}.json.gz"


def write_scene_dataset(
    path: Path,
    episodes: Sequence[Dict[str, Any]],
    instruction_vocab: Dict[str, Any],
) -> None:
    localized_episodes = []
    for episode in episodes:
        localized = dict(episode)
        localized["scene_id"] = local_scene_id(str(episode["scene_id"]))
        localized_episodes.append(localized)
    atomic_write_json_gz(
        path,
        {"episodes": localized_episodes, "instruction_vocab": instruction_vocab},
    )


def configure_environment(
    get_config,
    read_write,
    args: argparse.Namespace,
    scene_dataset_path: Path,
):
    config = get_config(args.config_path)
    with read_write(config):
        config.habitat.dataset.data_path = str(scene_dataset_path)
        config.habitat.dataset.scenes_dir = args.scene_root
        config.habitat.environment.iterator_options.shuffle = False
        config.habitat.environment.max_episode_steps = args.max_episode_steps
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = args.gpu_device_id
        config.habitat.task.measurements.success.success_distance = args.goal_radius
        if args.minimal_observations:
            config.habitat.simulator.agents.main_agent.sim_sensors = {}
            config.habitat.task.lab_sensors = {}
    return config


def generate_gt(args: argparse.Namespace, habitat_runtime: Tuple) -> Dict[str, int]:
    habitat, get_config, read_write, follower_cls = habitat_runtime
    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    journal_path = (
        Path(args.journal)
        if args.journal
        else Path(str(output_path) + ".journal.jsonl")
    )
    work_directory = output_path.parent

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Missing dataset: {dataset_path}")
    if args.overwrite:
        output_path.unlink(missing_ok=True)
        journal_path.unlink(missing_ok=True)
    elif output_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"GT output exists; use --resume or --overwrite: {output_path}"
            )
    elif journal_path.exists() and not args.resume:
        raise FileExistsError(f"GT journal exists; use --resume or --overwrite: {journal_path}")

    with gzip.open(dataset_path, "rt", encoding="utf-8") as handle:
        dataset = json.load(handle)
    all_episodes = list(dataset.get("episodes", []))
    if not all_episodes:
        raise ValueError(f"Dataset has no episodes: {dataset_path}")
    all_ids = validate_unique_episode_ids(all_episodes)
    assigned_episode_ids = getattr(args, "_assigned_episode_ids", None)
    if assigned_episode_ids is None:
        selected = slice_and_shard_episodes(
            all_episodes,
            args.start_index,
            args.max_episodes,
            args.rank,
            args.world_size,
        )
    else:
        assigned_id_set = {str(episode_id) for episode_id in assigned_episode_ids}
        selected = [
            episode
            for episode in all_episodes
            if episode_id_of(episode) in assigned_id_set
        ]
        missing_assignments = assigned_id_set - {
            episode_id_of(episode) for episode in selected
        }
        if missing_assignments:
            raise ValueError(
                "GT worker assignment contains episode IDs absent from its dataset: "
                f"{sorted(missing_assignments)[:10]}"
            )
    if not selected:
        raise ValueError(f"Episode selection is empty: {dataset_path}")
    if args.sort_by_scene:
        selected.sort(
            key=lambda episode: (
                str(episode["scene_id"]),
                str(episode.get("trajectory_id", episode["episode_id"])),
            )
        )
    selected_ids = [episode_id_of(episode) for episode in selected]
    selection_is_full = len(selected_ids) == len(all_ids) and set(selected_ids) == set(all_ids)
    expected_final_ids = all_ids if selection_is_full else selected_ids

    if output_path.exists():
        final = load_final_gt(output_path)
        missing = [episode_id for episode_id in expected_final_ids if episode_id not in final]
        if missing:
            raise RuntimeError(
                f"Existing formal GT is incomplete/corrupt ({len(missing)} missing); "
                f"use --overwrite after inspection: {output_path}"
            )
        print(f"[generate-gt] already complete: {output_path}", flush=True)
        return {"processed": 0, "skipped": len(selected), "selected": len(selected)}

    journal_records = read_journal(journal_path, repair_trailing_partial=args.resume)
    dataset_id_set = set(all_ids)
    unknown = sorted(set(journal_records) - dataset_id_set)
    if unknown:
        raise ValueError(
            f"GT journal contains episode IDs absent from dataset: {unknown[:10]}"
        )

    scene_batches = group_episodes_by_scene(selected)
    if args.sort_by_scene:
        scene_batches.sort(key=lambda item: item[0])
    work_directory.mkdir(parents=True, exist_ok=True)
    processed = 0
    skipped = sum(episode_id in journal_records for episode_id in selected_ids)
    started_at = time.time()
    progress = (
        tqdm(
            total=len(selected),
            initial=skipped,
            desc="GT",
            unit="ep",
            dynamic_ncols=True,
        )
        if tqdm is not None and not bool(getattr(args, "_disable_progress", False))
        else None
    )

    try:
        with journal_path.open("a", encoding="utf-8") as journal_handle:
            for scene_id, scene_episodes in scene_batches:
                pending = [
                    episode
                    for episode in scene_episodes
                    if episode_id_of(episode) not in journal_records
                ]
                if not pending:
                    continue
                batch_path = scene_batch_path(work_directory, scene_id)
                write_scene_dataset(
                    batch_path,
                    pending,
                    dict(dataset.get("instruction_vocab", {})),
                )
                try:
                    config = configure_environment(
                        get_config, read_write, args, batch_path
                    )
                    env = habitat.Env(config=config)
                    try:
                        emitted: Set[str] = set()
                        for episode in env.episodes:
                            episode_id = episode_id_of(episode)
                            record = generate_episode_gt(
                                env,
                                episode,
                                follower_cls,
                                args.goal_radius,
                            )
                            row = {
                                "episode_id": episode_id,
                                "trajectory_id": str(episode.trajectory_id),
                                "data": record,
                            }
                            append_journal_record(journal_handle, row)
                            journal_records[episode_id] = row
                            emitted.add(episode_id)
                            processed += 1
                            if progress is not None:
                                progress.update(1)
                                progress.set_postfix(
                                    scene=Path(scene_id).stem,
                                    done=f"{processed + skipped}/{len(selected)}",
                                    refresh=False,
                                )
                            if (
                                processed % args.gt_log_every == 0
                                and not bool(getattr(args, "_quiet_output", False))
                            ):
                                print(
                                    f"[generate-gt] scene={scene_id} processed={processed} "
                                    f"skipped={skipped} elapsed={time.time() - started_at:.1f}s",
                                    flush=True,
                                )
                        pending_ids = {episode_id_of(episode) for episode in pending}
                        if emitted != pending_ids:
                            raise RuntimeError(
                                f"Habitat episode loading mismatch for scene={scene_id}: "
                                f"missing={sorted(pending_ids - emitted)[:10]}, "
                                f"unexpected={sorted(emitted - pending_ids)[:10]}"
                            )
                    finally:
                        env.close()
                finally:
                    batch_path.unlink(missing_ok=True)

        if not args.skip_finalize:
            finalize_gt_records(journal_records, expected_final_ids, output_path)
            if not args.keep_jsonl:
                journal_path.unlink()
            if not bool(getattr(args, "_quiet_output", False)):
                print(
                    f"[generate-gt] atomically finalized {output_path} "
                    f"processed={processed} skipped={skipped} "
                    f"elapsed={time.time() - started_at:.1f}s",
                    flush=True,
                )
        return {"processed": processed, "skipped": skipped, "selected": len(selected)}
    finally:
        if progress is not None:
            progress.close()
        batch_directory = work_directory / ".gt_scene_batches"
        try:
            if batch_directory.exists() and not any(batch_directory.iterdir()):
                batch_directory.rmdir()
        except OSError:
            # Another GT worker may create or remove a batch file concurrently.
            pass


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    from data_create.trajectory.parallel_gt import (  # noqa: WPS433
        generate_gt_parallel,
        habitat_process_slots,
    )

    slots = habitat_process_slots(args)
    if len(slots) == 1:
        args.gpu_device_id = int(slots[0])
        runtime = import_habitat_runtime(args.repo_root)
        generate_gt(args, runtime)
    else:
        generate_gt_parallel(args, slots)


if __name__ == "__main__":
    main()
