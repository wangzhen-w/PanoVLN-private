#!/usr/bin/env python3
"""Scene-sharded multi-process orchestration for Habitat expert GT generation."""

from __future__ import annotations

import argparse
import contextlib
import copy
import gzip
import hashlib
import json
import math
import multiprocessing
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional presentation dependency
    tqdm = None

from data_create.trajectory.generate_gt import (
    atomic_write_json_gz,
    episode_id_of,
    finalize_gt_records,
    generate_gt,
    import_habitat_runtime,
    load_final_gt,
    read_journal,
    slice_and_shard_episodes,
    validate_unique_episode_ids,
)


PARALLEL_GT_SCHEMA = "panovln-parallel-gt-v1"


def habitat_process_slots(args: argparse.Namespace) -> List[int]:
    """Expand GPU IDs into one entry per requested Habitat process."""

    raw_ids = getattr(args, "gpu_device_ids", None)
    if raw_ids is None or not str(raw_ids).strip():
        gpu_ids = [int(args.gpu_device_id)]
    else:
        try:
            gpu_ids = [
                int(item.strip())
                for item in str(raw_ids).split(",")
                if item.strip()
            ]
        except ValueError as error:
            raise ValueError("--gpu-device-ids must be comma-separated integers") from error
        if not gpu_ids:
            raise ValueError("--gpu-device-ids must contain at least one GPU ID")
        if len(set(gpu_ids)) != len(gpu_ids):
            raise ValueError("--gpu-device-ids must not contain duplicates")
        if min(gpu_ids) < 0:
            raise ValueError("GPU device IDs must be non-negative")
    processes_per_gpu = int(args.processes_per_gpu)
    if processes_per_gpu < 1:
        raise ValueError("--processes-per-gpu must be >= 1")
    return [gpu_id for gpu_id in gpu_ids for _ in range(processes_per_gpu)]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.flush()
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _run_manifest(args: argparse.Namespace, slots: Sequence[int]) -> Dict[str, Any]:
    dataset_path = Path(args.dataset).resolve()
    config_path = Path(args.config_path).resolve()
    configuration = {
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256_file(dataset_path),
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "scene_root": str(Path(args.scene_root).resolve()),
        "repo_root": str(Path(args.repo_root).resolve()),
        "goal_radius": float(args.goal_radius),
        "max_episode_steps": int(args.max_episode_steps),
        "minimal_observations": bool(args.minimal_observations),
        "start_index": int(args.start_index),
        "max_episodes": args.max_episodes,
        "rank": int(args.rank),
        "world_size": int(args.world_size),
        "gpu_process_slots": [int(slot) for slot in slots],
    }
    encoded = json.dumps(
        configuration,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": PARALLEL_GT_SCHEMA,
        "fingerprint": hashlib.sha256(encoded).hexdigest(),
        "configuration": configuration,
    }


def _episode_cost(episode: Dict[str, Any]) -> float:
    info = episode.get("info") or {}
    try:
        reference_length = float(info.get("reference_length", 0.0))
    except (TypeError, ValueError):
        reference_length = 0.0
    if not math.isfinite(reference_length) or reference_length <= 0.0:
        reference_length = float(max(1, len(episode.get("reference_path") or [])))
    return max(1.0, reference_length)


def balance_scenes(
    episodes: Sequence[Dict[str, Any]],
    worker_count: int,
) -> List[List[Dict[str, Any]]]:
    """Assign each complete scene to one worker using deterministic LPT balancing."""

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for episode in episodes:
        grouped.setdefault(str(episode["scene_id"]), []).append(episode)
    if not grouped:
        return []
    active_workers = min(int(worker_count), len(grouped))
    assignments: List[List[Dict[str, Any]]] = [[] for _ in range(active_workers)]
    loads = [0.0] * active_workers
    scene_groups = [
        (scene_id, scene_episodes, sum(_episode_cost(row) for row in scene_episodes))
        for scene_id, scene_episodes in grouped.items()
    ]
    scene_groups.sort(key=lambda item: (-item[2], item[0]))
    for _, scene_episodes, cost in scene_groups:
        worker_id = min(range(active_workers), key=lambda index: (loads[index], index))
        scene_episodes.sort(
            key=lambda row: str(row.get("trajectory_id", row["episode_id"]))
        )
        assignments[worker_id].extend(scene_episodes)
        loads[worker_id] += cost
    return assignments


def _merge_journals(
    paths: Sequence[Path],
    trajectory_id_by_episode: Dict[str, str],
    repair_trailing_partial: bool,
) -> Tuple[Dict[str, Dict[str, Any]], int]:
    merged: Dict[str, Dict[str, Any]] = {}
    identical_duplicates = 0
    for path in paths:
        records = read_journal(path, repair_trailing_partial=repair_trailing_partial)
        for episode_id, row in records.items():
            expected_trajectory_id = trajectory_id_by_episode.get(episode_id)
            if expected_trajectory_id is None:
                raise ValueError(
                    f"GT journal contains episode_id absent from dataset: {episode_id} ({path})"
                )
            actual_trajectory_id = str(row.get("trajectory_id", expected_trajectory_id))
            if actual_trajectory_id != expected_trajectory_id:
                raise ValueError(
                    "GT journal trajectory_id mismatch for "
                    f"episode_id={episode_id}: expected={expected_trajectory_id}, "
                    f"actual={actual_trajectory_id}, journal={path}"
                )
            previous = merged.get(episode_id)
            if previous is not None:
                if previous.get("data") != row.get("data"):
                    raise ValueError(
                        f"Conflicting GT records for episode_id={episode_id} across journals"
                    )
                identical_duplicates += 1
                continue
            merged[episode_id] = row
    return merged, identical_duplicates


class JournalProgressTracker:
    """Count newly committed JSONL rows without IPC backpressure.

    Worker journals are already the durable resume state, so they are also a
    more accurate progress source than sending one queue message per episode.
    Only complete newline-terminated records advance an offset.
    """

    def __init__(self, paths: Sequence[Path]) -> None:
        self._offsets = {
            path: path.stat().st_size if path.is_file() else 0 for path in paths
        }

    def poll(self) -> int:
        completed = 0
        for path, offset in self._offsets.items():
            if not path.is_file():
                continue
            size = path.stat().st_size
            if size < offset:
                raise RuntimeError(f"GT worker journal was truncated while running: {path}")
            if size == offset:
                continue
            with path.open("rb") as handle:
                handle.seek(offset)
                appended = handle.read()
            last_newline = appended.rfind(b"\n")
            if last_newline < 0:
                continue
            committed = appended[: last_newline + 1]
            completed += committed.count(b"\n")
            self._offsets[path] = offset + last_newline + 1
        return completed


def _worker_entry(worker_args: argparse.Namespace) -> None:
    worker_log = Path(worker_args._worker_log)
    worker_log.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.environ.setdefault("MAGNUM_LOG", "quiet")
        os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
        with worker_log.open("a", encoding="utf-8") as log_handle:
            with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(
                log_handle
            ):
                runtime = import_habitat_runtime(worker_args.repo_root)
                generate_gt(worker_args, runtime)
    except BaseException as error:
        with worker_log.open("a", encoding="utf-8") as log_handle:
            traceback.print_exc(file=log_handle)
        print(
            f"[generate-gt] worker failed ({type(error).__name__}: {error}); "
            f"see {worker_log}",
            file=sys.stderr,
            flush=True,
        )
        raise


def _read_dataset(path: Path) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        dataset = json.load(handle)
    if not isinstance(dataset, dict) or not isinstance(dataset.get("episodes"), list):
        raise ValueError(f"Invalid trajectory dataset: {path}")
    return dataset


def generate_gt_parallel(args: argparse.Namespace, slots: Sequence[int]) -> Dict[str, int]:
    """Generate GT in scene-exclusive workers and atomically merge their journals."""

    started_at = time.time()
    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    base_journal = (
        Path(args.journal)
        if args.journal
        else Path(str(output_path) + ".journal.jsonl")
    )
    worker_root = Path(str(output_path) + ".workers")
    state_path = worker_root / "state.json"
    if args.overwrite:
        output_path.unlink(missing_ok=True)
        base_journal.unlink(missing_ok=True)
        if worker_root.exists():
            shutil.rmtree(worker_root)

    dataset = _read_dataset(dataset_path)
    all_episodes = list(dataset["episodes"])
    if not all_episodes:
        raise ValueError(f"Dataset has no episodes: {dataset_path}")
    all_ids = validate_unique_episode_ids(all_episodes)
    trajectory_id_by_episode = {
        episode_id_of(row): str(row.get("trajectory_id", row["episode_id"]))
        for row in all_episodes
    }
    selected = slice_and_shard_episodes(
        all_episodes,
        args.start_index,
        args.max_episodes,
        args.rank,
        args.world_size,
    )
    if not selected:
        raise ValueError(f"Episode selection is empty: {dataset_path}")
    selected_ids = [episode_id_of(row) for row in selected]

    if output_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"GT output exists; use --resume or --overwrite: {output_path}"
            )
        final = load_final_gt(output_path)
        missing = [episode_id for episode_id in selected_ids if episode_id not in final]
        if missing:
            raise RuntimeError(
                f"Existing formal GT is incomplete/corrupt ({len(missing)} missing); "
                f"use --overwrite after inspection: {output_path}"
            )
        print(f"[generate-gt] already complete: {output_path}", flush=True)
        return {"processed": 0, "skipped": len(selected), "selected": len(selected)}

    expected_manifest = _run_manifest(args, slots)
    if worker_root.exists():
        if not args.resume:
            raise FileExistsError(
                f"Parallel GT state exists; rerun with --resume: {worker_root}"
            )
        if not state_path.is_file():
            raise ValueError(f"Parallel GT state manifest is missing: {state_path}")
        with state_path.open("r", encoding="utf-8") as handle:
            actual_manifest = json.load(handle)
        if actual_manifest != expected_manifest:
            raise ValueError(
                "Parallel GT resume configuration changed. Remove the worker state or "
                f"rerun with its original dataset/settings: {state_path}"
            )
    else:
        worker_root.mkdir(parents=True, exist_ok=False)
        _atomic_json(state_path, expected_manifest)

    worker_journals = sorted(worker_root.glob("worker_*.journal.jsonl"))
    journal_paths = [base_journal, *worker_journals]
    completed, duplicate_count = _merge_journals(
        journal_paths,
        trajectory_id_by_episode,
        repair_trailing_partial=bool(args.resume),
    )
    selected_id_set = set(selected_ids)
    completed_selected = selected_id_set & set(completed)
    pending_id_set = selected_id_set - completed_selected

    assignments = balance_scenes(selected, len(slots))
    instruction_vocab = dict(dataset.get("instruction_vocab", {}))
    context = multiprocessing.get_context("spawn")
    processes: List[multiprocessing.Process] = []
    active_slots: List[int] = []
    worker_logs: Dict[str, Path] = {}
    progress_tracker = JournalProgressTracker(
        [
            worker_root / f"worker_{worker_id:03d}.journal.jsonl"
            for worker_id in range(len(assignments))
        ]
    )
    progress = (
        tqdm(
            total=len(selected),
            initial=len(completed_selected),
            desc="GT",
            unit="ep",
            dynamic_ncols=True,
        )
        if tqdm is not None
        else None
    )
    try:
        for worker_id, assigned_episodes in enumerate(assignments):
            pending_ids = [
                episode_id_of(row)
                for row in assigned_episodes
                if episode_id_of(row) in pending_id_set
            ]
            if not pending_ids:
                continue
            worker_dataset = worker_root / f"worker_{worker_id:03d}.dataset.json.gz"
            worker_journal = worker_root / f"worker_{worker_id:03d}.journal.jsonl"
            worker_output = worker_root / f"worker_{worker_id:03d}.unused.json.gz"
            if not worker_dataset.is_file():
                atomic_write_json_gz(
                    worker_dataset,
                    {
                        "episodes": assigned_episodes,
                        "instruction_vocab": instruction_vocab,
                    },
                )
            worker_args = copy.deepcopy(args)
            worker_args.dataset = str(worker_dataset)
            worker_args.output = str(worker_output)
            worker_args.journal = str(worker_journal)
            worker_args.gpu_device_id = int(slots[worker_id])
            worker_args.gpu_device_ids = None
            worker_args.processes_per_gpu = 1
            worker_args.resume = True
            worker_args.overwrite = False
            worker_args.keep_jsonl = True
            worker_args.skip_finalize = True
            worker_args.allow_partial_finalize = False
            worker_args.rank = 0
            worker_args.world_size = 1
            worker_args.start_index = 0
            worker_args.max_episodes = None
            worker_args.sort_by_scene = True
            worker_args._assigned_episode_ids = pending_ids
            worker_args._disable_progress = True
            worker_args._quiet_output = True
            worker_args._worker_log = str(
                worker_root / f"worker_{worker_id:03d}.log"
            )
            process = context.Process(
                target=_worker_entry,
                args=(worker_args,),
                name=f"gt-gpu{worker_args.gpu_device_id}-worker{worker_id}",
            )
            process.start()
            processes.append(process)
            active_slots.append(int(slots[worker_id]))
            worker_logs[process.name] = Path(worker_args._worker_log)

        # Worker datasets now own the full episode payloads. Keep only IDs and
        # trajectory IDs in the coordinator while child Habitat processes run.
        del assigned_episodes, assignments, selected, all_episodes, dataset

        while any(process.is_alive() for process in processes):
            newly_completed = progress_tracker.poll()
            if progress is not None and newly_completed:
                progress.update(newly_completed)
            failed = [
                process
                for process in processes
                if process.exitcode not in (None, 0)
            ]
            if failed:
                raise RuntimeError(
                    "Parallel GT worker failed: "
                    + ", ".join(
                        f"{process.name}={process.exitcode} "
                        f"(log={worker_logs[process.name]})"
                        for process in failed
                    )
                )
            if progress is not None:
                progress.set_postfix(
                    workers=len(processes),
                    alive=sum(process.is_alive() for process in processes),
                    gpus=",".join(str(gpu) for gpu in sorted(set(active_slots))),
                    refresh=False,
                )
            time.sleep(0.25)

        for process in processes:
            process.join()
        newly_completed = progress_tracker.poll()
        if progress is not None and newly_completed:
            progress.update(newly_completed)
        failed = [process for process in processes if process.exitcode != 0]
        if failed:
            raise RuntimeError(
                "Parallel GT worker failed: "
                + ", ".join(
                    f"{process.name}={process.exitcode} "
                    f"(log={worker_logs[process.name]})"
                    for process in failed
                )
            )
    except (Exception, KeyboardInterrupt):
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=10)
        raise
    finally:
        if progress is not None:
            progress.close()

    worker_journals = sorted(worker_root.glob("worker_*.journal.jsonl"))
    completed, duplicate_count = _merge_journals(
        [base_journal, *worker_journals],
        trajectory_id_by_episode,
        repair_trailing_partial=True,
    )
    missing = [episode_id for episode_id in selected_ids if episode_id not in completed]
    if missing:
        raise RuntimeError(
            f"Parallel GT completed with {len(missing)} missing episodes; "
            f"examples={missing[:10]}"
        )

    if not args.skip_finalize:
        finalize_gt_records(completed, selected_ids, output_path)
        if not args.keep_jsonl:
            base_journal.unlink(missing_ok=True)
            shutil.rmtree(worker_root)

    summary = {
        "schema_version": PARALLEL_GT_SCHEMA,
        "status": "complete" if not args.skip_finalize else "journal_complete",
        "dataset_episodes": len(all_ids),
        "selected": len(selected_ids),
        "processed": len(selected_id_set - completed_selected),
        "resumed": len(completed_selected),
        "workers": len(processes),
        "gpu_process_slots": active_slots,
        "identical_duplicate_records": duplicate_count,
        "output": None if args.skip_finalize else str(output_path),
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return {
        "processed": int(summary["processed"]),
        "skipped": int(summary["resumed"]),
        "selected": len(selected_ids),
    }
