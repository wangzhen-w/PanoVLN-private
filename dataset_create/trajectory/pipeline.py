#!/usr/bin/env python3
"""End-to-end HM3D navigation-region and decision-rich route collection."""

from __future__ import annotations

import argparse
import copy
import contextlib
import gzip
import json
import math
import multiprocessing
import os
import shutil
import sys
import tempfile
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from tqdm.auto import tqdm

from dataset_create.trajectory import SCHEMA_VERSION
from dataset_create.trajectory.decisions import mine_decision_relations
from dataset_create.trajectory.hm3d import (
    discover_scenes,
    make_simulator,
    package_versions,
    resolve_scene_path,
    shortest_path,
)
from dataset_create.trajectory.io_utils import (
    atomic_json_dump,
    atomic_json_gz_dump,
    load_json,
    load_json_gz,
    stable_id,
)
from dataset_create.trajectory.quality import compact_trajectory
from dataset_create.trajectory.regions import build_region_graph, partition_navigation_regions
from dataset_create.trajectory.replay import replay_episode_actions
from dataset_create.trajectory.routes import collect_routes
from dataset_create.trajectory.schema import validate_dataset
from dataset_create.trajectory.visualize import render_scene_visualizations


DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "default.json"


def _default_run_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["output_root"])


def _scene_output_dir(run_root: Path, record: dict[str, str]) -> Path:
    return run_root / record["scene_key"]


def _config_id(config: dict[str, Any]) -> str:
    normalized = copy.deepcopy(config)
    normalized.get("runtime", {}).pop("gpu_device_id", None)
    return stable_id("config", normalized, length=20)


def _collection_diagnosis(
    scene_graph: dict[str, Any],
    decision_stats: dict[str, Any],
    route_stats: dict[str, Any],
    navigable_area: float,
) -> dict[str, Any]:
    graph = build_region_graph(scene_graph)
    import networkx as nx

    components = sorted(
        (len(component) for component in nx.connected_components(graph)), reverse=True
    )
    # MultiGraph degree counts parallel portals into one neighboring region as
    # separate branches. Decision topology is defined over distinct neighbors.
    degrees = [len(set(graph.neighbors(node))) for node in graph.nodes]
    junction_count = sum(degree >= 3 for degree in degrees)
    valid_relations = int(decision_stats["valid_relations"])
    route_groups = int(route_stats.get("region_level_route_groups", 0))
    replay_verified = int(route_stats.get("replay_verified_trajectory_count", 0))
    quality_eligible = int(route_stats.get("quality_eligible_trajectory_count", 0))
    final_count = int(route_stats.get("final_trajectory_count", 0))
    if junction_count == 0:
        outcome = "no_region_graph_junction"
    elif valid_relations == 0:
        outcome = "junctions_fail_decision_geometry_or_visibility"
    elif route_groups == 0:
        outcome = "valid_relations_not_used_by_natural_shortest_paths"
    elif replay_verified == 0:
        outcome = "natural_routes_fail_primitive_replay_verification"
    elif quality_eligible == 0:
        outcome = "replayed_routes_fail_quality_gates"
    elif final_count < 10:
        outcome = "valid_but_structurally_sparse"
    else:
        outcome = "healthy"
    return {
        "outcome": outcome,
        "partition": {
            "connected_component_count": len(components),
            "component_region_counts": components,
            "junction_region_count": junction_count,
            "maximum_region_degree": max(degrees, default=0),
            "regions_per_100_m2": round(
                100.0 * len(scene_graph["regions"]) / max(navigable_area, 1e-6), 3
            ),
        },
        "decision": {
            "relation_proposals": int(decision_stats["relation_proposals"]),
            "valid_relations": valid_relations,
            "rejections": decision_stats["rejected"],
        },
        "route": {
            "available_endpoint_region_pairs": int(
                route_stats.get("available_endpoint_region_pairs", 0)
            ),
            "sampled_endpoint_region_pairs": int(
                route_stats.get("sampled_endpoint_region_pairs", 0)
            ),
            "region_level_route_groups": route_groups,
            "pre_replay_quality_eligible_group_count": int(
                route_stats.get("pre_replay_quality_eligible_group_count", 0)
            ),
            "pre_replay_quality_rejections": route_stats.get(
                "pre_replay_quality_rejections", {}
            ),
            "route_family_count": int(route_stats.get("route_family_count", 0)),
            "family_replay_attempted_group_count": int(
                route_stats.get("family_replay_attempted_group_count", 0)
            ),
            "replay_verified_trajectory_count": replay_verified,
            "quality_eligible_trajectory_count": quality_eligible,
            "replay_quality_rejections": route_stats.get(
                "replay_quality_rejections", {}
            ),
            "pre_replay_family_redundancy_rejections": int(
                route_stats.get("pre_replay_family_redundancy_rejections", 0)
            ),
            "replay_family_redundancy_rejections": int(
                route_stats.get("replay_family_redundancy_rejections", 0)
            ),
            "route_family_replay_failure_count": int(
                route_stats.get("route_family_replay_failure_count", 0)
            ),
            "replay_family_remapping_count": int(
                route_stats.get("replay_family_remapping_count", 0)
            ),
            "replay_rejections": route_stats.get("replay_rejections", {}),
        },
    }


@contextlib.contextmanager
def _silence_native_worker_output():
    """Hide per-scene Magnum logs; worker failures are returned as tracebacks."""

    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    with open(os.devnull, "w", encoding="utf-8") as null:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(null.fileno(), 1)
            os.dup2(null.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)


def _terminate_executors(executors: Sequence[ProcessPoolExecutor]) -> None:
    """Terminate active workers immediately after an interactive interrupt."""

    processes = []
    for executor in executors:
        process_map = getattr(executor, "_processes", None) or {}
        processes.extend(process_map.values())
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=2.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=1.0)
    for executor in executors:
        executor.shutdown(wait=False, cancel_futures=True)


def process_scene(
    record: dict[str, str],
    config: dict[str, Any],
    run_root: str | Path,
    *,
    visualization_root: str | Path | None,
    overwrite: bool,
) -> dict[str, Any]:
    started = time.time()
    output_dir = _scene_output_dir(Path(run_root), record)
    stats_path = output_dir / "stats.json"
    trajectory_path = output_dir / "trajectories.json.gz"
    current_config_id = _config_id(config)
    if stats_path.is_file() and trajectory_path.is_file() and not overwrite:
        existing = load_json(stats_path)
        if (
            existing.get("schema_version") == SCHEMA_VERSION
            and existing.get("config_id") == current_config_id
        ):
            return {**existing, "status": "resumed"}

    simulator = make_simulator(
        record["glb_path"],
        config,
        with_sensors=True,
        enable_physics=bool(config["runtime"].get("enable_physics_for_ray_cast", False)),
    )
    try:
        pathfinder = simulator.pathfinder
        if not pathfinder.is_loaded:
            if not pathfinder.load_nav_mesh(record["navmesh_path"]):
                raise RuntimeError(f"Cannot load NavMesh: {record['navmesh_path']}")
        partition_started = time.time()
        scene_graph_core, arrays = partition_navigation_regions(
            pathfinder, record["scene_id"], config
        )
        partition_seconds = time.time() - partition_started
        decision_started = time.time()
        relations, decision_stats = mine_decision_relations(
            simulator,
            scene_graph_core,
            arrays,
            record["scene_id"],
            config,
        )
        decision_seconds = time.time() - decision_started
        scene_graph_core["decision_relations"] = relations
        scene_graph_core["decision_stats"] = decision_stats
        route_started = time.time()
        trajectories, route_stats, dedup_examples = collect_routes(
            simulator,
            scene_graph_core,
            arrays,
            relations,
            decision_stats,
            record["scene_id"],
            config,
        )
        route_seconds = time.time() - route_started
        compact_trajectories = [
            compact_trajectory(trajectory) for trajectory in trajectories
        ]
        scene_dataset = {
            "schema_version": SCHEMA_VERSION,
            "metadata": {
                "kind": "single_scene_trajectory_shard",
                "scene": {
                    "scene_key": record["scene_key"],
                    "scene_id": record["scene_id"],
                },
                "config_id": current_config_id,
                "action_mapping": {
                    "stop": 0,
                    "move_forward": 1,
                    "turn_left": 2,
                    "turn_right": 3,
                },
            },
            "episodes": compact_trajectories,
        }
        validate_dataset(scene_dataset)
        visualization_manifest = {}
        if visualization_root is not None:
            visualization_manifest = render_scene_visualizations(
                simulator,
                scene_graph_core,
                arrays,
                trajectories,
                dedup_examples,
                Path(visualization_root) / record["scene_key"],
            )
        stats = {
            "schema_version": SCHEMA_VERSION,
            "kind": "scene_collection_stats",
            "status": "success",
            "scene_key": record["scene_key"],
            "scene_id": record["scene_id"],
            "config_id": current_config_id,
            "elapsed_seconds": time.time() - started,
            "module_seconds": {
                "partition": partition_seconds,
                "decision_relations": decision_seconds,
                "route_collection": route_seconds,
            },
            "navigable_area": float(pathfinder.navigable_area),
            "region_count": scene_graph_core["region_count"],
            "connection_count": scene_graph_core["connection_count"],
            "decision_relation_count": len(relations),
            "region_level_route_groups": int(
                route_stats.get("region_level_route_groups", 0)
            ),
            "replay_verified_trajectory_count": int(
                route_stats.get("replay_verified_trajectory_count", 0)
            ),
            "quality_eligible_trajectory_count": int(
                route_stats.get("quality_eligible_trajectory_count", 0)
            ),
            "final_trajectory_count": len(compact_trajectories),
            "decision_events_per_trajectory": route_stats[
                "decision_events_per_trajectory"
            ],
            "route_structure_retention_rate": route_stats[
                "route_structure_retention_rate"
            ],
            "diagnosis": _collection_diagnosis(
                scene_graph_core,
                decision_stats,
                route_stats,
                float(pathfinder.navigable_area),
            ),
        }
        if visualization_manifest:
            stats["visualizations"] = visualization_manifest
        atomic_json_gz_dump(scene_dataset, trajectory_path)
        atomic_json_dump(stats, stats_path)
        return stats
    finally:
        simulator.close()


def _process_scene_worker(payload: tuple) -> dict[str, Any]:
    record, config, run_root, visualization_root, overwrite = payload
    try:
        with _silence_native_worker_output():
            return process_scene(
                record,
                config,
                run_root,
                visualization_root=visualization_root,
                overwrite=overwrite,
            )
    except Exception as error:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "scene_key": record["scene_key"],
            "scene_id": record["scene_id"],
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }


def aggregate_shards(
    records: Sequence[dict[str, str]],
    config: dict[str, Any],
    work_root: str | Path,
    output_root: str | Path,
    dataset_name: str,
) -> tuple[Path, dict[str, Any]]:
    work_root = Path(work_root)
    output_root = Path(output_root)
    scene_stats = []
    shard_paths = []
    missing = []
    for record in records:
        output_dir = _scene_output_dir(work_root, record)
        trajectory_path = output_dir / "trajectories.json.gz"
        stats_path = output_dir / "stats.json"
        if not trajectory_path.is_file() or not stats_path.is_file():
            missing.append(record["scene_key"])
            continue
        shard = load_json_gz(trajectory_path)
        validate_dataset(shard)
        shard_paths.append(trajectory_path)
        stats = load_json(stats_path)
        route_groups = int(stats.get("region_level_route_groups", 0))
        trajectory_count = int(stats.get("final_trajectory_count", 0))
        stats["route_structure_retention_rate"] = trajectory_count / max(
            1, route_groups
        )
        stats["replay_success_rate"] = 1.0 if trajectory_count else None
        scene_stats.append(stats)
    if missing:
        raise RuntimeError(f"Missing scene shards: {missing}")
    versions = package_versions()
    metadata = {
        "kind": "hm3d_decision_rich_trajectories",
        "dataset_name": dataset_name,
        "purpose": "training",
        "scene_count": len(scene_stats),
        "habitat_sim_version": versions["habitat-sim"],
        "action_mapping": {
            "stop": 0,
            "move_forward": 1,
            "turn_left": 2,
            "turn_right": 3,
        },
        "action_parameters": config["agent"],
        "erp_sensor": {
            key: config["erp"][key] for key in ("width", "height", "max_depth")
        },
        "coordinate_frame": "Habitat world: +Y up; agent forward is local -Z",
        "decision_action_index": (
            "number of primitive actions already executed at the stored decision pose"
        ),
        "selection_policy": {
            "quality_gates": config["quality"],
            "minimum_route_length_m": float(
                config["route"]["min_geodesic_distance"]
            ),
            "strategy": "one replay-verified representative per directed route family",
            "fixed_global_or_per_scene_quota": False,
            "route_family": {
                "decision_structure": "ordered decision relations and region core",
                **config["route"]["family"],
            },
        },
    }
    dataset_header = {
        "schema_version": SCHEMA_VERSION,
        "metadata": metadata,
    }
    dataset_path = output_root / f"{dataset_name}.json.gz"
    output_root.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=dataset_path.parent, prefix=f".{dataset_path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    decision_histogram = Counter()
    region_count_histogram = Counter()
    trajectory_lengths = []
    total_trajectories = 0
    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            prefix = json.dumps(dataset_header, ensure_ascii=False, separators=(",", ":"))
            handle.write(prefix[:-1])
            handle.write(',"episodes":[')
            first_episode = True
            for shard_path in shard_paths:
                shard = load_json_gz(shard_path)
                for episode in shard["episodes"]:
                    if not first_episode:
                        handle.write(",")
                    json.dump(episode, handle, ensure_ascii=False, separators=(",", ":"))
                    first_episode = False
                    total_trajectories += 1
                    decision_histogram[
                        int(episode["metrics"]["decision_event_count"])
                    ] += 1
                    region_count_histogram[
                        int(episode["metrics"]["region_count"])
                    ] += 1
                    trajectory_lengths.append(float(episode["metrics"]["length_m"]))
            handle.write("]}")
        os.replace(temporary, dataset_path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    total_route_groups = sum(
        int(stats.get("region_level_route_groups", 0)) for stats in scene_stats
    )
    per_scene_counts = sorted(
        int(stats.get("final_trajectory_count", 0)) for stats in scene_stats
    )

    def percentile(fraction: float) -> int:
        if not per_scene_counts:
            return 0
        return per_scene_counts[int((len(per_scene_counts) - 1) * fraction)]

    diagnosis_counts = Counter(
        stats.get("diagnosis", {}).get("outcome", "unavailable")
        for stats in scene_stats
    )
    total_route_families = sum(
        int(stats.get("diagnosis", {}).get("route", {}).get("route_family_count", 0))
        for stats in scene_stats
    )
    compact_scene_stats = []
    for stats in scene_stats:
        diagnosis = stats.get("diagnosis", {})
        partition = diagnosis.get("partition", {})
        decision = diagnosis.get("decision", {})
        route = diagnosis.get("route", {})
        scene_record = {
            "scene_id": stats["scene_id"],
            "regions": int(stats["region_count"]),
            "connections": int(stats["connection_count"]),
            "junction_regions": int(partition.get("junction_region_count", 0)),
            "decision_relations": int(stats["decision_relation_count"]),
            "route_candidates": int(stats["region_level_route_groups"]),
            "route_families": int(route.get("route_family_count", 0)),
            "trajectories": int(stats["final_trajectory_count"]),
            "diagnosis": diagnosis.get("outcome", "unavailable"),
        }
        if decision.get("rejections"):
            scene_record["decision_rejections"] = decision["rejections"]
        if route.get("replay_rejections"):
            scene_record["replay_rejections"] = route["replay_rejections"]
        compact_scene_stats.append(scene_record)

    def numeric_summary(values: Sequence[float]) -> dict[str, float]:
        if not values:
            return {"minimum": 0.0, "median": 0.0, "p90": 0.0, "maximum": 0.0, "mean": 0.0}
        array = np.asarray(values, dtype=float)
        return {
            "minimum": float(np.min(array)),
            "median": float(np.percentile(array, 50)),
            "p90": float(np.percentile(array, 90)),
            "maximum": float(np.max(array)),
            "mean": float(np.mean(array)),
        }
    report = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset_path.name,
        "scene_count": len(scene_stats),
        "regions": sum(int(stats["region_count"]) for stats in scene_stats),
        "connections": sum(int(stats["connection_count"]) for stats in scene_stats),
        "decision_relations": sum(
            int(stats["decision_relation_count"]) for stats in scene_stats
        ),
        "route_candidates": total_route_groups,
        "route_families": total_route_families,
        "trajectories": total_trajectories,
        "decision_event_histogram": {
            str(key): value for key, value in sorted(decision_histogram.items())
        },
        "region_count_histogram": {
            str(key): value for key, value in sorted(region_count_histogram.items())
        },
        "length_m": numeric_summary(trajectory_lengths),
        "trajectories_per_scene": {
            "minimum": min(per_scene_counts, default=0),
            "p10": percentile(0.10),
            "p25": percentile(0.25),
            "median": percentile(0.50),
            "p75": percentile(0.75),
            "p90": percentile(0.90),
            "maximum": max(per_scene_counts, default=0),
            "mean": total_trajectories / max(1, len(per_scene_counts)),
        },
        "diagnoses": dict(sorted(diagnosis_counts.items())),
        "scenes": compact_scene_stats,
    }
    atomic_json_dump(report, output_root / f"{dataset_name}_stats.json")
    return dataset_path, report


def _rotation_error_degrees(first: Sequence[float], second: Sequence[float]) -> float:
    first_value = np.asarray(first, dtype=float)
    second_value = np.asarray(second, dtype=float)
    first_value /= np.linalg.norm(first_value)
    second_value /= np.linalg.norm(second_value)
    cosine = float(np.clip(abs(np.dot(first_value, second_value)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(cosine))


def replay_validate_dataset(
    dataset_path: str | Path, config: dict[str, Any]
) -> dict[str, Any]:
    dataset = load_json_gz(dataset_path)
    validate_dataset(dataset)
    installed_version = package_versions()["habitat-sim"]
    recorded_version = dataset["metadata"].get("habitat_sim_version")
    if recorded_version != installed_version:
        raise RuntimeError(
            f"Habitat-Sim version mismatch: dataset={recorded_version}, "
            f"installed={installed_version}"
        )
    recorded_agent = dataset["metadata"].get("action_parameters", {})
    for key in (
        "height",
        "radius",
        "sensor_height",
        "forward_step_size",
        "turn_angle_degrees",
        "goal_radius",
    ):
        if not math.isclose(
            float(recorded_agent.get(key, math.nan)),
            float(config["agent"][key]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError(f"Replay agent parameter mismatch: {key}")
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for episode in dataset["episodes"]:
        by_scene.setdefault(episode["scene_id"], []).append(episode)
    failures = []
    checked = 0
    maximum_errors = {
        "decision_position_m": 0.0,
        "decision_rotation_degrees": 0.0,
        "final_position_m": 0.0,
        "final_rotation_degrees": 0.0,
        "goal_geodesic_m": 0.0,
    }
    position_tolerance = float(config["route"]["replay_position_tolerance"])
    rotation_tolerance = float(
        config["route"]["replay_rotation_tolerance_degrees"]
    )
    ordered_scenes = sorted(by_scene.items())
    validation_progress = tqdm(
        total=len(ordered_scenes),
        desc="Replay",
        unit="scene",
        dynamic_ncols=True,
    )
    for scene_id, episodes in ordered_scenes:
        scene_path = resolve_scene_path(config["paths"]["scene_root"], scene_id)
        with _silence_native_worker_output():
            simulator = make_simulator(
                scene_path, config, with_sensors=False
            )
        try:
            for episode in episodes:
                observed_states, collision_count = replay_episode_actions(
                    simulator, episode
                )
                decision_position_errors = [
                    float(
                        np.linalg.norm(
                            np.asarray(
                                observed_states[int(event["action_index"])]["position"]
                            )
                            - np.asarray(event["position"])
                        )
                    )
                    for event in episode["decision_events"]
                ]
                decision_rotation_errors = [
                    _rotation_error_degrees(
                        observed_states[int(event["action_index"])]["rotation_xyzw"],
                        event["rotation_xyzw"],
                    )
                    for event in episode["decision_events"]
                ]
                decision_position_error = max(decision_position_errors, default=0.0)
                decision_rotation_error = max(decision_rotation_errors, default=0.0)
                final_position_error = float(
                    np.linalg.norm(
                        np.asarray(observed_states[-1]["position"], dtype=float)
                        - np.asarray(episode["final_position"], dtype=float)
                    )
                )
                final_rotation_error = _rotation_error_degrees(
                    observed_states[-1]["rotation_xyzw"],
                    episode["final_rotation_xyzw"],
                )
                goal_path = shortest_path(
                    simulator.pathfinder,
                    observed_states[-1]["position"],
                    episode["goal_position"],
                )
                goal_error = math.inf if goal_path is None else goal_path["distance"]
                maximum_errors["decision_position_m"] = max(
                    maximum_errors["decision_position_m"], decision_position_error
                )
                maximum_errors["decision_rotation_degrees"] = max(
                    maximum_errors["decision_rotation_degrees"], decision_rotation_error
                )
                maximum_errors["final_position_m"] = max(
                    maximum_errors["final_position_m"], final_position_error
                )
                maximum_errors["final_rotation_degrees"] = max(
                    maximum_errors["final_rotation_degrees"], final_rotation_error
                )
                maximum_errors["goal_geodesic_m"] = max(
                    maximum_errors["goal_geodesic_m"], goal_error
                )
                if (
                    decision_position_error > position_tolerance
                    or decision_rotation_error > rotation_tolerance
                    or final_position_error > position_tolerance
                    or final_rotation_error > rotation_tolerance
                    or goal_error > float(config["agent"]["goal_radius"]) + 1e-4
                    or collision_count > 0
                ):
                    failures.append(
                        {
                            "trajectory_id": episode["trajectory_id"],
                            "decision_position_error_m": decision_position_error,
                            "decision_rotation_error_degrees": decision_rotation_error,
                            "final_position_error_m": final_position_error,
                            "final_rotation_error_degrees": final_rotation_error,
                            "goal_geodesic_error_m": goal_error,
                            "collisions": collision_count,
                        }
                    )
                checked += 1
        finally:
            simulator.close()
        validation_progress.set_postfix(
            scene=scene_path.parent.name,
            trajectories=checked,
            refresh=False,
        )
        validation_progress.update(1)
    validation_progress.close()
    return {
        "trajectories": checked,
        "successful": checked - len(failures),
        "success_rate": (checked - len(failures)) / max(1, checked),
        "maximum_errors": maximum_errors,
        "failures": failures,
    }


def _parse_scene_ids(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = [value.strip() for value in raw.replace(",", " ").split() if value.strip()]
    return values or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="collect all available scenes as training data")
    collect.add_argument("--scene-root", required=True)
    collect.add_argument("--output-root", required=True)
    collect.add_argument("--scene-ids", default=None)
    collect.add_argument("--dataset-name", default="trajectories")
    collect.add_argument("--work-dir", default=None)
    collect.add_argument("--inspect", action="store_true", help="show scene selection without collecting")
    collect.add_argument("--gpu-device-ids", default="0")
    collect.add_argument("--processes-per-gpu", type=int, default=1)
    collect.add_argument("--visualize", action="store_true")
    collect.add_argument("--overwrite", action="store_true")

    validate = subparsers.add_parser("validate", help="independently replay a dataset")
    validate.add_argument("--scene-root", required=True)
    validate.add_argument("--dataset", required=True)
    validate.add_argument("--stats", required=True)
    validate.add_argument("--gpu-device-id", type=int, default=0)
    validate.add_argument("--cleanup-work-dir", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_json(args.config)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Config schema {config.get('schema_version')} does not match {SCHEMA_VERSION}"
        )

    if args.command == "collect":
        config["paths"] = {
            "scene_root": str(Path(args.scene_root).resolve()),
            "output_root": str(Path(args.output_root).resolve()),
        }
        records = discover_scenes(config["paths"]["scene_root"], _parse_scene_ids(args.scene_ids))
        output_root = _default_run_root(config)
        dataset_name = args.dataset_name
        work_root = Path(args.work_dir).resolve() if args.work_dir else output_root.parent / ".work" / "trajectory"
        if args.inspect:
            print(json.dumps({"scenes": len(records), "purpose": "training",
                              "dataset": str(output_root / f"{dataset_name}.json.gz"),
                              "work_dir": str(work_root)}, indent=2))
            return 0
        dataset_path = output_root / f"{dataset_name}.json.gz"
        if dataset_path.is_file() and not args.overwrite:
            existing = load_json_gz(dataset_path)
            validate_dataset(existing)
            stats_path = output_root / f"{dataset_name}_stats.json"
            stats = load_json(stats_path)
            if ({s["scene_id"] for s in stats["scenes"]} != {r["scene_id"] for r in records}
                    or stats["trajectories"] != len(existing["episodes"])):
                raise ValueError("Existing collection belongs to another selection; use a new output or --overwrite")
            print(json.dumps({"status": "already_collected", "dataset": str(dataset_path),
                              "trajectories": len(existing["episodes"]),
                              "note": "Existing trajectories are retained; --overwrite explicitly recollects them."}))
            return 0
        visualization_root = (
            output_root / "visualizations" / dataset_name
            if args.visualize
            else None
        )
        gpu_ids = [
            int(value) for value in args.gpu_device_ids.replace(",", " ").split()
        ]
        if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
            raise ValueError("--gpu-device-ids must be a non-empty unique list")
        if args.processes_per_gpu != 1:
            raise ValueError(
                "The validated resource policy permits exactly one Habitat process per GPU"
            )
        active_gpu_ids = gpu_ids[: min(len(gpu_ids), len(records))]
        payloads = []
        for index, record in enumerate(records):
            worker_config = copy.deepcopy(config)
            worker_config["runtime"]["gpu_device_id"] = active_gpu_ids[
                index % len(active_gpu_ids)
            ]
            payloads.append(
                (
                    record,
                    worker_config,
                    str(work_root),
                    None if visualization_root is None else str(visualization_root),
                    bool(args.overwrite),
                )
            )
        results = []
        collected_trajectories = 0
        collection_progress = tqdm(
            total=len(payloads),
            desc="Collect",
            unit="scene",
            dynamic_ncols=True,
        )
        if len(active_gpu_ids) == 1:
            for payload in payloads:
                result = _process_scene_worker(payload)
                results.append(result)
                collected_trajectories += int(
                    result.get("final_trajectory_count", 0)
                )
                collection_progress.set_postfix(
                    scene=result["scene_key"],
                    status=result["status"],
                    trajectories=collected_trajectories,
                    refresh=False,
                )
                collection_progress.update(1)
        else:
            context = multiprocessing.get_context("spawn")
            executors = [
                ProcessPoolExecutor(max_workers=1, mp_context=context)
                for _ in active_gpu_ids
            ]
            futures = []
            try:
                futures = [
                    executors[index % len(executors)].submit(
                        _process_scene_worker, payload
                    )
                    for index, payload in enumerate(payloads)
                ]
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    collected_trajectories += int(
                        result.get("final_trajectory_count", 0)
                    )
                    collection_progress.set_postfix(
                        scene=result["scene_key"],
                        status=result["status"],
                        trajectories=collected_trajectories,
                        refresh=False,
                    )
                    collection_progress.update(1)
            except KeyboardInterrupt:
                for future in futures:
                    future.cancel()
                _terminate_executors(executors)
                collection_progress.close()
                print("Collection interrupted; completed scene shards were kept.")
                return 130
            else:
                for executor in executors:
                    executor.shutdown(wait=True, cancel_futures=True)
        collection_progress.close()
        failures = [result for result in results if result["status"] == "failed"]
        failure_path = output_root / f"{dataset_name}_failures.json"
        if failures:
            atomic_json_dump(failures, failure_path)
            print(
                f"failed scenes: {len(failures)}; details: {failure_path}",
                file=sys.stderr,
            )
            return 2
        failure_path.unlink(missing_ok=True)
        dataset_path, report = aggregate_shards(
            records, config, work_root, output_root, dataset_name
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"dataset: {dataset_path.resolve()}")
        return 0

    if args.command == "validate":
        config["paths"] = {"scene_root": str(Path(args.scene_root).resolve())}
        config["runtime"]["gpu_device_id"] = int(args.gpu_device_id)
        report = replay_validate_dataset(args.dataset, config)
        stats_path = Path(args.stats)
        stats = load_json(stats_path)
        stats["independent_replay"] = {
            "trajectories": report["trajectories"],
            "success_rate": report["success_rate"],
            "maximum_errors": report["maximum_errors"],
        }
        if report["failures"]:
            stats["independent_replay"]["failures"] = report["failures"]
        atomic_json_dump(stats, stats_path)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if report["failures"]:
            return 2
        if args.cleanup_work_dir:
            work_dir = Path(args.cleanup_work_dir).resolve()
            expected_parent = (Path(args.dataset).resolve().parent.parent / ".work").resolve()
            if work_dir.parent != expected_parent or not work_dir.name:
                raise ValueError(f"Refusing to clean unexpected work directory: {work_dir}")
            if work_dir.is_dir():
                shutil.rmtree(work_dir)
            try:
                expected_parent.rmdir()
            except OSError:
                pass
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
