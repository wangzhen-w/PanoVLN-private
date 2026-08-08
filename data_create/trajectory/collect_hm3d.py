#!/usr/bin/env python3
"""Collect and validate high-quality PanoVLN-HM3D trajectories.

The collector samples directly from each official HM3D navmesh. It emits
benchmark-compatible VLN-CE episodes plus immutable geometry metrics. R2R-like
routes are shortest paths; RxR-like routes optionally visit an off-shortest-path
anchor to reproduce the instruction-fidelity challenge absent from pure PointNav.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import multiprocessing
import os
import random
import shutil
import sqlite3
import statistics
import time
import traceback
from pathlib import Path
from queue import Empty
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm.auto import tqdm

from data_create.trajectory.scene_paths import public_scene_id, resolve_scene_path


DEFAULT_SCENE_ROOT = "/workspace/data1/dataset/general_VLN_data/HM3D"
DEFAULT_PROFILE_CONFIG = str(
    Path(__file__).resolve().parents[1] / "config" / "trajectory_profiles.json"
)

EMPTY_INSTRUCTION_VOCAB = {
    "word_list": [],
    "word2idx_dict": {},
    "stoi": {},
    "itos": [],
    "num_vocab": 0,
    "UNK_INDEX": 1,
    "PAD_INDEX": 0,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="sample HM3D routes")
    collect.add_argument(
        "--scene-root",
        default=DEFAULT_SCENE_ROOT,
        help="Split-aware HM3D root containing train/ and val/ directories.",
    )
    collect.add_argument("--split", default="train")
    collect.add_argument("--scene-ids", default=None, help="comma-separated folder ids")
    collect.add_argument(
        "--num-trajectories",
        type=int,
        default=100,
        help=(
            "Desired trajectory count. Quality gates are never relaxed if the "
            "eligible scenes cannot supply the full target."
        ),
    )
    collect.add_argument(
        "--max-recovery-rounds",
        type=int,
        default=1,
        help=(
            "Additional all-scene sampling rounds after the initial scene-sharded "
            "round. After this limit, publish the largest ratio-balanced set of "
            "accepted trajectories instead of weakening quality gates."
        ),
    )
    collect.add_argument(
        "--r2r-ratio",
        type=float,
        default=0.40,
        help="Requested fraction of accepted trajectories using the R2R family.",
    )
    collect.add_argument(
        "--rxr-ratio",
        type=float,
        default=0.60,
        help="Requested fraction of accepted trajectories using the RxR family.",
    )
    collect.add_argument(
        "--trajectory-profile-config",
        default=DEFAULT_PROFILE_CONFIG,
        help="Repo-local aggregate R2R/RxR forward-distance histograms.",
    )
    collect.add_argument("--rxr-detour-fraction", type=float, default=0.445)
    collect.add_argument("--detour-ratio-min", type=float, default=1.12)
    collect.add_argument(
        "--detour-ratio-max",
        type=float,
        default=2.0,
        help=(
            "Per-detour upper bound. The RxR ~1.274 reference/shortest figure "
            "is a corpus-level descriptive mean, not this per-route bound."
        ),
    )
    collect.add_argument("--min-distance", type=float, default=5.0)
    collect.add_argument("--r2r-max-distance", type=float, default=20.0)
    collect.add_argument("--rxr-max-distance", type=float, default=40.0)
    collect.add_argument("--target-tolerance", type=float, default=2.0)
    collect.add_argument("--endpoint-clearance", type=float, default=0.22)
    collect.add_argument("--min-island-radius", type=float, default=1.5)
    collect.add_argument("--max-tortuosity", type=float, default=4.0)
    collect.add_argument(
        "--max-reference-turn-degrees",
        type=float,
        default=120.0,
        help="Reject route polylines with an unexplained cusp sharper than this.",
    )
    collect.add_argument(
        "--max-anchor-turn-degrees",
        type=float,
        default=90.0,
        help="Stricter turn bound at a synthetic detour anchor.",
    )
    collect.add_argument("--revisit-radius", type=float, default=0.4)
    collect.add_argument("--revisit-min-path-separation", type=float, default=1.0)
    collect.add_argument("--min-anchor-closure-ratio", type=float, default=0.25)
    collect.add_argument("--max-edge-geodesic-ratio", type=float, default=1.05)
    collect.add_argument("--review-heading-change-per-meter", type=float, default=30.0)
    collect.add_argument("--max-heading-change-per-meter", type=float, default=45.0)
    collect.add_argument("--near-goal-threshold", type=float, default=1.0)
    collect.add_argument("--max-near-goal-departure", type=float, default=0.75)
    collect.add_argument("--keypoint-spacing", type=float, default=1.5)
    collect.add_argument("--visual-check-width", type=int, default=512)
    collect.add_argument("--visual-check-height", type=int, default=256)
    collect.add_argument("--visual-check-sensor-height", type=float, default=1.25)
    collect.add_argument("--max-visual-checkpoints", type=int, default=24)
    collect.add_argument(
        "--max-black-ratio",
        type=float,
        default=0.10,
        help="Reject routes whose low-resolution visual precheck exposes scan voids.",
    )
    collect.add_argument("--floor-height", type=float, default=2.5)
    collect.add_argument("--max-floors", type=int, default=2)
    collect.add_argument("--goal-radius", type=float, default=0.3)
    collect.add_argument("--max-attempts-per-trajectory", type=int, default=500)
    collect.add_argument(
        "--max-sampling-failures-per-scene",
        type=int,
        default=8,
        help=(
            "Move to the next scene after this many exhausted trajectory searches; "
            "prevents one impossible family/detour request from looping globally."
        ),
    )
    collect.add_argument("--seed", type=int, default=42)
    collect.add_argument("--gpu-device-id", type=int, default=0)
    collect.add_argument(
        "--gpu-device-ids",
        default=None,
        help=(
            "Comma-separated Habitat GPU IDs. With multiple process slots, scenes "
            "are partitioned so one scene is owned by only one worker."
        ),
    )
    collect.add_argument(
        "--processes-per-gpu",
        type=int,
        default=1,
        help="Independent Habitat collection processes assigned to each GPU.",
    )
    collect.add_argument(
        "--output",
        required=True,
        help="Internal trajectory dataset JSON.GZ used by later stages.",
    )
    collect.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume trajectory sampling from the single-worker SQLite checkpoint or "
            "the per-worker checkpoint directory. State is removed after publication."
        ),
    )
    collect.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing trajectory dataset from a previous run.",
    )

    validate = subparsers.add_parser(
        "validate-gt", help="validate ShortestPathFollower GT generated for collected routes"
    )
    validate.add_argument("--dataset", required=True)
    validate.add_argument("--gt", required=True)
    validate.add_argument(
        "--generation-jsonl",
        default=None,
        help=(
            "Optional instruction-generation input JSONL. Rows omit instruction "
            "and contain the replayed actions plus terminal stop."
        ),
    )
    validate.add_argument("--goal-radius", type=float, default=0.3)
    validate.add_argument("--max-actions", type=int, default=500)
    validate.add_argument(
        "--drop-invalid",
        action="store_true",
        help=(
            "Replace the trajectory and GT datasets with atomically written "
            "subsets that pass every GT replay check. This is intended for "
            "dataset creation, where quality is preferred over an exact count."
        ),
    )

    render = subparsers.add_parser(
        "render-panoramas",
        help="replay validated GT and save one 2:1 RGB panorama per low-level state",
    )
    render.add_argument("--dataset", required=True)
    render.add_argument("--gt", required=True)
    render.add_argument(
        "--scene-root",
        default=DEFAULT_SCENE_ROOT,
        help="The same split-aware HM3D root used by collect.",
    )
    render.add_argument("--output-root", required=True)
    render.add_argument("--width", type=int, default=2048)
    render.add_argument("--height", type=int, default=1024)
    render.add_argument("--sensor-height", type=float, default=1.25)
    render.add_argument("--forward-step-size", type=float, default=0.25)
    render.add_argument("--turn-angle", type=float, default=15.0)
    render.add_argument(
        "--image-format",
        choices=("jpg", "jpeg", "png"),
        default="jpeg",
        help="Published panorama format. PanoVLN defaults to JPEG.",
    )
    render.add_argument("--jpeg-quality", type=int, default=75)
    render.add_argument(
        "--png-compress-level",
        type=int,
        choices=tuple(range(10)),
        default=6,
        help="PNG DEFLATE level; every level is pixel-lossless.",
    )
    render.add_argument("--max-black-ratio", type=float, default=0.10)
    render.add_argument(
        "--pose-tolerance",
        type=float,
        default=0.02,
        help="Maximum replay-position deviation in meters from GT locations.",
    )
    render.add_argument("--gpu-device-id", type=int, default=0)
    render.add_argument(
        "--gpu-device-ids",
        default=None,
        help="Comma-separated Habitat GPU IDs used by panorama workers.",
    )
    render.add_argument(
        "--processes-per-gpu",
        type=int,
        default=1,
        help="Independent panorama-rendering processes assigned to each GPU.",
    )
    render.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from the per-episode render journal and validated directories "
            "left in OUTPUT_ROOT.rendering."
        ),
    )
    render.add_argument("--max-episodes", type=int, default=None)
    render.add_argument(
        "--allow-partial-render",
        action="store_true",
        help=(
            "Allow --max-episodes to validate only a subset and exit successfully. "
            "The printed summary still marks selection_complete=false."
        ),
    )
    render.add_argument(
        "--generation-jsonl",
        default=None,
        help=(
            "Instruction-generation input JSONL to keep aligned when invalid "
            "rendered episodes are filtered."
        ),
    )
    render.add_argument(
        "--drop-invalid",
        action="store_true",
        help=(
            "Publish only episodes whose panorama replay passes every hard check, "
            "and filter dataset, GT, and generation JSONL to the same IDs."
        ),
    )
    return parser


def habitat_process_slots(args: argparse.Namespace) -> List[int]:
    """Expand GPU IDs into fixed process slots without changing CUDA visibility."""

    raw_ids = getattr(args, "gpu_device_ids", None)
    if raw_ids is None or not str(raw_ids).strip():
        gpu_ids = [int(args.gpu_device_id)]
    else:
        try:
            gpu_ids = [int(item.strip()) for item in str(raw_ids).split(",") if item.strip()]
        except ValueError as error:
            raise ValueError("--gpu-device-ids must be comma-separated integers") from error
        if not gpu_ids:
            raise ValueError("--gpu-device-ids must contain at least one GPU ID")
        if len(set(gpu_ids)) != len(gpu_ids):
            raise ValueError("--gpu-device-ids must not contain duplicates")
        if min(gpu_ids) < 0:
            raise ValueError("GPU device IDs must be non-negative")
    processes_per_gpu = int(getattr(args, "processes_per_gpu", 1))
    if processes_per_gpu <= 0:
        raise ValueError("--processes-per-gpu must be positive")
    return [gpu_id for gpu_id in gpu_ids for _ in range(processes_per_gpu)]


def read_profile_distances(
    path: str,
    profile_name: str,
    minimum: float,
    maximum: float,
) -> List[float]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != "panovln-trajectory-profiles-v1":
        raise ValueError(f"Unsupported trajectory profile schema: {path}")
    profile = (payload.get("profiles") or {}).get(profile_name)
    if not isinstance(profile, dict):
        raise ValueError(f"Missing trajectory profile {profile_name!r}: {path}")
    histogram = profile.get("histogram")
    if not isinstance(histogram, dict) or not histogram:
        raise ValueError(f"Profile {profile_name!r} has no histogram: {path}")
    unit = profile.get("distance_unit_m")
    if isinstance(unit, bool) or not isinstance(unit, (int, float)) or unit <= 0:
        raise ValueError(f"Profile {profile_name!r} has invalid distance_unit_m")
    distances: List[float] = []
    total_count = 0
    for raw_distance, count in histogram.items():
        try:
            distance = float(raw_distance)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Profile {profile_name!r} has invalid distance {raw_distance!r}"
            ) from error
        if not math.isfinite(distance) or distance <= 0:
            raise ValueError(f"Profile {profile_name!r} has invalid distance {distance}")
        if abs(distance / float(unit) - round(distance / float(unit))) > 1e-6:
            raise ValueError(
                f"Profile {profile_name!r} distance {distance} is off its {unit}m grid"
            )
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(
                f"Profile {profile_name!r} distance {distance} has invalid count {count!r}"
            )
        total_count += count
        if minimum <= distance <= maximum:
            distances.extend([distance] * count)
    if profile.get("sample_count") != total_count:
        raise ValueError(f"Profile {profile_name!r} sample_count does not match histogram")
    if not distances:
        raise ValueError(
            f"No {profile_name} profile distances in [{minimum}, {maximum}] from {path}"
        )
    return distances


def discover_scenes(
    root: str,
    split: str,
    scene_ids: Optional[str],
) -> List[Path]:
    wanted = {
        item.strip() for item in (scene_ids or "").split(",") if item.strip()
    }
    split_root = Path(root).resolve() / split
    scenes = sorted(split_root.glob("*/*.basis.glb"))
    missing_navmeshes = [
        scene
        for scene in scenes
        if not scene.with_name(scene.name.replace(".basis.glb", ".basis.navmesh")).is_file()
    ]
    if missing_navmeshes:
        raise FileNotFoundError(
            f"HM3D scenes are missing basis navmeshes: {missing_navmeshes[:10]}"
        )
    if wanted:
        scenes = [scene for scene in scenes if scene.parent.name in wanted]
        missing = sorted(wanted - {scene.parent.name for scene in scenes})
        if missing:
            raise FileNotFoundError(f"Requested HM3D scenes not found: {missing}")
    if not scenes:
        raise FileNotFoundError(f"No *.basis.glb scenes under {split_root}")
    return scenes


def make_simulator(scene: Path, args: argparse.Namespace, seed: int):
    import habitat_sim

    simulator_config = habitat_sim.SimulatorConfiguration()
    simulator_config.scene_id = str(scene)
    simulator_config.enable_physics = False
    simulator_config.gpu_device_id = int(args.gpu_device_id)
    sensor = habitat_sim.EquirectangularSensorSpec()
    sensor.uuid = "quality_rgb"
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.resolution = [int(args.visual_check_height), int(args.visual_check_width)]
    sensor.position = [0.0, float(args.visual_check_sensor_height), 0.0]
    agent_config = habitat_sim.agent.AgentConfiguration()
    agent_config.sensor_specifications = [sensor]
    simulator = habitat_sim.Simulator(
        habitat_sim.Configuration(simulator_config, [agent_config])
    )
    if not simulator.pathfinder.is_loaded:
        simulator.close()
        raise RuntimeError(f"Navmesh failed to load for {scene}")
    simulator.pathfinder.seed(int(seed))
    return simulator


def route_visual_metrics(
    simulator,
    route_keypoints: Sequence[Sequence[float]],
    max_checkpoints: int,
) -> Dict[str, Any]:
    """Render sparse low-resolution panoramas before accepting a route."""

    from habitat_sim import AgentState

    if not route_keypoints:
        raise ValueError("Route visual precheck requires at least one keypoint")
    points = list(route_keypoints)
    limit = max(2, int(max_checkpoints))
    if len(points) > limit:
        indices = sorted(
            set(
                int(round(index))
                for index in np.linspace(0, len(points) - 1, num=limit)
            )
        )
        points = [points[index] for index in indices]

    agent = simulator.get_agent(0)
    original_state = agent.get_state()
    black_ratios: List[float] = []
    try:
        for point in points:
            state = AgentState()
            state.position = np.asarray(point, dtype=np.float32)
            state.rotation = original_state.rotation
            agent.set_state(state, reset_sensors=True)
            observation = simulator.get_sensor_observations()["quality_rgb"]
            rgb = np.asarray(observation)[..., :3]
            black_ratios.append(float(np.mean(np.max(rgb, axis=2) <= 3)))
    finally:
        agent.set_state(original_state, reset_sensors=True)

    return {
        "visual_checkpoints": len(points),
        "visual_black_ratio_mean": statistics.mean(black_ratios),
        "visual_black_ratio_max": max(black_ratios),
    }


def shortest_path(pathfinder, start: Sequence[float], goal: Sequence[float]):
    import habitat_sim

    query = habitat_sim.ShortestPath()
    query.requested_start = start
    query.requested_end = goal
    if not pathfinder.find_path(query) or not math.isfinite(query.geodesic_distance):
        return None
    return query


def main_navigable_island(pathfinder, minimum_radius: float) -> int:
    candidates = [
        index
        for index in range(int(pathfinder.num_islands))
        if float(pathfinder.island_radius(index)) >= minimum_radius
    ]
    if not candidates:
        raise RuntimeError(
            f"No navmesh island has radius >= {minimum_radius}m; "
            f"num_islands={pathfinder.num_islands}"
        )
    return max(candidates, key=lambda index: float(pathfinder.island_area(index)))


def vec(point: Sequence[float]) -> np.ndarray:
    return np.asarray([float(value) for value in point], dtype=np.float64)


def path_length(points: Sequence[Sequence[float]]) -> float:
    if len(points) < 2:
        return 0.0
    return float(
        sum(np.linalg.norm(vec(right) - vec(left)) for left, right in zip(points[:-1], points[1:]))
    )


def concatenate_paths(*paths: Sequence[Sequence[float]]) -> List[List[float]]:
    output: List[List[float]] = []
    for path in paths:
        for point in path:
            converted = [float(value) for value in point]
            if output and np.linalg.norm(vec(output[-1]) - vec(converted)) < 1e-5:
                continue
            output.append(converted)
    return output


def resample_polyline(
    points: Sequence[Sequence[float]], spacing: float
) -> List[List[float]]:
    if len(points) < 2:
        return [[float(value) for value in point] for point in points]
    result = [[float(value) for value in points[0]]]
    carry = 0.0
    for left_raw, right_raw in zip(points[:-1], points[1:]):
        left = vec(left_raw)
        right = vec(right_raw)
        delta = right - left
        length = float(np.linalg.norm(delta))
        if length < 1e-8:
            continue
        direction = delta / length
        travelled = spacing - carry if carry > 1e-8 else spacing
        while travelled < length:
            result.append((left + direction * travelled).tolist())
            travelled += spacing
        carry = max(0.0, length - (travelled - spacing))
        if carry >= spacing:
            carry %= spacing
    final = [float(value) for value in points[-1]]
    if np.linalg.norm(vec(result[-1]) - vec(final)) > 1e-5:
        result.append(final)
    return result


def heading_change_degrees(points: Sequence[Sequence[float]]) -> float:
    directions = []
    for left, right in zip(points[:-1], points[1:]):
        delta = vec(right) - vec(left)
        planar = np.asarray([delta[0], delta[2]])
        norm = float(np.linalg.norm(planar))
        if norm > 1e-5:
            directions.append(planar / norm)
    change = 0.0
    for left, right in zip(directions[:-1], directions[1:]):
        cosine = float(np.clip(np.dot(left, right), -1.0, 1.0))
        change += math.degrees(math.acos(cosine))
    return change


def planar_turn_angles(points: Sequence[Sequence[float]]) -> List[float]:
    angles = []
    for left, center, right in zip(points[:-2], points[1:-1], points[2:]):
        incoming_3d = vec(center) - vec(left)
        outgoing_3d = vec(right) - vec(center)
        incoming = np.asarray([incoming_3d[0], incoming_3d[2]])
        outgoing = np.asarray([outgoing_3d[0], outgoing_3d[2]])
        left_norm = float(np.linalg.norm(incoming))
        right_norm = float(np.linalg.norm(outgoing))
        if left_norm < 0.1 or right_norm < 0.1:
            continue
        cosine = float(
            np.clip(np.dot(incoming, outgoing) / (left_norm * right_norm), -1.0, 1.0)
        )
        angles.append(math.degrees(math.acos(cosine)))
    return angles


def nonlocal_revisit_count(
    points: Sequence[Sequence[float]],
    radius: float,
    min_path_separation: float,
) -> int:
    sampled = resample_polyline(points, min(0.25, max(0.1, radius)))
    if len(sampled) < 3:
        return 0
    cumulative = [0.0]
    for left, right in zip(sampled[:-1], sampled[1:]):
        cumulative.append(cumulative[-1] + float(np.linalg.norm(vec(right) - vec(left))))
    revisited = set()
    for left_index in range(len(sampled)):
        for right_index in range(left_index + 2, len(sampled)):
            if cumulative[right_index] - cumulative[left_index] < min_path_separation:
                continue
            if float(np.linalg.norm(vec(sampled[right_index]) - vec(sampled[left_index]))) < radius:
                revisited.add(right_index)
                break
    return len(revisited)


def junction_turn_degrees(
    first_path: Sequence[Sequence[float]],
    second_path: Sequence[Sequence[float]],
) -> float:
    if len(first_path) < 2 or len(second_path) < 2:
        return 180.0
    return max(
        planar_turn_angles([first_path[-2], first_path[-1], second_path[1]])
        or [180.0]
    )


def grid_duplicate_rate(points: Sequence[Sequence[float]], cell_size: float = 0.5) -> float:
    if not points:
        return 0.0
    cells = [
        (
            round(float(point[0]) / cell_size),
            round(float(point[1]) / cell_size),
            round(float(point[2]) / cell_size),
        )
        for point in resample_polyline(points, cell_size)
    ]
    return 1.0 - len(set(cells)) / max(1, len(cells))


def route_hash(scene_id: str, points: Sequence[Sequence[float]]) -> str:
    payload = {
        "scene": scene_id,
        "points": [[round(float(value), 2) for value in point] for point in points],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def yaw_facing(start: Sequence[float], next_point: Sequence[float]) -> float:
    delta = vec(next_point) - vec(start)
    return math.atan2(-float(delta[0]), -float(delta[2]))


def sample_start_yaw(
    points: Sequence[Sequence[float]],
    family: str,
    rng: random.Random,
) -> Tuple[float, Dict[str, Any]]:
    forward_probability = 0.109 if family == "r2r" else 0.056
    base = yaw_facing(points[0], points[1])
    if rng.random() < forward_probability:
        offset_degrees = rng.uniform(-7.0, 7.0)
        intended = "forward"
    else:
        magnitude = rng.choice((30, 45, 60, 75, 90, 105, 120, 135, 150, 165))
        sign = -1 if rng.random() < 0.5 else 1
        offset_degrees = sign * magnitude
        intended = "left_or_right"
    yaw = base + math.radians(offset_degrees)
    return yaw, {
        "path_facing_yaw_radians": base,
        "start_yaw_offset_degrees": offset_degrees,
        "intended_initial_action_family": intended,
    }


def quaternion_from_yaw(yaw: float) -> List[float]:
    return [0.0, math.sin(yaw / 2.0), 0.0, math.cos(yaw / 2.0)]


def point_distance_to_path(point: Sequence[float], points: Sequence[Sequence[float]]) -> float:
    target = vec(point)
    if len(points) < 2:
        return min(float(np.linalg.norm(target - vec(candidate))) for candidate in points)
    distances = []
    for left_raw, right_raw in zip(points[:-1], points[1:]):
        left = vec(left_raw)
        right = vec(right_raw)
        delta = right - left
        denominator = float(np.dot(delta, delta))
        if denominator < 1e-12:
            projection = left
        else:
            fraction = float(np.clip(np.dot(target - left, delta) / denominator, 0.0, 1.0))
            projection = left + fraction * delta
        distances.append(float(np.linalg.norm(target - projection)))
    return min(distances)


def anchor_local_closure_ratio(
    first_path: Sequence[Sequence[float]],
    second_path: Sequence[Sequence[float]],
) -> float:
    if len(first_path) < 2 or len(second_path) < 2:
        return 0.0
    incoming = float(np.linalg.norm(vec(first_path[-1]) - vec(first_path[-2])))
    outgoing = float(np.linalg.norm(vec(second_path[1]) - vec(second_path[0])))
    closure = float(np.linalg.norm(vec(first_path[-2]) - vec(second_path[1])))
    return closure / max(incoming + outgoing, 1e-6)


def path_edges_are_directly_connected(
    pathfinder,
    points: Sequence[Sequence[float]],
    maximum_ratio: float,
) -> bool:
    for left, right in zip(points[:-1], points[1:]):
        euclidean = float(np.linalg.norm(vec(right) - vec(left)))
        if euclidean < 1e-6:
            return False
        query = shortest_path(pathfinder, left, right)
        if query is None:
            return False
        if float(query.geodesic_distance) / euclidean > maximum_ratio:
            return False
    return True


def goal_progress_metrics(
    pathfinder,
    points: Sequence[Sequence[float]],
    goal: Sequence[float],
    near_goal_threshold: float,
    max_near_goal_departure: float,
) -> Dict[str, Any]:
    distances = []
    for point in points:
        query = shortest_path(pathfinder, point, goal)
        if query is None:
            return {
                "goal_distance_trace": [],
                "max_goal_distance_increase": math.inf,
                "near_goal_departure": True,
            }
        distances.append(float(query.geodesic_distance))
    minimum_so_far = math.inf
    max_increase = 0.0
    near_goal_departure = False
    for distance in distances:
        if minimum_so_far <= near_goal_threshold:
            if distance - minimum_so_far > max_near_goal_departure:
                near_goal_departure = True
        max_increase = max(max_increase, distance - minimum_so_far)
        minimum_so_far = min(minimum_so_far, distance)
    return {
        "goal_distance_trace": distances,
        "max_goal_distance_increase": max(0.0, max_increase),
        "near_goal_departure": near_goal_departure,
    }


def sample_detour(
    pathfinder,
    start: Sequence[float],
    goal: Sequence[float],
    direct_points: Sequence[Sequence[float]],
    direct_distance: float,
    target_distance: float,
    island_index: int,
    args: argparse.Namespace,
) -> Optional[Tuple[List[List[float]], float, List[float], float, float]]:
    for _ in range(80):
        anchor = pathfinder.get_random_navigable_point(10, island_index)
        if point_distance_to_path(anchor, direct_points) < 1.25:
            continue
        first = shortest_path(pathfinder, start, anchor)
        second = shortest_path(pathfinder, anchor, goal)
        if first is None or second is None:
            continue
        total = float(first.geodesic_distance + second.geodesic_distance)
        ratio = total / max(direct_distance, 1e-6)
        if not args.detour_ratio_min <= ratio <= args.detour_ratio_max:
            continue
        if abs(total - target_distance) > max(args.target_tolerance, target_distance * 0.20):
            continue
        anchor_turn = junction_turn_degrees(first.points, second.points)
        if anchor_turn > args.max_anchor_turn_degrees:
            continue
        closure_ratio = anchor_local_closure_ratio(first.points, second.points)
        if closure_ratio < args.min_anchor_closure_ratio:
            continue
        points = concatenate_paths(first.points, second.points)
        if nonlocal_revisit_count(
            points,
            radius=args.revisit_radius,
            min_path_separation=args.revisit_min_path_separation,
        ):
            continue
        if max(planar_turn_angles(points) or [0.0]) > args.max_reference_turn_degrees:
            continue
        return points, total, [float(value) for value in anchor], anchor_turn, closure_ratio
    return None


def trajectory_metrics(
    points: Sequence[Sequence[float]],
    shortest_distance: float,
    start_clearance: float,
    goal_clearance: float,
    keypoint_spacing: float,
    floor_height: float,
    revisit_radius: float,
    revisit_min_path_separation: float,
    anchor_turn_degrees: Optional[float],
    anchor_closure_ratio: Optional[float],
) -> Dict[str, Any]:
    length = path_length(points)
    euclidean = float(np.linalg.norm(vec(points[-1]) - vec(points[0])))
    keypoints = resample_polyline(points, keypoint_spacing)
    vertical_values = [float(point[1]) for point in points]
    vertical_span = max(vertical_values) - min(vertical_values)
    floor_count_proxy = max(1, int(round(vertical_span / max(floor_height, 1e-6))) + 1)
    turn_angles = planar_turn_angles(points)
    resampled_heading_change = heading_change_degrees(resample_polyline(points, 0.5))
    heading_change_per_meter = resampled_heading_change / max(length, 1e-6)
    return {
        "reference_length": length,
        "shortest_distance": float(shortest_distance),
        "detour_ratio": length / max(float(shortest_distance), 1e-6),
        "euclidean_distance": euclidean,
        "tortuosity": length / max(euclidean, 1e-6),
        "vertical_displacement": abs(float(points[-1][1]) - float(points[0][1])),
        "vertical_span": vertical_span,
        "floor_count_proxy": floor_count_proxy,
        "multi_floor_proxy": floor_count_proxy > 1,
        "heading_change_degrees": heading_change_degrees(points),
        "resampled_heading_change_degrees": resampled_heading_change,
        "heading_change_per_meter": heading_change_per_meter,
        "max_planar_turn_degrees": max(turn_angles or [0.0]),
        "sharp_turn_count": sum(angle > 100.0 for angle in turn_angles),
        "nonlocal_revisit_count": nonlocal_revisit_count(
            points,
            radius=revisit_radius,
            min_path_separation=revisit_min_path_separation,
        ),
        "detour_anchor_turn_degrees": anchor_turn_degrees,
        "detour_anchor_closure_ratio": anchor_closure_ratio,
        "navmesh_corner_count": len(points),
        "route_keypoint_count": len(keypoints),
        "grid_duplicate_rate": grid_duplicate_rate(points),
        "start_clearance": float(start_clearance),
        "goal_clearance": float(goal_clearance),
    }


def sample_one(
    simulator,
    family: str,
    target_distance: float,
    make_detour: bool,
    rng: random.Random,
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    pathfinder = simulator.pathfinder
    island_index = main_navigable_island(pathfinder, args.min_island_radius)
    maximum = args.r2r_max_distance if family == "r2r" else args.rxr_max_distance
    for _ in range(args.max_attempts_per_trajectory):
        start = pathfinder.get_random_navigable_point(10, island_index)
        goal = pathfinder.get_random_navigable_point(10, island_index)
        start_clearance = float(pathfinder.distance_to_closest_obstacle(start))
        goal_clearance = float(pathfinder.distance_to_closest_obstacle(goal))
        if min(start_clearance, goal_clearance) < args.endpoint_clearance:
            continue
        direct = shortest_path(pathfinder, start, goal)
        if direct is None:
            continue
        direct_distance = float(direct.geodesic_distance)
        if direct_distance < args.min_distance or direct_distance > maximum:
            continue
        if not make_detour and abs(direct_distance - target_distance) > max(
            args.target_tolerance, target_distance * 0.18
        ):
            continue
        points = concatenate_paths(direct.points)
        reference_distance = direct_distance
        anchor = None
        anchor_turn = None
        anchor_closure = None
        if make_detour:
            detour = sample_detour(
                pathfinder,
                start,
                goal,
                points,
                direct_distance,
                target_distance,
                island_index,
                args,
            )
            if detour is None:
                continue
            points, reference_distance, anchor, anchor_turn, anchor_closure = detour
        if len(points) < 2:
            continue
        metrics = trajectory_metrics(
            points,
            shortest_distance=direct_distance,
            start_clearance=start_clearance,
            goal_clearance=goal_clearance,
            keypoint_spacing=args.keypoint_spacing,
            floor_height=args.floor_height,
            revisit_radius=args.revisit_radius,
            revisit_min_path_separation=args.revisit_min_path_separation,
            anchor_turn_degrees=anchor_turn,
            anchor_closure_ratio=anchor_closure,
        )
        if metrics["tortuosity"] > args.max_tortuosity:
            continue
        if metrics["floor_count_proxy"] > args.max_floors:
            continue
        if metrics["max_planar_turn_degrees"] > args.max_reference_turn_degrees:
            continue
        if metrics["heading_change_per_meter"] > args.max_heading_change_per_meter:
            continue
        if metrics["nonlocal_revisit_count"]:
            continue
        if not path_edges_are_directly_connected(
            pathfinder, points, args.max_edge_geodesic_ratio
        ):
            continue
        progress_metrics = goal_progress_metrics(
            pathfinder,
            points,
            goal,
            near_goal_threshold=max(
                args.near_goal_threshold, args.goal_radius * 3.0
            ),
            max_near_goal_departure=args.max_near_goal_departure,
        )
        if progress_metrics["near_goal_departure"]:
            continue
        metrics.update(progress_metrics)
        if not args.min_distance <= reference_distance <= maximum:
            continue
        yaw, yaw_metadata = sample_start_yaw(points, family, rng)
        return {
            "family": family,
            "is_detour": bool(make_detour),
            "target_distance": float(target_distance),
            "reference_path": points,
            "shortest_path": concatenate_paths(direct.points),
            "route_keypoints": resample_polyline(points, args.keypoint_spacing),
            "start_position": points[0],
            "goal_position": points[-1],
            "start_yaw": yaw,
            "start_rotation": quaternion_from_yaw(yaw),
            "detour_anchor": anchor,
            "metrics": metrics,
            "geometry_review_required": (
                metrics["heading_change_per_meter"]
                > args.review_heading_change_per_meter
            ),
            "yaw_metadata": yaw_metadata,
            "navmesh_island": {
                "index": island_index,
                "area": float(pathfinder.island_area(island_index)),
                "radius": float(pathfinder.island_radius(island_index)),
            },
        }
    return None


def distribute_families(
    r2r_ratio: float,
    rxr_ratio: float,
    count: int,
    rng: random.Random,
) -> List[str]:
    if count <= 0:
        raise ValueError(f"--num-trajectories must be positive, got {count}")
    if not all(math.isfinite(ratio) for ratio in (r2r_ratio, rxr_ratio)):
        raise ValueError("--r2r-ratio and --rxr-ratio must be finite")
    if r2r_ratio < 0.0 or rxr_ratio < 0.0:
        raise ValueError("--r2r-ratio and --rxr-ratio must be non-negative")
    if not math.isclose(r2r_ratio + rxr_ratio, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "--r2r-ratio and --rxr-ratio must sum to 1.0; "
            f"got {r2r_ratio + rxr_ratio:.12g}"
        )

    # The requested split is exact whenever count * r2r_ratio is integral. For
    # other totals, round R2R to the nearest trajectory and assign the remainder
    # to RxR so the overall count never changes.
    r2r_count = int(math.floor(count * r2r_ratio + 0.5))
    rxr_count = count - r2r_count
    families = ["r2r"] * r2r_count + ["rxr"] * rxr_count
    rng.shuffle(families)
    return families


def distribute_detours(
    families: Sequence[str], fraction: float, rng: random.Random
) -> List[bool]:
    rxr_indices = [index for index, family in enumerate(families) if family == "rxr"]
    detour_count = int(round(len(rxr_indices) * min(1.0, max(0.0, fraction))))
    chosen = set(rng.sample(rxr_indices, detour_count)) if detour_count else set()
    return [index in chosen for index in range(len(families))]


def atomic_json_gz(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


COLLECT_CHECKPOINT_SCHEMA = "panovln-collect-checkpoint-v1"


def nested_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(nested_tuple(item) for item in value)
    return value


def collect_run_manifest(
    args: argparse.Namespace,
    scenes: Sequence[Path],
) -> Dict[str, Any]:
    # Hardware scheduling can be changed when resuming without changing the
    # requested dataset. Sampling parameters and scene ownership remain hashed.
    excluded = {
        "command",
        "output",
        "overwrite",
        "resume",
        "gpu_device_id",
        "gpu_device_ids",
        "processes_per_gpu",
    }
    configuration = {
        key: value
        for key, value in sorted(vars(args).items())
        if key not in excluded and not key.startswith("_")
    }
    configuration["scene_root"] = str(Path(args.scene_root).resolve())
    profile_path = Path(args.trajectory_profile_config).resolve()
    configuration["trajectory_profile_config"] = str(profile_path)
    configuration["trajectory_profile_sha256"] = sha256_file(profile_path)
    configuration["ordered_scenes"] = [
        public_scene_id(scene, args.scene_root) for scene in scenes
    ]
    encoded = json.dumps(
        configuration, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "schema_version": COLLECT_CHECKPOINT_SCHEMA,
        "fingerprint": hashlib.sha256(encoded).hexdigest(),
        "configuration": configuration,
    }


def candidate_sampling_seed(base_seed: int, scene_index: int, attempt_index: int) -> int:
    encoded = f"{base_seed}:{scene_index}:{attempt_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "little")


def remove_collect_checkpoint(path: Path) -> None:
    for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        candidate.unlink(missing_ok=True)


class CollectCheckpoint:
    """Transactional trajectory rows plus compact resumable sampler state."""

    def __init__(
        self,
        path: Path,
        manifest: Dict[str, Any],
        initial_state: Dict[str, Any],
        resume: bool,
    ) -> None:
        existed = path.exists()
        if existed and not resume:
            raise FileExistsError(
                f"Trajectory checkpoint exists; rerun with --resume: {path}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS trajectories "
            "(episode_id INTEGER PRIMARY KEY, row_json TEXT NOT NULL)"
        )
        self.connection.commit()

        if existed:
            try:
                stored_manifest = self._read_metadata("manifest")
                if stored_manifest.get("schema_version") != COLLECT_CHECKPOINT_SCHEMA:
                    raise ValueError(f"Unsupported trajectory checkpoint schema: {path}")
                if stored_manifest.get("fingerprint") != manifest["fingerprint"]:
                    raise ValueError(
                        "Trajectory resume inputs or sampling parameters changed. Preserve "
                        f"the old checkpoint for inspection or remove it: {path}"
                    )
                self.state = self._read_metadata("state")
            except Exception:
                self.connection.close()
                raise
        else:
            self.state = initial_state
            with self.connection:
                self._write_metadata("manifest", manifest)
                self._write_metadata("state", initial_state)

    def _read_metadata(self, key: str) -> Dict[str, Any]:
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            raise ValueError(f"Trajectory checkpoint is missing metadata {key!r}")
        value = json.loads(row[0])
        if not isinstance(value, dict):
            raise ValueError(f"Trajectory checkpoint metadata {key!r} is invalid")
        return value

    def _write_metadata(self, key: str, value: Dict[str, Any]) -> None:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        self.connection.execute(
            "INSERT INTO metadata(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, encoded),
        )

    def load_trajectories(self) -> List[Dict[str, Any]]:
        rows = []
        for expected_id, (episode_id, row_json) in enumerate(
            self.connection.execute(
                "SELECT episode_id, row_json FROM trajectories ORDER BY episode_id"
            )
        ):
            if episode_id != expected_id:
                raise ValueError(
                    "Trajectory checkpoint episode IDs are not contiguous: "
                    f"expected={expected_id}, actual={episode_id}"
                )
            row = json.loads(row_json)
            if not isinstance(row, dict) or row.get("episode_id") != episode_id:
                raise ValueError(
                    f"Trajectory checkpoint row {episode_id} is malformed"
                )
            rows.append(row)
        if int(self.state.get("collected_count", -1)) != len(rows):
            raise ValueError(
                "Trajectory checkpoint state/row count mismatch: "
                f"state={self.state.get('collected_count')}, rows={len(rows)}"
            )
        return rows

    def commit(
        self,
        state: Dict[str, Any],
        trajectory: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self.connection:
            if trajectory is not None:
                episode_id = int(trajectory["episode_id"])
                self.connection.execute(
                    "INSERT INTO trajectories(episode_id, row_json) VALUES (?, ?)",
                    (
                        episode_id,
                        json.dumps(
                            trajectory,
                            ensure_ascii=False,
                            allow_nan=False,
                            separators=(",", ":"),
                        ),
                    ),
                )
            self._write_metadata("state", state)
        self.state = state

    def close(self) -> None:
        self.connection.close()


def summarize_trajectories(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def dist(key: str, subset: Sequence[Dict[str, Any]] = rows) -> Dict[str, float]:
        values = [float(row["metrics"][key]) for row in subset]
        ordered = sorted(values)
        if not ordered:
            return {}
        quantile = lambda p: ordered[int(round((len(ordered) - 1) * p))]
        return {
            "mean": statistics.mean(values),
            "p10": quantile(0.10),
            "p50": quantile(0.50),
            "p90": quantile(0.90),
        }

    family_counts = {
        family: sum(row["family"] == family for row in rows)
        for family in ("r2r", "rxr")
    }
    family_ratios = {
        family: count / max(1, len(rows)) for family, count in family_counts.items()
    }
    r2r_rows = [row for row in rows if row["family"] == "r2r"]
    rxr_rows = [row for row in rows if row["family"] == "rxr"]
    scene_counts: Dict[str, int] = {}
    for row in rows:
        scene_id = str(row["scene_id"])
        scene_counts[scene_id] = scene_counts.get(scene_id, 0) + 1
    return {
        "trajectories": len(rows),
        "family_counts": family_counts,
        "family_ratios": family_ratios,
        "scenes": len(scene_counts),
        "scene_counts": dict(sorted(scene_counts.items())),
        "scene_count_range": (
            [min(scene_counts.values()), max(scene_counts.values())]
            if scene_counts
            else []
        ),
        "reference_length": dist("reference_length"),
        "shortest_distance": dist("shortest_distance"),
        "detour_ratio": dist("detour_ratio"),
        "route_keypoint_count": dist("route_keypoint_count"),
        "heading_change_degrees": dist("heading_change_degrees"),
        "heading_change_per_meter": dist("heading_change_per_meter"),
        "max_planar_turn_degrees": dist("max_planar_turn_degrees"),
        "nonlocal_revisit_count": dist("nonlocal_revisit_count"),
        "visual_black_ratio_max": dist("visual_black_ratio_max"),
        "multi_floor_proxy_rate": sum(row["metrics"]["multi_floor_proxy"] for row in rows)
        / max(1, len(rows)),
        "rxr_detour_rate": sum(row["is_detour"] for row in rxr_rows) / max(1, len(rxr_rows)),
        "geometry_review_count": sum(
            bool(row.get("geometry_review_required")) for row in rows
        ),
        "by_family": {
            "r2r": {
                "reference_length": dist("reference_length", r2r_rows),
                "route_keypoint_count": dist("route_keypoint_count", r2r_rows),
                "multi_floor_proxy_rate": sum(
                    row["metrics"]["multi_floor_proxy"] for row in r2r_rows
                )
                / max(1, len(r2r_rows)),
            },
            "rxr": {
                "reference_length": dist("reference_length", rxr_rows),
                "route_keypoint_count": dist("route_keypoint_count", rxr_rows),
                "detour_ratio": dist("detour_ratio", rxr_rows),
                "multi_floor_proxy_rate": sum(
                    row["metrics"]["multi_floor_proxy"] for row in rxr_rows
                )
                / max(1, len(rxr_rows)),
            },
        },
        "unique_route_hashes": len({row["route_hash"] for row in rows}),
    }


def _balanced_counts(total: int, parts: int) -> List[int]:
    if total < 0 or parts <= 0:
        raise ValueError("Balanced allocation requires total >= 0 and parts > 0")
    quotient, remainder = divmod(total, parts)
    return [quotient + (index < remainder) for index in range(parts)]


def _collect_worker_entry(worker_args: argparse.Namespace) -> None:
    """Spawn-safe collection worker; its output/checkpoint belongs to one shard."""

    _collect_sequential(worker_args)


def _collect_shard_progress(output: Path, checkpoint: Path, expected: int) -> int:
    if output.is_file():
        return expected
    if not checkpoint.is_file():
        return 0
    try:
        connection = sqlite3.connect(f"file:{checkpoint}?mode=ro", uri=True, timeout=0.1)
        try:
            row = connection.execute("SELECT COUNT(*) FROM trajectories").fetchone()
            return min(expected, int(row[0]) if row else 0)
        finally:
            connection.close()
    except sqlite3.Error:
        return 0


def _largest_ratio_balanced_set(
    episodes: Sequence[Dict[str, Any]],
    desired_total: int,
    r2r_ratio: float,
) -> Tuple[List[Dict[str, Any]], int]:
    """Return the largest unique subset that preserves the requested family ratio."""

    unique: List[Dict[str, Any]] = []
    seen_hashes = set()
    duplicate_count = 0
    for episode in episodes:
        route_hash = (episode.get("info") or {}).get("route_hash")
        if not route_hash:
            raise RuntimeError("Collected trajectory is missing route_hash")
        route_hash = str(route_hash)
        if route_hash in seen_hashes:
            duplicate_count += 1
            continue
        family = (episode.get("info") or {}).get("family")
        if family not in {"r2r", "rxr"}:
            raise RuntimeError(f"Collected trajectory has invalid family: {family!r}")
        seen_hashes.add(route_hash)
        unique.append(episode)

    available = {
        family: sum((episode.get("info") or {}).get("family") == family for episode in unique)
        for family in ("r2r", "rxr")
    }
    final_total = min(int(desired_total), len(unique))
    while final_total > 0:
        r2r_count = int(math.floor(final_total * r2r_ratio + 0.5))
        rxr_count = final_total - r2r_count
        if r2r_count <= available["r2r"] and rxr_count <= available["rxr"]:
            break
        final_total -= 1
    if final_total <= 0:
        raise RuntimeError(
            "Accepted trajectories cannot form a non-empty set at the requested "
            f"R2R/RxR ratio; available={available}"
        )

    required_r2r = int(math.floor(final_total * r2r_ratio + 0.5))
    required = {"r2r": required_r2r, "rxr": final_total - required_r2r}
    kept = {"r2r": 0, "rxr": 0}
    selected: List[Dict[str, Any]] = []
    for episode in unique:
        family = str((episode.get("info") or {}).get("family"))
        if kept[family] >= required[family]:
            continue
        selected.append(episode)
        kept[family] += 1
    if kept != required:
        raise RuntimeError(
            f"Failed to construct ratio-balanced set: required={required}, kept={kept}"
        )
    return selected, duplicate_count


def _collect_parallel(args: argparse.Namespace, slots: Sequence[int]) -> None:
    """Collect scene shards and publish the largest quality-passing balanced set."""

    recovery_depth = int(getattr(args, "_recovery_depth", 0))
    output_path = Path(args.output)
    partial_path = Path(str(output_path) + ".partial")
    checkpoint_path = Path(str(output_path) + ".collect.sqlite3")
    shard_root = Path(str(output_path) + ".collect_workers")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if args.overwrite:
        output_path.unlink(missing_ok=True)
        partial_path.unlink(missing_ok=True)
        remove_collect_checkpoint(checkpoint_path)
        if shard_root.exists():
            shutil.rmtree(shard_root)
    if output_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"Trajectory output already exists; use --resume or --overwrite: {output_path}"
            )
        with gzip.open(output_path, "rt", encoding="utf-8") as handle:
            existing_dataset = json.load(handle)
        existing_episodes = existing_dataset.get("episodes")
        if not isinstance(existing_episodes, list) or (
            not existing_episodes and recovery_depth == 0
        ):
            raise ValueError(
                f"Existing trajectory output does not contain valid episodes: {output_path}"
            )
        if not existing_episodes:
            if shard_root.exists():
                shutil.rmtree(shard_root)
            print(f"[collect] recovery round already recorded no progress: {output_path}")
            return
        balanced, duplicates = _largest_ratio_balanced_set(
            existing_episodes,
            min(len(existing_episodes), int(args.num_trajectories)),
            float(args.r2r_ratio),
        )
        if len(balanced) != len(existing_episodes) or duplicates:
            raise ValueError(
                "Existing trajectory output is not unique and ratio-balanced for the "
                f"current request: {output_path}"
            )
        if shard_root.exists():
            shutil.rmtree(shard_root)
        remove_collect_checkpoint(checkpoint_path)
        partial_path.unlink(missing_ok=True)
        print(
            f"[collect] already published: {output_path} "
            f"({len(existing_episodes)}/{args.num_trajectories} desired)",
            flush=True,
        )
        return
    if checkpoint_path.exists():
        raise ValueError(
            "A single-process trajectory checkpoint already exists. Resume once with "
            "one collection process, or use --overwrite before changing to parallel collection: "
            f"{checkpoint_path}"
        )
    if shard_root.exists() and not args.resume:
        raise FileExistsError(
            f"Parallel collection state exists; rerun with --resume: {shard_root}"
        )
    if partial_path.exists() and not shard_root.exists():
        raise FileExistsError(
            "A failed partial trajectory file exists without parallel checkpoints; "
            f"inspect it and use --overwrite to restart: {partial_path}"
        )

    assignment_rng = random.Random(args.seed)
    scenes = discover_scenes(args.scene_root, args.split, args.scene_ids)
    assignment_rng.shuffle(scenes)
    worker_count = min(len(slots), len(scenes), int(args.num_trajectories))
    if worker_count <= 1:
        single_args = copy.deepcopy(args)
        single_args.gpu_device_id = int(slots[0])
        single_args.gpu_device_ids = None
        single_args.processes_per_gpu = 1
        collect(single_args)
        return

    active_slots = list(slots[:worker_count])
    scene_shards = [scenes[index::worker_count] for index in range(worker_count)]
    trajectory_counts = _balanced_counts(int(args.num_trajectories), worker_count)
    total_r2r = int(math.floor(args.num_trajectories * args.r2r_ratio + 0.5))
    family_plan = ["r2r"] * total_r2r + ["rxr"] * (args.num_trajectories - total_r2r)
    assignment_rng.shuffle(family_plan)

    shard_root.mkdir(parents=True, exist_ok=True)
    worker_args_list: List[argparse.Namespace] = []
    shard_outputs: List[Path] = []
    offset = 0
    for worker_id, (gpu_id, count, scene_shard) in enumerate(
        zip(active_slots, trajectory_counts, scene_shards)
    ):
        shard_families = family_plan[offset : offset + count]
        offset += count
        r2r_count = shard_families.count("r2r")
        shard_output = shard_root / f"worker_{worker_id:03d}.json.gz"
        shard_outputs.append(shard_output)
        worker_args = copy.deepcopy(args)
        worker_args.output = str(shard_output)
        worker_args.num_trajectories = count
        worker_args.r2r_ratio = r2r_count / count
        worker_args.rxr_ratio = 1.0 - worker_args.r2r_ratio
        worker_args.scene_ids = ",".join(scene.parent.name for scene in scene_shard)
        worker_args.seed = candidate_sampling_seed(args.seed, worker_id, count)
        worker_args.gpu_device_id = int(gpu_id)
        worker_args.gpu_device_ids = None
        worker_args.processes_per_gpu = 1
        worker_args.overwrite = False
        worker_args._disable_progress = True
        worker_args._quiet_output = True
        worker_args._allow_partial = True
        worker_args_list.append(worker_args)

    context = multiprocessing.get_context("spawn")
    processes: List[multiprocessing.Process] = []
    progress = tqdm(
        total=args.num_trajectories,
        desc="trajectory",
        unit="ep",
        dynamic_ncols=True,
    )
    observed = [0] * worker_count
    try:
        for worker_id, worker_args in enumerate(worker_args_list):
            process = context.Process(
                target=_collect_worker_entry,
                args=(worker_args,),
                name=f"trajectory-gpu{worker_args.gpu_device_id}-worker{worker_id}",
            )
            process.start()
            processes.append(process)

        while any(process.is_alive() for process in processes):
            for worker_id, (shard_output, expected) in enumerate(
                zip(shard_outputs, trajectory_counts)
            ):
                current = _collect_shard_progress(
                    shard_output,
                    Path(str(shard_output) + ".collect.sqlite3"),
                    expected,
                )
                if current > observed[worker_id]:
                    progress.update(current - observed[worker_id])
                    observed[worker_id] = current
            progress.set_postfix(
                workers=worker_count,
                alive=sum(process.is_alive() for process in processes),
                gpus=",".join(str(gpu) for gpu in sorted(set(active_slots))),
                refresh=False,
            )
            failed = [
                process
                for process in processes
                if process.exitcode not in (None, 0)
            ]
            if failed:
                raise RuntimeError(
                    "Parallel trajectory worker failed: "
                    + ", ".join(f"{process.name}={process.exitcode}" for process in failed)
                )
            time.sleep(0.25)
        for process in processes:
            process.join()
        failed = [process for process in processes if process.exitcode != 0]
        if failed:
            raise RuntimeError(
                "Parallel trajectory worker failed: "
                + ", ".join(f"{process.name}={process.exitcode}" for process in failed)
            )
        # Capture the last transactions committed immediately before each
        # worker exited. A capacity-exhausted shard can legitimately be short.
        for worker_id, expected in enumerate(trajectory_counts):
            current = _collect_shard_progress(
                shard_outputs[worker_id],
                Path(str(shard_outputs[worker_id]) + ".collect.sqlite3"),
                expected,
            )
            if current > observed[worker_id]:
                progress.update(current - observed[worker_id])
                observed[worker_id] = current
    except (Exception, KeyboardInterrupt):
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=10)
        raise
    finally:
        progress.close()

    merged: List[Dict[str, Any]] = []
    for shard_output, expected in zip(shard_outputs, trajectory_counts):
        partial_shard = Path(str(shard_output) + ".partial")
        published_shard = shard_output if shard_output.is_file() else partial_shard
        if not published_shard.is_file():
            raise FileNotFoundError(
                f"Collection worker did not publish complete or partial data: {shard_output}"
            )
        with gzip.open(published_shard, "rt", encoding="utf-8") as handle:
            shard_dataset = json.load(handle)
        shard_episodes = shard_dataset.get("episodes")
        if (
            not isinstance(shard_episodes, list)
            or len(shard_episodes) > expected
        ):
            raise ValueError(
                f"Invalid collection shard {published_shard}: "
                f"expected at most {expected} episodes"
            )
        merged.extend(shard_episodes)

    family_counts = {
        family: sum((episode.get("info") or {}).get("family") == family for episode in merged)
        for family in ("r2r", "rxr")
    }
    expected_family_counts = {
        "r2r": total_r2r,
        "rxr": args.num_trajectories - total_r2r,
    }
    remaining_family_counts = {
        family: expected_family_counts[family] - family_counts[family]
        for family in ("r2r", "rxr")
    }
    if min(remaining_family_counts.values()) < 0:
        raise RuntimeError(
            "Collection shards exceeded a requested family count: "
            f"expected={expected_family_counts}, actual={family_counts}"
        )
    missing_count = sum(remaining_family_counts.values())
    recovered_count = 0
    if missing_count:
        max_recovery_rounds = int(args.max_recovery_rounds)
        if recovery_depth < max_recovery_rounds:
            print(
                "[collect] scene shards exhausted before reaching the desired count; "
                f"starting recovery round {recovery_depth + 1}/{max_recovery_rounds} "
                f"for up to {missing_count} trajectories "
                f"(r2r={remaining_family_counts['r2r']}, "
                f"rxr={remaining_family_counts['rxr']}).",
                flush=True,
            )
            recovery_output = shard_root / "recovery.json.gz"
            recovery_args = copy.deepcopy(args)
            recovery_args.output = str(recovery_output)
            recovery_args.num_trajectories = missing_count
            recovery_args.r2r_ratio = remaining_family_counts["r2r"] / missing_count
            recovery_args.rxr_ratio = remaining_family_counts["rxr"] / missing_count
            recovery_args.seed = candidate_sampling_seed(
                args.seed,
                10_000 + recovery_depth,
                missing_count,
            )
            recovery_args.resume = True
            recovery_args.overwrite = False
            recovery_args._recovery_depth = recovery_depth + 1
            _collect_parallel(recovery_args, slots)
            with gzip.open(recovery_output, "rt", encoding="utf-8") as handle:
                recovery_dataset = json.load(handle)
            recovery_episodes = recovery_dataset.get("episodes")
            if (
                not isinstance(recovery_episodes, list)
                or len(recovery_episodes) > missing_count
            ):
                raise RuntimeError(
                    f"Trajectory recovery produced an invalid dataset: {recovery_output}"
                )
            merged.extend(recovery_episodes)
            recovered_count = len(recovery_episodes)
        else:
            print(
                "[collect] desired count is unavailable after the configured recovery "
                "rounds; publishing the largest quality-passing balanced dataset.",
                flush=True,
            )

    if not merged:
        if recovery_depth == 0:
            raise RuntimeError("Parallel collection made no progress in any scene shard")
        atomic_json_gz(
            output_path,
            {"episodes": [], "instruction_vocab": EMPTY_INSTRUCTION_VOCAB},
        )
        partial_path.unlink(missing_ok=True)
        shutil.rmtree(shard_root)
        print(
            "[collect] recovery round produced no additional quality-passing trajectories.",
            flush=True,
        )
        return

    merged, duplicate_count = _largest_ratio_balanced_set(
        merged,
        int(args.num_trajectories),
        float(args.r2r_ratio),
    )
    family_counts = {
        family: sum((episode.get("info") or {}).get("family") == family for episode in merged)
        for family in ("r2r", "rxr")
    }

    for episode_id, episode in enumerate(merged):
        episode["episode_id"] = episode_id
        episode["trajectory_id"] = episode_id
    hashes = [str((episode.get("info") or {}).get("route_hash")) for episode in merged]
    if len(set(hashes)) != len(hashes) or "None" in hashes:
        raise RuntimeError("Merged trajectory shards contain missing or duplicate route hashes")

    atomic_json_gz(
        output_path,
        {"episodes": merged, "instruction_vocab": EMPTY_INSTRUCTION_VOCAB},
    )
    partial_path.unlink(missing_ok=True)
    shutil.rmtree(shard_root)
    print(
        json.dumps(
            {
                "schema_version": "panovln-hm3d-trajectory-parallel-v1",
                "status": (
                    "target_reached"
                    if len(merged) == args.num_trajectories
                    else "quality_limited"
                ),
                "dataset": str(output_path),
                "desired_trajectories": int(args.num_trajectories),
                "trajectories": len(merged),
                "family_counts": family_counts,
                "scenes": len({episode["scene_id"] for episode in merged}),
                "workers": worker_count,
                "gpu_process_slots": active_slots,
                "recovered_trajectories": recovered_count,
                "deduplicated_trajectories": duplicate_count,
                "all_hard_checks_pass": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def collect(args: argparse.Namespace) -> None:
    if int(args.max_recovery_rounds) < 0:
        raise ValueError("--max-recovery-rounds must be non-negative")
    slots = habitat_process_slots(args)
    if len(slots) == 1:
        args.gpu_device_id = int(slots[0])
        args._allow_partial = True
        _collect_sequential(args)
        output_path = Path(args.output)
        partial_path = Path(str(output_path) + ".partial")
        if output_path.is_file():
            return
        if not partial_path.is_file():
            raise RuntimeError("Single-process collector did not publish any trajectories")
        with gzip.open(partial_path, "rt", encoding="utf-8") as handle:
            partial_dataset = json.load(handle)
        partial_episodes = partial_dataset.get("episodes")
        if not isinstance(partial_episodes, list):
            raise RuntimeError(f"Invalid partial trajectory dataset: {partial_path}")
        selected, duplicate_count = _largest_ratio_balanced_set(
            partial_episodes,
            int(args.num_trajectories),
            float(args.r2r_ratio),
        )
        for episode_id, episode in enumerate(selected):
            episode["episode_id"] = episode_id
            episode["trajectory_id"] = episode_id
        atomic_json_gz(
            output_path,
            {"episodes": selected, "instruction_vocab": EMPTY_INSTRUCTION_VOCAB},
        )
        partial_path.unlink(missing_ok=True)
        remove_collect_checkpoint(Path(str(output_path) + ".collect.sqlite3"))
        print(
            json.dumps(
                {
                    "schema_version": "panovln-hm3d-trajectory-single-v1",
                    "status": "quality_limited",
                    "dataset": str(output_path),
                    "desired_trajectories": int(args.num_trajectories),
                    "trajectories": len(selected),
                    "deduplicated_trajectories": duplicate_count,
                    "all_hard_checks_pass": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        _collect_parallel(args, slots)


def _collect_sequential(args: argparse.Namespace) -> None:
    os.environ.setdefault("MAGNUM_LOG", "quiet")
    os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")

    output_path = Path(args.output)
    partial_path = Path(str(output_path) + ".partial")
    checkpoint_path = Path(str(output_path) + ".collect.sqlite3")
    if args.overwrite:
        output_path.unlink(missing_ok=True)
        partial_path.unlink(missing_ok=True)
        remove_collect_checkpoint(checkpoint_path)
    if output_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"Trajectory output already exists; use --resume or --overwrite: {output_path}"
            )
        with gzip.open(output_path, "rt", encoding="utf-8") as handle:
            existing_dataset = json.load(handle)
        existing_episodes = existing_dataset.get("episodes")
        if not isinstance(existing_episodes, list) or not existing_episodes:
            raise ValueError(
                f"Existing trajectory output does not contain valid episodes: {output_path}"
            )
        balanced, duplicates = _largest_ratio_balanced_set(
            existing_episodes,
            min(len(existing_episodes), int(args.num_trajectories)),
            float(args.r2r_ratio),
        )
        if len(balanced) != len(existing_episodes) or duplicates:
            raise ValueError(
                "Existing trajectory output is not unique and ratio-balanced for the "
                f"current request: {output_path}"
            )
        remove_collect_checkpoint(checkpoint_path)
        partial_path.unlink(missing_ok=True)
        print(
            f"[collect] already published: {output_path} "
            f"({len(existing_episodes)}/{args.num_trajectories} desired)",
            flush=True,
        )
        return
    if partial_path.exists() and not checkpoint_path.exists():
        raise FileExistsError(
            "A legacy/failed partial trajectory file exists without a resumable "
            f"checkpoint; inspect it and use --overwrite to restart: {partial_path}"
        )

    rng = random.Random(args.seed)
    scenes = discover_scenes(
        args.scene_root,
        args.split,
        args.scene_ids,
    )
    # A seeded shuffle avoids lexicographic scene bias while remaining reproducible.
    rng.shuffle(scenes)
    eligible_scene_count = len(scenes)
    families = distribute_families(
        args.r2r_ratio,
        args.rxr_ratio,
        args.num_trajectories,
        rng,
    )
    expected_family_counts = {
        family: families.count(family) for family in ("r2r", "rxr")
    }
    r2r_distances = (
        read_profile_distances(
            args.trajectory_profile_config,
            "r2r",
            args.min_distance,
            args.r2r_max_distance,
        )
        if expected_family_counts["r2r"]
        else []
    )
    rxr_distances = (
        read_profile_distances(
            args.trajectory_profile_config,
            "rxr",
            args.min_distance,
            args.rxr_max_distance,
        )
        if expected_family_counts["rxr"]
        else []
    )
    planned_detours = distribute_detours(families, args.rxr_detour_fraction, rng)
    initial_state = {
        "scene_index": 0,
        "scene_quota": 0,
        "accepted_in_scene": 0,
        "consecutive_failures_in_scene": 0,
        "attempt_index": 0,
        "collected_count": 0,
        "sampling_failure_count": 0,
        "sampling_failure_preview": [],
        "rng_state": rng.getstate(),
    }
    checkpoint = CollectCheckpoint(
        checkpoint_path,
        collect_run_manifest(args, scenes),
        initial_state,
        resume=bool(args.resume),
    )
    try:
        state = checkpoint.state
        collected = checkpoint.load_trajectories()
    except Exception:
        checkpoint.close()
        raise
    if not 0 <= int(state.get("scene_index", -1)) <= len(scenes):
        checkpoint.close()
        raise ValueError("Trajectory checkpoint has an invalid scene index")
    if len(collected) > args.num_trajectories:
        checkpoint.close()
        raise ValueError("Trajectory checkpoint exceeds requested trajectory count")
    rng.setstate(nested_tuple(state["rng_state"]))
    hashes = {str(row["route_hash"]) for row in collected}
    if len(hashes) != len(collected):
        checkpoint.close()
        raise ValueError("Trajectory checkpoint contains duplicate route hashes")
    family_counts = {
        family: sum(row["family"] == family for row in collected)
        for family in ("r2r", "rxr")
    }
    progress = tqdm(
        total=args.num_trajectories,
        initial=len(collected),
        desc=str(getattr(args, "_progress_desc", "trajectory")),
        unit="ep",
        dynamic_ncols=True,
        disable=bool(getattr(args, "_disable_progress", False)),
    )

    def update_state(
        *,
        scene_index: int,
        scene_quota: int,
        accepted_in_scene: int,
        consecutive_failures: int,
        attempt_index: int,
        collected_count: int,
        failure_count: int,
        failure_preview: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "scene_index": scene_index,
            "scene_quota": scene_quota,
            "accepted_in_scene": accepted_in_scene,
            "consecutive_failures_in_scene": consecutive_failures,
            "attempt_index": attempt_index,
            "collected_count": collected_count,
            "sampling_failure_count": failure_count,
            "sampling_failure_preview": list(failure_preview),
            "rng_state": rng.getstate(),
        }

    try:
        scene_index = int(state["scene_index"])
        while scene_index < len(scenes) and len(collected) < args.num_trajectories:
            remaining_scenes = len(scenes) - scene_index
            remaining_routes = args.num_trajectories - len(collected)
            scene_quota = int(state.get("scene_quota", 0))
            if scene_quota <= 0:
                scene_quota = math.ceil(remaining_routes / remaining_scenes)
                state = update_state(
                    scene_index=scene_index,
                    scene_quota=scene_quota,
                    accepted_in_scene=0,
                    consecutive_failures=0,
                    attempt_index=int(state["attempt_index"]),
                    collected_count=len(collected),
                    failure_count=int(state["sampling_failure_count"]),
                    failure_preview=state["sampling_failure_preview"],
                )
                checkpoint.commit(state)

            scene = scenes[scene_index]
            scene_id = public_scene_id(scene, args.scene_root)
            accepted_here = int(state["accepted_in_scene"])
            exhausted_here = int(state["consecutive_failures_in_scene"])
            attempt_index = int(state["attempt_index"])
            failure_count = int(state["sampling_failure_count"])
            failure_preview = list(state["sampling_failure_preview"])
            simulator = make_simulator(scene, args, args.seed + scene_index)
            try:
                while accepted_here < scene_quota and len(collected) < args.num_trajectories:
                    episode_index = len(collected)
                    family = families[episode_index]
                    distances = r2r_distances if family == "r2r" else rxr_distances
                    target_distance = rng.choice(distances)
                    make_detour = planned_detours[episode_index]
                    progress.set_postfix(
                        scene=scene.parent.name,
                        family=family,
                        r2r=family_counts["r2r"],
                        rxr=family_counts["rxr"],
                        failures=failure_count,
                        attempts=attempt_index,
                        refresh=False,
                    )
                    simulator.pathfinder.seed(
                        candidate_sampling_seed(args.seed, scene_index, attempt_index)
                    )
                    sampled = sample_one(
                        simulator,
                        family,
                        target_distance,
                        make_detour,
                        rng,
                        args,
                    )
                    attempt_index += 1
                    failure: Optional[Dict[str, Any]] = None
                    if sampled is None:
                        exhausted_here += 1
                        failure = {
                            "scene_id": scene_id,
                            "family": family,
                            "target_distance": target_distance,
                            "detour": make_detour,
                            "reason": "sampling_attempts_exhausted",
                        }
                    else:
                        visual_metrics = route_visual_metrics(
                            simulator,
                            sampled["route_keypoints"],
                            args.max_visual_checkpoints,
                        )
                        sampled["metrics"].update(visual_metrics)
                        if visual_metrics["visual_black_ratio_max"] > args.max_black_ratio:
                            exhausted_here += 1
                            failure = {
                                "scene_id": scene_id,
                                "family": family,
                                "target_distance": target_distance,
                                "detour": make_detour,
                                "reason": "visual_scan_void",
                                **visual_metrics,
                            }

                    digest = None
                    if sampled is not None and failure is None:
                        digest = route_hash(scene_id, sampled["reference_path"])
                        if digest in hashes:
                            exhausted_here += 1
                            failure = {
                                "scene_id": scene_id,
                                "family": family,
                                "target_distance": target_distance,
                                "detour": make_detour,
                                "reason": "duplicate_route_hash",
                            }

                    if failure is not None:
                        failure_count += 1
                        if len(failure_preview) < 20:
                            failure_preview.append(failure)
                        state = update_state(
                            scene_index=scene_index,
                            scene_quota=scene_quota,
                            accepted_in_scene=accepted_here,
                            consecutive_failures=exhausted_here,
                            attempt_index=attempt_index,
                            collected_count=len(collected),
                            failure_count=failure_count,
                            failure_preview=failure_preview,
                        )
                        checkpoint.commit(state)
                        if exhausted_here >= args.max_sampling_failures_per_scene:
                            break
                        continue

                    assert sampled is not None and digest is not None
                    episode_id = len(collected)
                    sampled.update(
                        {
                            "episode_id": episode_id,
                            # Physical trajectory IDs also key panorama directories.
                            "trajectory_id": episode_id,
                            "scene_id": scene_id,
                            "scene_path": str(scene),
                            "route_hash": digest,
                            "seed": args.seed,
                        }
                    )
                    accepted_here += 1
                    exhausted_here = 0
                    state = update_state(
                        scene_index=scene_index,
                        scene_quota=scene_quota,
                        accepted_in_scene=accepted_here,
                        consecutive_failures=exhausted_here,
                        attempt_index=attempt_index,
                        collected_count=len(collected) + 1,
                        failure_count=failure_count,
                        failure_preview=failure_preview,
                    )
                    checkpoint.commit(state, trajectory=sampled)
                    collected.append(sampled)
                    hashes.add(digest)
                    family_counts[family] += 1
                    progress.set_postfix(
                        scene=scene.parent.name,
                        family=family,
                        r2r=family_counts["r2r"],
                        rxr=family_counts["rxr"],
                        failures=failure_count,
                        attempts=attempt_index,
                        refresh=False,
                    )
                    progress.update(1)
            finally:
                simulator.close()

            scene_index += 1
            state = update_state(
                scene_index=scene_index,
                scene_quota=0,
                accepted_in_scene=0,
                consecutive_failures=0,
                attempt_index=attempt_index,
                collected_count=len(collected),
                failure_count=failure_count,
                failure_preview=failure_preview,
            )
            checkpoint.commit(state)
    finally:
        progress.close()
        checkpoint.close()

    episodes = []
    for row in collected:
        info = dict(row["metrics"])
        info.update(
            {
                "family": row["family"],
                "is_detour": row["is_detour"],
                "detour_anchor": row["detour_anchor"],
                "route_hash": row["route_hash"],
                "sampling_seed": row["seed"],
                "route_keypoints": row["route_keypoints"],
                "navmesh_island": row["navmesh_island"],
                **row["yaw_metadata"],
            }
        )
        episodes.append(
            {
                "episode_id": row["episode_id"],
                "trajectory_id": row["trajectory_id"],
                "scene_id": row["scene_id"],
                "start_position": row["start_position"],
                "start_rotation": row["start_rotation"],
                "info": info,
                "goals": [
                    {"position": row["goal_position"], "radius": args.goal_radius}
                ],
                "instruction": {
                    "instruction_text": "",
                    "instruction_tokens": None,
                },
                "reference_path": row["reference_path"],
            }
        )

    summary = summarize_trajectories(collected)
    hard_checks = {
        "family_counts_match_request": (
            summary["family_counts"] == expected_family_counts
        ),
        "all_unique_hashes": summary["unique_route_hashes"] == len(collected),
        "all_endpoint_clearance": all(
            min(row["metrics"]["start_clearance"], row["metrics"]["goal_clearance"])
            >= args.endpoint_clearance
            for row in collected
        ),
        "all_distance_bounds": all(
            args.min_distance <= row["metrics"]["reference_length"]
            <= (args.r2r_max_distance if row["family"] == "r2r" else args.rxr_max_distance)
            for row in collected
        ),
        "all_reference_turns": all(
            row["metrics"]["max_planar_turn_degrees"]
            <= args.max_reference_turn_degrees
            for row in collected
        ),
        "no_nonlocal_revisits": all(
            row["metrics"]["nonlocal_revisit_count"] == 0 for row in collected
        ),
        "all_detour_anchor_turns": all(
            not row["is_detour"]
            or float(row["metrics"]["detour_anchor_turn_degrees"])
            <= args.max_anchor_turn_degrees
            for row in collected
        ),
        "all_detour_anchor_closures": all(
            not row["is_detour"]
            or float(row["metrics"]["detour_anchor_closure_ratio"])
            >= args.min_anchor_closure_ratio
            for row in collected
        ),
        "no_near_goal_departures": all(
            not row["metrics"]["near_goal_departure"] for row in collected
        ),
        "all_heading_density_below_hard_limit": all(
            row["metrics"]["heading_change_per_meter"]
            <= args.max_heading_change_per_meter
            for row in collected
        ),
        "all_visual_prechecks_pass": all(
            row["metrics"]["visual_black_ratio_max"] <= args.max_black_ratio
            for row in collected
        ),
    }
    complete = len(collected) == args.num_trajectories and all(hard_checks.values())
    dataset_path = output_path if complete else partial_path
    publication = {
        "schema_version": "panovln-hm3d-trajectory-v1",
        "status": "complete" if complete else "failed_partial",
        "dataset": str(dataset_path),
        "scene_root": str(Path(args.scene_root).resolve()),
        "split": args.split,
        "eligible_scenes": eligible_scene_count,
        "seed": args.seed,
        "requested_family_ratios": {
            "r2r": args.r2r_ratio,
            "rxr": args.rxr_ratio,
        },
        "expected_family_counts": expected_family_counts,
        "summary": summary,
        "sampling_failures": int(state["sampling_failure_count"]),
        "sampling_failure_preview": state["sampling_failure_preview"],
        "hard_checks": hard_checks,
    }
    atomic_json_gz(
        dataset_path,
        {"episodes": episodes, "instruction_vocab": EMPTY_INSTRUCTION_VOCAB},
    )
    if complete:
        partial_path.unlink(missing_ok=True)
        remove_collect_checkpoint(checkpoint_path)
    if not bool(getattr(args, "_quiet_output", False)):
        print(json.dumps(publication, ensure_ascii=False, indent=2))
    if len(collected) != args.num_trajectories:
        if bool(getattr(args, "_allow_partial", False)):
            return
        raise RuntimeError(
            f"Collected {len(collected)}/{args.num_trajectories}; "
            f"sampling_failures={state['sampling_failure_count']}"
        )
    if not all(hard_checks.values()):
        raise RuntimeError(f"Hard collection checks failed: {hard_checks}")


def oscillation_count(actions: Sequence[int]) -> int:
    turns = [int(action) for action in actions if int(action) in {2, 3}]
    return sum(left != right for left, right in zip(turns[:-1], turns[1:]))


def validate_gt(args: argparse.Namespace) -> None:
    with gzip.open(args.dataset, "rt", encoding="utf-8") as handle:
        dataset = json.load(handle)
    with gzip.open(args.gt, "rt", encoding="utf-8") as handle:
        gt = json.load(handle)
    episodes = {str(row["episode_id"]): row for row in dataset["episodes"]}
    rows = []
    for episode_id, episode in episodes.items():
        record = gt.get(episode_id)
        issues = []
        if record is None:
            issues.append("missing_gt")
            actions: List[int] = []
            locations: List[List[float]] = []
        else:
            actions = [int(action) for action in record.get("actions", [])]
            locations = record.get("locations") or []
        invalid_actions = sorted(set(actions) - {1, 2, 3})
        if invalid_actions:
            issues.append(f"invalid_or_terminal_actions_{invalid_actions}")
        if len(actions) > args.max_actions:
            issues.append("too_many_actions")
        forward_actions = actions.count(1)
        forward_steps = int(record.get("forward_steps", -1)) if record else -1
        if forward_actions != forward_steps:
            issues.append("forward_collision_or_accounting_mismatch")
        if locations and len(locations) != forward_steps + 1:
            issues.append("location_count_mismatch")
        if not locations:
            issues.append("missing_locations")
            final_error = None
        else:
            goal = vec(episode["goals"][0]["position"])
            final_error = float(np.linalg.norm(vec(locations[-1]) - goal))
            if final_error > args.goal_radius + 1e-4:
                issues.append("goal_radius_failure")
        leading_turns = 0
        for action in actions:
            if action not in {2, 3}:
                break
            leading_turns += 1
        if leading_turns * 15 >= 360:
            issues.append("leading_full_spin")
        rows.append(
            {
                "episode_id": episode_id,
                "family": episode.get("info", {}).get("family"),
                "is_detour": episode.get("info", {}).get("is_detour"),
                "action_count": len(actions),
                "forward_actions": forward_actions,
                "turn_actions": sum(action in {2, 3} for action in actions),
                "leading_turn_degrees": leading_turns * 15,
                "turn_direction_changes": oscillation_count(actions),
                "final_euclidean_error": final_error,
                "issues": issues,
            }
        )
    issue_counts: Dict[str, int] = {}
    for row in rows:
        for issue in row["issues"]:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
    action_counts = [row["action_count"] for row in rows]
    extra_gt_ids = sorted(set(gt) - set(episodes))
    if extra_gt_ids:
        issue_counts["extra_gt_records"] = len(extra_gt_ids)
    source_all_hard_checks_pass = not issue_counts and len(episodes) == len(gt)
    valid_episode_ids = [row["episode_id"] for row in rows if not row["issues"]]
    invalid_episode_ids = [row["episode_id"] for row in rows if row["issues"]]
    filtered = False
    if args.drop_invalid and not source_all_hard_checks_pass:
        if not valid_episode_ids:
            raise RuntimeError(
                "Every trajectory failed GT validation; refusing to publish an empty dataset"
            )
        valid_id_set = set(valid_episode_ids)
        filtered_dataset = dict(dataset)
        filtered_dataset["episodes"] = [
            episode
            for episode in dataset["episodes"]
            if str(episode["episode_id"]) in valid_id_set
        ]
        filtered_gt = {
            episode_id: record
            for episode_id, record in gt.items()
            if episode_id in valid_id_set
        }
        atomic_json_gz(Path(args.gt), filtered_gt)
        atomic_json_gz(Path(args.dataset), filtered_dataset)
        filtered = True

    command_succeeded = source_all_hard_checks_pass or filtered
    summary = {
        "episodes": len(episodes),
        "gt_records": len(gt),
        "passed": sum(not row["issues"] for row in rows),
        "failed": sum(bool(row["issues"]) for row in rows),
        "issue_counts": issue_counts,
        "action_count_mean": statistics.mean(action_counts) if action_counts else None,
        "action_count_max": max(action_counts) if action_counts else None,
        "source_all_hard_checks_pass": source_all_hard_checks_pass,
        "filtered_invalid_episodes": len(invalid_episode_ids) if filtered else 0,
        "filtered_extra_gt_records": len(extra_gt_ids) if filtered else 0,
        "published_episodes": len(valid_episode_ids) if filtered else len(episodes),
        "dropped_episode_preview": invalid_episode_ids[:20] if filtered else [],
        "all_hard_checks_pass": command_succeeded,
    }
    generation_rows = []
    if args.generation_jsonl:
        validation_by_id = {row["episode_id"]: row for row in rows}
        for episode_id, episode in episodes.items():
            if validation_by_id[episode_id]["issues"]:
                continue
            gt_record = gt[episode_id]
            actions = [int(action) for action in gt_record.get("actions", [])]
            locations = [vec(item) for item in (gt_record.get("locations") or [])]
            elevations = [float(item[1]) for item in locations]
            elevation_deltas = [
                right - left for left, right in zip(elevations[:-1], elevations[1:])
            ]
            upward_travel = sum(max(0.0, delta) for delta in elevation_deltas)
            downward_travel = sum(max(0.0, -delta) for delta in elevation_deltas)
            net_elevation_change = (
                elevations[-1] - elevations[0] if elevations else 0.0
            )
            if upward_travel >= 0.5 and downward_travel >= 0.5:
                vertical_motion = "mixed"
            elif net_elevation_change >= 0.5:
                vertical_motion = "ascending"
            elif net_elevation_change <= -0.5:
                vertical_motion = "descending"
            else:
                vertical_motion = "level"
            generation_rows.append(
                {
                    "episode_id": episode["episode_id"],
                    "trajectory_id": episode.get("trajectory_id"),
                    "actions": actions + [0],
                    "trajectory_metadata": {
                        "vertical_motion": vertical_motion,
                        "net_elevation_change_m": round(net_elevation_change, 4),
                        "upward_travel_m": round(upward_travel, 4),
                        "downward_travel_m": round(downward_travel, 4),
                    },
                }
            )
        generation_path = Path(args.generation_jsonl)
        if command_succeeded:
            Path(str(generation_path) + ".partial").unlink(missing_ok=True)
            write_jsonl(generation_path, generation_rows)
            summary["generation_jsonl"] = str(generation_path)
        else:
            generation_path.unlink(missing_ok=True)
            partial_path = Path(str(generation_path) + ".partial")
            write_jsonl(partial_path, generation_rows)
            summary["generation_jsonl_partial"] = str(partial_path)
        summary["generation_rows"] = len(generation_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not command_succeeded:
        raise RuntimeError(f"Trajectory GT validation failed: {issue_counts}")


def make_panorama_simulator(scene: Path, args: argparse.Namespace):
    import habitat_sim

    simulator_config = habitat_sim.SimulatorConfiguration()
    simulator_config.scene_id = str(scene)
    simulator_config.enable_physics = False
    simulator_config.gpu_device_id = int(args.gpu_device_id)
    sensor = habitat_sim.EquirectangularSensorSpec()
    sensor.uuid = "rgb"
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.resolution = [int(args.height), int(args.width)]
    sensor.position = [0.0, float(args.sensor_height), 0.0]
    agent_config = habitat_sim.agent.AgentConfiguration()
    agent_config.sensor_specifications = [sensor]
    agent_config.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward",
            habitat_sim.agent.ActuationSpec(amount=float(args.forward_step_size)),
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left",
            habitat_sim.agent.ActuationSpec(amount=float(args.turn_angle)),
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right",
            habitat_sim.agent.ActuationSpec(amount=float(args.turn_angle)),
        ),
    }
    simulator = habitat_sim.Simulator(
        habitat_sim.Configuration(simulator_config, [agent_config])
    )
    if not simulator.pathfinder.is_loaded:
        simulator.close()
        raise RuntimeError(f"Navmesh failed to load for panorama rendering: {scene}")
    simulator.initialize_agent(0)
    return simulator


def normalize_panorama_image_format(image_format: str) -> str:
    normalized = str(image_format).strip().lower()
    if normalized in {"jpg", "jpeg"}:
        return "jpeg"
    if normalized == "png":
        return "png"
    raise ValueError(f"Unsupported panorama image format: {image_format!r}")


def panorama_frame_filename(frame_index: int, image_format: str) -> str:
    extension = (
        ".png"
        if normalize_panorama_image_format(image_format) == "png"
        else ".jpg"
    )
    return f"frame_{int(frame_index)}{extension}"


def save_panorama(
    observation: np.ndarray,
    path: Path,
    image_format: str,
    jpeg_quality: int,
    png_compress_level: int,
) -> float:
    from PIL import Image

    rgb = np.asarray(observation)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"Unexpected RGB panorama shape: {rgb.shape}")
    rgb = rgb[:, :, :3].astype(np.uint8)
    black_ratio = float(np.mean(np.max(rgb, axis=2) <= 3))
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(rgb)
    if image_format == "png":
        image.save(path, format="PNG", compress_level=int(png_compress_level))
    else:
        image.save(path, format="JPEG", quality=int(jpeg_quality))
    return black_ratio


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, allow_nan=False)
            handle.flush()
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def render_run_manifest(
    args: argparse.Namespace,
    episodes: Sequence[Dict[str, Any]],
    dataset_total: int,
) -> Dict[str, Any]:
    """Fingerprint every input that can change the rendered observations."""

    dataset_path = Path(args.dataset).resolve()
    gt_path = Path(args.gt).resolve()
    configuration = {
        "dataset_sha256": sha256_file(dataset_path),
        "gt_sha256": sha256_file(gt_path),
        "scene_root": str(Path(args.scene_root).resolve()),
        "dataset_total": dataset_total,
        "selected_episode_ids": [str(episode["episode_id"]) for episode in episodes],
        "width": int(args.width),
        "height": int(args.height),
        "sensor_height": float(args.sensor_height),
        "forward_step_size": float(args.forward_step_size),
        "turn_angle": float(args.turn_angle),
        "image_format": str(args.image_format),
        "jpeg_quality": int(args.jpeg_quality),
        "png_compress_level": int(args.png_compress_level),
        "max_black_ratio": float(args.max_black_ratio),
        "pose_tolerance": float(args.pose_tolerance),
    }
    encoded = json.dumps(
        configuration, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "schema_version": "panovln-render-resume-v1",
        "fingerprint": hashlib.sha256(encoded).hexdigest(),
        "configuration": configuration,
    }


def read_render_journal(path: Path) -> Dict[str, Dict[str, Any]]:
    """Read the latest episode records and repair a crash-truncated final line."""

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
            if not isinstance(row.get("issues"), list):
                raise ValueError(f"Render journal row {index + 1} has no issues list")
            records[episode_id] = row
            valid_bytes += len(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            if index != len(raw_lines) - 1:
                raise ValueError(
                    f"Invalid render journal at line {index + 1}: {path}"
                ) from error
            with path.open("r+b") as handle:
                handle.truncate(valid_bytes)
            break
    return records


def append_render_journal(handle, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()


def completed_render_is_reusable(
    result: Optional[Dict[str, Any]],
    output_root: Path,
    episode_id: str,
    gt_record: Optional[Dict[str, Any]],
    image_format: str,
    include_failed: bool = False,
) -> bool:
    """Trust only an atomically published episode directory with matching journal data."""

    if result is None or not isinstance(gt_record, dict):
        return False
    failed = bool(result.get("issues"))
    if failed and not include_failed:
        return False
    actions = gt_record.get("actions")
    if not isinstance(actions, list):
        return False
    expected_frames = len(actions) + 1
    if (
        result.get("frame_count") != expected_frames
        or result.get("expected_frame_count") != expected_frames
    ):
        return False
    directory = (
        output_root / f".episode_{episode_id}.failed"
        if failed
        else output_root / episode_id
    )
    first_frame = panorama_frame_filename(0, image_format)
    last_frame = panorama_frame_filename(expected_frames - 1, image_format)
    return (
        directory.is_dir()
        and (directory / first_frame).is_file()
        and (directory / last_frame).is_file()
    )


def _stage_json_gz(path: Path, payload: Dict[str, Any]) -> Path:
    temporary = path.with_name(f".{path.name}.render-filter.{os.getpid()}")
    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, allow_nan=False)
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _stage_filtered_generation_jsonl(
    path: Path,
    selected_ids: Set[str],
    successful_ids: Set[str],
) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing generation JSONL required for filtering: {path}")
    seen: Set[str] = set()
    temporary = path.with_name(f".{path.name}.render-filter.{os.getpid()}")
    retained = 0
    try:
        with path.open("r", encoding="utf-8") as source, temporary.open(
            "w", encoding="utf-8"
        ) as destination:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    episode_id = str(row["episode_id"])
                except (json.JSONDecodeError, KeyError, TypeError) as error:
                    raise ValueError(
                        f"Invalid generation JSONL row {path}:{line_number}"
                    ) from error
                if episode_id in seen:
                    raise ValueError(
                        f"Duplicate generation episode_id={episode_id} in {path}"
                    )
                if episode_id not in selected_ids:
                    raise ValueError(
                        f"Generation episode_id={episode_id} is absent from render dataset"
                    )
                seen.add(episode_id)
                if episode_id in successful_ids:
                    destination.write(
                        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                    )
                    retained += 1
        missing = selected_ids - seen
        if missing:
            raise ValueError(
                f"Generation JSONL is missing {len(missing)} rendered episodes; "
                f"examples={sorted(missing)[:10]}"
            )
        if retained != len(successful_ids):
            raise RuntimeError(
                f"Filtered generation row count mismatch: {retained} != "
                f"{len(successful_ids)}"
            )
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def publish_filtered_render_sources(
    args: argparse.Namespace,
    dataset: Dict[str, Any],
    gt: Dict[str, Any],
    selected_ids: Set[str],
    successful_ids: Set[str],
) -> None:
    """Stage aligned trajectory, GT, and agent inputs before replacing sources."""

    if not args.generation_jsonl:
        raise ValueError("--drop-invalid requires --generation-jsonl")
    filtered_dataset = dict(dataset)
    filtered_dataset["episodes"] = [
        episode
        for episode in dataset["episodes"]
        if str(episode["episode_id"]) in successful_ids
    ]
    filtered_gt = {
        episode_id: record
        for episode_id, record in gt.items()
        if episode_id in successful_ids
    }
    if len(filtered_dataset["episodes"]) != len(successful_ids):
        raise RuntimeError("Filtered trajectory dataset does not match successful images")
    if len(filtered_gt) != len(successful_ids):
        raise RuntimeError("Filtered GT dataset does not match successful images")

    dataset_path = Path(args.dataset)
    gt_path = Path(args.gt)
    generation_path = Path(args.generation_jsonl)
    staged: List[Tuple[Path, Path]] = []
    try:
        staged.append((dataset_path, _stage_json_gz(dataset_path, filtered_dataset)))
        staged.append((gt_path, _stage_json_gz(gt_path, filtered_gt)))
        staged.append(
            (
                generation_path,
                _stage_filtered_generation_jsonl(
                    generation_path, selected_ids, successful_ids
                ),
            )
        )
        for destination, temporary in staged:
            os.replace(temporary, destination)
    finally:
        for _, temporary in staged:
            temporary.unlink(missing_ok=True)


def replace_directory_atomically(staging: Path, destination: Path) -> None:
    """Publish a complete render tree without exposing per-episode partial state."""
    backup = Path(str(destination) + ".previous")
    if backup.exists():
        shutil.rmtree(backup)
    destination.parent.mkdir(parents=True, exist_ok=True)
    had_destination = destination.exists()
    if had_destination:
        os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except Exception:
        if had_destination and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def _render_one_panorama_episode(
    simulator,
    episode: Dict[str, Any],
    record: Optional[Dict[str, Any]],
    args: argparse.Namespace,
    output_root: Path,
) -> Dict[str, Any]:
    """Render and atomically publish one episode inside a scene-owned worker."""

    from habitat_sim import AgentState
    from habitat_sim.utils.common import quat_from_coeffs

    episode_id = str(episode["episode_id"])
    scene_id = str(episode["scene_id"])
    episode_dir = output_root / episode_id
    temporary_dir = output_root / f".episode_{episode_id}.tmp"
    failed_dir = output_root / f".episode_{episode_id}.failed"
    for stale_path in (episode_dir, temporary_dir, failed_dir):
        if stale_path.exists():
            shutil.rmtree(stale_path)

    if not isinstance(record, dict):
        return {
            "episode_id": episode_id,
            "scene_id": scene_id,
            "frame_count": 0,
            "expected_frame_count": 0,
            "collision_count": 0,
            "pose_error_max": None,
            "final_goal_error": None,
            "black_ratio_mean": None,
            "black_ratio_max": None,
            "issues": ["missing_gt"],
        }

    action_name = {1: "move_forward", 2: "turn_left", 3: "turn_right"}
    actions = [int(action) for action in record.get("actions", [])]
    locations = [vec(location) for location in record.get("locations", [])]
    state = AgentState()
    state.position = np.asarray(episode["start_position"], dtype=np.float32)
    state.rotation = quat_from_coeffs(
        np.asarray(episode["start_rotation"], dtype=np.float64)
    )
    simulator.get_agent(0).set_state(state, reset_sensors=True)
    temporary_dir.mkdir(parents=True, exist_ok=True)
    black_ratios: List[float] = []
    issues: List[str] = []
    pose_errors: List[float] = []
    start_pose = np.asarray(
        simulator.get_agent(0).get_state().position, dtype=np.float64
    )
    if not locations:
        issues.append("missing_locations")
    else:
        start_error = float(np.linalg.norm(start_pose - locations[0]))
        pose_errors.append(start_error)
        if start_error > args.pose_tolerance:
            issues.append("start_pose_mismatch")
    observation = simulator.get_sensor_observations()["rgb"]
    black_ratios.append(
        save_panorama(
            observation,
            temporary_dir / panorama_frame_filename(0, args.image_format),
            args.image_format,
            args.jpeg_quality,
            args.png_compress_level,
        )
    )
    collision_count = 0
    location_index = 0
    for action_index, raw_action in enumerate(actions, start=1):
        action = int(raw_action)
        if action not in action_name:
            issues.append(f"unsupported_action_{action}")
            break
        before = np.asarray(
            simulator.get_agent(0).get_state().position, dtype=np.float64
        )
        observations = simulator.step(action_name[action])
        after = np.asarray(
            simulator.get_agent(0).get_state().position, dtype=np.float64
        )
        if action == 1 and float(np.linalg.norm(after - before)) < (
            args.forward_step_size * 0.5
        ):
            collision_count += 1
        if action == 1:
            location_index += 1
            if location_index >= len(locations):
                issues.append("missing_gt_location_for_forward")
            else:
                pose_error = float(np.linalg.norm(after - locations[location_index]))
                pose_errors.append(pose_error)
                if pose_error > args.pose_tolerance:
                    issues.append("replay_pose_mismatch")
        black_ratios.append(
            save_panorama(
                observations["rgb"],
                temporary_dir
                / panorama_frame_filename(action_index, args.image_format),
                args.image_format,
                args.jpeg_quality,
                args.png_compress_level,
            )
        )
    if locations and location_index != len(locations) - 1:
        issues.append("unused_gt_locations")
    final_position = np.asarray(
        simulator.get_agent(0).get_state().position, dtype=np.float64
    )
    goal = vec(episode["goals"][0]["position"])
    goal_radius = float(episode["goals"][0].get("radius", 0.3))
    final_goal_error = float(np.linalg.norm(final_position - goal))
    if final_goal_error > goal_radius + args.pose_tolerance:
        issues.append("render_goal_radius_failure")
    expected_frames = len(actions) + 1
    frame_extension = ".png" if args.image_format == "png" else ".jpg"
    actual_frames = len(list(temporary_dir.glob(f"frame_*{frame_extension}")))
    if actual_frames != expected_frames:
        issues.append("frame_count_mismatch")
    if collision_count:
        issues.append("render_replay_collision")
    if black_ratios and max(black_ratios) > args.max_black_ratio:
        issues.append("excessive_black_pixels")
    issues = sorted(set(issues))
    result = {
        "episode_id": episode_id,
        "scene_id": scene_id,
        "frame_count": actual_frames,
        "expected_frame_count": expected_frames,
        "collision_count": collision_count,
        "pose_error_max": max(pose_errors) if pose_errors else None,
        "final_goal_error": final_goal_error,
        "black_ratio_mean": statistics.mean(black_ratios),
        "black_ratio_max": max(black_ratios),
        "issues": issues,
    }
    target_dir = episode_dir if not issues else failed_dir
    os.replace(temporary_dir, target_dir)
    return result


def _panorama_worker_entry(
    worker_id: int,
    gpu_device_id: int,
    input_path: str,
    args: argparse.Namespace,
    output_root: str,
    result_queue,
) -> None:
    """Persistent Habitat process: each listed scene is loaded exactly once."""

    os.environ.setdefault("MAGNUM_LOG", "quiet")
    os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
    worker_args = copy.deepcopy(args)
    worker_args.gpu_device_id = int(gpu_device_id)
    simulator = None
    try:
        with gzip.open(input_path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        for batch in payload["scene_batches"]:
            scene_id = str(batch["scene_id"])
            scene_path = resolve_scene_path(worker_args.scene_root, scene_id)
            simulator = make_panorama_simulator(scene_path, worker_args)
            try:
                for item in batch["items"]:
                    result = _render_one_panorama_episode(
                        simulator,
                        item["episode"],
                        item.get("gt"),
                        worker_args,
                        Path(output_root),
                    )
                    result_queue.put({"type": "result", "result": result})
            finally:
                simulator.close()
                simulator = None
        result_queue.put({"type": "done", "worker_id": worker_id})
    except BaseException as error:
        if simulator is not None:
            simulator.close()
        result_queue.put(
            {
                "type": "error",
                "worker_id": worker_id,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
        raise


def _parallel_render_batches(
    pending_episodes: Sequence[Dict[str, Any]],
    gt: Dict[str, Any],
    args: argparse.Namespace,
    output_root: Path,
    journal_handle,
    results_by_id: Dict[str, Dict[str, Any]],
    progress,
    resumed_episodes: int,
) -> None:
    """Assign whole scenes to fixed GPU workers and journal results in the parent."""

    slots = habitat_process_slots(args)
    by_scene: Dict[str, List[Dict[str, Any]]] = {}
    for episode in pending_episodes:
        by_scene.setdefault(str(episode["scene_id"]), []).append(episode)
    worker_count = min(len(slots), len(by_scene))
    if worker_count <= 0:
        return
    active_slots = slots[:worker_count]
    assignments: List[List[Tuple[str, List[Dict[str, Any]]]]] = [
        [] for _ in range(worker_count)
    ]
    loads = [0] * worker_count
    weighted_scenes = []
    for scene_id, scene_episodes in by_scene.items():
        weight = sum(
            len((gt.get(str(episode["episode_id"])) or {}).get("actions", [])) + 1
            for episode in scene_episodes
        )
        weighted_scenes.append((weight, scene_id, scene_episodes))
    for weight, scene_id, scene_episodes in sorted(
        weighted_scenes, key=lambda item: (-item[0], item[1])
    ):
        worker_id = min(range(worker_count), key=lambda index: (loads[index], index))
        assignments[worker_id].append((scene_id, scene_episodes))
        loads[worker_id] += weight

    worker_root = output_root / ".render_workers"
    if worker_root.exists():
        shutil.rmtree(worker_root)
    worker_root.mkdir(parents=True)
    input_paths: List[Path] = []
    for worker_id, batches in enumerate(assignments):
        input_path = worker_root / f"worker_{worker_id:03d}.json.gz"
        atomic_json_gz(
            input_path,
            {
                "scene_batches": [
                    {
                        "scene_id": scene_id,
                        "items": [
                            {
                                "episode": episode,
                                "gt": gt.get(str(episode["episode_id"])),
                            }
                            for episode in scene_episodes
                        ],
                    }
                    for scene_id, scene_episodes in batches
                ]
            },
        )
        input_paths.append(input_path)

    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    processes: List[multiprocessing.Process] = []
    completed_workers = set()
    try:
        for worker_id, (gpu_id, input_path) in enumerate(zip(active_slots, input_paths)):
            process = context.Process(
                target=_panorama_worker_entry,
                args=(
                    worker_id,
                    gpu_id,
                    str(input_path),
                    args,
                    str(output_root),
                    result_queue,
                ),
                name=f"panorama-gpu{gpu_id}-worker{worker_id}",
            )
            process.start()
            processes.append(process)

        while len(completed_workers) < worker_count:
            try:
                message = result_queue.get(timeout=0.5)
            except Empty:
                failed = [
                    process
                    for process in processes
                    if process.exitcode not in (None, 0)
                ]
                if failed:
                    raise RuntimeError(
                        "Panorama worker exited before reporting completion: "
                        + ", ".join(
                            f"{process.name}={process.exitcode}" for process in failed
                        )
                    )
                continue
            message_type = message.get("type")
            if message_type == "error":
                raise RuntimeError(
                    f"Panorama worker {message.get('worker_id')} failed: "
                    f"{message.get('error')}\n{message.get('traceback')}"
                )
            if message_type == "done":
                completed_workers.add(int(message["worker_id"]))
                continue
            if message_type != "result" or not isinstance(message.get("result"), dict):
                raise RuntimeError(f"Invalid panorama worker message: {message}")
            result = message["result"]
            episode_id = str(result["episode_id"])
            if episode_id in results_by_id:
                raise RuntimeError(f"Panorama worker returned duplicate episode {episode_id}")
            append_render_journal(journal_handle, result)
            results_by_id[episode_id] = result
            progress.update(1)
            progress.set_postfix(
                workers=worker_count,
                gpus=",".join(str(gpu) for gpu in sorted(set(active_slots))),
                resumed=resumed_episodes,
                refresh=False,
            )

        for process in processes:
            process.join()
        failed = [process for process in processes if process.exitcode != 0]
        if failed:
            raise RuntimeError(
                "Panorama worker failed: "
                + ", ".join(f"{process.name}={process.exitcode}" for process in failed)
            )
    except (Exception, KeyboardInterrupt):
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=10)
        raise
    finally:
        result_queue.close()
        result_queue.join_thread()
        if worker_root.exists():
            shutil.rmtree(worker_root)


def render_panorama_episodes(args: argparse.Namespace) -> None:
    args.image_format = normalize_panorama_image_format(args.image_format)
    if not 1 <= int(args.jpeg_quality) <= 100:
        raise ValueError(f"jpeg_quality must be in [1, 100], got {args.jpeg_quality}")
    if not 0 <= int(args.png_compress_level) <= 9:
        raise ValueError(
            "png_compress_level must be in [0, 9], "
            f"got {args.png_compress_level}"
        )
    os.environ.setdefault("MAGNUM_LOG", "quiet")
    os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
    with gzip.open(args.dataset, "rt", encoding="utf-8") as handle:
        dataset = json.load(handle)
    with gzip.open(args.gt, "rt", encoding="utf-8") as handle:
        gt = json.load(handle)
    episodes = list(dataset["episodes"])
    dataset_total = len(episodes)
    if args.max_episodes is not None and args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]
    episodes.sort(key=lambda episode: (str(episode["scene_id"]), int(episode["episode_id"])))
    final_output_root = Path(args.output_root)
    output_root = Path(str(final_output_root) + ".rendering")
    legacy_failed_root = Path(str(final_output_root) + ".failed")
    if output_root.exists() and legacy_failed_root.exists():
        raise FileExistsError(
            f"Both render resume roots exist: {output_root}, {legacy_failed_root}"
        )
    if legacy_failed_root.exists():
        if not args.resume:
            raise FileExistsError(
                f"Previous render state exists; rerun with --resume: {legacy_failed_root}"
            )
        os.replace(legacy_failed_root, output_root)
    if output_root.exists() and not args.resume:
        raise FileExistsError(
            f"Previous render state exists; rerun with --resume: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    state_path = output_root / ".render_state.json"
    journal_path = output_root / ".render_journal.jsonl"
    expected_manifest = render_run_manifest(args, episodes, dataset_total)
    if state_path.exists():
        with state_path.open("r", encoding="utf-8") as handle:
            existing_manifest = json.load(handle)
        if existing_manifest.get("fingerprint") != expected_manifest["fingerprint"]:
            raise ValueError(
                "Render resume inputs or settings changed. Preserve the old directory for "
                f"inspection or remove it before starting a new render: {output_root}"
            )
    else:
        leftovers = list(output_root.iterdir())
        if leftovers:
            raise ValueError(
                f"Render resume root has no valid state manifest: {output_root}"
            )
        atomic_json(state_path, expected_manifest)

    journal_records = read_render_journal(journal_path)
    selected_ids = {str(episode["episode_id"]) for episode in episodes}
    unknown_journal_ids = sorted(set(journal_records) - selected_ids)
    if unknown_journal_ids:
        raise ValueError(
            f"Render journal contains episodes outside the current selection: "
            f"{unknown_journal_ids[:10]}"
        )
    results_by_id: Dict[str, Dict[str, Any]] = {}
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        result = journal_records.get(episode_id)
        if completed_render_is_reusable(
            result,
            output_root,
            episode_id,
            gt.get(episode_id),
            args.image_format,
            include_failed=bool(args.drop_invalid),
        ):
            results_by_id[episode_id] = result  # type: ignore[assignment]

    resumed_episodes = len(results_by_id)
    pending_episodes = [
        episode
        for episode in episodes
        if str(episode["episode_id"]) not in results_by_id
    ]
    progress = tqdm(
        total=len(episodes),
        initial=resumed_episodes,
        desc="panorama",
        unit="ep",
        dynamic_ncols=True,
    )
    try:
        with journal_path.open("a", encoding="utf-8") as journal_handle:
            slots = habitat_process_slots(args)
            if len(slots) > 1 and pending_episodes:
                _parallel_render_batches(
                    pending_episodes,
                    gt,
                    args,
                    output_root,
                    journal_handle,
                    results_by_id,
                    progress,
                    resumed_episodes,
                )
            else:
                active_scene = None
                simulator = None
                args.gpu_device_id = int(slots[0])
                try:
                    for episode in pending_episodes:
                        episode_id = str(episode["episode_id"])
                        scene_id = str(episode["scene_id"])
                        if scene_id != active_scene:
                            if simulator is not None:
                                simulator.close()
                            scene_path = resolve_scene_path(args.scene_root, scene_id)
                            simulator = make_panorama_simulator(scene_path, args)
                            active_scene = scene_id
                        assert simulator is not None
                        result = _render_one_panorama_episode(
                            simulator,
                            episode,
                            gt.get(episode_id),
                            args,
                            output_root,
                        )
                        append_render_journal(journal_handle, result)
                        results_by_id[episode_id] = result
                        progress.update(1)
                        progress.set_postfix(
                            scene=Path(scene_id).stem,
                            resumed=resumed_episodes,
                            refresh=False,
                        )
                finally:
                    if simulator is not None:
                        simulator.close()
    except (Exception, KeyboardInterrupt) as error:
        fatal_summary = {
            "dataset_total": dataset_total,
            "selected": len(episodes),
            "episodes_completed": len(results_by_id),
            "resumed_episodes": resumed_episodes,
            "image_root": str(final_output_root.resolve()),
            "resume_root": str(output_root.resolve()),
            "fatal_error": f"{type(error).__name__}: {error}",
            "all_hard_checks_pass": False,
        }
        print(json.dumps(fatal_summary, ensure_ascii=False, indent=2))
        raise
    finally:
        progress.close()

    results = [
        results_by_id[str(episode["episode_id"])]
        for episode in episodes
        if str(episode["episode_id"]) in results_by_id
    ]
    issue_counts: Dict[str, int] = {}
    for result in results:
        for issue in result.get("issues", []):
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
    selected_hard_checks_pass = (
        bool(results) and len(results) == len(episodes) and not issue_counts
    )
    selection_complete = len(episodes) == dataset_total
    successful_ids = [
        str(result["episode_id"])
        for result in results
        if not result.get("issues")
    ]
    successful_id_set = set(successful_ids)
    rendered_frame_count = sum(
        int(result.get("frame_count") or 0)
        for result in results
        if not result.get("issues")
    )
    all_results_complete = bool(results) and len(results) == len(episodes)
    can_filter = bool(
        args.drop_invalid
        and selection_complete
        and all_results_complete
        and successful_id_set
    )
    command_succeeded = (
        selected_hard_checks_pass or can_filter
    ) and (
        selection_complete or args.allow_partial_render
    )
    summary = {
        "dataset_total": dataset_total,
        "selected": len(episodes),
        "selection_complete": selection_complete,
        "episodes": len(results),
        "resumed_episodes": resumed_episodes,
        "rendered_this_run": len(results) - resumed_episodes,
        "passed": sum(not result.get("issues") for result in results),
        "failed": sum(bool(result.get("issues")) for result in results),
        "issue_counts": issue_counts,
        "failed_episode_preview": [
            {
                "episode_id": result["episode_id"],
                "scene_id": result["scene_id"],
                "issues": result["issues"],
                "collision_count": result["collision_count"],
                "pose_error_max": result["pose_error_max"],
                "final_goal_error": result["final_goal_error"],
                "black_ratio_max": result["black_ratio_max"],
            }
            for result in results
            if result.get("issues")
        ][:50],
        "image_root": str(final_output_root.resolve()),
        "rendered_frame_count": rendered_frame_count,
        "filtered_invalid_episodes": (
            len(results) - len(successful_ids) if can_filter else 0
        ),
        "published_episodes": len(successful_ids) if can_filter else len(results),
        "selected_hard_checks_pass": selected_hard_checks_pass,
        "full_dataset_hard_checks_pass": (
            selected_hard_checks_pass and selection_complete
        ),
        "all_hard_checks_pass": command_succeeded,
    }
    if command_succeeded:
        if can_filter and not selected_hard_checks_pass:
            publish_filtered_render_sources(
                args,
                dataset,
                gt,
                selected_ids,
                successful_id_set,
            )
            for result in results:
                if result.get("issues"):
                    shutil.rmtree(
                        output_root / f".episode_{result['episode_id']}.failed",
                        ignore_errors=True,
                    )
        replace_directory_atomically(output_root, final_output_root)
        (final_output_root / journal_path.name).unlink(missing_ok=True)
        (final_output_root / state_path.name).unlink(missing_ok=True)
    else:
        summary["resume_root"] = str(output_root.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not command_succeeded:
        raise RuntimeError(f"Panorama rendering validation failed: {issue_counts}")


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "collect":
        collect(args)
    elif args.command == "validate-gt":
        validate_gt(args)
    else:
        render_panorama_episodes(args)


if __name__ == "__main__":
    main()
