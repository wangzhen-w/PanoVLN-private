#!/usr/bin/env python3
"""Collect and validate high-quality PanoVLN-HM3D trajectories.

The collector samples directly from each official HM3D navmesh. It emits
benchmark-compatible VLN-CE episodes plus immutable geometry metrics. R2R-like
routes are shortest paths; RxR-like routes optionally visit an off-shortest-path
anchor to reproduce the instruction-fidelity challenge absent from pure PointNav.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import shutil
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

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
        "--require-semantic-assets",
        action="store_true",
        help=(
            "Use only scenes that have both .semantic.glb and .semantic.txt. "
            "Optional when a downstream semantic-grounding stage needs them; "
            "the RGB/VLM pipeline itself does not."
        ),
    )
    collect.add_argument("--num-trajectories", type=int, default=100)
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
        "--output",
        required=True,
        help="Internal trajectory dataset JSON.GZ used by later stages.",
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
    render.add_argument("--jpeg-quality", type=int, default=90)
    render.add_argument("--max-black-ratio", type=float, default=0.10)
    render.add_argument(
        "--pose-tolerance",
        type=float,
        default=0.02,
        help="Maximum replay-position deviation in meters from GT locations.",
    )
    render.add_argument("--gpu-device-id", type=int, default=0)
    render.add_argument("--max-episodes", type=int, default=None)
    render.add_argument(
        "--allow-partial-render",
        action="store_true",
        help=(
            "Allow --max-episodes to validate only a subset and exit successfully. "
            "The printed summary still marks selection_complete=false."
        ),
    )
    return parser


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
    require_semantic_assets: bool = False,
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
    if require_semantic_assets:
        scenes = [
            scene
            for scene in scenes
            if scene.with_name(scene.name.replace(".basis.glb", ".semantic.glb")).is_file()
            and scene.with_name(
                scene.name.replace(".basis.glb", ".semantic.txt")
            ).is_file()
        ]
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


def collect(args: argparse.Namespace) -> None:
    os.environ.setdefault("MAGNUM_LOG", "quiet")
    os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
    rng = random.Random(args.seed)
    scenes = discover_scenes(
        args.scene_root,
        args.split,
        args.scene_ids,
        args.require_semantic_assets,
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
    output_path = Path(args.output)
    partial_path = Path(str(output_path) + ".partial")
    existing = [path for path in (output_path, partial_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Trajectory output already exists; use --overwrite to replace it: "
            + ", ".join(str(path) for path in existing)
        )
    if args.overwrite:
        output_path.unlink(missing_ok=True)
        partial_path.unlink(missing_ok=True)
    collected: List[Dict[str, Any]] = []
    hashes = set()
    failures = []

    for scene_index, scene in enumerate(scenes):
        remaining_scenes = len(scenes) - scene_index
        remaining_routes = args.num_trajectories - len(collected)
        if remaining_routes <= 0:
            break
        quota = math.ceil(remaining_routes / remaining_scenes)
        simulator = make_simulator(scene, args, args.seed + scene_index)
        scene_id = public_scene_id(scene, args.scene_root)
        accepted_here = 0
        exhausted_here = 0
        try:
            while accepted_here < quota and len(collected) < args.num_trajectories:
                family = families[len(collected)]
                distances = r2r_distances if family == "r2r" else rxr_distances
                target_distance = rng.choice(distances)
                make_detour = planned_detours[len(collected)]
                sampled = sample_one(
                    simulator,
                    family,
                    target_distance,
                    make_detour,
                    rng,
                    args,
                )
                if sampled is None:
                    exhausted_here += 1
                    failures.append(
                        {
                            "scene_id": scene_id,
                            "family": family,
                            "target_distance": target_distance,
                            "detour": make_detour,
                            "reason": "sampling_attempts_exhausted",
                        }
                    )
                    if exhausted_here >= args.max_sampling_failures_per_scene:
                        break
                    continue
                visual_metrics = route_visual_metrics(
                    simulator,
                    sampled["route_keypoints"],
                    args.max_visual_checkpoints,
                )
                sampled["metrics"].update(visual_metrics)
                if visual_metrics["visual_black_ratio_max"] > args.max_black_ratio:
                    exhausted_here += 1
                    failures.append(
                        {
                            "scene_id": scene_id,
                            "family": family,
                            "target_distance": target_distance,
                            "detour": make_detour,
                            "reason": "visual_scan_void",
                            **visual_metrics,
                        }
                    )
                    if exhausted_here >= args.max_sampling_failures_per_scene:
                        break
                    continue
                digest = route_hash(scene_id, sampled["reference_path"])
                if digest in hashes:
                    exhausted_here += 1
                    if exhausted_here >= args.max_sampling_failures_per_scene:
                        break
                    continue
                hashes.add(digest)
                episode_id = len(collected)
                sampled.update(
                    {
                        "episode_id": episode_id,
                        # Rendering and export key physical trajectories by the
                        # source episode ID, matching R2R/ScaleVLN conventions.
                        "trajectory_id": episode_id,
                        "scene_id": scene_id,
                        "scene_path": str(scene),
                        "route_hash": digest,
                        "seed": args.seed,
                    }
                )
                collected.append(sampled)
                accepted_here += 1
                exhausted_here = 0
        finally:
            simulator.close()

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
        "sampling_failures": len(failures),
        "sampling_failure_preview": failures[:20],
        "hard_checks": hard_checks,
    }
    atomic_json_gz(
        dataset_path,
        {"episodes": episodes, "instruction_vocab": EMPTY_INSTRUCTION_VOCAB},
    )
    if complete:
        partial_path.unlink(missing_ok=True)
    print(json.dumps(publication, ensure_ascii=False, indent=2))
    if len(collected) != args.num_trajectories:
        raise RuntimeError(
            f"Collected {len(collected)}/{args.num_trajectories}; "
            f"sampling_failures={len(failures)}"
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
    summary = {
        "episodes": len(episodes),
        "gt_records": len(gt),
        "passed": sum(not row["issues"] for row in rows),
        "failed": sum(bool(row["issues"]) for row in rows),
        "issue_counts": issue_counts,
        "action_count_mean": statistics.mean(action_counts) if action_counts else None,
        "action_count_max": max(action_counts) if action_counts else None,
        "all_hard_checks_pass": not issue_counts and len(episodes) == len(gt),
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
        if summary["all_hard_checks_pass"]:
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
    if not summary["all_hard_checks_pass"]:
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


def save_panorama(observation: np.ndarray, path: Path, jpeg_quality: int) -> float:
    from PIL import Image

    rgb = np.asarray(observation)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"Unexpected RGB panorama shape: {rgb.shape}")
    rgb = rgb[:, :, :3].astype(np.uint8)
    black_ratio = float(np.mean(np.max(rgb, axis=2) <= 3))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path, quality=jpeg_quality)
    return black_ratio


def rendered_frame_fingerprint(
    output_root: Path,
    episode_ids: Sequence[str],
) -> Tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for episode_id in sorted(episode_ids, key=lambda value: int(value)):
        for path in sorted(
            (output_root / episode_id).glob("frame_*.jpg"),
            key=lambda item: int(item.stem.split("_")[-1]),
        ):
            relative = path.relative_to(output_root).as_posix()
            digest.update(relative.encode("utf-8") + b"\0")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            count += 1
    return digest.hexdigest(), count


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


def render_panorama_episodes(args: argparse.Namespace) -> None:
    from habitat_sim import AgentState
    from habitat_sim.utils.common import quat_from_coeffs

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
    action_name = {1: "move_forward", 2: "turn_left", 3: "turn_right"}
    final_output_root = Path(args.output_root)
    output_root = Path(str(final_output_root) + ".rendering")
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    active_scene = None
    simulator = None
    try:
        for episode in episodes:
            episode_id = str(episode["episode_id"])
            scene_id = str(episode["scene_id"])
            if scene_id != active_scene:
                if simulator is not None:
                    simulator.close()
                scene_path = resolve_scene_path(args.scene_root, scene_id)
                simulator = make_panorama_simulator(scene_path, args)
                active_scene = scene_id
            assert simulator is not None
            record = gt.get(episode_id)
            if record is None:
                results.append({"episode_id": episode_id, "issues": ["missing_gt"]})
                continue
            actions = [int(action) for action in record.get("actions", [])]
            locations = [vec(location) for location in record.get("locations", [])]
            state = AgentState()
            state.position = np.asarray(episode["start_position"], dtype=np.float32)
            state.rotation = quat_from_coeffs(
                np.asarray(episode["start_rotation"], dtype=np.float64)
            )
            simulator.get_agent(0).set_state(state, reset_sensors=True)
            episode_dir = output_root / episode_id
            temporary_dir = output_root / f".episode_{episode_id}.tmp"
            failed_dir = output_root / f".episode_{episode_id}.failed"
            temporary_dir.mkdir(parents=True, exist_ok=True)
            black_ratios = []
            issues = []
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
                    temporary_dir / "frame_0.jpg",
                    args.jpeg_quality,
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
                        pose_error = float(
                            np.linalg.norm(after - locations[location_index])
                        )
                        pose_errors.append(pose_error)
                        if pose_error > args.pose_tolerance:
                            issues.append("replay_pose_mismatch")
                black_ratios.append(
                    save_panorama(
                        observations["rgb"],
                        temporary_dir / f"frame_{action_index}.jpg",
                        args.jpeg_quality,
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
            actual_frames = len(list(temporary_dir.glob("frame_*.jpg")))
            if actual_frames != expected_frames:
                issues.append("frame_count_mismatch")
            if collision_count:
                issues.append("render_replay_collision")
            if black_ratios and max(black_ratios) > args.max_black_ratio:
                issues.append("excessive_black_pixels")
            issues = sorted(set(issues))
            target_dir = episode_dir if not issues else failed_dir
            os.replace(temporary_dir, target_dir)
            results.append(
                {
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
            )
    except Exception as error:
        failed_root = Path(str(final_output_root) + ".failed")
        if failed_root.exists():
            shutil.rmtree(failed_root)
        if output_root.exists():
            os.replace(output_root, failed_root)
        fatal_summary = {
            "dataset_total": dataset_total,
            "selected": len(episodes),
            "episodes_completed": len(results),
            "image_root": str(final_output_root.resolve()),
            "failed_artifact_root": str(failed_root.resolve()),
            "fatal_error": f"{type(error).__name__}: {error}",
            "all_hard_checks_pass": False,
        }
        print(json.dumps(fatal_summary, ensure_ascii=False, indent=2))
        raise
    finally:
        if simulator is not None:
            simulator.close()
    issue_counts: Dict[str, int] = {}
    for result in results:
        for issue in result.get("issues", []):
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
    selected_hard_checks_pass = bool(results) and not issue_counts
    selection_complete = len(episodes) == dataset_total
    successful_ids = [
        str(result["episode_id"])
        for result in results
        if not result.get("issues")
    ]
    image_sha256, rendered_frame_count = rendered_frame_fingerprint(
        output_root, successful_ids
    )
    command_succeeded = selected_hard_checks_pass and (
        selection_complete or args.allow_partial_render
    )
    summary = {
        "dataset_total": dataset_total,
        "selected": len(episodes),
        "selection_complete": selection_complete,
        "episodes": len(results),
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
        "rendered_frames_sha256": image_sha256,
        "selected_hard_checks_pass": selected_hard_checks_pass,
        "full_dataset_hard_checks_pass": (
            selected_hard_checks_pass and selection_complete
        ),
        "all_hard_checks_pass": command_succeeded,
    }
    if command_succeeded:
        replace_directory_atomically(output_root, final_output_root)
    else:
        failed_root = Path(str(final_output_root) + ".failed")
        if failed_root.exists():
            shutil.rmtree(failed_root)
        os.replace(output_root, failed_root)
        summary["failed_artifact_root"] = str(failed_root.resolve())
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
