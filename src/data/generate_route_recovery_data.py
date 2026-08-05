"""Generate continuous R2R/RxR route-recovery trajectories.

For each physical reference route, one instruction variant is selected with a
stable random seed.  The saved panorama sequence is collected continuously
from the episode start, through the clean reference prefix, an injected
off-route excursion, and the ordered return through the actual forward
breadcrumbs.  The excursion actions are executed only to create causal visual
history; a separate preparation script decides which actions receive loss.

The collector uses the saved R2R/RxR annotations only to select the established
route and instruction population.  It regenerates each clean prefix from the
episode reference path with Habitat's shortest-path follower at a 0.3m waypoint
radius.  It does not use a navigation-model rollout or saved train_gt
actions/locations.  Its public output follows the standard PanoVLN layout::

    sub_dataset/r2r_rxr_recovery.jsonl
    images/r2r_rxr_recovery/<trajectory_id>/frame_*.jpg

Per-worker manifests and incomplete image directories live only in hidden
``.inprogress`` locations and are removed after a successful final merge.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import heapq
import json
import math
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_INPUT_ROOT = Path("/workspace/z_PanoVLN")
DEFAULT_OUTPUT_ROOT = Path("/workspace/z_PanoVLN")
DEFAULT_OUTPUT_DATASET_NAME = "r2r_rxr_recovery"
DEFAULT_CONNECTIVITY_ROOT = Path(
    "/workspace/data2/dataset/general_VLN_data/ScaleVLN_total/connectivity"
)
DEFAULT_SCENE_ROOT = Path(
    "/workspace/data2/dataset/janusvln_data/scene_datasets"
)

DATASET_SPECS = {
    "r2r": {
        "annotation_name": "r2r.jsonl",
        "episode_path": Path(
            "/workspace/data2/dataset/general_VLN_data/"
            "R2R_VLNCE_v1-3_preprocessed/train/train.json.gz"
        ),
        "config_path": PROJECT_ROOT / "config/vln_r2r_train.yaml",
        "habitat_data_path": (
            "/workspace/data2/dataset/janusvln_data/"
            "datasets/r2r/{split}/{split}.json.gz"
        ),
    },
    "rxr": {
        "annotation_name": "rxr.jsonl",
        "episode_path": Path(
            "/workspace/data2/dataset/general_VLN_data/"
            "RxR_VLNCE_v0/train/train_guide.json.gz"
        ),
        "config_path": PROJECT_ROOT / "config/vln_rxr_train.yaml",
        "habitat_data_path": (
            "/workspace/data2/dataset/janusvln_data/"
            "datasets/rxr/{split}/{split}_{role}.json.gz"
        ),
    },
}


@contextmanager
def silence_external_output(enabled: bool = True):
    if not enabled:
        yield
        return
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    with open(os.devnull, "w") as null_handle:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(null_handle.fileno(), 1)
            os.dup2(null_handle.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)


with silence_external_output():
    import habitat
    import habitat_sim
    from habitat.config.default import get_config
    from habitat.sims.habitat_simulator.actions import HabitatSimActions
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

    from habitat_extensions import measures, task  # noqa: F401


STOP_ACTION = int(HabitatSimActions.stop)
FORWARD_ACTION = int(HabitatSimActions.move_forward)
LEFT_ACTION = int(HabitatSimActions.turn_left)
RIGHT_ACTION = int(HabitatSimActions.turn_right)
SUPPORTED_ACTIONS = {STOP_ACTION, FORWARD_ACTION, LEFT_ACTION, RIGHT_ACTION}

# Match the 0.3m shortest-path-follower setting used by the clean EBS data.
# Standard prefix actions are generated from the reference path inside this
# collector; the current 0.5m action lists are used only to select the existing
# R2R/RxR route and instruction population.
STANDARD_WAYPOINT_RADIUS = 0.3
LOCAL_TARGET_RADIUS = 0.2
ROUTE_CONTACT_RADIUS = 0.25
RETURN_POSITION_TOLERANCE = 0.25
BREADCRUMB_RETURN_TOLERANCE = 0.10
MAX_ROUTE_CONTACT_FORWARD_STEPS = 2
MIN_TRANSLATION = 0.05
DEFAULT_MIN_FREE_DISK_GB = 200.0


class PilotGenerationError(RuntimeError):
    pass


def pilot_failure_code(error: PilotGenerationError) -> str:
    """Collapse route-specific text into a stable diagnostic category."""

    message = str(error)
    exact_codes = {
        "route_recontact",
        "late_route_departure",
        "non_finite_route_distance",
        "requested_depth_reached",
        "local_follower_error",
        "local_follower_stopped_early",
        "forward_collision",
        "episode_ended",
        "local_action_limit",
    }
    if message in exact_codes:
        return message
    if message.startswith("recovery failed: "):
        suffix = message.removeprefix("recovery failed: ")
        return f"recovery_{suffix}" if suffix in exact_codes else "recovery_failed"
    prefixes = (
        ("mapped reference path contains non-adjacent", "non_adjacent_reference_mapping"),
        ("reference point ", "reference_mapping_error"),
        ("standard follower ", "standard_follower_error"),
        ("standard route ", "standard_route_error"),
        ("episode start is ", "standard_start_mismatch"),
        ("branch did not reach", "branch_depth_not_reached"),
        ("deviation endpoint is already", "deviation_near_task_goal"),
        ("recovery returned ", "recovery_anchor_error"),
        ("recovery forward count ", "recovery_forward_count_mismatch"),
        ("recovery left reversed breadcrumbs", "recovery_breadcrumb_error"),
    )
    for prefix, code in prefixes:
        if message.startswith(prefix):
            return code
    normalized = re.sub(r"[^a-z0-9]+", "_", message.lower()).strip("_")
    return normalized[:80] or "pilot_generation_error"


@dataclass(frozen=True)
class GraphCandidate:
    anchor_node: int
    first_branch_node: int
    anchor_reference_index: int
    node_path: Tuple[int, ...]
    graph_path_length: float
    graph_distance_to_route: float


@dataclass
class StandardTrace:
    actions: List[int]
    state_positions: List[np.ndarray]
    state_rotations: List[List[float]]
    forward_positions: List[np.ndarray]
    reference_boundaries: Dict[int, int]


@dataclass
class SegmentRollout:
    candidate: GraphCandidate
    requested_depth: float
    prefix_action_count: int
    anchor_position: np.ndarray
    anchor_rotation: List[float]
    endpoint_position: np.ndarray
    deviation_actions: List[int]
    recovery_actions: List[int]
    deviation_forward_positions: List[np.ndarray]
    recovery_forward_positions: List[np.ndarray]
    distance_trace: List[float]
    departure_forward_step: int
    return_error: float
    max_breadcrumb_return_error: float
    recovery_arc_ratio: float
    max_snap_displacement: float


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_worker_jsonl_for_resume(path: Path) -> List[Dict[str, Any]]:
    """Read a rank manifest, repairing only an interrupted trailing write."""

    if not path.is_file():
        return []
    payload = path.read_bytes()
    lines = payload.splitlines(keepends=True)
    rows = []
    needs_rewrite = bool(payload) and not payload.endswith(b"\n")
    for line_index, raw_line in enumerate(lines):
        if not raw_line.strip():
            continue
        try:
            rows.append(json.loads(raw_line.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            if line_index != len(lines) - 1:
                raise PilotGenerationError(
                    f"corrupt non-terminal JSONL row in {path} at line "
                    f"{line_index + 1}"
                ) from error
            needs_rewrite = True
            print(
                f"discarding interrupted trailing JSONL row: {path}",
                flush=True,
            )
    if needs_rewrite:
        atomic_write_jsonl(path, rows)
    return rows


def load_episode_dicts(path: Path) -> List[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["episodes"]


def position_array(value: Any) -> np.ndarray:
    if hasattr(value, "position"):
        value = value.position
    return np.asarray(value, dtype=np.float64)


def position_list(value: Any) -> List[float]:
    return [float(item) for item in position_array(value)]


def rotation_list(rotation: Any) -> List[float]:
    return [
        float(rotation.x),
        float(rotation.y),
        float(rotation.z),
        float(rotation.w),
    ]


def euclidean_distance(first: Any, second: Any) -> float:
    return float(np.linalg.norm(position_array(first) - position_array(second)))


def instruction_text(episode_dict: Dict[str, Any]) -> str:
    instruction = episode_dict.get("instruction", "")
    if isinstance(instruction, dict):
        return str(instruction.get("instruction_text", ""))
    return str(instruction)


def stable_seed(base_seed: int, *parts: Any) -> int:
    payload = "|".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def shuffle_candidates_by_anchor(
    candidates: Sequence[GraphCandidate], rng: random.Random
) -> List[GraphCandidate]:
    """Choose a decision point before choosing one of its outgoing edges."""

    by_anchor: Dict[Tuple[int, int], List[GraphCandidate]] = defaultdict(list)
    for candidate in candidates:
        key = (candidate.anchor_node, candidate.anchor_reference_index)
        by_anchor[key].append(candidate)
    anchor_keys = list(by_anchor)
    rng.shuffle(anchor_keys)
    ordered = []
    for key in anchor_keys:
        group = by_anchor[key]
        rng.shuffle(group)
        ordered.extend(group)
    return ordered


def scan_id(scene_id: str) -> str:
    return Path(scene_id).stem


def validate_actions(actions: Sequence[Any], context: str) -> List[int]:
    result = [int(action) for action in actions]
    if not result or result[-1] != STOP_ACTION:
        raise PilotGenerationError(f"{context}: standard actions must end in stop")
    invalid = sorted(set(result) - SUPPORTED_ACTIONS)
    if invalid:
        raise PilotGenerationError(f"{context}: invalid actions {invalid}")
    return result


class ConnectivityGraph:
    def __init__(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as handle:
            self.records = json.load(handle)
        included = [
            index for index, record in enumerate(self.records) if record["included"]
        ]
        included_set = set(included)
        self.coordinates = {
            index: np.asarray(
                [
                    self.records[index]["pose"][3],
                    self.records[index]["pose"][11],
                    -self.records[index]["pose"][7],
                ],
                dtype=np.float64,
            )
            for index in included
        }
        self.adjacency: Dict[int, List[int]] = {index: [] for index in included}
        for index in included:
            unobstructed = self.records[index]["unobstructed"]
            for neighbor, is_open in enumerate(unobstructed):
                if not is_open or neighbor not in included_set:
                    continue
                reverse = self.records[neighbor]["unobstructed"]
                if index < len(reverse) and reverse[index]:
                    self.adjacency[index].append(neighbor)

    def edge_length(self, first: int, second: int) -> float:
        return euclidean_distance(self.coordinates[first], self.coordinates[second])

    def map_reference_path(
        self,
        reference_path: Sequence[Sequence[float]],
        max_mapping_error: float,
    ) -> Tuple[List[Tuple[int, int]], Dict[str, float]]:
        nodes = list(self.coordinates)
        mapped: List[Tuple[int, int]] = []
        nearest_errors = []
        nearest_ratios = []
        for reference_index, point in enumerate(reference_path):
            point_array = position_array(point)
            distances = sorted(
                (
                    float(
                        np.linalg.norm(
                            self.coordinates[node][[0, 2]] - point_array[[0, 2]]
                        )
                    ),
                    node,
                )
                for node in nodes
            )
            nearest_error, nearest_node = distances[0]
            if nearest_error > max_mapping_error:
                raise PilotGenerationError(
                    f"reference point {reference_index} maps {nearest_error:.3f}m "
                    f"from connectivity graph"
                )
            second_error = distances[1][0] if len(distances) > 1 else math.inf
            nearest_errors.append(nearest_error)
            nearest_ratios.append(nearest_error / max(second_error, 1e-9))
            if not mapped or mapped[-1][0] != nearest_node:
                mapped.append((nearest_node, reference_index))
            else:
                mapped[-1] = (nearest_node, reference_index)
        for (first_node, _), (second_node, _) in zip(mapped, mapped[1:]):
            if second_node not in self.adjacency[first_node]:
                raise PilotGenerationError(
                    "mapped reference path contains non-adjacent connectivity nodes"
                )
        return mapped, {
            "max_mapping_error": max(nearest_errors, default=0.0),
            "max_nearest_ratio": max(nearest_ratios, default=0.0),
        }

    def distance_to_route(self, route_nodes: set[int]) -> Dict[int, float]:
        distances = {node: math.inf for node in self.coordinates}
        queue: List[Tuple[float, int]] = []
        for node in route_nodes:
            distances[node] = 0.0
            heapq.heappush(queue, (0.0, node))
        while queue:
            current_distance, node = heapq.heappop(queue)
            if current_distance != distances[node]:
                continue
            for neighbor in self.adjacency[node]:
                new_distance = current_distance + self.edge_length(node, neighbor)
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    heapq.heappush(queue, (new_distance, neighbor))
        return distances

    def candidate_branches(
        self,
        mapped_route: Sequence[Tuple[int, int]],
        requested_depth: float,
    ) -> List[GraphCandidate]:
        route_nodes = [node for node, _ in mapped_route]
        route_node_set = set(route_nodes)
        graph_route_distance = self.distance_to_route(route_node_set)
        candidates = []
        for anchor_node, reference_index in mapped_route[1:-1]:
            for first_branch_node in self.adjacency[anchor_node]:
                if first_branch_node in route_node_set:
                    continue
                initial_length = self.edge_length(anchor_node, first_branch_node)
                best_distance = {first_branch_node: initial_length}
                previous: Dict[int, Optional[int]] = {first_branch_node: None}
                queue = [(initial_length, first_branch_node)]
                target_node = None
                while queue:
                    path_length, node = heapq.heappop(queue)
                    if path_length != best_distance[node]:
                        continue
                    if graph_route_distance[node] >= requested_depth:
                        target_node = node
                        break
                    for neighbor in self.adjacency[node]:
                        if neighbor in route_node_set:
                            continue
                        new_length = path_length + self.edge_length(node, neighbor)
                        if new_length < best_distance.get(neighbor, math.inf):
                            best_distance[neighbor] = new_length
                            previous[neighbor] = node
                            heapq.heappush(queue, (new_length, neighbor))
                if target_node is None:
                    continue
                reverse_path = []
                node: Optional[int] = target_node
                while node is not None:
                    reverse_path.append(node)
                    node = previous[node]
                branch_path = tuple([anchor_node, *reversed(reverse_path)])
                candidates.append(
                    GraphCandidate(
                        anchor_node=anchor_node,
                        first_branch_node=first_branch_node,
                        anchor_reference_index=reference_index,
                        node_path=branch_path,
                        graph_path_length=float(best_distance[target_node]),
                        graph_distance_to_route=float(graph_route_distance[target_node]),
                    )
                )
        return candidates


def build_env_config(dataset_name: str, gpu_id: int, image_width: int, image_height: int):
    spec = DATASET_SPECS[dataset_name]
    config = get_config(str(spec["config_path"]))
    with habitat.config.read_write(config):
        config.habitat.dataset.data_path = spec["habitat_data_path"]
        config.habitat.dataset.scenes_dir = str(DEFAULT_SCENE_ROOT)
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = int(gpu_id)
        sensor = config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor
        sensor.width = int(image_width)
        sensor.height = int(image_height)
        config.habitat.environment.max_episode_steps = 3000
        config.habitat.task.measurements = {}
    return config


def build_route_records(
    dataset_name: str,
    instruction_seed: int,
    input_root: Path = DEFAULT_INPUT_ROOT,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    spec = DATASET_SPECS[dataset_name]
    annotation_path = input_root / "sub_dataset" / spec["annotation_name"]
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)
    action_rows = read_jsonl(annotation_path)
    annotation_episode_ids = {str(row["episode_id"]) for row in action_rows}
    if len(annotation_episode_ids) != len(action_rows):
        raise PilotGenerationError(
            f"{dataset_name}: duplicate episode ids in {annotation_path}"
        )
    episode_dicts = load_episode_dicts(spec["episode_path"])
    episode_by_id = {
        str(episode["episode_id"]): episode
        for episode in episode_dicts
        if str(episode["episode_id"]) in annotation_episode_ids
    }
    missing = sorted(annotation_episode_ids - set(episode_by_id), key=int)
    if missing:
        raise PilotGenerationError(
            f"{dataset_name}: {len(missing)} action rows have no source episode"
        )
    groups: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for episode_id, episode in episode_by_id.items():
        key = (str(episode["scene_id"]), str(episode["trajectory_id"]))
        groups[key].append(episode_id)
    route_records = []
    for (scene_id, trajectory_id), episode_ids in groups.items():
        episode_ids.sort(key=int)
        canonical_id = episode_ids[0]
        episode = episode_by_id[canonical_id]
        canonical_reference_path = episode.get("reference_path", [])
        canonical_start_position = episode.get("start_position", [])
        canonical_start_rotation = episode.get("start_rotation", [])
        for variant_id in episode_ids[1:]:
            variant = episode_by_id[variant_id]
            if (
                variant.get("reference_path", []) != canonical_reference_path
                or variant.get("start_position", []) != canonical_start_position
                or variant.get("start_rotation", []) != canonical_start_rotation
            ):
                raise PilotGenerationError(
                    f"{dataset_name}:{trajectory_id} instruction variants have "
                    "different route geometry"
                )
        instruction_rng = random.Random(
            stable_seed(
                instruction_seed,
                dataset_name,
                scene_id,
                trajectory_id,
                "instruction",
            )
        )
        selected_instruction_id = instruction_rng.choice(episode_ids)
        route_records.append(
            {
                "dataset": dataset_name,
                "scene_id": scene_id,
                "trajectory_id": trajectory_id,
                "canonical_episode_id": canonical_id,
                "selected_instruction_episode_id": selected_instruction_id,
                "source_episode_ids": episode_ids,
                "episode": episode,
                "instruction": instruction_text(
                    episode_by_id[selected_instruction_id]
                ),
            }
        )
    return route_records, episode_by_id


def agent_state(env) -> Tuple[np.ndarray, List[float]]:
    state = env.sim.get_agent_state()
    return position_array(state.position), rotation_list(state.rotation)


def step_without_render(env, action: int) -> None:
    """Apply a discrete action without drawing any configured sensors."""

    env.sim.get_agent(0).act(int(action))


def generate_standard_trace(env, episode) -> StandardTrace:
    """Generate a 0.3m reference-path rollout without rendering RGB frames."""

    env.current_episode = episode
    env.reset()
    start_position, start_rotation = agent_state(env)
    references = [position_array(point) for point in episode.reference_path]
    if len(references) < 2:
        raise PilotGenerationError("standard route has fewer than two reference points")
    start_error = euclidean_distance(start_position, references[0])
    if start_error > 1e-4:
        raise PilotGenerationError(
            f"episode start is {start_error:.3f}m from reference-path start"
        )

    follower = ShortestPathFollower(
        env.sim,
        goal_radius=STANDARD_WAYPOINT_RADIUS,
        return_one_hot=False,
        stop_on_error=True,
    )
    actions: List[int] = []
    state_positions = [start_position.copy()]
    state_rotations = [list(start_rotation)]
    forward_positions = [start_position.copy()]
    boundaries = {0: 0}
    reference_index = 1
    max_standard_actions = 3000
    while reference_index < len(references):
        current_position, _ = agent_state(env)
        target = references[reference_index]
        if euclidean_distance(current_position, target) <= STANDARD_WAYPOINT_RADIUS:
            boundaries[reference_index] = len(actions)
            reference_index += 1
            continue
        try:
            action = follower.get_next_action(np.asarray(target, dtype=np.float32))
        except habitat_sim.errors.GreedyFollowerError as error:
            raise PilotGenerationError(
                f"standard follower failed at reference point {reference_index}"
            ) from error
        if action is None:
            raise PilotGenerationError(
                f"standard follower failed at reference point {reference_index}"
            )
        action = int(action)
        if action == STOP_ACTION:
            raise PilotGenerationError(
                f"standard follower stopped before reference point {reference_index}"
            )
        if action not in SUPPORTED_ACTIONS:
            raise PilotGenerationError(f"standard follower returned action {action}")
        before, _ = agent_state(env)
        step_without_render(env, action)
        after, rotation = agent_state(env)
        actions.append(action)
        state_positions.append(after.copy())
        state_rotations.append(list(rotation))
        if action == FORWARD_ACTION:
            if euclidean_distance(before, after) <= MIN_TRANSLATION:
                raise PilotGenerationError("standard follower forward action collided")
            forward_positions.append(after.copy())
        if len(actions) >= max_standard_actions:
            raise PilotGenerationError(
                f"standard route exceeded {max_standard_actions} actions"
            )
    actions.append(STOP_ACTION)

    return StandardTrace(
        actions=actions,
        state_positions=state_positions,
        state_rotations=state_rotations,
        forward_positions=forward_positions,
        reference_boundaries=boundaries,
    )


def reset_and_replay_prefix(
    env,
    episode,
    actions: Sequence[int],
) -> None:
    env.current_episode = episode
    env.reset()
    for action in actions:
        step_without_render(env, int(action))
        if env.episode_over:
            raise PilotGenerationError("prefix replay ended the episode")


def multi_goal_distance(query, pathfinder, point: np.ndarray) -> float:
    query.requested_start = np.asarray(point, dtype=np.float32)
    if not pathfinder.find_path(query):
        return math.inf
    distance = float(query.geodesic_distance)
    return distance if math.isfinite(distance) else math.inf


def snap_graph_target(
    pathfinder,
    graph: ConnectivityGraph,
    node: int,
    anchor_node: int,
    anchor_position: np.ndarray,
    island_index: int,
) -> Tuple[np.ndarray, float]:
    raw_target = graph.coordinates[node].copy()
    raw_target[1] = anchor_position[1] + (
        graph.coordinates[node][1] - graph.coordinates[anchor_node][1]
    )
    snapped = np.asarray(
        pathfinder.snap_point(raw_target, island_index=island_index),
        dtype=np.float64,
    )
    if not np.isfinite(snapped).all():
        raise PilotGenerationError("connectivity target did not snap to the navmesh")
    return snapped, euclidean_distance(raw_target, snapped)


def follow_local_target(
    env,
    follower,
    target: np.ndarray,
    actions: List[int],
    forward_positions: List[np.ndarray],
    max_actions: int,
    after_forward=None,
    target_radius: float = LOCAL_TARGET_RADIUS,
) -> Optional[str]:
    for _ in range(max_actions):
        current_position, _ = agent_state(env)
        if euclidean_distance(current_position, target) <= target_radius:
            return None
        try:
            raw_action = follower.get_next_action(
                np.asarray(target, dtype=np.float32)
            )
        except habitat_sim.errors.GreedyFollowerError:
            return "local_follower_error"
        if raw_action is None:
            return "local_follower_error"
        action = int(raw_action)
        if action == STOP_ACTION:
            current_position, _ = agent_state(env)
            if euclidean_distance(current_position, target) <= target_radius:
                return None
            return "local_follower_stopped_early"
        before, _ = agent_state(env)
        step_without_render(env, action)
        actions.append(action)
        after, _ = agent_state(env)
        if action == FORWARD_ACTION:
            if euclidean_distance(before, after) <= MIN_TRANSLATION:
                return "forward_collision"
            forward_positions.append(after.copy())
            if after_forward is not None:
                event = after_forward(after)
                if event is not None:
                    return event
        if env.episode_over:
            return "episode_ended"
    return "local_action_limit"


def task_success_radius(episode) -> float:
    goals = getattr(episode, "goals", None)
    if not goals:
        return 3.0
    radius = getattr(goals[0], "radius", None)
    return float(radius) if radius is not None else 3.0


def task_goal_position(episode) -> np.ndarray:
    goals = getattr(episode, "goals", None)
    if not goals:
        raise PilotGenerationError("episode has no task goal")
    return position_array(goals[0].position)


def try_candidate(
    env,
    episode,
    standard_trace: StandardTrace,
    graph: ConnectivityGraph,
    candidate: GraphCandidate,
    requested_depth: float,
) -> SegmentRollout:
    if candidate.anchor_reference_index not in standard_trace.reference_boundaries:
        raise PilotGenerationError("candidate anchor has no standard action boundary")
    prefix_action_count = standard_trace.reference_boundaries[
        candidate.anchor_reference_index
    ]
    reset_and_replay_prefix(
        env,
        episode,
        standard_trace.actions[:prefix_action_count],
    )
    anchor_position, anchor_rotation = agent_state(env)
    expected_anchor = standard_trace.state_positions[prefix_action_count]
    if euclidean_distance(anchor_position, expected_anchor) > 1e-4:
        raise PilotGenerationError("standard prefix did not replay deterministically")

    pathfinder = env.sim.pathfinder
    island_index = int(pathfinder.get_island(anchor_position))
    if island_index < 0 or island_index >= int(pathfinder.num_islands):
        raise PilotGenerationError("anchor has no valid navmesh island")
    route_query = habitat_sim.MultiGoalShortestPath()
    route_query.requested_ends = np.asarray(
        standard_trace.forward_positions,
        dtype=np.float32,
    )
    start_route_distance = multi_goal_distance(route_query, pathfinder, anchor_position)
    if start_route_distance > ROUTE_CONTACT_RADIUS + 1e-4:
        raise PilotGenerationError(
            f"anchor is {start_route_distance:.3f}m from replayed standard route"
        )

    follower = ShortestPathFollower(
        env.sim,
        goal_radius=LOCAL_TARGET_RADIUS,
        return_one_hot=False,
        stop_on_error=True,
    )
    deviation_actions: List[int] = []
    deviation_forward_positions = [anchor_position.copy()]
    distance_trace = [start_route_distance]
    left_route_contact = False
    deviation_forward_count = 0
    departure_forward_step: Optional[int] = None
    reached_depth = False
    max_snap_displacement = 0.0

    def after_deviation_forward(point: np.ndarray) -> Optional[str]:
        nonlocal left_route_contact
        nonlocal deviation_forward_count
        nonlocal departure_forward_step
        nonlocal reached_depth
        deviation_forward_count += 1
        distance = multi_goal_distance(route_query, pathfinder, point)
        distance_trace.append(distance)
        if not math.isfinite(distance):
            return "non_finite_route_distance"
        if distance > ROUTE_CONTACT_RADIUS:
            if not left_route_contact:
                departure_forward_step = deviation_forward_count
            left_route_contact = True
        elif left_route_contact:
            return "route_recontact"
        elif deviation_forward_count >= MAX_ROUTE_CONTACT_FORWARD_STEPS:
            return "late_route_departure"
        if distance >= requested_depth:
            reached_depth = True
            return "requested_depth_reached"
        return None

    for node in candidate.node_path[1:]:
        target, snap_displacement = snap_graph_target(
            pathfinder,
            graph,
            node,
            candidate.anchor_node,
            anchor_position,
            island_index,
        )
        max_snap_displacement = max(max_snap_displacement, snap_displacement)
        event = follow_local_target(
            env,
            follower,
            target,
            deviation_actions,
            deviation_forward_positions,
            max_actions=300,
            after_forward=after_deviation_forward,
        )
        if event == "requested_depth_reached":
            break
        if event is not None:
            raise PilotGenerationError(event)
    if not reached_depth:
        raise PilotGenerationError("branch did not reach requested deviation depth")
    if len(deviation_forward_positions) < 2:
        raise PilotGenerationError("deviation contains no translated step")

    endpoint_position, _ = agent_state(env)
    goal_distance_query = habitat_sim.ShortestPath()
    goal_distance_query.requested_start = np.asarray(endpoint_position, dtype=np.float32)
    goal_distance_query.requested_end = np.asarray(
        task_goal_position(episode), dtype=np.float32
    )
    if pathfinder.find_path(goal_distance_query):
        if float(goal_distance_query.geodesic_distance) <= task_success_radius(episode):
            raise PilotGenerationError("deviation endpoint is already inside task goal radius")

    recovery_actions: List[int] = []
    recovery_forward_positions = [endpoint_position.copy()]
    for breadcrumb in reversed(deviation_forward_positions[:-1]):
        event = follow_local_target(
            env,
            follower,
            breadcrumb,
            recovery_actions,
            recovery_forward_positions,
            max_actions=120,
        )
        if event is not None:
            raise PilotGenerationError(f"recovery failed: {event}")
    returned_position, _ = agent_state(env)
    return_error = euclidean_distance(returned_position, anchor_position)
    if return_error > RETURN_POSITION_TOLERANCE:
        raise PilotGenerationError(
            f"recovery returned {return_error:.3f}m from branch point"
        )
    expected_return_positions = list(reversed(deviation_forward_positions))
    if len(recovery_forward_positions) != len(expected_return_positions):
        raise PilotGenerationError(
            "recovery forward count differs from reversed deviation breadcrumbs"
        )
    breadcrumb_errors = [
        euclidean_distance(actual, expected)
        for actual, expected in zip(
            recovery_forward_positions, expected_return_positions, strict=True
        )
    ]
    max_breadcrumb_return_error = max(breadcrumb_errors, default=0.0)
    if max_breadcrumb_return_error > BREADCRUMB_RETURN_TOLERANCE:
        raise PilotGenerationError(
            "recovery left reversed breadcrumbs by "
            f"{max_breadcrumb_return_error:.3f}m"
        )
    deviation_arc = action_arc_length(deviation_forward_positions)
    recovery_arc = action_arc_length(recovery_forward_positions)
    recovery_arc_ratio = recovery_arc / max(deviation_arc, 1e-9)
    return SegmentRollout(
        candidate=candidate,
        requested_depth=requested_depth,
        prefix_action_count=prefix_action_count,
        anchor_position=anchor_position,
        anchor_rotation=anchor_rotation,
        endpoint_position=endpoint_position,
        deviation_actions=deviation_actions,
        recovery_actions=recovery_actions,
        deviation_forward_positions=deviation_forward_positions,
        recovery_forward_positions=recovery_forward_positions,
        distance_trace=distance_trace,
        departure_forward_step=int(departure_forward_step or 0),
        return_error=return_error,
        max_breadcrumb_return_error=max_breadcrumb_return_error,
        recovery_arc_ratio=recovery_arc_ratio,
        max_snap_displacement=max_snap_displacement,
    )


def save_rgb(
    observation: Dict[str, Any],
    path: Path,
    image_size: Tuple[int, int],
) -> None:
    """Save with the same Pillow JPEG defaults as the standard VLN collector."""

    frame = Image.fromarray(observation["rgb"])
    if frame.mode != "RGB":
        frame = frame.convert("RGB")
    if frame.size != image_size:
        frame = frame.resize(image_size)
    frame.save(path)


def replay_and_save_trajectory(
    env,
    episode,
    standard_trace: StandardTrace,
    rollout: SegmentRollout,
    image_directory: Path,
    image_size: Tuple[int, int],
) -> Dict[str, Any]:
    if image_directory.exists():
        raise PilotGenerationError(f"image directory already exists: {image_directory}")
    image_directory.mkdir(parents=True)
    env.current_episode = episode
    observation = env.reset()
    start_position, start_rotation = agent_state(env)

    prefix_actions = standard_trace.actions[: rollout.prefix_action_count]
    all_actions = [
        *prefix_actions,
        *rollout.deviation_actions,
        *rollout.recovery_actions,
    ]
    deviation_start = len(prefix_actions)
    recovery_start = deviation_start + len(rollout.deviation_actions)
    recovery_end = len(all_actions)

    image_paths: List[Path] = []
    frame_path = image_directory / "frame_0.jpg"
    save_rgb(observation, frame_path, image_size=image_size)
    image_paths.append(frame_path)
    action_positions = [start_position.copy()]
    action_rotations = [start_rotation]

    for frame_index, action in enumerate(all_actions, start=1):
        observation = env.step(int(action))
        after, rotation = agent_state(env)
        action_positions.append(after.copy())
        action_rotations.append(rotation)
        frame_path = image_directory / f"frame_{frame_index}.jpg"
        save_rgb(observation, frame_path, image_size=image_size)
        image_paths.append(frame_path)
        if env.episode_over:
            raise PilotGenerationError(
                f"continuous replay ended at action {frame_index}/{len(all_actions)}"
            )

    anchor_position = action_positions[deviation_start]
    anchor_rotation = action_rotations[deviation_start]
    endpoint_position = action_positions[recovery_start]
    final_position = action_positions[recovery_end]
    if euclidean_distance(anchor_position, rollout.anchor_position) > 1e-4:
        raise PilotGenerationError("continuous replay reached a different branch point")
    if anchor_rotation != rollout.anchor_rotation:
        rotation_error = max(
            abs(float(actual) - float(expected))
            for actual, expected in zip(
                anchor_rotation,
                rollout.anchor_rotation,
                strict=True,
            )
        )
        if rotation_error > 1e-5:
            raise PilotGenerationError(
                "continuous replay reached a different branch orientation"
            )
    if euclidean_distance(endpoint_position, rollout.endpoint_position) > 1e-3:
        raise PilotGenerationError("saved deviation endpoint differs from dry run")
    if euclidean_distance(final_position, rollout.anchor_position) > RETURN_POSITION_TOLERANCE:
        raise PilotGenerationError("saved recovery did not return to branch point")
    return {
        "image_paths": image_paths,
        "actions": all_actions,
        "standard_prefix_end": deviation_start,
        "deviation_start": deviation_start,
        "recovery_start": recovery_start,
        "recovery_end": recovery_end,
        "action_positions": action_positions,
        "action_rotations": action_rotations,
    }


def action_arc_length(positions: Sequence[np.ndarray]) -> float:
    return sum(
        euclidean_distance(positions[index - 1], positions[index])
        for index in range(1, len(positions))
    )


def initial_turn_prefix_length(actions: Sequence[int]) -> int:
    if not actions or actions[0] not in {LEFT_ACTION, RIGHT_ACTION}:
        return 0
    first_action = actions[0]
    prefix_length = 0
    for action in actions:
        if action != first_action:
            break
        prefix_length += 1
    return prefix_length


def make_row(
    sample_id: str,
    route_record: Dict[str, Any],
    rollout: SegmentRollout,
    saved: Dict[str, Any],
) -> Dict[str, Any]:
    common = {
        "schema_version": 2,
        "episode_id": sample_id,
        "trajectory_id": sample_id,
        "source_dataset": route_record["dataset"],
        "source_episode_id": route_record["selected_instruction_episode_id"],
        "canonical_episode_id": route_record["canonical_episode_id"],
        "source_episode_ids": route_record["source_episode_ids"],
        "source_trajectory_id": route_record["trajectory_id"],
        "scene_id": route_record["scene_id"],
        "instruction": route_record["instruction"],
        "anchor_reference_index": rollout.candidate.anchor_reference_index,
        "standard_prefix_action_count": rollout.prefix_action_count,
        "requested_deviation_depth": rollout.requested_depth,
        "actual_deviation_depth": max(rollout.distance_trace),
        "standard_waypoint_radius": STANDARD_WAYPOINT_RADIUS,
        "standard_action_source": "reference_path_shortest_path_follower",
        "trajectory_algorithm_version": 3,
        "departure_forward_step": rollout.departure_forward_step,
        "route_mapping": route_record.get("route_mapping_metrics", {}),
    }
    combined = {
        **common,
        "episode_start_position": position_list(saved["action_positions"][0]),
        "episode_start_rotation": saved["action_rotations"][0],
        "anchor_position": position_list(rollout.anchor_position),
        "anchor_rotation": rollout.anchor_rotation,
        "actions": saved["actions"],
        "standard_actions": route_record["standard_actions"],
        "standard_prefix_end": saved["standard_prefix_end"],
        "deviation_start": saved["deviation_start"],
        "recovery_start": saved["recovery_start"],
        "recovery_end": saved["recovery_end"],
        "deviation_forward_path": [
            position_list(point) for point in rollout.deviation_forward_positions
        ],
        "recovery_forward_path": [
            position_list(point) for point in rollout.recovery_forward_positions
        ],
        "distance_to_standard_route": rollout.distance_trace,
        "return_error": rollout.return_error,
        "max_breadcrumb_return_error": rollout.max_breadcrumb_return_error,
        "recovery_arc_ratio": rollout.recovery_arc_ratio,
        "ambiguous_initial_turn_prefix": initial_turn_prefix_length(
            rollout.recovery_actions
        ),
        "action_positions": [
            position_list(point) for point in saved["action_positions"]
        ],
        "action_rotations": saved["action_rotations"],
        "graph_branch": {
            "anchor_node": rollout.candidate.anchor_node,
            "first_branch_node": rollout.candidate.first_branch_node,
            "node_path": list(rollout.candidate.node_path),
            "graph_path_length": rollout.candidate.graph_path_length,
            "graph_distance_to_route": rollout.candidate.graph_distance_to_route,
            "max_snap_displacement": rollout.max_snap_displacement,
        },
    }
    return combined


def summarize(rows: Sequence[Dict[str, Any]], failure_counts: Dict[str, int]) -> Dict[str, Any]:
    if not rows:
        return {"samples": 0, "failure_counts": failure_counts}
    prefix_lengths = [row["standard_prefix_end"] for row in rows]
    deviation_lengths = [
        row["recovery_start"] - row["deviation_start"] for row in rows
    ]
    recovery_lengths = [
        row["recovery_end"] - row["recovery_start"] for row in rows
    ]
    return {
        "samples": len(rows),
        "by_dataset": {
            dataset: sum(row["source_dataset"] == dataset for row in rows)
            for dataset in sorted({row["source_dataset"] for row in rows})
        },
        "by_requested_depth": {
            str(depth): sum(row["requested_deviation_depth"] == depth for row in rows)
            for depth in sorted({row["requested_deviation_depth"] for row in rows})
        },
        "actual_depth": {
            "min": min(row["actual_deviation_depth"] for row in rows),
            "mean": sum(row["actual_deviation_depth"] for row in rows) / len(rows),
            "max": max(row["actual_deviation_depth"] for row in rows),
        },
        "standard_prefix_actions": {
            "min": min(prefix_lengths),
            "mean": sum(prefix_lengths) / len(prefix_lengths),
            "max": max(prefix_lengths),
        },
        "deviation_actions": {
            "min": min(deviation_lengths),
            "mean": sum(deviation_lengths) / len(deviation_lengths),
            "max": max(deviation_lengths),
        },
        "recovery_actions": {
            "min": min(recovery_lengths),
            "mean": sum(recovery_lengths) / len(recovery_lengths),
            "max": max(recovery_lengths),
        },
        "departure_forward_step": {
            "min": min(row["departure_forward_step"] for row in rows),
            "mean": sum(row["departure_forward_step"] for row in rows) / len(rows),
            "max": max(row["departure_forward_step"] for row in rows),
        },
        "return_error": {
            "max": max(row["return_error"] for row in rows),
            "mean": sum(row["return_error"] for row in rows) / len(rows),
        },
        "max_breadcrumb_return_error": max(
            row["max_breadcrumb_return_error"] for row in rows
        ),
        "recovery_arc_ratio": {
            "min": min(row["recovery_arc_ratio"] for row in rows),
            "max": max(row["recovery_arc_ratio"] for row in rows),
        },
        "ambiguous_initial_turn_prefix": {
            "min": min(row["ambiguous_initial_turn_prefix"] for row in rows),
            "mean": sum(row["ambiguous_initial_turn_prefix"] for row in rows)
            / len(rows),
            "max": max(row["ambiguous_initial_turn_prefix"] for row in rows),
        },
        "failure_counts": dict(sorted(failure_counts.items())),
    }


def physical_route_key(route_record: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(route_record["dataset"]),
        str(route_record["scene_id"]),
        str(route_record["trajectory_id"]),
    )


def row_route_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(row["source_dataset"]),
        str(row["scene_id"]),
        str(row["source_trajectory_id"]),
    )


def route_belongs_to_worker(
    route_record: Dict[str, Any],
    scene_worker_assignment: Dict[str, int],
    worker_index: int,
) -> bool:
    scene = scan_id(route_record["scene_id"])
    if scene not in scene_worker_assignment:
        raise KeyError(f"scene has no worker assignment: {scene}")
    return scene_worker_assignment[scene] == worker_index


def build_scene_worker_assignment(
    route_inventory: Dict[str, Sequence[Dict[str, Any]]],
    num_workers: int,
    seed: int,
) -> Tuple[Dict[str, int], List[int]]:
    """Assign each scene once, greedily balancing physical-route counts."""

    scene_counts: Counter[str] = Counter()
    for route_records in route_inventory.values():
        scene_counts.update(scan_id(route["scene_id"]) for route in route_records)
    ordered_scenes = sorted(
        scene_counts,
        key=lambda scene: (
            -scene_counts[scene],
            stable_seed(seed, scene, "scene_worker_assignment"),
            scene,
        ),
    )
    worker_loads = [0 for _ in range(num_workers)]
    assignment: Dict[str, int] = {}
    for scene in ordered_scenes:
        worker_index = min(
            range(num_workers),
            key=lambda index: (worker_loads[index], index),
        )
        assignment[scene] = worker_index
        worker_loads[worker_index] += scene_counts[scene]
    return assignment, worker_loads


def select_worker_routes(
    route_records: Sequence[Dict[str, Any]],
    dataset_name: str,
    scene_worker_assignment: Dict[str, int],
    worker_index: int,
    seed: int,
    route_order: str,
    max_routes_to_try: Optional[int],
) -> List[Dict[str, Any]]:
    selected = [
        route
        for route in route_records
        if route_belongs_to_worker(
            route,
            scene_worker_assignment=scene_worker_assignment,
            worker_index=worker_index,
        )
    ]
    route_rng = random.Random(stable_seed(seed, dataset_name, "routes"))
    route_rng.shuffle(selected)
    if route_order == "scene":
        scene_order = list(dict.fromkeys(scan_id(route["scene_id"]) for route in selected))
        scene_rank = {scene: index for index, scene in enumerate(scene_order)}
        selected.sort(key=lambda route: scene_rank[scan_id(route["scene_id"])])
    if max_routes_to_try is not None:
        selected = selected[:max_routes_to_try]
    return selected


def build_assigned_route_inventory(
    route_inventory: Dict[str, Sequence[Dict[str, Any]]],
    dataset_names: Sequence[str],
    scene_worker_assignment: Dict[str, int],
    num_workers: int,
    seed: int,
    route_order: str,
    max_routes_to_try: Optional[int],
) -> Dict[int, Dict[str, List[Dict[str, Any]]]]:
    return {
        worker_index: {
            dataset_name: select_worker_routes(
                route_inventory[dataset_name],
                dataset_name=dataset_name,
                scene_worker_assignment=scene_worker_assignment,
                worker_index=worker_index,
                seed=seed,
                route_order=route_order,
                max_routes_to_try=max_routes_to_try,
            )
            for dataset_name in dataset_names
        }
        for worker_index in range(num_workers)
    }


def route_inventory_digest(route_records: Sequence[Dict[str, Any]]) -> str:
    entries = []
    for route in sorted(route_records, key=physical_route_key):
        episode = route["episode"]
        entries.append(
            {
                "route_key": physical_route_key(route),
                "canonical_episode_id": route["canonical_episode_id"],
                "selected_instruction_episode_id": route[
                    "selected_instruction_episode_id"
                ],
                "instruction": route["instruction"],
                "start_position": episode.get("start_position"),
                "start_rotation": episode.get("start_rotation"),
                "reference_path": episode.get("reference_path"),
            }
        )
    payload = json.dumps(
        entries,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def depth_attempt_order(
    depth_choices: Sequence[float],
    rng: random.Random,
) -> List[float]:
    """Choose one weighted preferred depth, then deterministic fallbacks."""

    preferred = float(rng.choice(list(depth_choices)))
    remaining = list(dict.fromkeys(float(depth) for depth in depth_choices))
    rng.shuffle(remaining)
    return [preferred, *(depth for depth in remaining if depth != preferred)]


def safe_identifier(value: Any) -> str:
    raw = str(value)
    return "".join(character if character.isalnum() else "_" for character in raw)


def generate_dataset_samples(
    dataset_name: str,
    assigned_route_records: Sequence[Dict[str, Any]],
    depth_choices: Sequence[float],
    image_dataset_root: Path,
    gpu_id: int,
    image_width: int,
    image_height: int,
    seed: int,
    max_samples_per_dataset: Optional[int],
    max_mapping_error: float,
    worker_index: int,
    existing_route_keys: set[Tuple[str, str, str]],
    route_outcomes: Dict[Tuple[str, str, str], Dict[str, Any]],
    row_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
    outcome_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
    resume: bool = False,
    min_free_disk_gb: float = DEFAULT_MIN_FREE_DISK_GB,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    route_records = list(assigned_route_records)
    assigned_route_keys = {physical_route_key(route) for route in route_records}

    def record_outcome(
        route_record: Dict[str, Any],
        status: str,
        failure_code: Optional[str] = None,
        failure_detail: Optional[str] = None,
        candidate_failure_counts: Optional[Dict[str, int]] = None,
    ) -> None:
        outcome = make_route_outcome(
            route_record,
            worker_index=worker_index,
            status=status,
            failure_code=failure_code,
            failure_detail=failure_detail,
            candidate_failure_counts=candidate_failure_counts,
        )
        key = route_outcome_key(outcome)
        if key in route_outcomes:
            raise ValueError(f"attempted to overwrite terminal route outcome: {key}")
        if outcome_sink is not None:
            outcome_sink(outcome)
        route_outcomes[key] = outcome

    config = build_env_config(dataset_name, gpu_id, image_width, image_height)
    with silence_external_output():
        dataset = habitat.datasets.make_dataset(
            id_dataset=config.habitat.dataset.type,
            config=config.habitat.dataset,
        )
    habitat_episode_by_id = {
        str(episode.episode_id): episode for episode in dataset.episodes
    }
    pending_route_records = [
        route
        for route in route_records
        if physical_route_key(route) not in route_outcomes
        and physical_route_key(route) not in existing_route_keys
    ]
    failure_counts: Dict[str, int] = defaultdict(int)
    valid_pending_route_records = []
    for route_record in pending_route_records:
        episode_id = route_record["canonical_episode_id"]
        if episode_id not in habitat_episode_by_id:
            failure_counts["missing_habitat_episode"] += 1
            record_outcome(
                route_record,
                status="terminal_failure",
                failure_code="missing_habitat_episode",
                failure_detail=f"canonical episode {episode_id} is absent",
            )
        else:
            valid_pending_route_records.append(route_record)
    pending_route_records = valid_pending_route_records
    dataset.episodes = [
        habitat_episode_by_id[route["canonical_episode_id"]]
        for route in pending_route_records
    ]
    if not pending_route_records:
        return [], dict(failure_counts)

    graph_cache: Dict[str, ConnectivityGraph] = {}
    rows: List[Dict[str, Any]] = []
    existing_dataset_count = sum(
        key in assigned_route_keys for key in existing_route_keys
    )
    env = None
    try:
        with silence_external_output():
            env = habitat.Env(config=config.habitat, dataset=dataset)
        for route_record in pending_route_records:
            if (
                max_samples_per_dataset is not None
                and existing_dataset_count + len(rows) >= max_samples_per_dataset
            ):
                break
            scene = scan_id(route_record["scene_id"])
            episode_id = route_record["canonical_episode_id"]
            episode = habitat_episode_by_id[episode_id]
            graph = graph_cache.get(scene)
            if graph is None:
                connectivity_path = DEFAULT_CONNECTIVITY_ROOT / f"{scene}_connectivity.json"
                if not connectivity_path.is_file():
                    failure_counts["missing_connectivity"] += 1
                    record_outcome(
                        route_record,
                        status="terminal_failure",
                        failure_code="missing_connectivity",
                        failure_detail=str(connectivity_path),
                    )
                    continue
                graph = ConnectivityGraph(connectivity_path)
                graph_cache[scene] = graph

            route_failure_counts: Dict[str, int] = defaultdict(int)
            try:
                mapped_route, mapping_metrics = graph.map_reference_path(
                    route_record["episode"]["reference_path"],
                    max_mapping_error=max_mapping_error,
                )
                route_record["route_mapping_metrics"] = mapping_metrics
                standard_trace = generate_standard_trace(env, episode)
                route_record["standard_actions"] = standard_trace.actions
            except PilotGenerationError as error:
                code = pilot_failure_code(error)
                failure_counts[code] += 1
                route_failure_counts[code] += 1
                record_outcome(
                    route_record,
                    status="terminal_failure",
                    failure_code="route_setup_failed",
                    failure_detail=str(error),
                    candidate_failure_counts=dict(route_failure_counts),
                )
                continue

            depth_rng = random.Random(
                stable_seed(seed, *physical_route_key(route_record), "depth")
            )
            rollout: Optional[SegmentRollout] = None
            for requested_depth in depth_attempt_order(depth_choices, depth_rng):
                candidates = graph.candidate_branches(
                    mapped_route,
                    requested_depth=requested_depth,
                )
                if not candidates:
                    code = f"no_branch_at_{requested_depth:g}m"
                    failure_counts[code] += 1
                    route_failure_counts[code] += 1
                    continue
                candidate_rng = random.Random(
                    stable_seed(
                        seed,
                        dataset_name,
                        route_record["scene_id"],
                        route_record["trajectory_id"],
                        requested_depth,
                    )
                )
                candidates = shuffle_candidates_by_anchor(candidates, candidate_rng)
                for candidate in candidates:
                    try:
                        rollout = try_candidate(
                            env,
                            episode,
                            standard_trace,
                            graph,
                            candidate,
                            requested_depth,
                        )
                        break
                    except PilotGenerationError as error:
                        code = pilot_failure_code(error)
                        failure_counts[code] += 1
                        route_failure_counts[code] += 1
                if rollout is not None:
                    break
            if rollout is None:
                failure_counts["no_valid_recovery_for_route"] += 1
                record_outcome(
                    route_record,
                    status="terminal_failure",
                    failure_code="no_valid_recovery_for_route",
                    failure_detail="no branch candidate passed the trajectory checks",
                    candidate_failure_counts=dict(route_failure_counts),
                )
                continue

            free_bytes = shutil.disk_usage(image_dataset_root).free
            required_free_bytes = int(min_free_disk_gb * (1024 ** 3))
            if free_bytes < required_free_bytes:
                raise OSError(
                    f"free disk space fell below {min_free_disk_gb:g} GiB at "
                    f"{image_dataset_root}: {free_bytes / (1024 ** 3):.1f} GiB left"
                )

            sample_id = (
                f"{dataset_name}_recovery_"
                f"{safe_identifier(scene)}_traj{safe_identifier(route_record['trajectory_id'])}_"
                f"src{safe_identifier(episode_id)}_d{rollout.requested_depth:.1f}"
            )
            sample_directory = image_dataset_root / sample_id
            temporary_directory = (
                image_dataset_root
                / ".inprogress"
                / f"{sample_id}.worker{worker_index:02d}.tmp"
            )
            if temporary_directory.exists():
                shutil.rmtree(temporary_directory)
            if sample_directory.exists():
                if not resume:
                    raise PilotGenerationError(
                        f"image directory already exists: {sample_directory}"
                    )
                print(
                    f"removing orphaned unindexed sample directory: {sample_directory}",
                    flush=True,
                )
                shutil.rmtree(sample_directory)
            try:
                saved = replay_and_save_trajectory(
                    env,
                    episode,
                    standard_trace,
                    rollout,
                    temporary_directory,
                    image_size=(image_width, image_height),
                )
                row = make_row(sample_id, route_record, rollout, saved)
                expected_frame_count = len(row["actions"]) + 1
                if len(saved["image_paths"]) != expected_frame_count:
                    raise PilotGenerationError(
                        f"{sample_id}: expected {expected_frame_count} frames, got "
                        f"{len(saved['image_paths'])}"
                    )
            except Exception:
                if temporary_directory.exists():
                    shutil.rmtree(temporary_directory)
                raise
            sample_directory.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary_directory, sample_directory)
            if row_sink is not None:
                row_sink(row)
            record_outcome(
                route_record,
                status="success",
                candidate_failure_counts=dict(route_failure_counts),
            )
            rows.append(row)
            existing_route_keys.add(physical_route_key(route_record))
            print(
                json.dumps(
                    {
                        "status": "saved",
                        "sample_id": sample_id,
                        "dataset": dataset_name,
                        "instruction_episode_id": row["source_episode_id"],
                        "requested_depth": rollout.requested_depth,
                        "actual_depth": row["actual_deviation_depth"],
                        "prefix_actions": rollout.prefix_action_count,
                        "deviation_actions": len(rollout.deviation_actions),
                        "recovery_actions": len(rollout.recovery_actions),
                        "return_error": rollout.return_error,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        if env is not None:
            with silence_external_output():
                env.close()
    return rows, dict(failure_counts)


def parse_depths(raw: Sequence[str]) -> List[float]:
    depths = [float(value) for value in raw]
    if not depths or any(not math.isfinite(value) or value < 2.0 for value in depths):
        raise ValueError("all deviation depths must be finite and at least 2.0m")
    return depths


def collection_paths(
    output_root: Path,
    output_dataset_name: str,
) -> Dict[str, Path]:
    manifest_path = output_root / "sub_dataset" / f"{output_dataset_name}.jsonl"
    return {
        "manifest": manifest_path,
        "summary": manifest_path.with_suffix(".summary.json"),
        "route_outcomes": manifest_path.with_suffix(".route_outcomes.jsonl"),
        "progress": Path(f"{manifest_path}.inprogress"),
        "images": output_root / "images" / output_dataset_name,
    }


def worker_manifest_path(progress_directory: Path, worker_index: int) -> Path:
    return progress_directory / f"rank_{worker_index:02d}.jsonl"


def worker_summary_path(progress_directory: Path, worker_index: int) -> Path:
    return progress_directory / f"rank_{worker_index:02d}.summary.json"


def worker_route_ledger_path(progress_directory: Path, worker_index: int) -> Path:
    return progress_directory / f"rank_{worker_index:02d}.routes.jsonl"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configuration_sha256(configuration: Dict[str, Any]) -> str:
    payload = json.dumps(
        configuration,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def is_uncapped_collection(configuration: Dict[str, Any]) -> bool:
    return (
        configuration.get("max_routes_to_try") is None
        and configuration.get("max_samples_per_dataset") is None
    )


def route_outcome_key(outcome: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(outcome["source_dataset"]),
        str(outcome["scene_id"]),
        str(outcome["source_trajectory_id"]),
    )


def make_route_outcome(
    route_record: Dict[str, Any],
    worker_index: int,
    status: str,
    failure_code: Optional[str] = None,
    failure_detail: Optional[str] = None,
    candidate_failure_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    if status not in {"success", "terminal_failure"}:
        raise ValueError(f"invalid route outcome status: {status}")
    outcome = {
        "schema_version": 1,
        "worker_index": int(worker_index),
        "source_dataset": str(route_record["dataset"]),
        "scene_id": str(route_record["scene_id"]),
        "source_trajectory_id": str(route_record["trajectory_id"]),
        "canonical_episode_id": str(route_record["canonical_episode_id"]),
        "status": status,
        "candidate_failure_counts": dict(
            sorted((candidate_failure_counts or {}).items())
        ),
    }
    if failure_code is not None:
        outcome["failure_code"] = str(failure_code)
    if failure_detail is not None:
        outcome["failure_detail"] = str(failure_detail)[:1000]
    return outcome


def validate_route_outcome(
    outcome: Dict[str, Any],
    expected_worker_index: int,
) -> None:
    context = route_outcome_key(outcome)
    if int(outcome.get("schema_version", 0)) != 1:
        raise ValueError(f"{context}: invalid route-outcome schema")
    if int(outcome.get("worker_index", -1)) != expected_worker_index:
        raise ValueError(f"{context}: route outcome belongs to another worker")
    status = outcome.get("status")
    if status not in {"success", "terminal_failure"}:
        raise ValueError(f"{context}: invalid route outcome status {status!r}")
    if status == "terminal_failure" and not outcome.get("failure_code"):
        raise ValueError(f"{context}: terminal failure has no failure_code")
    counts = outcome.get("candidate_failure_counts", {})
    if not isinstance(counts, dict) or any(int(value) < 0 for value in counts.values()):
        raise ValueError(f"{context}: invalid candidate failure counts")


def load_route_outcomes(
    path: Path,
    worker_index: int,
    resume: bool,
) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    rows = read_worker_jsonl_for_resume(path) if resume else read_jsonl(path)
    outcomes: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for outcome in rows:
        validate_route_outcome(outcome, worker_index)
        key = route_outcome_key(outcome)
        previous = outcomes.get(key)
        if previous is not None and previous != outcome:
            raise ValueError(f"conflicting route outcomes for {key}")
        if previous is not None:
            raise ValueError(f"duplicate route outcome for {key}")
        outcomes[key] = outcome
    return outcomes


def atomic_write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def atomic_write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def collection_configuration(
    args: argparse.Namespace,
    input_root: Path,
    route_inventory: Dict[str, Sequence[Dict[str, Any]]],
    scene_worker_assignment: Dict[str, int],
    worker_loads: Sequence[int],
) -> Dict[str, Any]:
    return {
        "trajectory_algorithm_version": 3,
        "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "schema_version": 2,
        "input_root": str(input_root),
        "output_dataset_name": args.output_dataset_name,
        "datasets": list(args.dataset_names),
        "depth_choices": parse_depths(args.depth_choices),
        "seed": args.seed,
        "num_workers": args.num_workers,
        "image_size": [args.image_width, args.image_height],
        "max_mapping_error": args.max_mapping_error,
        "route_order": args.route_order,
        "max_routes_to_try": args.max_routes_to_try,
        "max_samples_per_dataset": args.max_samples_per_dataset,
        "min_free_disk_gb": args.min_free_disk_gb,
        "route_inventory_counts": {
            dataset_name: len(route_inventory[dataset_name])
            for dataset_name in args.dataset_names
        },
        "route_inventory_sha256": {
            dataset_name: route_inventory_digest(route_inventory[dataset_name])
            for dataset_name in args.dataset_names
        },
        "worker_assignment": {
            "strategy": "scene_greedy_balanced_v1",
            "scene_count": len(scene_worker_assignment),
            "assignment_sha256": hashlib.sha256(
                json.dumps(
                    scene_worker_assignment,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "uncapped_route_loads": [int(load) for load in worker_loads],
        },
        "standard_action_source": "reference_path_shortest_path_follower",
        "standard_waypoint_radius": STANDARD_WAYPOINT_RADIUS,
        "local_target_radius": LOCAL_TARGET_RADIUS,
        "route_contact_radius": ROUTE_CONTACT_RADIUS,
        "return_position_tolerance": RETURN_POSITION_TOLERANCE,
        "breadcrumb_return_tolerance": BREADCRUMB_RETURN_TOLERANCE,
        "max_route_contact_forward_steps": MAX_ROUTE_CONTACT_FORWARD_STEPS,
        "min_translation": MIN_TRANSLATION,
    }


def require_matching_configuration(
    configuration_path: Path,
    requested: Dict[str, Any],
) -> None:
    if not configuration_path.is_file():
        raise FileNotFoundError(
            f"collection has not been initialized: {configuration_path}"
        )
    with configuration_path.open("r", encoding="utf-8") as handle:
        existing = json.load(handle)
    mismatches = {
        key: {"existing": existing.get(key), "requested": value}
        for key, value in requested.items()
        if existing.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "collection configuration changed while resuming: "
            f"{json.dumps(mismatches, ensure_ascii=False)}"
        )


def initialize_collection(
    paths: Dict[str, Path],
    configuration: Dict[str, Any],
    resume: bool,
) -> None:
    manifest_path = paths["manifest"]
    progress_directory = paths["progress"]
    image_dataset_root = paths["images"]
    configuration_path = progress_directory / "configuration.json"
    if resume:
        if not progress_directory.is_dir():
            raise FileNotFoundError(
                "--resume was requested but no in-progress collection exists: "
                f"{progress_directory}"
            )
    else:
        if manifest_path.exists():
            raise FileExistsError(
                f"final recovery annotation already exists: {manifest_path}"
            )
        if paths["summary"].exists():
            raise FileExistsError(
                f"recovery summary already exists: {paths['summary']}"
            )
        if paths["route_outcomes"].exists():
            raise FileExistsError(
                "recovery route-outcome ledger already exists: "
                f"{paths['route_outcomes']}"
            )
        if progress_directory.exists():
            raise FileExistsError(
                "unfinished recovery collection already exists; pass --resume: "
                f"{progress_directory}"
            )
        if image_dataset_root.exists() and any(image_dataset_root.iterdir()):
            raise FileExistsError(
                f"recovery image directory is not empty: {image_dataset_root}"
            )
    progress_directory.mkdir(parents=True, exist_ok=True)
    image_dataset_root.mkdir(parents=True, exist_ok=True)
    (image_dataset_root / ".inprogress").mkdir(parents=True, exist_ok=True)
    if configuration_path.exists():
        require_matching_configuration(configuration_path, configuration)
    else:
        atomic_write_json(configuration_path, configuration)
    print(
        json.dumps(
            {
                "status": "initialized",
                "manifest": str(manifest_path),
                "images": str(image_dataset_root),
                "resume": resume,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def validate_manifest_row(
    row: Dict[str, Any],
    image_dataset_root: Path,
    expected_image_size: Optional[Tuple[int, int]] = None,
    check_image_headers: bool = False,
) -> Dict[str, int]:
    context = str(row.get("episode_id", "<unknown>"))
    if Path(context).name != context or context in {"", ".", ".."}:
        raise ValueError(f"unsafe trajectory identifier: {context!r}")
    if int(row.get("schema_version", 0)) != 2:
        raise ValueError(f"{context}: expected schema_version=2")
    if int(row.get("trajectory_algorithm_version", 0)) != 3:
        raise ValueError(f"{context}: expected trajectory_algorithm_version=3")
    if float(row.get("standard_waypoint_radius", -1.0)) != STANDARD_WAYPOINT_RADIUS:
        raise ValueError(f"{context}: recovery trajectory is not the 0.3m version")
    if row.get("standard_action_source") != "reference_path_shortest_path_follower":
        raise ValueError(f"{context}: unexpected standard action source")
    if str(row.get("trajectory_id")) != context:
        raise ValueError(f"{context}: trajectory_id must equal episode_id")
    actions_raw = row.get("actions")
    if not isinstance(actions_raw, list) or not actions_raw:
        raise ValueError(f"{context}: actions must be a non-empty list")
    actions = [int(action) for action in actions_raw]
    invalid_actions = sorted(set(actions) - SUPPORTED_ACTIONS)
    if invalid_actions:
        raise ValueError(f"{context}: invalid actions {invalid_actions}")
    if STOP_ACTION in actions:
        raise ValueError(f"{context}: continuous prefix/deviation/recovery contains stop")
    standard_actions = validate_actions(
        row.get("standard_actions", []),
        f"{context}:standard_actions",
    )
    deviation_start = int(row.get("deviation_start", -1))
    recovery_start = int(row.get("recovery_start", -1))
    recovery_end = int(row.get("recovery_end", -1))
    if int(row.get("standard_prefix_end", -1)) != deviation_start:
        raise ValueError(f"{context}: inconsistent clean/deviation boundary")
    if int(row.get("standard_prefix_action_count", -1)) != deviation_start:
        raise ValueError(f"{context}: inconsistent standard-prefix action count")
    if not 0 <= deviation_start < recovery_start < recovery_end == len(actions):
        raise ValueError(f"{context}: invalid trajectory segment boundaries")
    if actions[:deviation_start] != standard_actions[:deviation_start]:
        raise ValueError(f"{context}: saved clean prefix differs from standard actions")
    if STOP_ACTION in actions[recovery_start:recovery_end]:
        raise ValueError(f"{context}: recovery interval contains stop")

    positions = row.get("action_positions")
    rotations = row.get("action_rotations")
    if not isinstance(positions, list) or len(positions) != len(actions) + 1:
        raise ValueError(f"{context}: action_positions must have actions+1 entries")
    if not isinstance(rotations, list) or len(rotations) != len(actions) + 1:
        raise ValueError(f"{context}: action_rotations must have actions+1 entries")
    for label, values, width in (
        ("action_positions", positions, 3),
        ("action_rotations", rotations, 4),
    ):
        for index, value in enumerate(values):
            if not isinstance(value, list) or len(value) != width:
                raise ValueError(f"{context}: {label}[{index}] must have {width} values")
            if any(not math.isfinite(float(item)) for item in value):
                raise ValueError(f"{context}: {label}[{index}] contains non-finite values")
    deviation_forward_path = row.get("deviation_forward_path")
    recovery_forward_path = row.get("recovery_forward_path")
    for label, path in (
        ("deviation_forward_path", deviation_forward_path),
        ("recovery_forward_path", recovery_forward_path),
    ):
        if not isinstance(path, list) or not path:
            raise ValueError(f"{context}: {label} must be non-empty")
        for index, point in enumerate(path):
            if (
                not isinstance(point, list)
                or len(point) != 3
                or any(not math.isfinite(float(item)) for item in point)
            ):
                raise ValueError(f"{context}: invalid {label}[{index}]")
    expected_deviation_forward_path = [positions[deviation_start]] + [
        positions[action_index + 1]
        for action_index in range(deviation_start, recovery_start)
        if actions[action_index] == FORWARD_ACTION
    ]
    expected_recovery_forward_path = [positions[recovery_start]] + [
        positions[action_index + 1]
        for action_index in range(recovery_start, recovery_end)
        if actions[action_index] == FORWARD_ACTION
    ]
    if deviation_forward_path != expected_deviation_forward_path:
        raise ValueError(f"{context}: stored deviation breadcrumbs disagree with actions")
    if recovery_forward_path != expected_recovery_forward_path:
        raise ValueError(f"{context}: stored recovery breadcrumbs disagree with actions")
    if len(deviation_forward_path) != len(recovery_forward_path):
        raise ValueError(f"{context}: deviation/recovery breadcrumb counts differ")
    breadcrumb_errors = [
        euclidean_distance(actual, expected)
        for actual, expected in zip(
            recovery_forward_path,
            reversed(deviation_forward_path),
            strict=True,
        )
    ]
    recomputed_breadcrumb_error = max(breadcrumb_errors, default=0.0)
    stored_breadcrumb_error = float(
        row.get("max_breadcrumb_return_error", math.inf)
    )
    if abs(recomputed_breadcrumb_error - stored_breadcrumb_error) > 1e-6:
        raise ValueError(f"{context}: breadcrumb error summary is inconsistent")
    for label, maximum in (
        ("return_error", RETURN_POSITION_TOLERANCE),
        ("max_breadcrumb_return_error", BREADCRUMB_RETURN_TOLERANCE),
    ):
        value = float(row.get(label, math.inf))
        if not math.isfinite(value) or value > maximum + 1e-6:
            raise ValueError(f"{context}: invalid {label}={value}")
    actual_depth = float(row.get("actual_deviation_depth", math.nan))
    requested_depth = float(row.get("requested_deviation_depth", math.nan))
    distance_trace = row.get("distance_to_standard_route")
    if (
        not isinstance(distance_trace, list)
        or len(distance_trace) != len(deviation_forward_path)
        or any(not math.isfinite(float(value)) for value in distance_trace)
    ):
        raise ValueError(f"{context}: invalid distance-to-route trace")
    if (
        not math.isfinite(actual_depth)
        or not math.isfinite(requested_depth)
        or actual_depth + 1e-6 < requested_depth
        or abs(actual_depth - max(float(value) for value in distance_trace)) > 1e-6
    ):
        raise ValueError(f"{context}: invalid deviation depth")
    departure_forward_step = int(row.get("departure_forward_step", 0))
    if not 1 <= departure_forward_step <= MAX_ROUTE_CONTACT_FORWARD_STEPS:
        raise ValueError(f"{context}: invalid departure_forward_step")
    image_directory = image_dataset_root / context
    if not image_directory.is_dir():
        raise FileNotFoundError(f"{context}: missing image directory {image_directory}")
    expected_frame_count = len(actions) + 1
    total_image_bytes = 0
    decode_indices = {0, expected_frame_count // 2, expected_frame_count - 1}
    for frame_index in range(expected_frame_count):
        frame_path = image_directory / f"frame_{frame_index}.jpg"
        if not frame_path.is_file():
            raise FileNotFoundError(f"{context}: missing {frame_path.name}")
        frame_bytes = frame_path.stat().st_size
        if frame_bytes <= 0:
            raise ValueError(f"{context}: empty image {frame_path.name}")
        total_image_bytes += frame_bytes
        if check_image_headers:
            with Image.open(frame_path) as frame:
                if frame.format != "JPEG":
                    raise ValueError(f"{context}: {frame_path.name} is not JPEG")
                if expected_image_size is not None and frame.size != expected_image_size:
                    raise ValueError(
                        f"{context}: {frame_path.name} has size {frame.size}, "
                        f"expected {expected_image_size}"
                    )
                if frame_index in decode_indices:
                    frame.load()
    actual_frame_count = sum(1 for _ in image_directory.glob("frame_*.jpg"))
    if actual_frame_count != expected_frame_count:
        raise ValueError(
            f"{context}: expected {expected_frame_count} frames, found "
            f"{actual_frame_count}"
        )
    return {"image_count": expected_frame_count, "image_bytes": total_image_bytes}


def finalize_collection(
    paths: Dict[str, Path],
    configuration: Dict[str, Any],
    assigned_route_inventory: Dict[int, Dict[str, List[Dict[str, Any]]]],
) -> Dict[str, Any]:
    progress_directory = paths["progress"]
    require_matching_configuration(
        progress_directory / "configuration.json",
        configuration,
    )
    num_workers = int(configuration["num_workers"])
    expected_worker_manifests = [
        worker_manifest_path(progress_directory, worker_index)
        for worker_index in range(num_workers)
    ]
    expected_worker_summaries = [
        worker_summary_path(progress_directory, worker_index)
        for worker_index in range(num_workers)
    ]
    expected_worker_ledgers = [
        worker_route_ledger_path(progress_directory, worker_index)
        for worker_index in range(num_workers)
    ]
    missing_worker_outputs = [
        str(path)
        for path in [
            *expected_worker_manifests,
            *expected_worker_summaries,
            *expected_worker_ledgers,
        ]
        if not path.is_file()
    ]
    if missing_worker_outputs:
        raise FileNotFoundError(
            "cannot finalize before every worker finishes; missing: "
            f"{missing_worker_outputs}"
        )
    actual_worker_manifests = {
        path
        for path in progress_directory.glob("rank_*.jsonl")
        if not path.name.endswith(".routes.jsonl")
    }
    unexpected_worker_manifests = sorted(
        str(path)
        for path in actual_worker_manifests - set(expected_worker_manifests)
    )
    if unexpected_worker_manifests:
        raise ValueError(
            f"unexpected worker manifests: {unexpected_worker_manifests}"
        )
    actual_worker_summaries = set(progress_directory.glob("rank_*.summary.json"))
    unexpected_worker_summaries = sorted(
        str(path)
        for path in actual_worker_summaries - set(expected_worker_summaries)
    )
    if unexpected_worker_summaries:
        raise ValueError(f"unexpected worker summaries: {unexpected_worker_summaries}")
    actual_worker_ledgers = set(progress_directory.glob("rank_*.routes.jsonl"))
    unexpected_worker_ledgers = sorted(
        str(path)
        for path in actual_worker_ledgers - set(expected_worker_ledgers)
    )
    if unexpected_worker_ledgers:
        raise ValueError(f"unexpected worker route ledgers: {unexpected_worker_ledgers}")

    rows_by_route: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    trajectory_ids: Dict[str, Tuple[str, str, str]] = {}
    all_outcomes: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    configuration_hash = configuration_sha256(configuration)
    image_count = 0
    image_bytes = 0
    worker_accounting: Dict[str, Dict[str, Any]] = {}
    candidate_failure_counts: Dict[str, int] = defaultdict(int)
    terminal_failure_counts: Dict[str, int] = defaultdict(int)

    for worker_index in range(num_workers):
        manifest_path = expected_worker_manifests[worker_index]
        ledger_path = expected_worker_ledgers[worker_index]
        summary_path = expected_worker_summaries[worker_index]
        expected_routes = {
            physical_route_key(route): route
            for dataset_routes in assigned_route_inventory[worker_index].values()
            for route in dataset_routes
        }
        worker_rows: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for row in read_jsonl(manifest_path):
            stats = validate_manifest_row(
                row,
                paths["images"],
                expected_image_size=tuple(configuration["image_size"]),
                check_image_headers=True,
            )
            route_key = row_route_key(row)
            if route_key not in expected_routes:
                raise ValueError(
                    f"worker {worker_index} emitted an unassigned route: {route_key}"
                )
            if route_key in worker_rows:
                raise ValueError(f"duplicate physical route in worker manifest: {route_key}")
            worker_rows[route_key] = row
            image_count += stats["image_count"]
            image_bytes += stats["image_bytes"]
            trajectory_id = str(row["trajectory_id"])
            previous_route = trajectory_ids.get(trajectory_id)
            if previous_route is not None and previous_route != route_key:
                raise ValueError(
                    f"trajectory_id collision {trajectory_id}: "
                    f"{previous_route} versus {route_key}"
                )
            trajectory_ids[trajectory_id] = route_key
            previous_row = rows_by_route.get(route_key)
            if previous_row is not None:
                raise ValueError(f"physical route assigned to multiple workers: {route_key}")
            rows_by_route[route_key] = row

        worker_outcomes = load_route_outcomes(
            ledger_path,
            worker_index=worker_index,
            resume=False,
        )
        unexpected_outcomes = sorted(set(worker_outcomes) - set(expected_routes))
        if unexpected_outcomes:
            raise ValueError(
                f"worker {worker_index} has unassigned route outcomes: "
                f"{unexpected_outcomes[:10]}"
            )
        for route_key, outcome in worker_outcomes.items():
            expected_route = expected_routes[route_key]
            if str(outcome["canonical_episode_id"]) != str(
                expected_route["canonical_episode_id"]
            ):
                raise ValueError(f"{route_key}: route outcome episode mismatch")
            if route_key in all_outcomes:
                raise ValueError(f"route outcome assigned to multiple workers: {route_key}")
            all_outcomes[route_key] = outcome
            for reason, count in outcome.get("candidate_failure_counts", {}).items():
                candidate_failure_counts[str(reason)] += int(count)
            if outcome["status"] == "terminal_failure":
                terminal_failure_counts[str(outcome["failure_code"])] += 1

        success_outcome_keys = {
            key for key, outcome in worker_outcomes.items() if outcome["status"] == "success"
        }
        terminal_failure_keys = {
            key
            for key, outcome in worker_outcomes.items()
            if outcome["status"] == "terminal_failure"
        }
        if success_outcome_keys != set(worker_rows):
            raise ValueError(
                f"worker {worker_index}: success ledger and manifest disagree"
            )
        covered_keys = success_outcome_keys | terminal_failure_keys
        unprocessed_keys = set(expected_routes) - covered_keys
        if configuration["max_samples_per_dataset"] is None and unprocessed_keys:
            raise ValueError(
                f"worker {worker_index} left {len(unprocessed_keys)} assigned routes "
                f"without a terminal outcome, for example {sorted(unprocessed_keys)[:5]}"
            )

        with summary_path.open("r", encoding="utf-8") as handle:
            worker_summary = json.load(handle)
        expected_accounting = {
            "assigned": len(expected_routes),
            "success": len(success_outcome_keys),
            "terminal_failure": len(terminal_failure_keys),
            "unprocessed": len(unprocessed_keys),
        }
        if worker_summary.get("complete") is not True:
            raise ValueError(f"worker {worker_index} summary is not a completion marker")
        if int(worker_summary.get("worker_index", -1)) != worker_index:
            raise ValueError(f"worker {worker_index} summary has the wrong rank")
        if worker_summary.get("configuration_sha256") != configuration_hash:
            raise ValueError(f"worker {worker_index} summary has the wrong configuration")
        if worker_summary.get("route_accounting") != expected_accounting:
            raise ValueError(f"worker {worker_index} route accounting disagrees")
        if int(worker_summary.get("manifest_rows", -1)) != len(worker_rows):
            raise ValueError(f"worker {worker_index} summary has the wrong row count")
        if worker_summary.get("manifest_sha256") != file_sha256(manifest_path):
            raise ValueError(f"worker {worker_index} manifest changed after completion")
        if int(worker_summary.get("route_ledger_rows", -1)) != len(worker_outcomes):
            raise ValueError(f"worker {worker_index} summary has the wrong ledger count")
        if worker_summary.get("route_ledger_sha256") != file_sha256(ledger_path):
            raise ValueError(f"worker {worker_index} route ledger changed after completion")
        worker_accounting[str(worker_index)] = expected_accounting

    rows = sorted(rows_by_route.values(), key=row_route_key)
    if paths["manifest"].is_file():
        existing_final_rows = sorted(read_jsonl(paths["manifest"]), key=row_route_key)
        if existing_final_rows != rows:
            raise ValueError("existing final manifest disagrees with worker manifests")
    expected_image_directories = set(trajectory_ids)
    actual_image_directories = {
        path.name
        for path in paths["images"].iterdir()
        if path.is_dir() and path.name != ".inprogress"
    }
    unexpected_image_directories = sorted(
        actual_image_directories - expected_image_directories
    )
    if unexpected_image_directories:
        preview = unexpected_image_directories[:10]
        raise ValueError(
            f"found {len(unexpected_image_directories)} unindexed recovery image "
            f"directories, for example: {preview}"
        )
    if configuration["max_samples_per_dataset"] is None:
        missing_success_datasets = [
            dataset_name
            for dataset_name in configuration["datasets"]
            if not any(row["source_dataset"] == dataset_name for row in rows)
        ]
        if missing_success_datasets:
            raise ValueError(
                f"datasets produced no successful trajectories: {missing_success_datasets}"
            )

    final_summary = summarize(rows, dict(candidate_failure_counts))
    by_dataset_accounting = {}
    for dataset_name in configuration["datasets"]:
        assigned_keys = {
            physical_route_key(route)
            for worker_routes in assigned_route_inventory.values()
            for route in worker_routes[dataset_name]
        }
        dataset_outcomes = {
            key: outcome
            for key, outcome in all_outcomes.items()
            if key in assigned_keys
        }
        success_count = sum(
            outcome["status"] == "success" for outcome in dataset_outcomes.values()
        )
        failure_count = sum(
            outcome["status"] == "terminal_failure"
            for outcome in dataset_outcomes.values()
        )
        by_dataset_accounting[dataset_name] = {
            "assigned": len(assigned_keys),
            "success": success_count,
            "terminal_failure": failure_count,
            "unprocessed": len(assigned_keys) - success_count - failure_count,
        }
    total_assigned = sum(item["assigned"] for item in worker_accounting.values())
    total_success = len(rows)
    total_terminal_failure = sum(terminal_failure_counts.values())
    if is_uncapped_collection(configuration):
        expected_inventory_counts = configuration["route_inventory_counts"]
        actual_inventory_counts = {
            dataset_name: by_dataset_accounting[dataset_name]["assigned"]
            for dataset_name in configuration["datasets"]
        }
        if actual_inventory_counts != expected_inventory_counts:
            raise ValueError(
                "uncapped worker assignment does not cover the source inventory: "
                f"expected {expected_inventory_counts}, got {actual_inventory_counts}"
            )
    final_summary["complete"] = (
        is_uncapped_collection(configuration)
        and total_assigned == total_success + total_terminal_failure
    )
    final_summary["route_accounting"] = {
        "assigned": total_assigned,
        "success": total_success,
        "terminal_failure": total_terminal_failure,
        "unprocessed": total_assigned - total_success - total_terminal_failure,
        "by_dataset": by_dataset_accounting,
        "by_worker": worker_accounting,
        "terminal_failure_counts": dict(sorted(terminal_failure_counts.items())),
    }
    final_summary["image_count"] = image_count
    final_summary["image_bytes"] = image_bytes
    final_summary["configuration"] = configuration
    final_summary["configuration_sha256"] = configuration_hash
    final_summary["manifest"] = str(paths["manifest"])
    final_summary["route_outcomes"] = str(paths["route_outcomes"])
    final_summary["images"] = str(paths["images"])

    temporary_image_root = paths["images"] / ".inprogress"
    if temporary_image_root.exists() and any(temporary_image_root.iterdir()):
        raise RuntimeError(
            "incomplete image directories remain; collection cannot be finalized: "
            f"{temporary_image_root}"
        )
    ordered_outcomes = [all_outcomes[key] for key in sorted(all_outcomes)]
    atomic_write_jsonl(paths["manifest"], rows)
    atomic_write_jsonl(paths["route_outcomes"], ordered_outcomes)
    final_summary["manifest_sha256"] = file_sha256(paths["manifest"])
    final_summary["route_outcomes_sha256"] = file_sha256(paths["route_outcomes"])
    atomic_write_json(paths["summary"], final_summary)
    shutil.rmtree(progress_directory)
    if temporary_image_root.exists():
        temporary_image_root.rmdir()
    print(json.dumps(final_summary, ensure_ascii=False, indent=2), flush=True)
    return final_summary


def verify_final_collection(
    paths: Dict[str, Path],
    configuration: Dict[str, Any],
    assigned_route_inventory: Dict[int, Dict[str, List[Dict[str, Any]]]],
) -> Dict[str, Any]:
    if paths["progress"].exists():
        raise RuntimeError(
            f"collection still has in-progress state: {paths['progress']}"
        )
    for label in ("manifest", "summary", "route_outcomes"):
        if not paths[label].is_file():
            raise FileNotFoundError(f"missing final {label}: {paths[label]}")
    with paths["summary"].open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("configuration") != configuration:
        raise ValueError("final collection configuration does not match this request")
    if summary.get("configuration_sha256") != configuration_sha256(configuration):
        raise ValueError("final collection configuration hash is invalid")
    if summary.get("manifest_sha256") != file_sha256(paths["manifest"]):
        raise ValueError("final recovery manifest hash mismatch")
    if summary.get("route_outcomes_sha256") != file_sha256(
        paths["route_outcomes"]
    ):
        raise ValueError("final route-outcome ledger hash mismatch")

    expected_routes: Dict[Tuple[str, str, str], Tuple[int, Dict[str, Any]]] = {}
    for worker_index, worker_routes in assigned_route_inventory.items():
        for dataset_routes in worker_routes.values():
            for route in dataset_routes:
                key = physical_route_key(route)
                if key in expected_routes:
                    raise ValueError(f"route assigned more than once: {key}")
                expected_routes[key] = (worker_index, route)

    rows_by_route: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    image_count = 0
    image_bytes = 0
    trajectory_ids = set()
    for row in read_jsonl(paths["manifest"]):
        stats = validate_manifest_row(
            row,
            paths["images"],
            expected_image_size=tuple(configuration["image_size"]),
            check_image_headers=True,
        )
        key = row_route_key(row)
        if key not in expected_routes:
            raise ValueError(f"final manifest contains an unassigned route: {key}")
        if key in rows_by_route:
            raise ValueError(f"duplicate final route: {key}")
        rows_by_route[key] = row
        trajectory_ids.add(str(row["trajectory_id"]))
        image_count += stats["image_count"]
        image_bytes += stats["image_bytes"]

    outcomes: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for outcome in read_jsonl(paths["route_outcomes"]):
        key = route_outcome_key(outcome)
        if key not in expected_routes:
            raise ValueError(f"final ledger contains an unassigned route: {key}")
        expected_worker, expected_route = expected_routes[key]
        validate_route_outcome(outcome, expected_worker)
        if str(outcome["canonical_episode_id"]) != str(
            expected_route["canonical_episode_id"]
        ):
            raise ValueError(f"{key}: final outcome episode mismatch")
        if key in outcomes:
            raise ValueError(f"duplicate final route outcome: {key}")
        outcomes[key] = outcome
    success_keys = {
        key for key, outcome in outcomes.items() if outcome["status"] == "success"
    }
    if success_keys != set(rows_by_route):
        raise ValueError("final success outcomes and manifest routes disagree")
    if configuration["max_samples_per_dataset"] is None:
        if set(outcomes) != set(expected_routes):
            raise ValueError("final production ledger does not cover every assigned route")
    if is_uncapped_collection(configuration):
        expected_counts = configuration["route_inventory_counts"]
        actual_counts = Counter(key[0] for key in expected_routes)
        if {name: actual_counts[name] for name in configuration["datasets"]} != expected_counts:
            raise ValueError("uncapped final assignment does not cover source inventory")
        if summary.get("complete") is not True:
            raise ValueError("final production summary is not marked complete")
    elif summary.get("complete") is not False:
        raise ValueError("a capped collection must not be marked complete")
    if int(summary.get("samples", -1)) != len(rows_by_route):
        raise ValueError("final summary sample count mismatch")
    if int(summary.get("image_count", -1)) != image_count:
        raise ValueError("final summary image count mismatch")
    if int(summary.get("image_bytes", -1)) != image_bytes:
        raise ValueError("final summary image byte count mismatch")
    actual_image_directories = {
        path.name
        for path in paths["images"].iterdir()
        if path.is_dir() and path.name != ".inprogress"
    }
    if actual_image_directories != trajectory_ids:
        raise ValueError("final image directories and manifest trajectories disagree")
    print(
        json.dumps(
            {
                "status": "verified",
                "samples": len(rows_by_route),
                "route_outcomes": len(outcomes),
                "image_count": image_count,
                "manifest": str(paths["manifest"]),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--output_dataset_name",
        default=DEFAULT_OUTPUT_DATASET_NAME,
        help="Dataset directory and sub_dataset JSONL stem.",
    )
    parser.add_argument("--dataset_names", nargs="+", default=["r2r", "rxr"])
    parser.add_argument(
        "--depth_choices",
        nargs="+",
        default=["2.0", "2.0", "2.5", "2.5", "3.0", "3.5"],
        help=(
            "Weighted deviation-depth choices in meters. Duplicate values act as "
            "weights; unavailable preferred depths fall back to another choice."
        ),
    )
    parser.add_argument("--gpu_id", type=int, default=2)
    parser.add_argument("--image_width", type=int, default=1280)
    parser.add_argument("--image_height", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_routes_to_try",
        type=int,
        default=None,
        help="Optional deterministic cap on physical routes attempted by this worker.",
    )
    parser.add_argument(
        "--max_samples_per_dataset",
        type=int,
        default=None,
        help="Optional pilot cap on successfully saved trajectories per dataset.",
    )
    parser.add_argument("--max_mapping_error", type=float, default=0.5)
    parser.add_argument(
        "--min_free_disk_gb",
        type=float,
        default=DEFAULT_MIN_FREE_DISK_GB,
        help="Fail before saving another trajectory below this free-space watermark.",
    )
    parser.add_argument(
        "--route_order",
        choices=("scene", "diverse"),
        default="scene",
        help=(
            "Process routes from the same scene together for production speed; "
            "use 'diverse' only for small multi-scene review pilots."
        ),
    )
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--worker_index", type=int, default=0)
    parser.add_argument("--initialize_only", action="store_true")
    parser.add_argument("--finalize_only", action="store_true")
    parser.add_argument("--verify_only", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted collection or verify an existing final output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unknown = sorted(set(args.dataset_names) - set(DATASET_SPECS))
    if unknown:
        raise ValueError(f"unsupported datasets: {unknown}")
    depth_choices = parse_depths(args.depth_choices)
    if args.output_dataset_name in {"", ".", ".."} or Path(
        args.output_dataset_name
    ).name != args.output_dataset_name:
        raise ValueError(
            f"invalid output_dataset_name: {args.output_dataset_name!r}"
        )
    if args.num_workers <= 0:
        raise ValueError(f"num_workers must be positive, got {args.num_workers}")
    if not 0 <= args.worker_index < args.num_workers:
        raise ValueError(
            f"worker_index must be in [0, {args.num_workers}), got "
            f"{args.worker_index}"
        )
    mode_count = sum(
        (args.initialize_only, args.finalize_only, args.verify_only)
    )
    if mode_count > 1:
        raise ValueError(
            "--initialize_only, --finalize_only and --verify_only are mutually exclusive"
        )
    for name in ("max_routes_to_try", "max_samples_per_dataset"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive when set, got {value}")
    if not math.isfinite(args.min_free_disk_gb) or args.min_free_disk_gb <= 0:
        raise ValueError(
            f"min_free_disk_gb must be finite and positive, got {args.min_free_disk_gb}"
        )
    output_root = args.output_root.resolve()
    input_root = args.input_root.resolve()
    if not input_root.is_dir():
        raise NotADirectoryError(input_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = collection_paths(output_root, args.output_dataset_name)
    route_inventory = {
        dataset_name: build_route_records(
            dataset_name,
            instruction_seed=args.seed,
            input_root=input_root,
        )[0]
        for dataset_name in args.dataset_names
    }
    scene_worker_assignment, worker_loads = build_scene_worker_assignment(
        route_inventory,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    assigned_route_inventory = build_assigned_route_inventory(
        route_inventory,
        dataset_names=args.dataset_names,
        scene_worker_assignment=scene_worker_assignment,
        num_workers=args.num_workers,
        seed=args.seed,
        route_order=args.route_order,
        max_routes_to_try=args.max_routes_to_try,
    )
    configuration = collection_configuration(
        args,
        input_root,
        route_inventory=route_inventory,
        scene_worker_assignment=scene_worker_assignment,
        worker_loads=worker_loads,
    )
    if args.initialize_only:
        initialize_collection(paths, configuration, resume=args.resume)
        return
    if args.finalize_only:
        finalize_collection(paths, configuration, assigned_route_inventory)
        return
    if args.verify_only:
        verify_final_collection(paths, configuration, assigned_route_inventory)
        return

    require_matching_configuration(
        paths["progress"] / "configuration.json",
        configuration,
    )
    manifest_path = worker_manifest_path(paths["progress"], args.worker_index)
    summary_path = worker_summary_path(paths["progress"], args.worker_index)
    route_ledger_path = worker_route_ledger_path(
        paths["progress"], args.worker_index
    )
    if (
        manifest_path.exists()
        or summary_path.exists()
        or route_ledger_path.exists()
    ) and not args.resume:
        raise FileExistsError(
            f"worker {args.worker_index} already has progress; pass --resume"
        )
    expected_routes = {
        physical_route_key(route): route
        for dataset_routes in assigned_route_inventory[args.worker_index].values()
        for route in dataset_routes
    }
    final_rows = (
        [
            row
            for row in read_jsonl(paths["manifest"])
            if row_route_key(row) in expected_routes
        ]
        if paths["manifest"].is_file()
        else []
    )
    worker_rows = (
        read_worker_jsonl_for_resume(manifest_path)
        if args.resume
        else []
    )
    rows_by_route: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in worker_rows:
        validate_manifest_row(row, paths["images"])
        key = row_route_key(row)
        if key not in expected_routes:
            raise ValueError(f"worker {args.worker_index} has unassigned row: {key}")
        previous = rows_by_route.get(key)
        if previous is not None and previous != row:
            raise ValueError(f"conflicting duplicate route row: {key}")
        rows_by_route[key] = row
    for row in final_rows:
        validate_manifest_row(row, paths["images"])
        key = row_route_key(row)
        if rows_by_route.get(key) != row:
            raise ValueError(
                f"existing final manifest is not reproducible from worker rows: {key}"
            )
    rows = list(rows_by_route.values())
    existing_route_keys = set(rows_by_route)
    route_outcomes = (
        load_route_outcomes(
            route_ledger_path,
            worker_index=args.worker_index,
            resume=True,
        )
        if route_ledger_path.exists()
        else {}
    )
    unexpected_outcomes = sorted(set(route_outcomes) - set(expected_routes))
    if unexpected_outcomes:
        raise ValueError(
            f"worker {args.worker_index} has unassigned outcomes: "
            f"{unexpected_outcomes[:10]}"
        )
    for key, outcome in route_outcomes.items():
        if outcome["status"] == "success" and key not in rows_by_route:
            raise ValueError(f"success outcome has no manifest row: {key}")
        if outcome["status"] == "terminal_failure" and key in rows_by_route:
            raise ValueError(f"failed route also has a manifest row: {key}")

    if summary_path.exists():
        summary_path.unlink()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        manifest_path.open("a", encoding="utf-8") as manifest_handle,
        route_ledger_path.open("a", encoding="utf-8") as route_ledger_handle,
    ):
        def persist_row(row: Dict[str, Any]) -> None:
            payload = (
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
            manifest_handle.write(payload)
            manifest_handle.flush()
            os.fsync(manifest_handle.fileno())

        def persist_outcome(outcome: Dict[str, Any]) -> None:
            payload = (
                json.dumps(outcome, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
            route_ledger_handle.write(payload)
            route_ledger_handle.flush()
            os.fsync(route_ledger_handle.fileno())

        for key, row in rows_by_route.items():
            existing_outcome = route_outcomes.get(key)
            if existing_outcome is None:
                repaired_outcome = make_route_outcome(
                    expected_routes[key],
                    worker_index=args.worker_index,
                    status="success",
                )
                persist_outcome(repaired_outcome)
                route_outcomes[key] = repaired_outcome

        for dataset_name in args.dataset_names:
            dataset_rows, failures = generate_dataset_samples(
                dataset_name=dataset_name,
                assigned_route_records=assigned_route_inventory[
                    args.worker_index
                ][dataset_name],
                depth_choices=depth_choices,
                image_dataset_root=paths["images"],
                gpu_id=args.gpu_id,
                image_width=args.image_width,
                image_height=args.image_height,
                seed=args.seed,
                max_samples_per_dataset=args.max_samples_per_dataset,
                max_mapping_error=args.max_mapping_error,
                worker_index=args.worker_index,
                existing_route_keys=existing_route_keys,
                route_outcomes=route_outcomes,
                row_sink=persist_row,
                outcome_sink=persist_outcome,
                resume=args.resume,
                min_free_disk_gb=args.min_free_disk_gb,
            )
            rows.extend(dataset_rows)
            del failures

    candidate_failure_counts: Dict[str, int] = defaultdict(int)
    for outcome in route_outcomes.values():
        for reason, count in outcome.get("candidate_failure_counts", {}).items():
            candidate_failure_counts[str(reason)] += int(count)
    success_keys = {
        key for key, outcome in route_outcomes.items() if outcome["status"] == "success"
    }
    failure_keys = {
        key
        for key, outcome in route_outcomes.items()
        if outcome["status"] == "terminal_failure"
    }
    if success_keys != existing_route_keys:
        raise ValueError("worker success ledger and manifest rows disagree")
    assigned_keys = set(expected_routes)
    unprocessed_keys = assigned_keys - success_keys - failure_keys
    route_accounting = {
        "assigned": len(assigned_keys),
        "success": len(success_keys),
        "terminal_failure": len(failure_keys),
        "unprocessed": len(unprocessed_keys),
    }
    summary = summarize(rows, dict(candidate_failure_counts))
    summary["complete"] = True
    summary["worker_index"] = args.worker_index
    summary["configuration_sha256"] = configuration_sha256(configuration)
    summary["route_accounting"] = route_accounting
    summary["manifest_rows"] = len(rows)
    summary["manifest_sha256"] = file_sha256(manifest_path)
    summary["route_ledger_rows"] = len(route_outcomes)
    summary["route_ledger_sha256"] = file_sha256(route_ledger_path)
    summary["configuration"] = configuration
    atomic_write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
