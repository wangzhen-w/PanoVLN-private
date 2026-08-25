"""Convert natural shortest paths to PanoVLN primitives and replay them."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Sequence

import numpy as np

from dataset_create.trajectory.decisions import (
    branch_anchor_map,
    observe_branches,
)
from dataset_create.trajectory.hm3d import set_agent_state, shortest_path, state_record, yaw_rotation_xyzw
from dataset_create.trajectory.regions import RegionLookup


ACTION_NAME_TO_ID = {
    "stop": 0,
    "move_forward": 1,
    "turn_left": 2,
    "turn_right": 3,
}
ACTION_ID_TO_NAME = {value: key for key, value in ACTION_NAME_TO_ID.items()}


def replay_episode_actions(
    simulator, episode: dict[str, Any]
) -> tuple[list[dict[str, list[float]]], int]:
    """Replay one serialized episode and return every reproducible agent state."""

    set_agent_state(
        simulator,
        episode["start_position"],
        episode["start_rotation_xyzw"],
    )
    states = [state_record(simulator.get_agent(0).get_state())]
    collision_count = 0
    for action_id in episode["action_ids"]:
        action_id = int(action_id)
        if action_id not in ACTION_ID_TO_NAME:
            raise ValueError(f"Unknown primitive action id: {action_id}")
        before = np.asarray(
            simulator.get_agent(0).get_state().position, dtype=float
        ).copy()
        if action_id != ACTION_NAME_TO_ID["stop"]:
            simulator.get_agent(0).act(ACTION_ID_TO_NAME[action_id])
        after_state = simulator.get_agent(0).get_state()
        after = np.asarray(after_state.position, dtype=float)
        if (
            action_id == ACTION_NAME_TO_ID["move_forward"]
            and np.linalg.norm(after - before) < 0.05
        ):
            collision_count += 1
        states.append(state_record(after_state))
    return states, collision_count


def _initial_rotation(shortest_path_points: Sequence[Sequence[float]]) -> list[float]:
    points = np.asarray(shortest_path_points, dtype=float)
    for point in points[1:]:
        direction = point - points[0]
        if np.linalg.norm(direction[[0, 2]]) > 0.1:
            return yaw_rotation_xyzw(direction)
    return yaw_rotation_xyzw([0.0, 0.0, -1.0])


def _compressed_labels(labels: Sequence[int]) -> list[int]:
    result = []
    for value in labels:
        value = int(value)
        if value < 0 or (result and result[-1] == value):
            continue
        result.append(value)
    return result


def _same_region_route(expected: Sequence[int], actual: Sequence[int]) -> bool:
    expected_values = list(map(int, expected))
    actual_values = _compressed_labels(actual)
    if actual_values == expected_values:
        return True
    # Action states are 0.25 m apart and can skip a very thin connection basin.
    expected_without_singletons = [
        value
        for index, value in enumerate(expected_values)
        if index in (0, len(expected_values) - 1)
        or value in actual_values
    ]
    return actual_values == expected_without_singletons


def replay_route(
    simulator,
    route: dict[str, Any],
    scene_graph: dict[str, Any],
    arrays: dict[str, np.ndarray],
    decision_stats: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    import habitat_sim

    start = np.asarray(route["start_position"], dtype=np.float32)
    goal = np.asarray(route["goal_position"], dtype=np.float32)
    rotation = _initial_rotation(route["shortest_path"])
    set_agent_state(simulator, start, rotation)
    follower = habitat_sim.GreedyGeodesicFollower(
        simulator.pathfinder,
        simulator.get_agent(0),
        goal_radius=float(config["agent"]["goal_radius"]),
        stop_key="stop",
        forward_key="move_forward",
        left_key="turn_left",
        right_key="turn_right",
        fix_thrashing=True,
    )
    try:
        planned = follower.find_path(goal)
    except habitat_sim.errors.GreedyFollowerError as error:
        return None, {"status": "planner_error", "detail": str(error)}
    actions = ["stop" if action is None else str(action) for action in planned]
    if not actions or actions[-1] != "stop":
        actions.append("stop")
    if len(actions) > int(config["route"]["max_actions"]):
        return None, {"status": "too_many_actions", "action_count": len(actions)}
    if any(action not in ACTION_NAME_TO_ID for action in actions):
        return None, {"status": "unknown_action", "actions": sorted(set(actions))}

    # Planning is documented to preserve the state, but explicitly restore it so the
    # serialized start pose is the single source of truth across Habitat versions.
    set_agent_state(simulator, start, rotation)
    states = [state_record(simulator.get_agent(0).get_state())]
    collision_count = 0
    for action in actions:
        before = np.asarray(simulator.get_agent(0).get_state().position, dtype=float).copy()
        # STOP is a task-level terminal action, not a duplicate zero-distance
        # move_forward control (the GreedyGeodesicFollower requires unique controls).
        if action != "stop":
            simulator.get_agent(0).act(action)
        after_state = simulator.get_agent(0).get_state()
        after = np.asarray(after_state.position, dtype=float).copy()
        if action == "move_forward" and np.linalg.norm(after - before) < 0.05:
            collision_count += 1
        states.append(state_record(after_state))

    final_position = np.asarray(states[-1]["position"], dtype=np.float32)
    remaining = shortest_path(simulator.pathfinder, final_position, goal)
    remaining_distance = math.inf if remaining is None else float(remaining["distance"])
    success = remaining_distance <= float(config["agent"]["goal_radius"]) + 1e-4
    if not success:
        return None, {
            "status": "goal_not_reached",
            "remaining_geodesic_distance": remaining_distance,
            "action_count": len(actions),
        }

    lookup = RegionLookup(arrays)
    replay_positions = np.asarray([state["position"] for state in states], dtype=float)
    replay_labels = lookup.regions_for_points(replay_positions)
    if not _same_region_route(route["region_sequence"], replay_labels):
        return None, {
            "status": "region_route_mismatch",
            "expected": route["region_sequence"],
            "actual": _compressed_labels(replay_labels),
        }

    branch_anchors = branch_anchor_map(decision_stats)
    replay_events = []
    event_rejections = Counter()
    for event in route["decision_events"]:
        canonical = np.asarray(event["canonical_position"], dtype=float)
        distances = np.linalg.norm(replay_positions - canonical, axis=1)
        state_index = int(np.argmin(distances))
        if float(distances[state_index]) > float(
            config["decision"]["max_canonical_route_offset"]
        ):
            event_rejections["canonical_offset"] += 1
            continue
        state = states[state_index]
        connection_ids = [
            event["incoming_connection_id"],
            event["selected_connection_id"],
            *event["alternative_connection_ids"],
        ]
        verification, _ = observe_branches(
            simulator,
            scene_graph,
            int(event["region_id"]),
            connection_ids,
            state["position"],
            state["rotation_xyzw"],
            config,
            branch_anchors,
        )
        selected = verification[event["selected_connection_id"]]
        visible_alternatives = [
            connection_id
            for connection_id in event["alternative_connection_ids"]
            if verification[connection_id]["visible"]
        ]
        if not selected["visible"] or not visible_alternatives:
            event_rejections["replay_panorama_visibility"] += 1
            continue
        alternative_regions = dict(
            zip(
                event["alternative_connection_ids"],
                event["alternative_region_ids"],
            )
        )
        retained_connection_ids = [
            event["incoming_connection_id"],
            event["selected_connection_id"],
            *visible_alternatives,
        ]
        replay_events.append(
            {
                **event,
                "state_index": state_index,
                "action_index": min(state_index, len(actions) - 1),
                "position": state["position"],
                "rotation_xyzw": state["rotation_xyzw"],
                "alternative_connection_ids": visible_alternatives,
                "alternative_region_ids": [
                    alternative_regions[connection_id]
                    for connection_id in visible_alternatives
                ],
                "branch_anchor_positions": {
                    connection_id: branch_anchors[
                        (int(event["region_id"]), connection_id)
                    ].astype(float).tolist()
                    for connection_id in retained_connection_ids
                },
                "branch_observations": {
                    connection_id: verification[connection_id]
                    for connection_id in retained_connection_ids
                },
            }
        )
    if not replay_events:
        return None, {
            "status": "no_replay_verified_decision",
            "event_rejections": dict(event_rejections),
        }

    replay_record = {
        "start_rotation_xyzw": rotation,
        "actions": actions,
        "action_ids": [ACTION_NAME_TO_ID[action] for action in actions],
        "states": states,
        "final_geodesic_error": remaining_distance,
        "collision_count": collision_count,
        "success": True,
    }
    result = dict(route)
    result["decision_events"] = replay_events
    result["replay"] = replay_record
    result["decision_count"] = len(replay_events)
    return result, {
        "status": "success",
        "action_count": len(actions),
        "decision_count": len(replay_events),
        "event_rejections": dict(event_rejections),
    }
