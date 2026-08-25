"""Route-family helpers, replay quality checks, and compact records."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from dataset_create.trajectory.io_utils import stable_id


def _wrapped_separation(first: float, second: float) -> float:
    return abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)


def event_quality(event: dict[str, Any]) -> dict[str, float]:
    observations = event["branch_observations"]
    selected = observations[event["selected_connection_id"]]
    alternatives = [
        observations[connection_id]
        for connection_id in event["alternative_connection_ids"]
        if connection_id in observations
    ]
    if not alternatives:
        return {
            "branch_separation_degrees": 0.0,
            "depth_margin_m": -math.inf,
            "canonical_offset_m": math.inf,
        }
    pairs = [
        (
            _wrapped_separation(
                selected["bearing_degrees"], alternative["bearing_degrees"]
            ),
            min(float(selected["depth_margin"]), float(alternative["depth_margin"])),
        )
        for alternative in alternatives
    ]
    # Angular separation is descriptive, not a validity threshold. Prefer the
    # topologically distinct alternative with the strongest geometry margin.
    separation, depth_margin = max(
        pairs,
        key=lambda value: (
            value[1],
            value[0],
        ),
    )
    canonical_offset = float(
        np.linalg.norm(
            np.asarray(event["position"], dtype=float)
            - np.asarray(event["canonical_position"], dtype=float)
        )
    )
    return {
        "branch_separation_degrees": float(separation),
        "depth_margin_m": float(depth_margin),
        "canonical_offset_m": canonical_offset,
    }


def trajectory_quality(
    trajectory: dict[str, Any], config: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    settings = config["quality"]
    qualities = [event_quality(event) for event in trajectory["decision_events"]]
    minimum_separation = min(
        (value["branch_separation_degrees"] for value in qualities), default=0.0
    )
    minimum_depth_margin = min(
        (value["depth_margin_m"] for value in qualities), default=-math.inf
    )
    maximum_offset = max(
        (value["canonical_offset_m"] for value in qualities), default=math.inf
    )
    collision_count = int(trajectory["replay"].get("collision_count", 0))
    reasons = []
    if minimum_depth_margin < float(settings["min_depth_margin_m"]):
        reasons.append("weak_depth_margin")
    if maximum_offset > float(settings["max_decision_state_offset_m"]):
        reasons.append("decision_state_offset")
    if collision_count > int(settings["max_collisions"]):
        reasons.append("collision")

    return (
        {
            "minimum_branch_separation_degrees": minimum_separation,
            "minimum_depth_margin_m": minimum_depth_margin,
            "maximum_decision_state_offset_m": maximum_offset,
            "event_quality": qualities,
        },
        reasons,
    )


def _bucket(value: float, boundaries: Sequence[float]) -> int:
    return sum(float(value) >= float(boundary) for boundary in boundaries)


def route_family_signature(
    route: dict[str, Any], config: dict[str, Any]
) -> tuple[Any, ...]:
    """Directed decision structure plus coarse, interpretable route context."""

    regions = tuple(map(int, route["region_sequence"]))
    events = route["decision_events"]
    indices = [
        int(event.get("sequence_index", event.get("region_sequence_index", -1)))
        for event in events
    ]
    if not events or any(index < 0 or index >= len(regions) for index in indices):
        raise ValueError("A route family requires valid decision sequence indices")
    first, last = min(indices), max(indices)
    settings = config["route"]["family"]
    length = float(
        route.get("geodesic_distance", route.get("metrics", {}).get("length_m", 0.0))
    )
    return (
        tuple(event["relation_id"] for event in events),
        regions[first : last + 1],
        _bucket(first, settings["context_hop_boundaries"]),
        _bucket(len(regions) - 1 - last, settings["context_hop_boundaries"]),
        _bucket(length, settings["length_boundaries_m"]),
    )


def trajectory_id_for_family(
    scene_id: str, family_signature: tuple[Any, ...]
) -> str:
    """Return the stable identity of one directed, replay-verified family."""

    return stable_id("trajectory", scene_id, family_signature)


def trajectory_quality_rank(trajectory: dict[str, Any]) -> tuple[float, ...]:
    """Lexicographic representative ranking without arbitrary score weights."""

    quality = trajectory["quality"]
    return (
        float(quality["minimum_depth_margin_m"]),
        -float(quality["maximum_decision_state_offset_m"]),
        float(trajectory.get("selection_score", {}).get("endpoint_clearance_sum", 0.0)),
        -float(len(trajectory.get("replay", {}).get("action_ids", []))),
    )


def _round_vector(value: Sequence[float], digits: int = 7) -> list[float]:
    return [round(float(component), digits) for component in value]


def compact_trajectory(trajectory: dict[str, Any]) -> dict[str, Any]:
    compact_events = []
    for event in trajectory["decision_events"]:
        anchors = event["branch_anchor_positions"]

        def branch(connection_id: str, region_id: int) -> dict[str, Any]:
            return {
                "connection_id": connection_id,
                "region_id": int(region_id),
                "anchor_position": _round_vector(anchors[connection_id]),
            }

        compact_events.append(
            {
                "relation_id": event["relation_id"],
                "region_id": int(event["region_id"]),
                "region_sequence_index": int(event["sequence_index"]),
                "action_index": int(event["action_index"]),
                "position": _round_vector(event["position"]),
                "rotation_xyzw": _round_vector(event["rotation_xyzw"], 9),
                "incoming": branch(
                    event["incoming_connection_id"], event["incoming_region_id"]
                ),
                "selected": branch(
                    event["selected_connection_id"], event["selected_region_id"]
                ),
                "alternatives": [
                    branch(connection_id, region_id)
                    for connection_id, region_id in zip(
                        event["alternative_connection_ids"],
                        event["alternative_region_ids"],
                    )
                ],
            }
        )
    replay = trajectory["replay"]
    decision_region_count = len(
        {int(event["region_id"]) for event in trajectory["decision_events"]}
    )
    return {
        "trajectory_id": trajectory["trajectory_id"],
        "scene_id": trajectory["scene_id"],
        "start_position": _round_vector(trajectory["start_position"]),
        "start_rotation_xyzw": _round_vector(replay["start_rotation_xyzw"], 9),
        "goal_position": _round_vector(trajectory["goal_position"]),
        "final_position": _round_vector(replay["states"][-1]["position"]),
        "final_rotation_xyzw": _round_vector(
            replay["states"][-1]["rotation_xyzw"], 9
        ),
        "shortest_path": [
            _round_vector(point) for point in trajectory["shortest_path"]
        ],
        "region_sequence": list(map(int, trajectory["region_sequence"])),
        "connection_sequence": list(trajectory["connection_sequence"]),
        "decision_events": compact_events,
        "action_ids": list(map(int, replay["action_ids"])),
        "metrics": {
            "length_m": round(float(trajectory["geodesic_distance"]), 4),
            "region_count": len(trajectory["region_sequence"]),
            "decision_region_count": decision_region_count,
            "decision_event_count": len(compact_events),
        },
    }

