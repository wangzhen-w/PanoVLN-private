"""Natural shortest-path enumeration, region-structure grouping, and collection."""

from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from typing import Any, Sequence

import numpy as np

from dataset_create.trajectory.decisions import relation_index
from dataset_create.trajectory.hm3d import interpolate_polyline, shortest_path
from dataset_create.trajectory.io_utils import stable_id
from dataset_create.trajectory.quality import (
    event_quality,
    route_family_signature,
    trajectory_id_for_family,
    trajectory_quality,
    trajectory_quality_rank,
)
from dataset_create.trajectory.regions import RegionLookup
from dataset_create.trajectory.replay import replay_route


def _compress_runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    runs = []
    start = 0
    for index in range(1, len(labels) + 1):
        if index == len(labels) or labels[index] != labels[start]:
            if int(labels[start]) >= 0:
                runs.append((int(labels[start]), start, index))
            start = index
    return runs


def _choose_connection(
    connections_by_pair: dict[tuple[int, int], list[dict[str, Any]]],
    first_region: int,
    second_region: int,
    crossing_position: np.ndarray,
) -> dict[str, Any] | None:
    options = connections_by_pair.get(tuple(sorted((first_region, second_region))), [])
    if not options:
        return None
    return min(
        options,
        key=lambda connection: float(
            np.linalg.norm(np.asarray(connection["position"]) - crossing_position)
        ),
    )


def map_path_to_structure(
    dense_points: np.ndarray,
    lookup: RegionLookup,
    scene_graph: dict[str, Any],
) -> dict[str, Any] | None:
    labels = lookup.regions_for_points(dense_points)
    runs = _compress_runs(labels)
    if len(runs) < 2:
        return None
    connections_by_pair: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for connection in scene_graph["connections"]:
        connections_by_pair[tuple(map(int, connection["regions"]))].append(connection)
    region_sequence = [run[0] for run in runs]
    connection_sequence = []
    transitions = []
    for first, second in zip(runs[:-1], runs[1:]):
        crossing = (dense_points[first[2] - 1] + dense_points[second[1]]) * 0.5
        connection = _choose_connection(
            connections_by_pair, first[0], second[0], crossing
        )
        if connection is None:
            return None
        connection_sequence.append(connection["connection_id"])
        transitions.append(
            {
                "from_region_id": first[0],
                "to_region_id": second[0],
                "connection_id": connection["connection_id"],
                "crossing_position": crossing.astype(float).tolist(),
            }
        )
    return {
        "region_sequence": region_sequence,
        "connection_sequence": connection_sequence,
        "region_runs": runs,
        "transitions": transitions,
    }


def _decision_events(
    structure: dict[str, Any], relations: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    index = relation_index(relations)
    events = []
    for sequence_index in range(1, len(structure["region_sequence"]) - 1):
        incoming = structure["connection_sequence"][sequence_index - 1]
        selected = structure["connection_sequence"][sequence_index]
        relation = index.get(
            (int(structure["region_sequence"][sequence_index]), incoming, selected)
        )
        if relation is None:
            continue
        events.append(
            {
                "relation_id": relation["relation_id"],
                "sequence_index": sequence_index,
                "region_id": relation["region_id"],
                "incoming_connection_id": incoming,
                "incoming_region_id": relation["incoming_region_id"],
                "selected_connection_id": selected,
                "selected_region_id": relation["selected_region_id"],
                "alternative_connection_ids": relation["alternative_connection_ids"],
                "alternative_region_ids": relation["alternative_region_ids"],
                "canonical_position": relation["canonical_state"]["position"],
                "canonical_rotation_xyzw": relation["canonical_state"]["rotation_xyzw"],
            }
        )
    return events


def _endpoint_pairs(
    start_positions: Sequence[Sequence[float]],
    goal_positions: Sequence[Sequence[float]],
    maximum: int,
):
    pairs = list(itertools.product(enumerate(start_positions), enumerate(goal_positions)))
    pairs.sort(key=lambda pair: (pair[0][0] + pair[1][0], pair[0][0], pair[1][0]))
    return pairs[:maximum]


def enumerate_route_candidates(
    pathfinder,
    scene_graph: dict[str, Any],
    arrays: dict[str, np.ndarray],
    relations: Sequence[dict[str, Any]],
    scene_id: str,
    config: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    lookup = RegionLookup(arrays)
    settings = config["route"]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counters = Counter()
    regions = scene_graph["regions"]
    region_pairs = [
        (start_region, goal_region)
        for start_region in regions
        for goal_region in regions
        if start_region["region_id"] != goal_region["region_id"]
        and start_region["island_id"] == goal_region["island_id"]
    ]
    counters["available_endpoint_region_pairs"] = len(region_pairs)
    region_pairs.sort(
        key=lambda pair: stable_id(
            "endpoint_pair",
            scene_id,
            pair[0]["region_id"],
            pair[1]["region_id"],
        )
    )
    counters["sampled_endpoint_region_pairs"] = len(region_pairs)
    for start_region, goal_region in region_pairs:
        for (start_index, start), (goal_index, goal) in _endpoint_pairs(
            start_region["representative_positions"],
            goal_region["representative_positions"],
            int(settings["max_endpoint_combinations"]),
        ):
            counters["endpoint_pairs"] += 1
            path = shortest_path(pathfinder, start, goal)
            if path is None:
                counters["no_path"] += 1
                continue
            if not float(settings["min_geodesic_distance"]) <= path["distance"] <= float(
                settings["max_geodesic_distance"]
            ):
                counters["distance_gate"] += 1
                continue
            dense = interpolate_polyline(
                path["points"], float(settings["path_sample_spacing"])
            )
            structure = map_path_to_structure(dense, lookup, scene_graph)
            if structure is None:
                counters["structure_mapping"] += 1
                continue
            if structure["region_sequence"][0] != start_region["region_id"] or structure[
                "region_sequence"
            ][-1] != goal_region["region_id"]:
                counters["endpoint_region_mismatch"] += 1
                continue
            events = _decision_events(structure, relations)
            if not events:
                counters["no_decision"] += 1
                continue
            signature_payload = {
                "scene_id": scene_id,
                "start_region_id": structure["region_sequence"][0],
                "region_sequence": structure["region_sequence"],
                "connection_sequence": structure["connection_sequence"],
                "decision_relation_ids": [event["relation_id"] for event in events],
                "goal_region_id": structure["region_sequence"][-1],
            }
            structure_id = stable_id("route", signature_payload)
            canonical_offsets = [
                float(
                    np.min(
                        np.linalg.norm(
                            dense - np.asarray(event["canonical_position"]), axis=1
                        )
                    )
                )
                for event in events
            ]
            clearance_score = float(
                pathfinder.distance_to_closest_obstacle(start, 3.0)
                + pathfinder.distance_to_closest_obstacle(goal, 3.0)
            )
            candidate_id = stable_id(
                "candidate", structure_id, start_index, goal_index, np.round(start, 3), np.round(goal, 3)
            )
            groups[structure_id].append(
                {
                    "candidate_id": candidate_id,
                    "route_structure_id": structure_id,
                    "route_structure": signature_payload,
                    "start_position": np.asarray(start, dtype=float).tolist(),
                    "goal_position": np.asarray(goal, dtype=float).tolist(),
                    "geodesic_distance": path["distance"],
                    "shortest_path": path["points"].astype(float).tolist(),
                    "region_sequence": structure["region_sequence"],
                    "connection_sequence": structure["connection_sequence"],
                    "decision_events": events,
                    "selection_score": {
                        "canonical_offset_sum": sum(canonical_offsets),
                        "canonical_offset_max": max(canonical_offsets),
                        "endpoint_clearance_sum": clearance_score,
                    },
                }
            )
            counters["accepted_candidate"] += 1
    for candidates in groups.values():
        candidates.sort(
            key=lambda candidate: (
                -len(candidate["decision_events"]),
                candidate["selection_score"]["canonical_offset_sum"],
                -candidate["selection_score"]["endpoint_clearance_sum"],
                candidate["candidate_id"],
            )
        )
    return dict(groups), {
        **dict(counters),
        "region_level_route_groups": len(groups),
        "coordinate_variants_per_group": {
            "min": min((len(values) for values in groups.values()), default=0),
            "max": max((len(values) for values in groups.values()), default=0),
            "mean": (
                sum(map(len, groups.values())) / len(groups) if groups else 0.0
            ),
        },
    }


def _pre_replay_quality(
    candidate: dict[str, Any], relations_by_id: dict[str, dict[str, Any]], config: dict[str, Any]
) -> tuple[tuple[float, ...], list[str]]:
    event_metrics = []
    for event in candidate["decision_events"]:
        relation = relations_by_id[event["relation_id"]]
        canonical_event = {
            **event,
            "position": event["canonical_position"],
            "branch_observations": relation["branch_observations"],
        }
        metrics = event_quality(canonical_event)
        event_metrics.append(metrics)
    minimum_separation = min(
        (metrics["branch_separation_degrees"] for metrics in event_metrics), default=0.0
    )
    minimum_depth = min(
        (metrics["depth_margin_m"] for metrics in event_metrics), default=-np.inf
    )
    maximum_offset = float(candidate["selection_score"]["canonical_offset_max"])
    settings = config["quality"]
    reasons = []
    if minimum_depth < float(settings["min_depth_margin_m"]):
        reasons.append("weak_depth_margin")
    if maximum_offset > float(settings["max_decision_state_offset_m"]):
        reasons.append("decision_state_offset")
    # Lexicographic ranking keeps the family rule explainable: prefer larger ERP
    # geometry margin, then better canonical alignment and endpoint clearance.
    return (
        float(minimum_depth),
        -maximum_offset,
        float(candidate["selection_score"]["endpoint_clearance_sum"]),
        float(minimum_separation),
    ), reasons


def collect_routes(
    simulator,
    scene_graph: dict[str, Any],
    arrays: dict[str, np.ndarray],
    relations: Sequence[dict[str, Any]],
    decision_stats: dict[str, Any],
    scene_id: str,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    groups, enumeration_stats = enumerate_route_candidates(
        simulator.pathfinder,
        scene_graph,
        arrays,
        relations,
        scene_id,
        config,
    )
    trajectories = []
    rejection_counts = Counter()
    candidate_quality_rejections = Counter()
    replay_quality_rejections = Counter()
    dedup_examples = {}
    relations_by_id = {relation["relation_id"]: relation for relation in relations}
    group_representatives = [candidates[0] for candidates in groups.values()]
    eligible_representatives = []
    for candidate in group_representatives:
        rank, reasons = _pre_replay_quality(
            candidate, relations_by_id, config
        )
        if reasons:
            candidate_quality_rejections.update(reasons)
            continue
        candidate["pre_replay_rank"] = rank
        eligible_representatives.append(candidate)

    route_families: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for candidate in eligible_representatives:
        route_families[route_family_signature(candidate, config)].append(candidate)
    for candidates in route_families.values():
        candidates.sort(
            key=lambda item: (
                *(-value for value in item["pre_replay_rank"]),
                item["route_structure_id"],
            )
        )

    replay_attempted_groups = 0
    replay_verified_count = 0
    replay_quality_eligible_count = 0
    failed_family_count = 0
    replay_family_remapping_count = 0
    actual_family_representatives: dict[
        tuple[Any, ...], dict[str, Any]
    ] = {}

    ordered_families = sorted(
        route_families.items(),
        key=lambda item: stable_id("route_family", scene_id, item[0]),
    )
    for family_signature, representatives in ordered_families:
        family_selected = None
        for representative in representatives:
            structure_id = representative["route_structure_id"]
            replay_attempted_groups += 1
            candidates = groups[structure_id]
            attempted = 0
            for candidate in candidates:
                attempted += 1
                replayed, replay_stats = replay_route(
                    simulator,
                    candidate,
                    scene_graph,
                    arrays,
                    decision_stats,
                    config,
                )
                if replayed is None:
                    rejection_counts[replay_stats["status"]] += 1
                    continue
                replay_verified_count += 1
                replayed["dense_shortest_path"] = interpolate_polyline(
                    replayed["shortest_path"],
                    float(config["route"]["path_sample_spacing"]),
                ).astype(float).tolist()
                replayed["scene_id"] = scene_id
                actual_payload = {
                    "scene_id": scene_id,
                    "regions": replayed["region_sequence"],
                    "connections": replayed["connection_sequence"],
                    "decisions": [
                        event["relation_id"] for event in replayed["decision_events"]
                    ],
                }
                actual_id = stable_id("route", actual_payload)
                replayed["route_structure_id"] = actual_id
                replayed["route_structure"]["decision_relation_ids"] = actual_payload[
                    "decisions"
                ]
                quality, reasons = trajectory_quality(replayed, config)
                if reasons:
                    replay_quality_rejections.update(reasons)
                    continue
                replayed["quality"] = quality
                replay_quality_eligible_count += 1
                actual_family = route_family_signature(replayed, config)
                replayed["trajectory_id"] = trajectory_id_for_family(
                    scene_id, actual_family
                )
                previous = actual_family_representatives.get(actual_family)
                if (
                    previous is None
                    or trajectory_quality_rank(replayed)
                    > trajectory_quality_rank(previous)
                    or (
                        trajectory_quality_rank(replayed)
                        == trajectory_quality_rank(previous)
                        and replayed["trajectory_id"] < previous["trajectory_id"]
                    )
                ):
                    actual_family_representatives[actual_family] = replayed
                if actual_family != family_signature:
                    replay_family_remapping_count += 1
                    continue
                replayed["deduplication"] = {
                    "coordinate_candidate_count": len(candidates),
                    "attempted_before_success": attempted,
                    "representative_candidate_id": replayed["candidate_id"],
                    "grouping_key": "directed_decision_core_and_route_context",
                }
                family_selected = replayed
                break
            if family_selected is not None:
                if len(candidates) > 1:
                    dedup_examples[structure_id] = {
                        "selected_candidate_id": family_selected["candidate_id"],
                        "candidate_paths": [
                            {
                                "candidate_id": candidate["candidate_id"],
                                "start_position": candidate["start_position"],
                                "goal_position": candidate["goal_position"],
                                "shortest_path": candidate["shortest_path"],
                            }
                            for candidate in candidates[:5]
                        ],
                    }
                break
        if family_selected is None:
            failed_family_count += 1

    # Final families are defined only by replay-verified decision annotations.
    # This sort is deterministic and does not impose a count target.
    trajectories = sorted(
        actual_family_representatives.values(),
        key=lambda item: item["trajectory_id"],
    )
    decision_histogram = Counter(trajectory["decision_count"] for trajectory in trajectories)
    stats = {
        **enumeration_stats,
        "pre_replay_quality_eligible_group_count": len(eligible_representatives),
        "pre_replay_quality_rejections": dict(sorted(candidate_quality_rejections.items())),
        "family_replay_attempted_group_count": replay_attempted_groups,
        "replay_verified_trajectory_count": replay_verified_count,
        "route_family_count": len(route_families),
        "pre_replay_family_redundancy_rejections": (
            len(eligible_representatives) - len(route_families)
        ),
        "route_family_replay_failure_count": failed_family_count,
        "replay_family_remapping_count": replay_family_remapping_count,
        "replay_quality_rejections": dict(sorted(replay_quality_rejections.items())),
        "quality_eligible_trajectory_count": replay_quality_eligible_count,
        "replay_family_redundancy_rejections": (
            replay_quality_eligible_count - len(trajectories)
        ),
        "final_trajectory_count": len(trajectories),
        "decision_events_per_trajectory": {
            str(key): value for key, value in sorted(decision_histogram.items())
        },
        "replay_success_rate": 1.0 if trajectories else None,
        "route_structure_retention_rate": (
            len(trajectories) / max(1, len(groups))
        ),
        "replay_rejections": dict(sorted(rejection_counts.items())),
    }
    return trajectories, stats, dedup_examples
