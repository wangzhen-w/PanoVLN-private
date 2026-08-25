"""Mine region-graph choices and verify them in a ground-truth ERP observation."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Sequence

import networkx as nx
import numpy as np

from dataset_create.trajectory.hm3d import (
    interpolate_polyline,
    point_along_polyline,
    set_agent_state,
    shortest_path,
    yaw_rotation_xyzw,
)
from dataset_create.trajectory.io_utils import stable_id
from dataset_create.trajectory.regions import RegionLookup, build_region_graph


def _wrap_degrees(angle: float) -> float:
    return (float(angle) + 180.0) % 360.0 - 180.0


def heading_vector(rotation_xyzw: Sequence[float]) -> np.ndarray:
    import habitat_sim

    quaternion = habitat_sim.utils.common.quat_from_coeffs(
        np.asarray(rotation_xyzw, dtype=np.float64)
    )
    return np.asarray(
        habitat_sim.utils.common.quat_rotate_vector(
            quaternion, np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
        ),
        dtype=np.float64,
    )


def relative_bearing_degrees(
    origin: Sequence[float], target: Sequence[float], rotation_xyzw: Sequence[float]
) -> float:
    vector = np.asarray(target, dtype=float) - np.asarray(origin, dtype=float)
    heading = heading_vector(rotation_xyzw)
    target_bearing = math.degrees(math.atan2(float(vector[0]), -float(vector[2])))
    heading_bearing = math.degrees(math.atan2(float(heading[0]), -float(heading[2])))
    return _wrap_degrees(target_bearing - heading_bearing)


def erp_pixel_for_target(
    origin: Sequence[float],
    target: Sequence[float],
    rotation_xyzw: Sequence[float],
    width: int,
    height: int,
) -> tuple[int, int, float, float]:
    vector = np.asarray(target, dtype=float) - np.asarray(origin, dtype=float)
    distance = float(np.linalg.norm(vector))
    bearing = relative_bearing_degrees(origin, target, rotation_xyzw)
    horizontal = (0.5 + bearing / 360.0) % 1.0
    horizontal_distance = float(np.linalg.norm(vector[[0, 2]]))
    elevation = math.degrees(math.atan2(float(vector[1]), max(horizontal_distance, 1e-8)))
    vertical = float(np.clip(0.5 - elevation / 180.0, 0.0, 1.0 - 1e-9))
    column = int(round(horizontal * width)) % width
    row = int(np.clip(round(vertical * height), 0, height - 1))
    return column, row, distance, bearing


def depth_visibility(
    depth: np.ndarray,
    origin: Sequence[float],
    target: Sequence[float],
    rotation_xyzw: Sequence[float],
    tolerance: float,
    patch_radius: int,
) -> dict[str, Any]:
    height, width = depth.shape[:2]
    column, row, target_distance, bearing = erp_pixel_for_target(
        origin, target, rotation_xyzw, width, height
    )
    rows = np.arange(max(0, row - patch_radius), min(height, row + patch_radius + 1))
    columns = np.asarray(
        [(column + offset) % width for offset in range(-patch_radius, patch_radius + 1)]
    )
    patch = depth[np.ix_(rows, columns)].astype(float)
    valid = patch[np.isfinite(patch) & (patch > 1e-4)]
    observed_depth = float(np.max(valid)) if len(valid) else 0.0
    return {
        "visible": bool(observed_depth > 0.0 and target_distance <= observed_depth + tolerance),
        "pixel": [column, row],
        "bearing_degrees": bearing,
        "target_distance": target_distance,
        "observed_depth": observed_depth,
        "depth_margin": observed_depth - target_distance,
    }


def _branch_separation(first: float, second: float) -> float:
    return abs(_wrap_degrees(first - second))


def _connection_maps(scene_graph: dict[str, Any]):
    connections = {
        connection["connection_id"]: connection for connection in scene_graph["connections"]
    }
    incident: dict[int, list[str]] = defaultdict(list)
    for connection in scene_graph["connections"]:
        for region_id in connection["regions"]:
            incident[int(region_id)].append(connection["connection_id"])
    for values in incident.values():
        values.sort()
    return connections, incident


def _other_region(connection: dict[str, Any], region_id: int) -> int:
    first, second = map(int, connection["regions"])
    if first == region_id:
        return second
    if second == region_id:
        return first
    raise ValueError(f"Connection {connection['connection_id']} is not incident to {region_id}")


def _connections_by_neighbor(
    connections: dict[str, dict[str, Any]],
    region_id: int,
    connection_ids: Sequence[str],
) -> dict[int, list[str]]:
    grouped: dict[int, list[str]] = defaultdict(list)
    for connection_id in connection_ids:
        grouped[_other_region(connections[connection_id], region_id)].append(
            connection_id
        )
    return dict(grouped)


def _branch_anchor(
    pathfinder,
    scene_graph: dict[str, Any],
    connection: dict[str, Any],
    from_region: int,
    distance: float,
) -> np.ndarray:
    neighbor = _other_region(connection, from_region)
    start = np.asarray(connection["side_positions"][str(neighbor)], dtype=np.float32)
    representatives = scene_graph["regions"][neighbor]["representative_positions"]
    ordered = sorted(
        representatives,
        key=lambda position: -float(np.linalg.norm(np.asarray(position) - start)),
    )
    for representative in ordered:
        path = shortest_path(pathfinder, start, representative)
        if path is not None and path["distance"] > 0.1:
            return point_along_polyline(path["points"], distance)
    return start


def _rejoin_distance(
    graph: nx.MultiGraph,
    decision_region: int,
    first_neighbor: int,
    second_neighbor: int,
) -> float:
    if first_neighbor == second_neighbor:
        return 0.0
    reduced = graph.copy()
    reduced.remove_node(decision_region)
    try:
        return float(
            nx.shortest_path_length(
                reduced, first_neighbor, second_neighbor, weight="weight"
            )
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return math.inf


def _geodesically_visible(pathfinder, origin: np.ndarray, target: np.ndarray) -> bool:
    path = shortest_path(pathfinder, origin, target)
    if path is None:
        return False
    euclidean = float(np.linalg.norm(target - origin))
    return path["distance"] <= max(euclidean * 1.25, euclidean + 0.35)


def _candidate_positions(
    pathfinder,
    lookup: RegionLookup,
    scene_graph: dict[str, Any],
    connections: dict[str, dict[str, Any]],
    region_id: int,
    incoming_id: str,
    selected_id: str,
    config: dict[str, Any],
) -> list[np.ndarray]:
    settings = config["decision"]
    incoming = connections[incoming_id]
    selected = connections[selected_id]
    start = np.asarray(incoming["side_positions"][str(region_id)], dtype=np.float32)
    end = np.asarray(selected["side_positions"][str(region_id)], dtype=np.float32)
    path = shortest_path(pathfinder, start, end)
    points = [] if path is None else list(
        interpolate_polyline(path["points"], float(settings["candidate_spacing"]))
    )
    points.extend(
        np.asarray(position, dtype=np.float32)
        for position in scene_graph["regions"][region_id]["representative_positions"]
    )
    incoming_position = np.asarray(incoming["position"], dtype=float)
    filtered = []
    seen = set()
    for point in points:
        distance = float(np.linalg.norm(np.asarray(point) - incoming_position))
        key = tuple(np.rint(np.asarray(point) / 0.1).astype(int).tolist())
        if key in seen or lookup.region_for_point(point) != region_id:
            continue
        if not float(settings["min_arrival_distance"]) <= distance <= float(
            settings["max_arrival_distance"]
        ):
            continue
        seen.add(key)
        filtered.append(np.asarray(point, dtype=np.float32))
    filtered.sort(
        key=lambda point: (
            abs(float(np.linalg.norm(point - incoming_position)) - 1.2),
            tuple(point.tolist()),
        )
    )
    return filtered[: int(settings["max_candidates_per_relation"])]


def observe_branches(
    simulator,
    scene_graph: dict[str, Any],
    region_id: int,
    connection_ids: Sequence[str],
    position: Sequence[float],
    rotation_xyzw: Sequence[float],
    config: dict[str, Any],
    branch_anchors: dict[tuple[int, str], np.ndarray],
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    connections = {
        connection["connection_id"]: connection for connection in scene_graph["connections"]
    }
    set_agent_state(simulator, position, rotation_xyzw)
    observations = simulator.get_sensor_observations()
    depth = np.asarray(observations["erp_depth"], dtype=np.float32)
    origin = np.asarray(position, dtype=float) + np.asarray(
        [0.0, float(config["agent"]["sensor_height"]), 0.0]
    )
    results: dict[str, dict[str, Any]] = {}
    for connection_id in connection_ids:
        anchor = branch_anchors[(region_id, connection_id)]
        target = np.asarray(anchor, dtype=float) + np.asarray(
            [0.0, float(config["agent"]["sensor_height"]), 0.0]
        )
        visibility = depth_visibility(
            depth,
            origin,
            target,
            rotation_xyzw,
            float(config["erp"]["visibility_depth_tolerance"]),
            int(config["erp"]["visibility_patch_radius"]),
        )
        visibility["depth_visible"] = visibility["visible"]
        visibility["geodesically_direct"] = _geodesically_visible(
            simulator.pathfinder, np.asarray(position), anchor
        )
        visibility["ray_cast_checked"] = False
        visibility["ray_geometry_clear"] = None
        if bool(config["runtime"].get("enable_physics_for_ray_cast", False)):
            try:
                import habitat_sim

                ray_vector = target - origin
                ray_distance = float(np.linalg.norm(ray_vector))
                ray = habitat_sim.geo.Ray(
                    origin.astype(np.float32),
                    (ray_vector / max(ray_distance, 1e-8)).astype(np.float32),
                )
                ray_result = simulator.cast_ray(
                    ray, max_distance=ray_distance, buffer_distance=0.02
                )
                first_hit = (
                    min(float(hit.ray_distance) for hit in ray_result.hits)
                    if ray_result.has_hits()
                    else math.inf
                )
                visibility["ray_cast_checked"] = True
                visibility["ray_first_hit_distance"] = (
                    None if math.isinf(first_hit) else first_hit
                )
                visibility["ray_geometry_clear"] = bool(
                    first_hit >= ray_distance - float(config["erp"]["visibility_depth_tolerance"])
                )
            except Exception as error:
                visibility["ray_cast_error"] = repr(error)
        visibility["visible"] = bool(
            visibility["visible"]
            and visibility["geodesically_direct"]
            and visibility["ray_geometry_clear"] is not False
            and visibility["target_distance"] <= float(config["erp"]["max_branch_distance"])
        )
        visibility["connection_id"] = connection_id
        visibility["to_region"] = _other_region(connections[connection_id], region_id)
        visibility["anchor_position"] = anchor.astype(float).tolist()
        results[connection_id] = visibility
    return results, observations


def mine_decision_relations(
    simulator,
    scene_graph: dict[str, Any],
    arrays: dict[str, np.ndarray],
    scene_id: str,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pathfinder = simulator.pathfinder
    lookup = RegionLookup(arrays)
    region_graph = build_region_graph(scene_graph)
    connections, incident = _connection_maps(scene_graph)
    anchor_distance = float(config["decision"]["branch_anchor_distance"])
    branch_anchors = {
        (region_id, connection_id): _branch_anchor(
            pathfinder,
            scene_graph,
            connections[connection_id],
            region_id,
            anchor_distance,
        )
        for region_id, connection_ids in incident.items()
        for connection_id in connection_ids
    }
    relations = []
    rejected = Counter()
    proposal_count = 0
    for region_id, connection_ids in sorted(incident.items()):
        connections_by_neighbor = _connections_by_neighbor(
            connections, region_id, connection_ids
        )
        neighbor_by_connection = {
            connection_id: neighbor
            for neighbor, grouped_ids in connections_by_neighbor.items()
            for connection_id in grouped_ids
        }
        # Parallel portals into the same adjacent region are one navigation
        # branch. Counting them independently creates U-turn pseudo-decisions
        # that immediately rejoin in the same open space.
        if len(connections_by_neighbor) < 3:
            continue
        for incoming_id in connection_ids:
            incoming_neighbor = neighbor_by_connection[incoming_id]
            outgoing = [
                value
                for value in connection_ids
                if neighbor_by_connection[value] != incoming_neighbor
            ]
            for selected_id in outgoing:
                proposal_count += 1
                selected_neighbor = neighbor_by_connection[selected_id]
                alternatives_by_neighbor: dict[int, list[str]] = defaultdict(list)
                rejoin_distances = {}
                for alternative_id in outgoing:
                    alternative_neighbor = neighbor_by_connection[alternative_id]
                    if alternative_neighbor == selected_neighbor:
                        continue
                    rejoin = _rejoin_distance(
                        region_graph, region_id, selected_neighbor, alternative_neighbor
                    )
                    rejoin_distances[alternative_id] = rejoin
                    if rejoin >= float(config["decision"]["min_rejoin_distance"]):
                        alternatives_by_neighbor[alternative_neighbor].append(alternative_id)
                if not alternatives_by_neighbor:
                    rejected["short_rejoin"] += 1
                    continue
                candidates = _candidate_positions(
                    pathfinder,
                    lookup,
                    scene_graph,
                    connections,
                    region_id,
                    incoming_id,
                    selected_id,
                    config,
                )
                if not candidates:
                    rejected["no_arrival_state"] += 1
                    continue
                incoming_position = np.asarray(connections[incoming_id]["position"], dtype=float)
                best = None
                selected_visible_at_any_candidate = False
                alternative_visible_at_any_candidate = False
                for position in candidates:
                    rotation = yaw_rotation_xyzw(position - incoming_position)
                    verification, _ = observe_branches(
                        simulator,
                        scene_graph,
                        region_id,
                        outgoing,
                        position,
                        rotation,
                        config,
                        branch_anchors,
                    )
                    if not verification[selected_id]["visible"]:
                        continue
                    selected_visible_at_any_candidate = True
                    visible_alternatives = []
                    for alternative_ids in alternatives_by_neighbor.values():
                        visible_for_neighbor = [
                            alternative_id
                            for alternative_id in alternative_ids
                            if verification[alternative_id]["visible"]
                        ]
                        if visible_for_neighbor:
                            visible_alternatives.append(
                                max(
                                    visible_for_neighbor,
                                    key=lambda alternative_id: (
                                        verification[alternative_id]["depth_margin"],
                                        _branch_separation(
                                            verification[selected_id]["bearing_degrees"],
                                            verification[alternative_id]["bearing_degrees"],
                                        ),
                                        alternative_id,
                                    ),
                                )
                            )
                    if not visible_alternatives:
                        continue
                    alternative_visible_at_any_candidate = True
                    minimum_depth_margin = min(
                        float(verification[selected_id]["depth_margin"]),
                        *(
                            float(verification[alternative_id]["depth_margin"])
                            for alternative_id in visible_alternatives
                        ),
                    )
                    minimum_separation = min(
                        _branch_separation(
                            verification[selected_id]["bearing_degrees"],
                            verification[alternative_id]["bearing_degrees"],
                        )
                        for alternative_id in visible_alternatives
                    )
                    score = (
                        len(visible_alternatives),
                        minimum_depth_margin,
                        -abs(float(np.linalg.norm(position - incoming_position)) - 1.2),
                        minimum_separation,
                    )
                    if best is None or score > best[0]:
                        best = (score, position, rotation, verification, visible_alternatives)
                if best is None:
                    if not selected_visible_at_any_candidate:
                        rejected["selected_branch_not_visible"] += 1
                    elif not alternative_visible_at_any_candidate:
                        rejected["alternative_not_visible"] += 1
                    else:
                        rejected["panorama_visibility"] += 1
                    continue
                _, position, rotation, verification, alternatives = best
                relation_id = stable_id(
                    "decision", scene_id, region_id, incoming_id, selected_id
                )
                relations.append(
                    {
                        "relation_id": relation_id,
                        "region_id": region_id,
                        "incoming_connection_id": incoming_id,
                        "incoming_region_id": _other_region(
                            connections[incoming_id], region_id
                        ),
                        "selected_connection_id": selected_id,
                        "selected_region_id": selected_neighbor,
                        "alternative_connection_ids": alternatives,
                        "alternative_region_ids": [
                            _other_region(connections[value], region_id)
                            for value in alternatives
                        ],
                        "canonical_state": {
                            "position": position.astype(float).tolist(),
                            "rotation_xyzw": rotation,
                        },
                        "branch_observations": {
                            value: verification[value]
                            for value in [selected_id, *alternatives]
                        },
                        "rejoin_distance_by_alternative": {
                            value: (
                                None
                                if math.isinf(rejoin_distances[value])
                                else float(rejoin_distances[value])
                            )
                            for value in alternatives
                        },
                        "verification_method": "habitat_erp_depth_bullet_ray_and_navmesh_geodesic",
                    }
                )
    relations.sort(key=lambda relation: relation["relation_id"])
    return relations, {
        "relation_proposals": proposal_count,
        "valid_relations": len(relations),
        "rejected": dict(sorted(rejected.items())),
        "branch_anchors": {
            f"{region_id}:{connection_id}": anchor.astype(float).tolist()
            for (region_id, connection_id), anchor in branch_anchors.items()
        },
    }


def relation_index(
    relations: Sequence[dict[str, Any]],
) -> dict[tuple[int, str, str], dict[str, Any]]:
    return {
        (
            int(relation["region_id"]),
            relation["incoming_connection_id"],
            relation["selected_connection_id"],
        ): relation
        for relation in relations
    }


def branch_anchor_map(decision_stats: dict[str, Any]) -> dict[tuple[int, str], np.ndarray]:
    result = {}
    for key, value in decision_stats["branch_anchors"].items():
        region, connection = key.split(":", 1)
        result[(int(region), connection)] = np.asarray(value, dtype=np.float32)
    return result
