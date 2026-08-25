"""Stable navigation-region partition over the intrinsic HM3D NavMesh graph."""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from dataset_create.trajectory.io_utils import stable_id


@dataclass
class _UnionFind:
    parent: dict[int, int]

    @classmethod
    def create(cls, values: Iterable[int]) -> "_UnionFind":
        return cls({int(value): int(value) for value in values})

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first: int, second: int) -> None:
        a, b = self.find(first), self.find(second)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def _edge_key(first: np.ndarray, second: np.ndarray, quantum: float) -> tuple:
    a = tuple(np.rint(first / quantum).astype(np.int64).tolist())
    b = tuple(np.rint(second / quantum).astype(np.int64).tolist())
    return tuple(sorted((a, b)))


def _extract_mesh(pathfinder, config: dict[str, Any]) -> dict[str, Any]:
    quantum = float(config["partition"]["coordinate_quantization"])
    minimum_island_area = float(config["partition"]["min_island_area"])
    triangles: list[np.ndarray] = []
    island_ids: list[int] = []
    for island_id in range(int(pathfinder.num_islands)):
        if float(pathfinder.island_area(island_id)) < minimum_island_area:
            continue
        vertices = np.asarray(
            pathfinder.build_navmesh_vertices(island_id), dtype=np.float32
        )
        if not len(vertices):
            continue
        island_triangles = vertices.reshape(-1, 3, 3)
        triangles.extend(island_triangles)
        island_ids.extend([island_id] * len(island_triangles))
    if not triangles:
        raise RuntimeError("No HM3D NavMesh island passes min_island_area")
    triangle_array = np.asarray(triangles, dtype=np.float32)
    islands = np.asarray(island_ids, dtype=np.int32)
    centers = triangle_array.mean(axis=1)
    areas = np.linalg.norm(
        np.cross(triangle_array[:, 1] - triangle_array[:, 0], triangle_array[:, 2] - triangle_array[:, 0]),
        axis=1,
    ) * 0.5
    clearances = np.asarray(
        [
            pathfinder.distance_to_closest_obstacle(center, 3.0)
            for center in centers
        ],
        dtype=np.float32,
    )
    clearances = np.nan_to_num(clearances, nan=0.0, posinf=3.0, neginf=0.0)

    edge_owners: dict[tuple, list[tuple[int, np.ndarray, np.ndarray]]] = defaultdict(list)
    for triangle_id, triangle in enumerate(triangle_array):
        for offset in range(3):
            first, second = triangle[offset], triangle[(offset + 1) % 3]
            edge_owners[_edge_key(first, second, quantum)].append(
                (triangle_id, first, second)
            )
    adjacency: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for owners in edge_owners.values():
        if len(owners) != 2:
            continue
        (first_id, edge_a, edge_b), (second_id, _, _) = owners
        if islands[first_id] != islands[second_id]:
            continue
        adjacency.append((first_id, second_id, edge_a.copy(), edge_b.copy()))
    return {
        "triangles": triangle_array,
        "triangle_island": islands,
        "centers": centers,
        "areas": areas.astype(np.float32),
        "clearances": clearances,
        "adjacency": adjacency,
    }


def _mesh_graph(mesh: dict[str, Any]) -> nx.Graph:
    graph = nx.Graph()
    for triangle_id, center in enumerate(mesh["centers"]):
        graph.add_node(triangle_id, position=center)
    for first, second, edge_a, edge_b in mesh["adjacency"]:
        graph.add_edge(
            first,
            second,
            weight=max(1e-4, float(np.linalg.norm(mesh["centers"][first] - mesh["centers"][second]))),
            portal_a=edge_a,
            portal_b=edge_b,
        )
    return graph


def _select_seeds(
    graph: nx.Graph,
    component: set[int],
    centers: np.ndarray,
    clearances: np.ndarray,
    config: dict[str, Any],
) -> list[int]:
    settings = config["partition"]
    minimum_clearance = float(settings["min_seed_clearance"])
    spacing = float(settings["seed_spacing"])
    max_coverage = float(settings["max_seed_coverage_distance"])
    candidates = []
    for node in component:
        neighbor_clearance = [clearances[neighbor] for neighbor in graph.neighbors(node)]
        if clearances[node] >= minimum_clearance and (
            not neighbor_clearance or clearances[node] >= max(neighbor_clearance) - 1e-4
        ):
            candidates.append(node)
    candidates.sort(key=lambda node: (-float(clearances[node]), node))
    seeds: list[int] = []
    for node in candidates:
        if all(float(np.linalg.norm(centers[node] - centers[seed])) >= spacing for seed in seeds):
            seeds.append(node)
    if not seeds:
        seeds = [max(component, key=lambda node: (float(clearances[node]), -node))]

    subgraph = graph.subgraph(component)
    while True:
        distances = nx.multi_source_dijkstra_path_length(subgraph, seeds, weight="weight")
        farthest = max(component, key=lambda node: (distances.get(node, math.inf), -node))
        if distances.get(farthest, math.inf) <= max_coverage:
            break
        seeds.append(farthest)
    return seeds


def _watershed_labels(
    graph: nx.Graph,
    component: set[int],
    seeds: Sequence[int],
    clearances: np.ndarray,
) -> tuple[dict[int, int], dict[int, float]]:
    labels: dict[int, int] = {}
    best: dict[int, tuple[float, float, int]] = {}
    queue: list[tuple[float, float, int, int]] = []
    peak_by_label: dict[int, float] = {}
    for label, node in enumerate(seeds):
        bottleneck = float(clearances[node])
        best[node] = (bottleneck, 0.0, label)
        peak_by_label[label] = bottleneck
        heapq.heappush(queue, (-bottleneck, 0.0, label, node))
    while queue:
        negative_bottleneck, distance, label, node = heapq.heappop(queue)
        bottleneck = -negative_bottleneck
        if best.get(node) != (bottleneck, distance, label):
            continue
        labels[node] = label
        for neighbor, attributes in graph[node].items():
            if neighbor not in component:
                continue
            next_bottleneck = min(bottleneck, float(clearances[neighbor]))
            next_distance = distance + float(attributes["weight"])
            previous = best.get(neighbor)
            candidate = (next_bottleneck, next_distance, label)
            is_better = previous is None or next_bottleneck > previous[0] + 1e-6
            if previous is not None and abs(next_bottleneck - previous[0]) <= 1e-6:
                is_better = (next_distance, label) < (previous[1], previous[2])
            if is_better:
                best[neighbor] = candidate
                heapq.heappush(queue, (-next_bottleneck, next_distance, label, neighbor))
    return labels, peak_by_label


def _merge_watershed_basins(
    graph: nx.Graph,
    labels: dict[int, int],
    peaks: dict[int, float],
    clearances: np.ndarray,
    ratio_threshold: float,
) -> dict[int, int]:
    saddle: dict[tuple[int, int], float] = defaultdict(float)
    for first, second in graph.edges:
        if first not in labels or second not in labels:
            continue
        a, b = labels[first], labels[second]
        if a == b:
            continue
        pair = tuple(sorted((a, b)))
        saddle[pair] = max(saddle[pair], min(float(clearances[first]), float(clearances[second])))
    union_find = _UnionFind.create(peaks)
    for (first, second), value in sorted(saddle.items(), key=lambda item: -item[1]):
        denominator = max(peaks[first], peaks[second], 1e-6)
        if value / denominator >= ratio_threshold:
            union_find.union(first, second)
    return {node: union_find.find(label) for node, label in labels.items()}


def _merge_small_regions(
    graph: nx.Graph,
    labels: dict[int, int],
    areas: np.ndarray,
    clearances: np.ndarray,
    minimum_area: float,
) -> dict[int, int]:
    labels = dict(labels)
    while True:
        members: dict[int, list[int]] = defaultdict(list)
        for node, label in labels.items():
            members[label].append(node)
        small = [
            label
            for label, nodes in members.items()
            if sum(float(areas[node]) for node in nodes) < minimum_area
        ]
        if not small or len(members) == 1:
            break
        changed = False
        for label in sorted(small):
            boundary: dict[int, tuple[float, int]] = {}
            for node in members.get(label, []):
                for neighbor in graph.neighbors(node):
                    other = labels[neighbor]
                    if other == label:
                        continue
                    value = min(float(clearances[node]), float(clearances[neighbor]))
                    previous = boundary.get(other, (-1.0, 0))
                    boundary[other] = (max(previous[0], value), previous[1] + 1)
            if boundary:
                target = max(boundary, key=lambda item: (boundary[item][0], boundary[item][1], -item))
                for node in members[label]:
                    labels[node] = target
                changed = True
        if not changed:
            break
    return labels


def _renumber_labels(
    labels: dict[int, int], triangle_island: np.ndarray
) -> tuple[np.ndarray, dict[int, int]]:
    label_keys = sorted(
        set(labels.values()),
        key=lambda label: (
            min(int(triangle_island[node]) for node, value in labels.items() if value == label),
            label,
        ),
    )
    mapping = {label: index for index, label in enumerate(label_keys)}
    result = np.full(len(triangle_island), -1, dtype=np.int32)
    for node, label in labels.items():
        result[node] = mapping[label]
    return result, mapping


def _representatives(
    nodes: Sequence[int],
    centers: np.ndarray,
    clearances: np.ndarray,
    count: int,
    spacing: float,
) -> list[int]:
    ordered = sorted(nodes, key=lambda node: (-float(clearances[node]), node))
    chosen = [ordered[0]]
    while len(chosen) < count:
        candidates = [node for node in nodes if node not in chosen]
        if not candidates:
            break
        best = max(
            candidates,
            key=lambda node: (
                min(float(np.linalg.norm(centers[node] - centers[item])) for item in chosen),
                float(clearances[node]),
                -node,
            ),
        )
        distance = min(float(np.linalg.norm(centers[best] - centers[item])) for item in chosen)
        if distance < spacing:
            break
        chosen.append(best)
    return chosen


def _cluster_connection_edges(
    records: list[tuple[int, int, np.ndarray, np.ndarray]], radius: float
) -> list[list[tuple[int, int, np.ndarray, np.ndarray]]]:
    if len(records) <= 1:
        return [records]
    midpoints = np.asarray([(edge_a + edge_b) * 0.5 for _, _, edge_a, edge_b in records])
    union_find = _UnionFind.create(range(len(records)))
    tree = cKDTree(midpoints)
    for first, second in tree.query_pairs(radius):
        union_find.union(int(first), int(second))
    groups: dict[int, list] = defaultdict(list)
    for index, record in enumerate(records):
        groups[union_find.find(index)].append(record)
    return [groups[key] for key in sorted(groups)]


def partition_navigation_regions(pathfinder, scene_id: str, config: dict[str, Any]):
    mesh = _extract_mesh(pathfinder, config)
    graph = _mesh_graph(mesh)
    all_labels: dict[int, int] = {}
    label_offset = 0
    for component_nodes in nx.connected_components(graph):
        component = set(component_nodes)
        seeds = _select_seeds(
            graph, component, mesh["centers"], mesh["clearances"], config
        )
        labels, peaks = _watershed_labels(graph, component, seeds, mesh["clearances"])
        labels = _merge_watershed_basins(
            graph,
            labels,
            peaks,
            mesh["clearances"],
            float(config["partition"]["watershed_merge_ratio"]),
        )
        unique = {value: index + label_offset for index, value in enumerate(sorted(set(labels.values())))}
        all_labels.update({node: unique[value] for node, value in labels.items()})
        label_offset += len(unique)
    all_labels = _merge_small_regions(
        graph,
        all_labels,
        mesh["areas"],
        mesh["clearances"],
        float(config["partition"]["min_region_area"]),
    )
    triangle_region, _ = _renumber_labels(all_labels, mesh["triangle_island"])

    members: dict[int, list[int]] = defaultdict(list)
    for triangle_id, region_id in enumerate(triangle_region):
        if region_id >= 0:
            members[int(region_id)].append(triangle_id)
    regions = []
    for region_id, nodes in sorted(members.items()):
        area_weights = mesh["areas"][nodes]
        centroid = np.average(mesh["centers"][nodes], axis=0, weights=area_weights)
        representatives = _representatives(
            nodes,
            mesh["centers"],
            mesh["clearances"],
            int(config["partition"]["representatives_per_region"]),
            float(config["partition"]["representative_spacing"]),
        )
        regions.append(
            {
                "region_id": region_id,
                "island_id": int(mesh["triangle_island"][nodes[0]]),
                "area": float(np.sum(area_weights)),
                "centroid": centroid.astype(float).tolist(),
                "max_clearance": float(np.max(mesh["clearances"][nodes])),
                "representative_positions": [
                    mesh["centers"][node].astype(float).tolist() for node in representatives
                ],
                "triangle_count": len(nodes),
            }
        )

    crossing_by_pair: dict[tuple[int, int], list] = defaultdict(list)
    for first, second, edge_a, edge_b in mesh["adjacency"]:
        region_a, region_b = int(triangle_region[first]), int(triangle_region[second])
        if region_a < 0 or region_a == region_b:
            continue
        pair = tuple(sorted((region_a, region_b)))
        crossing_by_pair[pair].append((first, second, edge_a, edge_b))

    connections = []
    radius = float(config["partition"]["connection_cluster_radius"])
    for pair, records in sorted(crossing_by_pair.items()):
        for cluster_index, cluster in enumerate(_cluster_connection_edges(records, radius)):
            segments = np.asarray([[edge_a, edge_b] for _, _, edge_a, edge_b in cluster])
            lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
            best_index = max(
                range(len(cluster)),
                key=lambda index: (
                    min(
                        float(mesh["clearances"][cluster[index][0]]),
                        float(mesh["clearances"][cluster[index][1]]),
                    ),
                    float(lengths[index]),
                ),
            )
            first, second, edge_a, edge_b = cluster[best_index]
            midpoint = (edge_a + edge_b) * 0.5
            side_positions: dict[str, list[float]] = {}
            for node in (first, second):
                region = int(triangle_region[node])
                side_positions[str(region)] = mesh["centers"][node].astype(float).tolist()
            connection_id = stable_id(
                "conn", scene_id, pair, np.round(midpoint, 3).tolist(), cluster_index
            )
            connections.append(
                {
                    "connection_id": connection_id,
                    "regions": list(pair),
                    "position": midpoint.astype(float).tolist(),
                    "side_positions": side_positions,
                    "boundary_width": float(np.sum(lengths)),
                    "bottleneck_clearance": max(
                        min(float(mesh["clearances"][a]), float(mesh["clearances"][b]))
                        for a, b, _, _ in cluster
                    ),
                    "boundary_segments": segments.astype(float).tolist(),
                }
            )

    graph_record = {
        "regions": regions,
        "connections": connections,
        "region_count": len(regions),
        "connection_count": len(connections),
    }
    arrays = {
        "triangles": mesh["triangles"],
        "triangle_centers": mesh["centers"],
        "triangle_areas": mesh["areas"],
        "triangle_clearance": mesh["clearances"],
        "triangle_island": mesh["triangle_island"],
        "triangle_region": triangle_region,
    }
    return graph_record, arrays


class RegionLookup:
    def __init__(self, arrays: dict[str, np.ndarray]):
        valid = arrays["triangle_region"] >= 0
        self.triangles = np.asarray(arrays["triangles"], dtype=np.float64)[valid]
        self.centers = np.asarray(arrays["triangle_centers"])[valid]
        self.regions = np.asarray(arrays["triangle_region"])[valid]
        self.tree = cKDTree(self.centers)
        radii = np.linalg.norm(
            self.triangles - self.centers[:, None, :], axis=2
        ).max(axis=1)
        self.maximum_triangle_radius = float(np.max(radii, initial=0.0))

    @staticmethod
    def _point_triangle_distance_squared(
        points: np.ndarray, triangles: np.ndarray
    ) -> np.ndarray:
        """Vectorized exact distance for matching ``[..., 3]`` batches.

        Habitat shortest-path samples lie on the NavMesh, but a large triangle's
        center can be several metres away.  Center distance is therefore only a
        candidate lookup mechanism and must not be used as the region test.
        """

        a, b, c = triangles[..., 0, :], triangles[..., 1, :], triangles[..., 2, :]
        ab, ac = b - a, c - a
        normal = np.cross(ab, ac)
        normal_squared = np.sum(normal * normal, axis=-1)
        ap = points - a
        signed_numerator = np.sum(ap * normal, axis=-1)
        safe_normal_squared = np.maximum(normal_squared, 1e-16)
        projected = points - (
            signed_numerator / safe_normal_squared
        )[..., None] * normal

        projected_offset = projected - a
        dot00 = np.sum(ab * ab, axis=-1)
        dot01 = np.sum(ab * ac, axis=-1)
        dot11 = np.sum(ac * ac, axis=-1)
        dot20 = np.sum(projected_offset * ab, axis=-1)
        dot21 = np.sum(projected_offset * ac, axis=-1)
        denominator = dot00 * dot11 - dot01 * dot01
        safe_denominator = np.where(
            np.abs(denominator) > 1e-16, denominator, 1.0
        )
        coordinate_b = (dot11 * dot20 - dot01 * dot21) / safe_denominator
        coordinate_c = (dot00 * dot21 - dot01 * dot20) / safe_denominator
        inside = (
            (normal_squared > 1e-16)
            & (coordinate_b >= -1e-7)
            & (coordinate_c >= -1e-7)
            & (coordinate_b + coordinate_c <= 1.0 + 1e-7)
        )
        plane_distance_squared = signed_numerator * signed_numerator / safe_normal_squared

        def segment_distance_squared(first: np.ndarray, second: np.ndarray) -> np.ndarray:
            edge = second - first
            length_squared = np.sum(edge * edge, axis=-1)
            fraction = np.sum((points - first) * edge, axis=-1) / np.maximum(
                length_squared, 1e-16
            )
            fraction = np.clip(fraction, 0.0, 1.0)
            closest = first + fraction[..., None] * edge
            return np.sum((points - closest) ** 2, axis=-1)

        edge_distance_squared = np.minimum.reduce(
            [
                segment_distance_squared(a, b),
                segment_distance_squared(b, c),
                segment_distance_squared(c, a),
            ]
        )
        return np.where(inside, plane_distance_squared, edge_distance_squared)

    def _lookup_batch(
        self, points: np.ndarray, max_distance: float
    ) -> tuple[np.ndarray, np.ndarray]:
        points = np.atleast_2d(np.asarray(points, dtype=np.float64))
        candidate_count = min(32, len(self.centers))
        _, candidate_indices = self.tree.query(points, k=candidate_count)
        if candidate_count == 1:
            candidate_indices = np.asarray(candidate_indices)[:, None]
        candidate_triangles = self.triangles[candidate_indices]
        distances_squared = self._point_triangle_distance_squared(
            points[:, None, :], candidate_triangles
        )
        best_offsets = np.argmin(distances_squared, axis=1)
        rows = np.arange(len(points))
        best_indices = candidate_indices[rows, best_offsets]
        best_distances_squared = distances_squared[rows, best_offsets]

        # Rare fallback for a very large triangle whose center was not among the
        # nearest 32. The radius bound guarantees that a containing triangle is
        # included without making the common path expensive.
        unresolved = np.flatnonzero(best_distances_squared > max_distance * max_distance)
        for row in unresolved:
            candidates = self.tree.query_ball_point(
                points[row], self.maximum_triangle_radius + max_distance
            )
            if not candidates:
                continue
            candidate_array = np.asarray(candidates, dtype=np.int64)
            exact = self._point_triangle_distance_squared(
                np.broadcast_to(points[row], (len(candidate_array), 3)),
                self.triangles[candidate_array],
            )
            offset = int(np.argmin(exact))
            if exact[offset] < best_distances_squared[row]:
                best_distances_squared[row] = exact[offset]
                best_indices[row] = candidate_array[offset]
        return best_indices, np.sqrt(np.maximum(best_distances_squared, 0.0))

    def region_for_point(self, point: Sequence[float], max_distance: float = 1.0) -> int:
        indices, distances = self._lookup_batch(np.asarray(point), max_distance)
        return int(self.regions[indices[0]]) if float(distances[0]) <= max_distance else -1

    def regions_for_points(
        self, points: Sequence[Sequence[float]], max_distance: float = 1.0
    ) -> np.ndarray:
        indices, distances = self._lookup_batch(np.asarray(points), max_distance)
        labels = self.regions[indices].astype(np.int32)
        labels[distances > max_distance] = -1
        return labels


def build_region_graph(scene_graph: dict[str, Any]) -> nx.MultiGraph:
    graph = nx.MultiGraph()
    for region in scene_graph["regions"]:
        graph.add_node(int(region["region_id"]), **region)
    for connection in scene_graph["connections"]:
        first, second = map(int, connection["regions"])
        weight = float(
            np.linalg.norm(
                np.asarray(scene_graph["regions"][first]["centroid"])
                - np.asarray(scene_graph["regions"][second]["centroid"])
            )
        )
        graph.add_edge(
            first,
            second,
            key=connection["connection_id"],
            weight=max(weight, 0.1),
            **connection,
        )
    return graph
