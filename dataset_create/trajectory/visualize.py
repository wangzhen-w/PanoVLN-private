"""Audit visualizations for partitions, routes, replay, decisions, and deduplication."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from PIL import Image, ImageDraw, ImageFont

from dataset_create.trajectory.hm3d import set_agent_state


def _floor_groups(scene_graph: dict[str, Any], gap: float = 1.2) -> list[list[int]]:
    ordered = sorted(
        ((region["region_id"], float(region["centroid"][1])) for region in scene_graph["regions"]),
        key=lambda item: item[1],
    )
    groups: list[list[int]] = []
    previous_height = None
    for region_id, height in ordered:
        if previous_height is None or height - previous_height > gap:
            groups.append([])
        groups[-1].append(int(region_id))
        previous_height = height
    return groups


def _draw_partition_axis(
    axis,
    scene_graph: dict[str, Any],
    arrays: dict[str, np.ndarray],
    region_ids: set[int] | None = None,
):
    triangles = np.asarray(arrays["triangles"])
    triangle_regions = np.asarray(arrays["triangle_region"])
    included = triangle_regions >= 0
    if region_ids is not None:
        included &= np.isin(triangle_regions, list(region_ids))
    polygons = triangles[included][:, :, [0, 2]]
    colors = plt.get_cmap("tab20")((triangle_regions[included] % 20) / 20.0)
    axis.add_collection(
        PolyCollection(polygons, facecolors=colors, edgecolors=(0, 0, 0, 0.12), linewidths=0.2)
    )
    region_by_id = {int(region["region_id"]): region for region in scene_graph["regions"]}
    for region_id, region in region_by_id.items():
        if region_ids is not None and region_id not in region_ids:
            continue
        x, _, z = region["centroid"]
        axis.text(x, z, f"R{region_id}", fontsize=7, ha="center", va="center")
    for connection in scene_graph["connections"]:
        first, second = map(int, connection["regions"])
        if region_ids is not None and not ({first, second} <= region_ids):
            continue
        x, _, z = connection["position"]
        axis.scatter([x], [z], s=14, c="black", marker="x", linewidths=0.8)
        first_center = np.asarray(region_by_id[first]["centroid"])[[0, 2]]
        second_center = np.asarray(region_by_id[second]["centroid"])[[0, 2]]
        axis.plot(
            [first_center[0], x, second_center[0]],
            [first_center[1], z, second_center[1]],
            color="black",
            alpha=0.25,
            linewidth=0.7,
        )
    axis.autoscale_view()
    axis.set_aspect("equal", adjustable="box")
    axis.invert_yaxis()
    axis.set_xlabel("Habitat x (m)")
    axis.set_ylabel("Habitat z (m)")


def plot_navigation_regions(
    scene_graph: dict[str, Any], arrays: dict[str, np.ndarray], output: str | Path
) -> None:
    groups = _floor_groups(scene_graph)
    columns = min(3, max(1, len(groups)))
    rows = int(math.ceil(len(groups) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(6 * columns, 5 * rows), squeeze=False)
    for index, axis in enumerate(axes.flat):
        if index >= len(groups):
            axis.axis("off")
            continue
        ids = set(groups[index])
        _draw_partition_axis(axis, scene_graph, arrays, ids)
        heights = [scene_graph["regions"][region_id]["centroid"][1] for region_id in ids]
        axis.set_title(
            f"floor group {index}: y={min(heights):.2f}..{max(heights):.2f} m"
        )
    figure.suptitle(
        f"Navigation Regions ({scene_graph['region_count']}) and connections ({scene_graph['connection_count']})"
    )
    figure.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_route(
    scene_graph: dict[str, Any],
    arrays: dict[str, np.ndarray],
    trajectory: dict[str, Any],
    output: str | Path,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 8))
    _draw_partition_axis(axis, scene_graph, arrays)
    points = np.asarray(trajectory["dense_shortest_path"])
    axis.plot(points[:, 0], points[:, 2], color="#147df5", linewidth=2.2, label="natural shortest path")
    axis.scatter(points[0, 0], points[0, 2], c="#00a878", s=70, marker="o", label="start")
    axis.scatter(points[-1, 0], points[-1, 2], c="#d7263d", s=80, marker="*", label="goal")
    for index, event in enumerate(trajectory["decision_events"]):
        position = np.asarray(event["position"])
        axis.scatter(position[0], position[2], c="#ff9f1c", s=60, marker="D")
        axis.text(position[0], position[2], f" D{index}", fontsize=8)
    axis.set_title(
        "Region route " + " → ".join(f"R{value}" for value in trajectory["region_sequence"])
    )
    axis.legend(loc="best")
    figure.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_replay(
    scene_graph: dict[str, Any],
    arrays: dict[str, np.ndarray],
    trajectory: dict[str, Any],
    output: str | Path,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 8))
    _draw_partition_axis(axis, scene_graph, arrays)
    ideal = np.asarray(trajectory["dense_shortest_path"])
    replay = np.asarray([state["position"] for state in trajectory["replay"]["states"]])
    axis.plot(ideal[:, 0], ideal[:, 2], color="gray", linestyle="--", linewidth=1.2, label="continuous shortest path")
    axis.plot(replay[:, 0], replay[:, 2], color="#6a00f4", linewidth=2.0, label="primitive replay")
    for index, event in enumerate(trajectory["decision_events"]):
        position = replay[int(event["state_index"])]
        axis.scatter(position[0], position[2], color="#ff9f1c", s=65, marker="D")
        axis.text(position[0], position[2], f" D{index}@{event['state_index']}", fontsize=8)
    axis.set_title(
        f"Replay: {len(trajectory['replay']['actions'])} actions, "
        f"{trajectory['decision_count']} verified decisions"
    )
    axis.legend(loc="best")
    figure.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_deduplication_example(
    scene_graph: dict[str, Any],
    arrays: dict[str, np.ndarray],
    example: dict[str, Any],
    output: str | Path,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 8))
    _draw_partition_axis(axis, scene_graph, arrays)
    selected_id = example["selected_candidate_id"]
    for candidate in example["candidate_paths"]:
        points = np.asarray(candidate["shortest_path"])
        selected = candidate["candidate_id"] == selected_id
        axis.plot(
            points[:, 0],
            points[:, 2],
            color="#147df5" if selected else "gray",
            alpha=1.0 if selected else 0.55,
            linewidth=2.5 if selected else 1.0,
            label="retained representative" if selected else None,
        )
        axis.scatter(points[0, 0], points[0, 2], s=18, color="#00a878", alpha=0.7)
        axis.scatter(points[-1, 0], points[-1, 2], s=18, color="#d7263d", alpha=0.7)
    axis.set_title(
        f"{len(example['candidate_paths'])} coordinate variants grouped by one directed region route"
    )
    axis.legend(loc="best")
    figure.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_decision_erp(
    simulator,
    trajectory: dict[str, Any],
    event: dict[str, Any],
    output: str | Path,
) -> None:
    set_agent_state(
        simulator, event["position"], event["rotation_xyzw"]
    )
    rgb = np.asarray(simulator.get_sensor_observations()["erp_rgb"])[..., :3]
    image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    styles = {
        event["incoming_connection_id"]: ("incoming", (245, 245, 245)),
        event["selected_connection_id"]: ("selected", (40, 220, 70)),
    }
    for index, connection_id in enumerate(event["alternative_connection_ids"]):
        styles[connection_id] = (f"alternative {index + 1}", (255, 160, 25))
    radius = max(7, image.width // 90)
    for connection_id, (label, color) in styles.items():
        observation = event["branch_observations"].get(connection_id)
        if observation is None:
            continue
        x, y = observation["pixel"]
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=4)
        text = f"{label}: R{observation['to_region']}"
        text_x = min(max(2, x + radius + 3), image.width - max(80, len(text) * 7))
        text_y = min(max(2, y - radius), image.height - 18)
        draw.rectangle((text_x - 2, text_y - 1, text_x + len(text) * 7 + 3, text_y + 15), fill=(0, 0, 0))
        draw.text((text_x, text_y), text, fill=color)
    footer = (
        f"{trajectory['trajectory_id']} | state {event['state_index']} | "
        f"decision region R{event['region_id']}"
    )
    draw.rectangle((0, image.height - 20, image.width, image.height), fill=(0, 0, 0))
    draw.text((5, image.height - 17), footer, fill=(255, 255, 255))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def render_scene_visualizations(
    simulator,
    scene_graph: dict[str, Any],
    arrays: dict[str, np.ndarray],
    trajectories: Sequence[dict[str, Any]],
    dedup_examples: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    plot_navigation_regions(scene_graph, arrays, root / "navigation_regions.png")
    manifest: dict[str, Any] = {"partition": "navigation_regions.png"}
    if not trajectories:
        return manifest
    representative = max(
        trajectories,
        key=lambda trajectory: (
            trajectory["decision_count"], trajectory["geodesic_distance"]
        ),
    )
    plot_route(scene_graph, arrays, representative, root / "region_route.png")
    plot_replay(scene_graph, arrays, representative, root / "replayed_trajectory.png")
    manifest.update(
        {
            "representative_trajectory_id": representative["trajectory_id"],
            "route": "region_route.png",
            "replay": "replayed_trajectory.png",
        }
    )
    erp_files = []
    for index, event in enumerate(representative["decision_events"][:4]):
        filename = f"decision_event_{index:02d}.png"
        render_decision_erp(simulator, representative, event, root / filename)
        erp_files.append(filename)
    manifest["decision_erps"] = erp_files
    example = dedup_examples.get(representative["route_structure_id"])
    if example is None and dedup_examples:
        example = dedup_examples[sorted(dedup_examples)[0]]
    if example is not None:
        plot_deduplication_example(
            scene_graph, arrays, example, root / "coordinate_deduplication.png"
        )
        manifest["coordinate_deduplication"] = "coordinate_deduplication.png"
    return manifest

