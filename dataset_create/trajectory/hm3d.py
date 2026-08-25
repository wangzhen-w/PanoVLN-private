"""HM3D discovery plus the Habitat-Sim 0.3.x adapter used by every stage."""

from __future__ import annotations

import importlib.metadata
import math
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

import numpy as np


def discover_scenes(
    scene_root: str | Path,
    split: str,
    scene_ids: Sequence[str] | None = None,
) -> list[dict[str, str]]:
    root = Path(scene_root).resolve()
    split_root = root / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"HM3D split does not exist: {split_root}")
    requested = set(scene_ids or [])
    records: list[dict[str, str]] = []
    for folder in sorted(path for path in split_root.iterdir() if path.is_dir()):
        short_id = folder.name.split("-", 1)[-1]
        if requested and folder.name not in requested and short_id not in requested:
            continue
        glbs = sorted(folder.glob("*.basis.glb"))
        navmeshes = sorted(folder.glob("*.basis.navmesh"))
        if len(glbs) != 1 or len(navmeshes) != 1:
            continue
        relative = glbs[0].relative_to(root).as_posix()
        records.append(
            {
                "scene_key": folder.name,
                "scene_id": f"hm3d/{relative}",
                "split": split,
                "glb_path": str(glbs[0].resolve()),
                "navmesh_path": str(navmeshes[0].resolve()),
                "glb_relative_path": relative,
                "navmesh_relative_path": navmeshes[0].relative_to(root).as_posix(),
            }
        )
    if requested:
        found = {record["scene_key"] for record in records} | {
            record["scene_key"].split("-", 1)[-1] for record in records
        }
        missing = requested - found
        if missing:
            raise FileNotFoundError(f"Unknown or incomplete HM3D scenes: {sorted(missing)}")
    return records


def resolve_scene_path(scene_root: str | Path, scene_id: str) -> Path:
    path = PurePosixPath(scene_id)
    parts = path.parts[1:] if path.parts and path.parts[0] == "hm3d" else path.parts
    if len(parts) != 3 or path.is_absolute() or ".." in parts:
        raise ValueError(f"Invalid portable HM3D scene id: {scene_id}")
    root = Path(scene_root).resolve()
    result = (root / Path(*parts)).resolve()
    result.relative_to(root)
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("habitat-sim", "habitat-lab", "numpy", "scipy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def make_simulator(
    scene_path: str | Path,
    config: dict[str, Any],
    *,
    with_sensors: bool = True,
    enable_physics: bool = False,
):
    import habitat_sim

    simulator_config = habitat_sim.SimulatorConfiguration()
    simulator_config.scene_id = str(Path(scene_path).resolve())
    simulator_config.enable_physics = bool(enable_physics)
    simulator_config.gpu_device_id = int(config["runtime"]["gpu_device_id"])

    agent_settings = config["agent"]
    agent_config = habitat_sim.agent.AgentConfiguration()
    agent_config.height = float(agent_settings["height"])
    agent_config.radius = float(agent_settings["radius"])
    if with_sensors:
        sensor_specs = []
        erp_settings = config["erp"]
        for uuid, sensor_type in (
            ("erp_rgb", habitat_sim.SensorType.COLOR),
            ("erp_depth", habitat_sim.SensorType.DEPTH),
        ):
            spec = habitat_sim.EquirectangularSensorSpec()
            spec.uuid = uuid
            spec.sensor_type = sensor_type
            spec.resolution = [int(erp_settings["height"]), int(erp_settings["width"])]
            spec.position = np.asarray(
                [0.0, float(agent_settings["sensor_height"]), 0.0], dtype=np.float32
            )
            spec.far = float(erp_settings["max_depth"])
            sensor_specs.append(spec)
        agent_config.sensor_specifications = sensor_specs
    step = float(agent_settings["forward_step_size"])
    turn = float(agent_settings["turn_angle_degrees"])
    agent_config.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=step)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=turn)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=turn)
        ),
    }
    return habitat_sim.Simulator(
        habitat_sim.Configuration(simulator_config, [agent_config])
    )


def set_agent_state(simulator, position: Sequence[float], rotation_xyzw: Sequence[float]):
    import habitat_sim

    state = habitat_sim.AgentState()
    state.position = np.asarray(position, dtype=np.float32)
    state.rotation = habitat_sim.utils.common.quat_from_coeffs(
        np.asarray(rotation_xyzw, dtype=np.float64)
    )
    simulator.get_agent(0).set_state(state, reset_sensors=True)
    return state


def yaw_rotation_xyzw(direction: Sequence[float]) -> list[float]:
    import habitat_sim

    vector = np.asarray(direction, dtype=np.float64)
    if np.linalg.norm(vector[[0, 2]]) < 1e-8:
        vector = np.asarray([0.0, 0.0, -1.0])
    yaw = math.atan2(-float(vector[0]), -float(vector[2]))
    quaternion = habitat_sim.utils.common.quat_from_angle_axis(
        yaw, np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    )
    return habitat_sim.utils.common.quat_to_coeffs(quaternion).astype(float).tolist()


def state_record(state) -> dict[str, list[float]]:
    import habitat_sim

    return {
        "position": np.asarray(state.position, dtype=float).tolist(),
        "rotation_xyzw": habitat_sim.utils.common.quat_to_coeffs(state.rotation)
        .astype(float)
        .tolist(),
    }


def shortest_path(pathfinder, start: Sequence[float], goal: Sequence[float]):
    import habitat_sim

    query = habitat_sim.ShortestPath()
    query.requested_start = np.asarray(start, dtype=np.float32)
    query.requested_end = np.asarray(goal, dtype=np.float32)
    if not pathfinder.find_path(query) or not np.isfinite(query.geodesic_distance):
        return None
    return {
        "distance": float(query.geodesic_distance),
        "points": np.asarray(query.points, dtype=np.float32),
    }


def interpolate_polyline(points: Sequence[Sequence[float]], spacing: float) -> np.ndarray:
    vertices = np.asarray(points, dtype=np.float32)
    if len(vertices) <= 1:
        return vertices.copy()
    result = [vertices[0]]
    for start, end in zip(vertices[:-1], vertices[1:]):
        distance = float(np.linalg.norm(end - start))
        steps = max(1, int(math.ceil(distance / spacing)))
        result.extend(start + (end - start) * (index / steps) for index in range(1, steps + 1))
    return np.asarray(result, dtype=np.float32)


def point_along_polyline(points: Sequence[Sequence[float]], distance: float) -> np.ndarray:
    vertices = np.asarray(points, dtype=np.float32)
    remaining = max(0.0, float(distance))
    for start, end in zip(vertices[:-1], vertices[1:]):
        length = float(np.linalg.norm(end - start))
        if remaining <= length and length > 1e-8:
            return start + (end - start) * (remaining / length)
        remaining -= length
    return vertices[-1].copy()
