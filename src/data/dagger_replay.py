"""Portable DAgger frame poses and model-free Habitat rendering helpers."""

import math
from pathlib import Path


def dagger_source(annotation):
    trajectory_id = annotation.get("trajectory_id", annotation.get("episode_id"))
    if not isinstance(trajectory_id, str):
        raise ValueError("DAgger trajectory_id must be source-prefixed, e.g. r2r_123")
    source, separator, episode_id = trajectory_id.partition("_")
    if source not in {"r2r", "rxr"} or not separator or not episode_id.isdecimal():
        raise ValueError(f"Invalid DAgger trajectory_id: {trajectory_id!r}")
    if annotation.get("source_dataset", source) != source or str(
        annotation.get("source_episode_id", episode_id)
    ) != episode_id:
        raise ValueError(f"DAgger source metadata disagrees with {trajectory_id}")
    return source, int(episode_id)


def portable_scene_id(scene_id, scenes_dir):
    scene = Path(scene_id)
    if scene.is_absolute():
        scene = scene.resolve().relative_to(Path(scenes_dir).resolve())
    if not scene.parts or any(part == ".." for part in scene.parts):
        raise ValueError(f"Invalid scene path: {scene_id!r}")
    return scene.as_posix()


def capture_frame_state(sim):
    from habitat_sim.utils.common import quat_to_coeffs

    def pose(state):
        return {
            "position": [float(value) for value in state.position],
            "rotation": quat_to_coeffs(state.rotation).tolist(),
        }

    state = sim.get_agent_state()
    frame = pose(state)
    frame["rgb_sensor"] = pose(state.sensor_states["rgb"])
    return frame


def capture_replay_config(env, image_size, image_format, jpeg_quality,
                          jpeg_subsampling, png_compress_level):
    import habitat
    import habitat_sim
    from omegaconf import OmegaConf

    config = env.sim.habitat_config
    return {
        "version": 1,
        "rotation_format": "xyzw",
        "rgb_sensor": OmegaConf.to_container(
            config.agents.main_agent.sim_sensors.rgb_sensor, resolve=True,
        ),
        "forward_step_size": float(config.forward_step_size),
        "turn_angle": float(config.turn_angle),
        "allow_sliding": bool(config.habitat_sim_v0.allow_sliding),
        "enable_physics": bool(config.habitat_sim_v0.enable_physics),
        "habitat_version": getattr(habitat, "__version__", "unknown"),
        "habitat_sim_version": getattr(habitat_sim, "__version__", "unknown"),
        "image_size": list(image_size),
        "image_format": image_format,
        "jpeg_quality": int(jpeg_quality),
        "jpeg_subsampling": int(jpeg_subsampling),
        "png_compress_level": int(png_compress_level),
    }


def validate_replay_metadata(annotation):
    """Return False for legacy rows; reject incomplete or invalid pose records."""
    if "frame_states" not in annotation and "replay_config" not in annotation:
        return False
    frames = annotation.get("frame_states")
    config = annotation.get("replay_config")
    if (
        not isinstance(config, dict)
        or config.get("version") != 1
        or config.get("rotation_format") != "xyzw"
        or not isinstance(config.get("rgb_sensor"), dict)
    ):
        raise ValueError("Invalid DAgger replay_config")
    image_size = config.get("image_size")
    if (
        not isinstance(image_size, list)
        or len(image_size) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in image_size)
    ):
        raise ValueError("Invalid replay image_size")
    scene_id = annotation.get("scene_id")
    if not isinstance(scene_id, str) or not scene_id or Path(scene_id).is_absolute():
        raise ValueError("Replay scene_id must be relative to the local scenes_dir")
    portable_scene_id(scene_id, ".")
    dagger_source(annotation)
    if not isinstance(frames, list) or not frames or len(frames) != len(annotation["actions"]):
        raise ValueError("frame_states must contain the initial frame and every non-STOP frame")
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"Invalid frame state at frame {index}")
        for pose in (frame, frame.get("rgb_sensor")):
            if not isinstance(pose, dict):
                raise ValueError(f"Missing RGB pose at frame {index}")
            for key, size in (("position", 3), ("rotation", 4)):
                values = pose.get(key)
                if (
                    not isinstance(values, list) or len(values) != size
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not math.isfinite(v) for v in values)
                ):
                    raise ValueError(f"Invalid {key} at frame {index}")
            if abs(sum(v * v for v in pose["rotation"]) - 1.0) > 1e-3:
                raise ValueError(f"Non-unit rotation at frame {index}")
    return True


def validate_replay_environment(annotation, env_config, episode):
    from omegaconf import OmegaConf

    if not validate_replay_metadata(annotation):
        return
    config = env_config.habitat
    scene_id = portable_scene_id(episode.scene_id, config.dataset.scenes_dir)
    if scene_id != annotation["scene_id"]:
        raise ValueError(f"Replay scene mismatch: {scene_id} != {annotation['scene_id']}")
    rgb_config = OmegaConf.to_container(
        config.simulator.agents.main_agent.sim_sensors.rgb_sensor, resolve=True,
    )
    if rgb_config != annotation["replay_config"]["rgb_sensor"]:
        raise ValueError("Replay RGB camera configuration differs from the collection configuration")


def render_frame_state(sim, frame):
    import numpy as np
    from habitat_sim.utils.common import quat_from_coeffs

    state = sim.get_agent_state()
    state.position = np.asarray(frame["position"], dtype=np.float32)
    state.rotation = quat_from_coeffs(frame["rotation"])
    rgb = state.sensor_states["rgb"]
    rgb.position = np.asarray(frame["rgb_sensor"]["position"], dtype=np.float32)
    rgb.rotation = quat_from_coeffs(frame["rgb_sensor"]["rotation"])
    state.sensor_states = {"rgb": rgb}
    agent = sim.get_agent(sim.habitat_config.default_agent_id)
    agent.set_state(state, reset_sensors=False, infer_sensor_states=False)
    observation = sim.get_observations_at(keep_agent_at_new_pose=True)
    if observation is None or "rgb" not in observation:
        raise RuntimeError("Habitat failed to render the recorded RGB pose")
    return observation
