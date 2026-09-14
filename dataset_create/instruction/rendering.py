"""On-demand Habitat replay, paired videos, and clean local review evidence."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dataset_create.logging_utils import quiet_native_logs
from dataset_create.instruction.geometry import (
    COMPASS, Camera, GroundRoute, densify_positions, look_rotation,
    overlay_route,
)
from dataset_create.instruction.segmentation import (
    EvidenceError, Segment, after_distance, before_distance, cumulative_distance,
    decision_spans, sample_frames,
)
from dataset_create.trajectory.hm3d import (
    package_versions, resolve_scene_path, set_agent_state,
    shortest_path, state_record,
)
from dataset_create.trajectory.io_utils import atomic_json_dump
from dataset_create.trajectory.replay import replay_episode_actions


def _font(size=18):
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(path, size) if Path(path).exists() else ImageFont.load_default()


def save_image(image, path, quality=92):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image).astype(np.uint8)).save(path, quality=quality)
    return str(path)


def video_from_frames(paths, destination, fps, hold=0):
    import imageio.v2 as imageio

    if not paths:
        raise EvidenceError("empty_video")
    with imageio.get_writer(str(destination), fps=fps, codec="libx264", quality=8,
                            macro_block_size=1, ffmpeg_log_level="error") as writer:
        for path in paths:
            frame = np.asarray(Image.open(path).convert("RGB"))
            writer.append_data(frame)
        for _ in range(hold):
            writer.append_data(frame)
    return str(destination)


def storyboard(paths, destination, maximum=12):
    indices = np.unique(np.rint(np.linspace(0, len(paths)-1, min(len(paths), maximum))).astype(int))
    width, height, band = 280, 210, 22
    canvas = Image.new("RGB", (width*3, (height+band)*int(math.ceil(len(indices)/3))), (29, 33, 38))
    draw = ImageDraw.Draw(canvas)
    for slot, index in enumerate(indices):
        x, y = slot % 3 * width, slot // 3 * (height+band)
        with Image.open(paths[index]) as frame:
            canvas.paste(frame.resize((width, height)), (x, y+band))
        draw.text((x+5, y+2), f"Frame {index}", font=_font(15), fill="white")
    return save_image(np.asarray(canvas), destination)


def compass_image(views):
    first = views["front"]
    height, width = first.shape[:2]
    band = 28
    canvas = Image.new("RGB", (width * 3, (height + band) * 3), (29, 33, 38))
    draw = ImageDraw.Draw(canvas)
    for name, _, row, col in COMPASS:
        x, y = col * width, row * (height + band)
        canvas.paste(Image.fromarray(views[name]), (x, y + band))
        draw.text((x+8, y+4), name.replace("_", " ").upper(), fill="white", font=_font(16))
    cx, cy = width * 1.5, (height + band) * 1.5
    draw.line((cx, cy+34, cx, cy-34), fill="white", width=6)
    draw.polygon([(cx, cy-50), (cx-16, cy-26), (cx+16, cy-26)], fill="white")
    draw.text((cx, cy+65), "CURRENT HEADING", font=_font(18), fill="white", anchor="mm")
    return np.asarray(canvas)


class EpisodeRenderer:
    """One persistent simulator per process. Never share its GL context between threads."""

    def __init__(self, scene_root, metadata, settings, gpu=0, erp_settings=None):
        self.scene_root = Path(scene_root)
        self.metadata = metadata
        self.settings = settings
        self.gpu = gpu
        self.sim = None
        self.scene_id = None
        self.erp_settings = erp_settings
        self.memory_frames = {}
        recorded = metadata.get("habitat_sim_version")
        installed = package_versions()["habitat-sim"]
        if recorded != installed:
            raise RuntimeError(f"Replay version mismatch: recorded={recorded}, installed={installed}")
        if metadata.get("action_mapping") != {"stop": 0, "move_forward": 1, "turn_left": 2, "turn_right": 3}:
            raise RuntimeError("Unsupported trajectory action mapping")

    def close(self):
        if self.sim is not None:
            with quiet_native_logs():
                self.sim.close()
            self.sim = None

    def load_scene(self, scene_id):
        if self.scene_id == scene_id and self.sim is not None:
            return
        self.close()
        import habitat_sim

        scene = resolve_scene_path(self.scene_root, scene_id)
        parameters = self.metadata["action_parameters"]
        sim_config = habitat_sim.SimulatorConfiguration()
        sim_config.scene_id = str(scene)
        sim_config.gpu_device_id = self.gpu
        sim_config.enable_physics = True
        agent = habitat_sim.agent.AgentConfiguration()
        agent.height = parameters["height"]
        agent.radius = parameters["radius"]
        agent.action_space = {
            name: habitat_sim.agent.ActionSpec(name, habitat_sim.agent.ActuationSpec(amount=amount))
            for name, amount in [("move_forward", parameters["forward_step_size"]),
                                 ("turn_left", parameters["turn_angle_degrees"]),
                                 ("turn_right", parameters["turn_angle_degrees"])]
        }
        specifications = []
        for uuid, kind in [("instruction_rgb", habitat_sim.SensorType.COLOR),
                           ("instruction_depth", habitat_sim.SensorType.DEPTH)]:
            spec = habitat_sim.CameraSensorSpec()
            spec.uuid = uuid
            spec.sensor_type = kind
            spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            spec.resolution = [self.settings["height"], self.settings["width"]]
            spec.position = np.asarray([0., self.metadata["action_parameters"]["sensor_height"], 0.], dtype=np.float32)
            spec.hfov = self.settings["hfov_degrees"]
            spec.orientation = np.asarray([math.radians(self.settings["camera_pitch_degrees"]), 0., 0.], dtype=np.float32)
            spec.near = self.settings["near_m"]
            spec.far = self.settings["far_m"]
            specifications.append(spec)
        agent.sensor_specifications = specifications
        agents = [agent]
        if self.erp_settings is not None:
            erp_agent = habitat_sim.agent.AgentConfiguration()
            erp_agent.height, erp_agent.radius = agent.height, agent.radius
            spec = habitat_sim.EquirectangularSensorSpec()
            spec.uuid = "training_erp"
            spec.sensor_type = habitat_sim.SensorType.COLOR
            spec.resolution = [self.erp_settings["height"], self.erp_settings["width"]]
            spec.position = np.array([0., parameters["sensor_height"], 0.], dtype=np.float32)
            spec.orientation = np.zeros(3, dtype=np.float32)
            erp_agent.sensor_specifications = [spec]
            agents.append(erp_agent)
        with quiet_native_logs():
            self.sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_config, agents))
        navmesh = scene.with_suffix(".navmesh")
        if not navmesh.exists() or not self.sim.pathfinder.load_nav_mesh(str(navmesh)):
            raise EvidenceError(f"navmesh_unavailable:{navmesh}")
        self.scene_id = scene_id

    def replay(self, episode):
        if episode["action_ids"].count(0) != 1:
            raise EvidenceError("trajectory_has_nonterminal_STOP")
        self.load_scene(episode["scene_id"])
        self.memory_frames.clear()
        states, collisions = replay_episode_actions(self.sim, episode)
        if collisions:
            raise EvidenceError(f"replay_collisions:{collisions}")
        checks = [(states[-1], episode["final_position"], episode["final_rotation_xyzw"])]
        checks += [(states[int(e["action_index"])], e["position"], e["rotation_xyzw"])
                   for e in episode["decision_events"]]
        for state, position, rotation in checks:
            error = np.linalg.norm(np.asarray(state["position"]) - position)
            a, b = np.asarray(state["rotation_xyzw"]), np.asarray(rotation)
            cosine = abs(float(a @ b)) / (np.linalg.norm(a) * np.linalg.norm(b))
            angular = math.degrees(2 * math.acos(float(np.clip(cosine, 0, 1))))
            if error > .001 or angular > .01:
                raise EvidenceError(f"replay_pose_mismatch:{error:.6f}m:{angular:.6f}deg")
        remaining = shortest_path(self.sim.pathfinder, states[-1]["position"], episode["goal_position"])
        if remaining is None or remaining["distance"] > self.metadata["action_parameters"]["goal_radius"] + 1e-4:
            raise EvidenceError("replay_did_not_reach_goal")
        return states

    def ground_route(self, states):
        import habitat_sim

        points, tangent, arc = densify_positions([s["position"] for s in states], self.settings["route_sample_m"])
        valid = np.zeros(len(points), dtype=bool)
        points = points.copy()
        for i, point in enumerate(points):
            origin = point + [0., self.settings["ground_ray_above_m"], 0.]
            ray = habitat_sim.geo.Ray(origin.astype(np.float32), np.array([0., -1., 0.], dtype=np.float32))
            result = self.sim.cast_ray(ray, max_distance=self.settings["ground_ray_above_m"] + self.settings["ground_ray_below_m"], buffer_distance=0.)
            hits = sorted(result.hits, key=lambda hit: hit.ray_distance)
            if not hits:
                continue
            hit = hits[0]
            surface = np.asarray(hit.point, dtype=float)
            normal = np.asarray(hit.normal, dtype=float)
            if abs(surface[1] - point[1]) <= self.settings["max_ground_error_m"] and abs(normal[1]) >= self.settings["floor_normal_min_y"]:
                points[i] = surface
                valid[i] = True
        # Unsupported samples are not drawn. Judge the resulting local material
        # by its visibility, not a trajectory-wide fraction of perfect scan mesh.
        return GroundRoute(points, tangent, arc, valid)

    def observe(self, state, bearing=0.):
        rotation = look_rotation(state["rotation_xyzw"], bearing)
        key = tuple(np.round(np.r_[state["position"], rotation], 7))
        if key in self.memory_frames:
            return self.memory_frames[key]
        set_agent_state(self.sim, state["position"], rotation)
        obs = self.sim.get_sensor_observations()
        sensor_state = self.sim.get_agent(0).get_state().sensor_states["instruction_rgb"]
        camera_state = state_record(sensor_state)
        camera = Camera(self.settings["width"], self.settings["height"], self.settings["hfov_degrees"],
                        np.asarray(camera_state["position"]), camera_state["rotation_xyzw"])
        value = (obs["instruction_rgb"][..., :3].copy(), obs["instruction_depth"].copy(), camera)
        # A bounded cache covers neighbouring segments without retaining a whole split.
        if len(self.memory_frames) >= 128:
            self.memory_frames.pop(next(iter(self.memory_frames)))
        self.memory_frames[key] = value
        return value

    def _compass(self, state, route=None):
        clean, marked, observations, pixel_counts = {}, {}, {}, {}
        for name, bearing, _, _ in COMPASS:
            rgb, depth, camera = self.observe(state, bearing)
            clean[name] = rgb
            observations[name] = (depth, camera)
            if route is not None:
                marked[name], stats = overlay_route(rgb, depth, camera, route, self.settings)
                pixel_counts[name] = stats["route_pixels"]
        return clean, marked, observations, {"total": sum(pixel_counts.values()), "per_view": pixel_counts}

    def observe_erp(self, state):
        """Render only the level ERP camera; instruction cameras keep their tilt."""
        import habitat_sim
        from habitat_sim.utils.common import quat_from_coeffs

        pose = habitat_sim.AgentState()
        pose.position = np.asarray(state["position"], dtype=np.float32)
        pose.rotation = quat_from_coeffs(state["rotation_xyzw"])
        self.sim.get_agent(1).set_state(pose, reset_sensors=True)
        return self.sim.get_sensor_observations(agent_ids=1)["training_erp"][..., :3].copy()

    def export_training(self, episode, episode_id, states, root, settings):
        """Finish geometry and ERP work together on the simulator's thread."""
        from dataset_create.instruction.training import export_clean_erp

        path = shortest_path(self.sim.pathfinder, states[0]["position"], states[-1]["position"])
        if path is None:
            raise EvidenceError("no_geodesic_path_to_real_stop")
        return path["distance"], export_clean_erp(self, episode, episode_id, states, root, settings)

    def decision_compass(self, episode, states, segment, decision_id, route, directory, revision=0):
        event = episode["decision_events"][int(decision_id[1:])]
        arc = cumulative_distance(states)
        span = decision_spans(episode, states, self.segmentation_settings)[int(decision_id[1:])]
        # The annotated choice and the branch anchor bound the approach. Graph
        # connection centroids are not necessarily visible architectural entrances.
        latest = min(int(event["action_index"]), before_distance(arc, span["anchor_index"], .4))
        candidates = [latest] + [before_distance(arc, latest, back)
                                 for back in (.35, .7, 1., self.settings["compass_backtrack_m"])]
        candidates = list(dict.fromkeys(candidates))
        if revision:
            candidates = candidates[1:] + candidates[:1]
        for index in candidates:
            future_route = route.subset(arc[index], arc[segment.end])
            clean, marked, _, stats = self._compass(states[index], future_route)
            if stats["total"] < self.settings["min_route_pixels"]:
                continue
            directory.mkdir(parents=True, exist_ok=True)
            return {"decision_id": decision_id, "state_index": index,
                    "position": states[index]["position"], "rotation_xyzw": states[index]["rotation_xyzw"],
                    "anchor_state_index": span["anchor_index"],
                    "clean_compass": save_image(compass_image(clean), directory / f"{decision_id}_clean.jpg"),
                    "marked_compass": save_image(compass_image(marked), directory / f"{decision_id}_route.jpg"),
                    "route_pixels": stats["total"], "route_pixels_per_view": stats["per_view"]}
        raise EvidenceError(f"decision_route_not_visible:{decision_id}")

    def stop_evidence(self, states, directory):
        """The final approach and actual stop define the local destination area."""
        stop = states[-1]
        clean, _, _, _ = self._compass(stop)
        actual = {"image": save_image(compass_image(clean), directory / "stop_actual.jpg"), "state": stop}
        return {"actual": actual}

    def render_segment(self, episode, states, segment, route, output, segmentation_settings, revision=0):
        self.segmentation_settings = segmentation_settings
        directory = Path(output) / segment.segment_id / f"attempt_{revision}"
        directory.mkdir(parents=True, exist_ok=True)
        arc = cumulative_distance(states)
        if revision:
            # A visual repair really changes the upstream evidence: wider clean
            # context and denser motion sampling, keeping the owned route intact.
            segment.context_start = before_distance(arc, segment.start, segmentation_settings["context_overlap_m"] + .75*revision)
            segment.context_end = after_distance(arc, segment.end, segmentation_settings["context_overlap_m"] + .75*revision)
        local_route = route.subset(arc[segment.start], arc[segment.end])
        decisions = [self.decision_compass(episode, states, segment, d, local_route, directory, revision)
                     for d in segment.decision_ids]
        if decisions:
            segment.context_start = min(segment.context_start, min(d["state_index"] for d in decisions))
        sampling = dict(self.settings)
        if revision:
            sampling.update(frame_distance_m=self.settings["frame_distance_m"]*.7,
                            frame_turn_degrees=self.settings["frame_turn_degrees"]*.7,
                            max_video_frames=self.settings["max_video_frames"]+16)
        required = [d["state_index"] for d in decisions]
        indices = sample_frames(segment, states, sampling, required=required)
        clean_paths, marked_paths, cameras, pixel_counts = [], [], [], []
        for j, i in enumerate(indices):
            rgb, depth, camera = self.observe(states[i])
            # Past/future overlap contains clean scenery only; marked route never
            # extends beyond this segment's owned action interval.
            future_route = local_route.subset(max(arc[i], arc[segment.start]), arc[segment.end])
            marked, stats = overlay_route(rgb, depth, camera, future_route, self.settings)
            clean_paths.append(save_image(rgb, directory / f"frame_{j:03d}_clean.jpg", self.settings["jpeg_quality"]))
            marked_paths.append(save_image(marked, directory / f"frame_{j:03d}_route.jpg", self.settings["jpeg_quality"]))
            # Preserve raw depth/calibration for reproducible geometry audits.
            np.savez_compressed(directory / f"frame_{j:03d}_depth.npz", depth=depth)
            cameras.append({"state_index": i, "position": camera.position.tolist(),
                            "rotation_xyzw": camera.rotation_xyzw, "hfov_degrees": camera.hfov_degrees,
                            "width": camera.width, "height": camera.height})
            pixel_counts.append(stats["route_pixels"])
        # Context remains available for audits, but never masquerades as motion
        # owned by this segment. Both native video and frames mode see only core.
        context = {"frame_state_indices": indices, "clean_frames": clean_paths}
        core = [j for j, i in enumerate(indices) if segment.start <= i <= segment.end]
        indices = [indices[j] for j in core]
        clean_paths = [clean_paths[j] for j in core]
        marked_paths = [marked_paths[j] for j in core]
        cameras = [cameras[j] for j in core]
        pixel_counts = [pixel_counts[j] for j in core]
        if max(pixel_counts, default=0) < self.settings["min_route_pixels"]:
            raise EvidenceError(f"route_not_visible:{segment.segment_id}")
        hold = self.settings["stop_hold_frames"] if segment.terminal else 0
        marked_video = video_from_frames(marked_paths, directory / "route.mp4", self.settings["video_fps"], hold)
        clean_video = video_from_frames(clean_paths, directory / "clean.mp4", self.settings["video_fps"], hold)
        evidence = {"segment": segment.to_dict(), "revision": revision, "frame_state_indices": indices,
                    "core_frame_indices": [j for j, i in enumerate(indices) if segment.start <= i <= segment.end],
                    "context": context,
                    "clean_frames": clean_paths, "marked_frames": marked_paths, "cameras": cameras,
                    "clean_video": clean_video, "marked_video": marked_video, "fps": self.settings["video_fps"],
                    "clean_storyboard": storyboard(clean_paths, directory / "clean_storyboard.jpg"),
                    "decisions": decisions, "route_pixels": pixel_counts,
                    "stop": self.stop_evidence(states, directory) if segment.terminal else None}
        atomic_json_dump(evidence, directory / "evidence.json")
        return evidence
