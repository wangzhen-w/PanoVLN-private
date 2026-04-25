from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
from fastdtw import fastdtw
from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.tasks.nav.nav import DistanceToGoal, Success
from numpy import ndarray
from omegaconf import DictConfig


ArrayLike3 = Union[List[float], Tuple[float, float, float], ndarray]


def euclidean_distance(
    pos_a: Union[List[float], ndarray], pos_b: Union[List[float], ndarray]
) -> float:
    return float(np.linalg.norm(np.asarray(pos_b) - np.asarray(pos_a), ord=2))



def _to_point_list(points: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[float(v) for v in p] for p in points]



def _get_goal_position(episode: Any) -> List[float]:
    if hasattr(episode, "goals") and episode.goals:
        goal = episode.goals[0]
        if hasattr(goal, "position") and goal.position is not None:
            return [float(v) for v in goal.position]
    if hasattr(episode, "reference_path") and episode.reference_path:
        last = episode.reference_path[-1]
        return [float(v) for v in last]
    raise ValueError(f"Cannot infer goal position for episode_id={episode.episode_id}")



def _get_sparse_reference_path(episode: Any) -> List[List[float]]:
    ref_path: List[List[float]] = []
    if hasattr(episode, "reference_path") and episode.reference_path is not None:
        ref_path = _to_point_list(episode.reference_path)

    if hasattr(episode, "start_position") and episode.start_position is not None:
        start_position = [float(v) for v in episode.start_position]
        if not ref_path:
            ref_path = [start_position]
        elif euclidean_distance(ref_path[0], start_position) > 1e-4:
            ref_path = [start_position] + ref_path

    goal_position = _get_goal_position(episode)
    if not ref_path:
        ref_path = [goal_position]
    elif euclidean_distance(ref_path[-1], goal_position) > 1e-4:
        ref_path.append(goal_position)

    dedup: List[List[float]] = []
    for point in ref_path:
        if not dedup or euclidean_distance(dedup[-1], point) > 1e-8:
            dedup.append(point)
    return dedup



def _resample_polyline(points: Sequence[Sequence[float]], step_size: float) -> List[List[float]]:
    if not points:
        return []
    if len(points) == 1:
        return [[float(v) for v in points[0]]]

    stride = max(float(step_size), 1e-6)
    dense: List[List[float]] = [[float(v) for v in points[0]]]

    for start, end in zip(points[:-1], points[1:]):
        start_np = np.asarray(start, dtype=np.float64)
        end_np = np.asarray(end, dtype=np.float64)
        segment = end_np - start_np
        seg_len = float(np.linalg.norm(segment, ord=2))
        if seg_len <= 1e-8:
            continue

        num_substeps = int(np.floor(seg_len / stride))
        for idx in range(1, num_substeps + 1):
            alpha = min((idx * stride) / seg_len, 1.0)
            point = (1.0 - alpha) * start_np + alpha * end_np
            point_list = [float(v) for v in point.tolist()]
            if euclidean_distance(dense[-1], point_list) > 1e-8:
                dense.append(point_list)

        end_list = [float(v) for v in end_np.tolist()]
        if euclidean_distance(dense[-1], end_list) > 1e-8:
            dense.append(end_list)

    return dense



def _build_dense_reference_path(sim: Simulator, episode: Any, step_size: float) -> List[List[float]]:
    sparse_points = _get_sparse_reference_path(episode)
    if len(sparse_points) <= 1:
        return sparse_points

    geodesic_polyline: List[List[float]] = []
    for start, end in zip(sparse_points[:-1], sparse_points[1:]):
        try:
            segment_points = sim.get_straight_shortest_path_points(start, end)
            segment_points = _to_point_list(segment_points)
        except Exception:
            segment_points = [list(start), list(end)]

        if not segment_points:
            segment_points = [list(start), list(end)]

        if not geodesic_polyline:
            geodesic_polyline.extend(segment_points)
        else:
            if euclidean_distance(geodesic_polyline[-1], segment_points[0]) <= 1e-8:
                geodesic_polyline.extend(segment_points[1:])
            else:
                geodesic_polyline.extend(segment_points)

    if not geodesic_polyline:
        geodesic_polyline = sparse_points

    dense_points = _resample_polyline(geodesic_polyline, step_size)
    if not dense_points:
        dense_points = geodesic_polyline

    dedup: List[List[float]] = []
    for point in dense_points:
        if not dedup or euclidean_distance(dedup[-1], point) > 1e-8:
            dedup.append(point)
    return dedup



def _robust_geodesic_distance(
    sim: Simulator,
    source: Sequence[float],
    target: Any,
    episode: Optional[Any] = None,
) -> float:
    # Do not pass `episode` into `sim.geodesic_distance` here.
    # HabitatSim reuses `episode._shortest_path_cache` and, when cache exists,
    # ignores newly provided targets (`position_b`). That can silently pin this
    # distance to previously queried endpoints (e.g., distance_to_goal).
    _ = episode
    try:
        dist_val = sim.geodesic_distance(source, target)
        if isinstance(dist_val, np.ndarray):
            dist_val = float(dist_val.item())
        dist_val = float(dist_val)
        if np.isfinite(dist_val):
            return dist_val
    except Exception:
        pass

    if isinstance(target, (list, tuple)) and len(target) > 0 and isinstance(target[0], (list, tuple, np.ndarray)):
        best = float("inf")
        for point in target:
            try:
                dist_val = sim.geodesic_distance(source, point)
                dist_val = float(dist_val)
            except Exception:
                dist_val = euclidean_distance(source, point)
            if np.isfinite(dist_val):
                best = min(best, dist_val)
        if np.isfinite(best):
            return best

    return euclidean_distance(source, target)


class _ReferencePathMixin:
    def __init__(self, *args: Any, sim: Simulator, config: Any, **kwargs: Any):
        self._sim = sim
        self._config = config
        self._dense_reference_path: List[List[float]] = []
        self._reference_path_cache_key: Optional[Tuple[str, str]] = None
        super().__init__()

    def _dense_step_size(self) -> float:
        # Custom measurement configs may be parsed as generic MeasurementConfig
        # under Habitat/Hydra structured config, which does not allow extra keys
        # like dense_step_size in yaml. Fall back to 0.25m, matching the simulator
        # forward step used in this evaluation setup.
        step_size = getattr(self._config, "dense_step_size", None)
        if step_size is None:
            step_size = getattr(self._config, "step_size", None)
        if step_size is None:
            step_size = 0.25
        return float(step_size)

    def _ensure_dense_reference_path(self, episode: Any) -> None:
        cache_key = (str(getattr(episode, "scene_id", "")), str(getattr(episode, "episode_id", "")))
        if self._reference_path_cache_key == cache_key and self._dense_reference_path:
            return
        self._dense_reference_path = _build_dense_reference_path(
            self._sim, episode, self._dense_step_size()
        )
        self._reference_path_cache_key = cache_key


@registry.register_measure
class PathLength(Measure):
    """Path length accumulated along the rollout."""

    cls_uuid: str = "path_length"

    def __init__(self, sim: Simulator, *args: Any, **kwargs: Any):
        self._sim = sim
        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any):
        self._previous_position = self._sim.get_agent_state().position
        self._metric = 0.0

    def update_metric(self, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position
        self._metric += euclidean_distance(current_position, self._previous_position)
        self._previous_position = current_position


@registry.register_measure
class OracleNavigationError(Measure):
    """Minimum distance-to-goal achieved along the rollout."""

    cls_uuid: str = "oracle_navigation_error"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        task.measurements.check_measure_dependencies(self.uuid, [DistanceToGoal.cls_uuid])
        self._metric = float("inf")
        self.update_metric(task=task)

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        distance_to_goal = float(
            task.measurements.measures[DistanceToGoal.cls_uuid].get_metric()
        )
        self._metric = min(float(self._metric), distance_to_goal)


@registry.register_measure
class OracleSuccess(Measure):
    """Whether the rollout ever enters the success radius."""

    cls_uuid: str = "oracle_success"

    def __init__(self, *args: Any, config: Any, **kwargs: Any):
        self._config = config
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        task.measurements.check_measure_dependencies(self.uuid, [DistanceToGoal.cls_uuid])
        self._metric = 0.0
        self.update_metric(task=task)

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        distance_to_goal = float(
            task.measurements.measures[DistanceToGoal.cls_uuid].get_metric()
        )
        success_measure = task.measurements.measures.get(Success.cls_uuid, None)
        success_distance = None
        if success_measure is not None:
            success_distance = getattr(getattr(success_measure, "_config", None), "success_distance", None)
        if success_distance is None:
            success_distance = getattr(self._config, "success_distance", 3.0)
        self._metric = float(bool(self._metric) or distance_to_goal <= float(success_distance))



@registry.register_measure
class PL(Measure):
    r"""Progress ratio = shortest distance from start to goal divided by agent path length."""

    def __init__(self, sim: Simulator, config: DictConfig, *args: Any, **kwargs: Any):
        self._previous_position: Union[None, np.ndarray, List[float]] = None
        self._start_end_episode_distance: Optional[float] = None
        self._agent_episode_distance: Optional[float] = None
        self._sim = sim
        self._config = config
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "pl"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid, Success.cls_uuid]
        )
        self._previous_position = self._sim.get_agent_state().position
        self._agent_episode_distance = 0.0
        self._start_end_episode_distance = float(
            task.measurements.measures[DistanceToGoal.cls_uuid].get_metric()
        )
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position
        self._agent_episode_distance += euclidean_distance(
            current_position, self._previous_position
        )
        self._previous_position = current_position
        self._metric = float(self._start_end_episode_distance) / max(
            float(self._start_end_episode_distance), float(self._agent_episode_distance)
        )


@registry.register_measure
class StepsTaken(Measure):
    """Counts how many actions were taken. STOP also counts."""

    cls_uuid: str = "steps_taken"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any):
        self._metric = 0.0

    def update_metric(self, *args: Any, **kwargs: Any):
        self._metric += 1.0


@registry.register_measure
class DistanceToReferencePath(_ReferencePathMixin, Measure):
    """Geodesic distance from current agent position to the dense GT reference path."""

    cls_uuid: str = "distance_to_reference_path"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, episode: Any, **kwargs: Any):
        self._ensure_dense_reference_path(episode)
        self.update_metric(episode=episode)

    def update_metric(self, *args: Any, episode: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position.tolist()
        self._metric = _robust_geodesic_distance(
            self._sim, current_position, self._dense_reference_path, episode=episode
        )


@registry.register_measure
class NDTW(_ReferencePathMixin, Measure):
    """Normalized Dynamic Time Warping against the dense GT reference path."""

    cls_uuid: str = "ndtw"

    def __init__(self, *args: Any, sim: Simulator, config: DictConfig, **kwargs: Any):
        self._agent_positions: List[List[float]] = []
        super().__init__(*args, sim=sim, config=config, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, episode: Any, **kwargs: Any):
        self._ensure_dense_reference_path(episode)
        current_position = self._sim.get_agent_state().position.tolist()
        self._agent_positions = [[float(v) for v in current_position]]
        self.update_metric(episode=episode)

    def update_metric(self, *args: Any, episode: Any, **kwargs: Any):
        current_position = [float(v) for v in self._sim.get_agent_state().position.tolist()]
        if not self._agent_positions or euclidean_distance(self._agent_positions[-1], current_position) > 1e-8:
            self._agent_positions.append(current_position)

        if not self._agent_positions or not self._dense_reference_path:
            self._metric = 0.0
            return

        def _dist(a: Sequence[float], b: Sequence[float]) -> float:
            return _robust_geodesic_distance(self._sim, a, b, episode=episode)

        dtw_distance = float(fastdtw(self._agent_positions, self._dense_reference_path, dist=_dist)[0])
        success_measure = kwargs.get("task").measurements.measures.get(Success.cls_uuid, None) if kwargs.get("task") is not None else None
        success_distance = None
        if success_measure is not None:
            success_distance = getattr(getattr(success_measure, "_config", None), "success_distance", None)
        if success_distance is None:
            success_distance = getattr(self._config, "success_distance", 3.0)
        normalizer = max(len(self._dense_reference_path) * float(success_distance), 1e-8)
        self._metric = float(np.exp(-dtw_distance / normalizer))


@registry.register_measure
class SDTW(Measure):
    """Success weighted nDTW."""

    cls_uuid: str = "sdtw"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [Success.cls_uuid, NDTW.cls_uuid]
        )
        self.update_metric(task=task)

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        success = float(task.measurements.measures[Success.cls_uuid].get_metric())
        ndtw = float(task.measurements.measures[NDTW.cls_uuid].get_metric())
        self._metric = success * ndtw
