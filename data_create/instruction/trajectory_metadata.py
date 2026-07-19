"""Trajectory metadata helpers for instruction grounding."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence


def _path_y_values(path: Any) -> list[float]:
    if not isinstance(path, Sequence) or isinstance(path, (str, bytes)):
        return []
    values: list[float] = []
    for point in path:
        if (
            isinstance(point, Sequence)
            and not isinstance(point, (str, bytes))
            and len(point) >= 3
        ):
            try:
                values.append(float(point[1]))
            except (TypeError, ValueError):
                return []
    return values


def vertical_motion_from_reference_path(path: Any) -> Dict[str, Any]:
    """Derive coarse vertical motion from a Habitat-style `[x, y, z]` path."""

    y_values = _path_y_values(path)
    if len(y_values) < 2:
        return {}
    start_y = y_values[0]
    end_y = y_values[-1]
    delta = end_y - start_y
    total_abs = sum(abs(right - left) for left, right in zip(y_values, y_values[1:]))
    if delta > 0.35:
        motion = "ascending"
    elif delta < -0.35:
        motion = "descending"
    elif total_abs > 0.75:
        motion = "mixed"
    else:
        motion = "level"
    return {
        "vertical_motion": motion,
        "vertical_delta_m": round(delta, 3),
        "start_y": round(start_y, 3),
        "end_y": round(end_y, 3),
        "min_y": round(min(y_values), 3),
        "max_y": round(max(y_values), 3),
        "total_abs_vertical_change_m": round(total_abs, 3),
    }


def enrich_trajectory_metadata(metadata: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Return metadata with derived physical constraints added when possible."""

    enriched: Dict[str, Any] = dict(metadata or {})
    if not enriched.get("vertical_motion"):
        enriched.update(vertical_motion_from_reference_path(enriched.get("reference_path")))
    return enriched
