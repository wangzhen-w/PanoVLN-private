"""Action-sequence utilities for VLN instruction generation."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


ACTION_NAMES = {0: "STOP", 1: "MOVE_FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}
TURN_DEGREES = 15.0


def validate_actions(actions: Sequence[Any], context: str = "actions") -> List[int]:
    if not isinstance(actions, list) or not actions:
        raise ValueError(f"{context} must be a non-empty list")
    try:
        normalized = [int(action) for action in actions]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must contain integer action IDs") from error
    invalid = sorted(set(normalized) - set(ACTION_NAMES))
    if invalid:
        raise ValueError(f"{context} contains invalid action IDs: {invalid}")
    if normalized[-1] != 0 or normalized.count(0) != 1:
        raise ValueError(f"{context} must contain exactly one terminal STOP=0")
    return normalized


def action_runs(actions: Sequence[int]) -> List[Tuple[int, int, int]]:
    """Return contiguous non-STOP action runs as (action, start_index, end_index)."""

    runs: List[Tuple[int, int, int]] = []
    start = 0
    while start < len(actions):
        action = int(actions[start])
        if action == 0:
            break
        end = start
        while end + 1 < len(actions) and int(actions[end + 1]) == action:
            end += 1
        runs.append((action, start, end))
        start = end + 1
    return runs


def translated_frame_indices(actions: Sequence[int], num_frames: int) -> List[int]:
    """Frames that correspond to distinct translated positions.

    `frame_0` is the start. Every non-STOP action produces one next frame, but
    in-place turns do not create new physical waypoints. Instructions should be
    based primarily on translated positions, with the final orientation handled
    separately by endpoint evidence.
    """

    indices = [0]
    for action_index, raw_action in enumerate(actions):
        action = int(raw_action)
        if action == 0:
            break
        if action == 1:
            indices.append(min(action_index + 1, num_frames - 1))
    return sorted(set(index for index in indices if 0 <= index < num_frames))


def select_route_frames(
    actions: Sequence[int], num_frames: int, max_waypoints: int
) -> List[int]:
    movement_frames = translated_frame_indices(actions, num_frames)
    if not movement_frames:
        return [0]
    max_waypoints = max(2, int(max_waypoints))
    if len(movement_frames) <= max_waypoints:
        return movement_frames

    importance: Dict[int, int] = {
        movement_frames[0]: 10_000,
        movement_frames[-1]: 10_000,
    }
    for action, start, end in action_runs(actions):
        if action not in {2, 3}:
            continue
        degrees = int((end - start + 1) * TURN_DEGREES)
        before = next((frame for frame in reversed(movement_frames) if frame <= start), None)
        after = next((frame for frame in movement_frames if frame > end), None)
        if before is None or after is None:
            continue
        importance[before] = max(importance.get(before, 0), degrees)
        importance[after] = max(importance.get(after, 0), degrees)

    selected = {movement_frames[0], movement_frames[-1]}
    rank_by_frame = {frame: index for index, frame in enumerate(movement_frames)}
    while len(selected) < max_waypoints:
        selected_ranks = [rank_by_frame[frame] for frame in selected]
        best_frame = None
        best_score = -1.0
        for frame in movement_frames:
            if frame in selected:
                continue
            rank = rank_by_frame[frame]
            coverage = min(abs(rank - selected_rank) for selected_rank in selected_ranks)
            event_bonus = 1.0 + min(3.0, importance.get(frame, 0) / 60.0)
            score = coverage * event_bonus
            if score > best_score:
                best_score = score
                best_frame = frame
        if best_frame is None:
            break
        selected.add(best_frame)
    return sorted(selected)


def select_start_frames(actions: Sequence[int], num_frames: int, window: int) -> List[int]:
    return translated_frame_indices(actions, num_frames)[: max(1, int(window))]


def select_endpoint_frames(actions: Sequence[int], num_frames: int, window: int) -> List[int]:
    frames = translated_frame_indices(actions, num_frames)
    if num_frames:
        frames = sorted(set(frames + [num_frames - 1]))
    return frames[-max(1, int(window)) :]


def build_heading_by_frame(actions: Sequence[int], num_frames: int) -> List[float]:
    heading = 0.0
    headings = [heading for _ in range(num_frames)]
    frame_index = 0
    for raw_action in actions:
        action = int(raw_action)
        if action == 0:
            break
        if action == 2:
            heading -= TURN_DEGREES
        elif action == 3:
            heading += TURN_DEGREES
        frame_index += 1
        if frame_index >= num_frames:
            break
        headings[frame_index] = heading
    return headings


def major_turns(actions: Sequence[int]) -> List[Dict[str, Any]]:
    turns = []
    for action, start, end in action_runs(actions):
        if action not in {2, 3}:
            continue
        degrees = int(round((end - start + 1) * TURN_DEGREES))
        if degrees < 45:
            continue
        turns.append(
            {
                "action_index_start": start,
                "action_index_end": end,
                "direction": "left" if action == 2 else "right",
                "degrees": degrees,
            }
        )
    return turns


def dead_reckoned_path(actions: Sequence[int], step_size: float = 0.25) -> List[List[float]]:
    """Planar egocentric path from actions, used only for complexity cues."""

    x = z = yaw = 0.0
    points = [[x, z]]
    for raw_action in actions:
        action = int(raw_action)
        if action == 0:
            break
        if action == 2:
            yaw += math.radians(TURN_DEGREES)
        elif action == 3:
            yaw -= math.radians(TURN_DEGREES)
        elif action == 1:
            x += math.sin(yaw) * step_size
            z += math.cos(yaw) * step_size
            points.append([x, z])
    return points


def path_complexity(actions: Sequence[int]) -> Dict[str, Any]:
    points = dead_reckoned_path(actions)
    if len(points) < 2:
        return {
            "dead_reckoned_length_m": 0.0,
            "dead_reckoned_displacement_m": 0.0,
            "dead_reckoned_tortuosity": 0.0,
            "possible_revisit": False,
        }
    array = np.asarray(points, dtype=np.float64)
    segment_lengths = np.linalg.norm(array[1:] - array[:-1], axis=1)
    length = float(segment_lengths.sum())
    displacement = float(np.linalg.norm(array[-1] - array[0]))
    tortuosity = length / max(displacement, 1e-6)
    possible_revisit = False
    if len(array) > 6:
        for left in range(len(array)):
            for right in range(left + 5, len(array)):
                if np.linalg.norm(array[left] - array[right]) < 0.5:
                    possible_revisit = True
                    break
            if possible_revisit:
                break
    return {
        "dead_reckoned_length_m": round(length, 3),
        "dead_reckoned_displacement_m": round(displacement, 3),
        "dead_reckoned_tortuosity": round(tortuosity, 3),
        "possible_revisit": possible_revisit,
    }


def compact_action_summary(actions: Sequence[int], num_frames: int) -> Dict[str, Any]:
    runs = action_runs(actions)
    first = runs[0] if runs else None
    initial = "starts with STOP"
    if first:
        action, start, end = first
        if action == 1:
            initial = "starts by moving forward"
        else:
            initial = (
                f"starts with a {'left' if action == 2 else 'right'} turn of "
                f"about {int((end - start + 1) * TURN_DEGREES)} degrees"
            )
    run_text = []
    for action, start, end in runs[:28]:
        name = ACTION_NAMES[action].replace("MOVE_FORWARD", "forward").replace(
            "TURN_", "turn "
        ).lower()
        run_text.append(f"{name} x{end - start + 1} (actions {start}-{end})")
    if len(runs) > 28:
        run_text.append(f"... {len(runs) - 28} more runs")
    forwards = sum(action == 1 for action in actions)
    turn_actions = sum(action in {2, 3} for action in actions)
    return {
        "low_level_action_count_including_stop": len(actions),
        "forward_action_count": forwards,
        "turn_action_count": turn_actions,
        "translated_position_count": len(translated_frame_indices(actions, num_frames)),
        "initial_motion": initial,
        "major_turns": major_turns(actions),
        "action_runs_compact": run_text,
        "complexity": path_complexity(actions),
    }
