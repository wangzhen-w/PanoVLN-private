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


def action_runs_in_span(
    actions: Sequence[int],
    start_action: int,
    end_action_exclusive: int,
) -> List[Tuple[int, int]]:
    """Return contiguous action runs inside an action-index span.

    The returned tuples are `(action, count)` and exclude STOP. This compact
    representation is intended as evidence for instruction generation, not as
    publishable text.
    """

    stop_index = next((index for index, action in enumerate(actions) if int(action) == 0), len(actions))
    start = max(0, min(int(start_action), stop_index))
    end = max(start, min(int(end_action_exclusive), stop_index))
    runs: List[Tuple[int, int]] = []
    cursor = start
    while cursor < end:
        action = int(actions[cursor])
        if action == 0:
            break
        count = 1
        while cursor + count < end and int(actions[cursor + count]) == action:
            count += 1
        runs.append((action, count))
        cursor += count
    return runs


def summarize_action_span(
    actions: Sequence[int],
    *,
    start_frame: int,
    end_frame: int,
    step_size: float = 0.25,
) -> Dict[str, Any]:
    """Summarize actions that carry the agent from one saved frame to another.

    Saved frame `k` is the observation after action `k - 1`; therefore actions
    `[start_frame, end_frame)` connect two frame labels. The summary helps the
    VLM distinguish true translation from camera/heading changes in panoramic
    evidence.
    """

    start = int(start_frame)
    end = int(end_frame)
    runs = action_runs_in_span(actions, start, end)
    forward = sum(count for action, count in runs if action == 1)
    left = sum(count for action, count in runs if action == 2)
    right = sum(count for action, count in runs if action == 3)
    turn_count = left + right
    run_text = []
    for action, count in runs[:16]:
        if action == 1:
            run_text.append(f"forward x{count}")
        elif action == 2:
            run_text.append(f"left turn x{count}")
        elif action == 3:
            run_text.append(f"right turn x{count}")
    if len(runs) > 16:
        run_text.append(f"... {len(runs) - 16} more runs")
    mostly_orientation = turn_count >= 4 and forward <= 2
    return {
        "from_frame_label": start,
        "to_frame_label": end,
        "forward_action_count": forward,
        "approx_translation_m": round(forward * step_size, 2),
        "left_turn_degrees": int(round(left * TURN_DEGREES)),
        "right_turn_degrees": int(round(right * TURN_DEGREES)),
        "mostly_orientation_or_alignment": mostly_orientation,
        "compact_runs": run_text,
    }


def segment_action_summary(
    actions: Sequence[int],
    frames: Sequence[int],
    *,
    step_size: float = 0.25,
) -> Dict[str, Any]:
    """Summarize real low-level motion for a selected visual segment."""

    clean_frames = list(dict.fromkeys(int(frame) for frame in frames))
    transitions = [
        summarize_action_span(
            actions,
            start_frame=left,
            end_frame=right,
            step_size=step_size,
        )
        for left, right in zip(clean_frames, clean_frames[1:])
    ]
    forward = sum(int(item["forward_action_count"]) for item in transitions)
    left_degrees = sum(int(item["left_turn_degrees"]) for item in transitions)
    right_degrees = sum(int(item["right_turn_degrees"]) for item in transitions)
    mostly_orientation_intervals = [
        {
            "from_frame_label": item["from_frame_label"],
            "to_frame_label": item["to_frame_label"],
            "approx_translation_m": item["approx_translation_m"],
        }
        for item in transitions
        if item["mostly_orientation_or_alignment"]
    ]
    return {
        "frame_labels": clean_frames,
        "forward_action_count": forward,
        "approx_translation_m": round(forward * step_size, 2),
        "left_turn_degrees_total": left_degrees,
        "right_turn_degrees_total": right_degrees,
        "mostly_orientation_or_alignment_intervals": mostly_orientation_intervals,
        "transition_summaries": transitions,
    }


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


def navigation_cues(actions: Sequence[int]) -> List[Dict[str, Any]]:
    """Return coarse route-shape cues that are easy to lose in long summaries.

    These cues are not publishable instructions by themselves.  They only tell
    the vision-language stages where the low-level trajectory contains a large
    heading change that may correspond to a route choice, a staircase alignment,
    or a final approach orientation.  The VLM must still ground any wording in
    the visual evidence.
    """

    runs = action_runs(actions)
    cues: List[Dict[str, Any]] = []
    first_forward_run = next(
        (index for index, (action, _, _) in enumerate(runs) if action == 1),
        None,
    )
    if first_forward_run is not None:
        left = right = 0
        start_index = None
        end_index = None
        for action, start, end in runs[:first_forward_run]:
            if action not in {2, 3}:
                continue
            start_index = start if start_index is None else min(start_index, start)
            end_index = end if end_index is None else max(end_index, end)
            if action == 2:
                left += end - start + 1
            else:
                right += end - start + 1
        if left + right >= 3:
            dominant = "left" if left >= right else "right"
            degrees = int(round((left if dominant == "left" else right) * TURN_DEGREES))
            cues.append(
                {
                    "kind": "initial_alignment_before_first_translation",
                    "action_index_start": start_index,
                    "action_index_end": end_index,
                    "dominant_direction": dominant,
                    "dominant_turn_degrees": degrees,
                    "use": (
                        "Mention this only if visual evidence shows it chooses the first "
                        "hallway, doorway, stair, or room direction."
                    ),
                }
            )

    for turn in major_turns(actions):
        cues.append(
            {
                "kind": "major_turn_between_translations",
                **turn,
                "use": (
                    "Ground this in the nearby visual waypoint when it corresponds to a "
                    "real route choice or space transition."
                ),
            }
        )

    last_forward_end = None
    for action, _, end in runs:
        if action == 1:
            last_forward_end = end
    if last_forward_end is not None:
        tail_runs = [
            (action, start, end)
            for action, start, end in runs
            if start > last_forward_end and action in {2, 3}
        ]
        left = sum(end - start + 1 for action, start, end in tail_runs if action == 2)
        right = sum(end - start + 1 for action, start, end in tail_runs if action == 3)
        if left + right >= 3:
            dominant = "left" if left >= right else "right"
            cues.append(
                {
                    "kind": "final_alignment_after_last_translation",
                    "action_index_start": tail_runs[0][1],
                    "action_index_end": tail_runs[-1][2],
                    "dominant_direction": dominant,
                    "dominant_turn_degrees": int(
                        round((left if dominant == "left" else right) * TURN_DEGREES)
                    ),
                    "use": (
                        "Use endpoint visual evidence for the destination semantics; do not turn "
                        "a side/back view into the destination."
                    ),
                }
            )
    return cues


def decision_boundary_cues(actions: Sequence[int]) -> List[Dict[str, Any]]:
    """Return coarse action-derived places where route instructions need care.

    The cues deliberately avoid semantic room names because actions alone cannot
    identify spaces.  They tell the VLM where visual evidence should be checked
    for a real doorway, stair landing, junction, room boundary, or final
    approach.  They are evidence for auditing route coverage, not text to copy.
    """

    runs = action_runs(actions)
    cues: List[Dict[str, Any]] = []
    cue_id = 1
    first_forward_run = next(
        (index for index, (action, _, _) in enumerate(runs) if action == 1),
        None,
    )
    if first_forward_run is not None:
        pre_turns = [
            (action, start, end)
            for action, start, end in runs[:first_forward_run]
            if action in {2, 3}
        ]
        turn_count = sum(end - start + 1 for _, start, end in pre_turns)
        if turn_count >= 3:
            cues.append(
                {
                    "cue_id": cue_id,
                    "kind": "initial_route_choice_before_translation",
                    "action_index_start": pre_turns[0][1],
                    "action_index_end": pre_turns[-1][2],
                    "guidance": (
                        "Check start evidence for the first real hallway, doorway, stair, "
                        "or room direction. Mention it only if visually grounded."
                    ),
                }
            )
            cue_id += 1

    for turn in major_turns(actions):
        cues.append(
            {
                "cue_id": cue_id,
                "kind": "possible_decision_boundary_after_translation",
                "action_index_start": turn["action_index_start"],
                "action_index_end": turn["action_index_end"],
                "direction": turn["direction"],
                "degrees": turn["degrees"],
                "guidance": (
                    "Check nearby translated views for a grounded branch, doorway, "
                    "stair/landing transition, or room boundary. Do not publish the "
                    "side direction if the evidence is ambiguous."
                ),
            }
        )
        cue_id += 1

    last_forward_end = None
    for action, _, end in runs:
        if action == 1:
            last_forward_end = end
    if last_forward_end is not None:
        tail_turns = [
            (action, start, end)
            for action, start, end in runs
            if start > last_forward_end and action in {2, 3}
        ]
        turn_count = sum(end - start + 1 for _, start, end in tail_turns)
        if turn_count >= 3:
            cues.append(
                {
                    "cue_id": cue_id,
                    "kind": "final_local_alignment_after_translation",
                    "action_index_start": tail_turns[0][1],
                    "action_index_end": tail_turns[-1][2],
                    "guidance": (
                        "Use FINAL STOP evidence for the endpoint. This may be only a "
                        "local facing change near the stop, not a route choice."
                    ),
                }
            )
    return cues


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
        "navigation_cues": navigation_cues(actions),
        "decision_boundary_cues": decision_boundary_cues(actions),
        "action_runs_compact": run_text,
        "complexity": path_complexity(actions),
    }
