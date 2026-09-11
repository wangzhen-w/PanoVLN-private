"""Partition at navigation events, never at an arbitrary frame/length budget.

State i is the pose AFTER actions[:i]. Segment core intervals share a boundary
state, but own disjoint action intervals [start, end). Context is visual only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


class EvidenceError(ValueError):
    """An upstream geometry, segmentation or visibility problem; do not rewrite."""


def rotation_matrix(xyzw) -> np.ndarray:
    q = np.asarray(xyzw, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError("Invalid xyzw quaternion")
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def yaw_degrees(xyzw) -> float:
    direction = rotation_matrix(xyzw) @ np.array([0., 0., -1.])
    return float(np.degrees(np.arctan2(direction[0], -direction[2])))


def camera_motion(cameras):
    """Measured changes between video frames, without interpreting scene landmarks."""
    headings = np.degrees(np.unwrap(np.radians([yaw_degrees(c["rotation_xyzw"]) for c in cameras])))
    positions = np.asarray([c["position"] for c in cameras])
    result = []
    for i in range(1, len(cameras)):
        turn = round(float(headings[i] - headings[i-1]), 1)
        result.append({"frames": [i-1, i],
                       "turn": "right" if turn > 0 else "left" if turn < 0 else "none",
                       "degrees": abs(turn),
                       "travel_m": round(float(np.linalg.norm(positions[i] - positions[i-1])), 2),
                       "rise_m": round(float(positions[i, 1] - positions[i-1, 1]), 2),
                       "height_from_start_m": round(float(positions[i, 1] - positions[0, 1]), 2)})
    return result


def angle_difference(a, b):
    return (a - b + 180.) % 360. - 180.


def cumulative_distance(states) -> np.ndarray:
    positions = np.asarray([s["position"] for s in states], dtype=float)
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]


def before_distance(arc, index, distance):
    # First state at the chosen distance includes rotations at the same position.
    chosen = max(0, min(index, int(np.searchsorted(arc, arc[index] - distance, side="right")) - 1))
    return int(np.searchsorted(arc, arc[chosen], side="left"))


def after_distance(arc, index, distance):
    return min(len(arc) - 1, max(index, int(np.searchsorted(arc, arc[index] + distance, side="right"))))


@dataclass
class Segment:
    segment_id: str
    start: int
    end: int
    context_start: int
    context_end: int
    decision_ids: list[str]
    kinds: list[str]
    terminal: bool = False

    def to_dict(self):
        return asdict(self)


def decision_spans(episode, states, settings):
    positions = np.asarray([s["position"] for s in states])
    arc = cumulative_distance(states)
    spans = []
    for number, event in enumerate(episode["decision_events"]):
        index = int(event["action_index"])
        anchor = np.asarray(event["selected"]["anchor_position"])
        # Do not match a later return to the same spatial point.
        next_index = min(
            [int(e["action_index"]) for e in episode["decision_events"]
             if int(e["action_index"]) > index] or [len(states) - 1]
        )
        distances = np.linalg.norm(positions[index:next_index + 1] - anchor, axis=1)
        entry = index + int(np.argmin(distances))
        if float(distances[entry-index]) > settings["max_anchor_route_distance_m"]:
            raise EvidenceError(f"decision_anchor_off_route:d{number}")
        # An anchor is inside a branch. Protect approach, entry and settling.
        start = before_distance(arc, index, settings["decision_approach_m"])
        end = after_distance(arc, entry, settings["decision_exit_m"])
        spans.append({"start": start, "end": end, "decision_id": f"d{number}",
                      "event_index": index, "anchor_index": entry})
    return spans


def segment_episode(episode: dict[str, Any], states, settings) -> list[Segment]:
    if len(states) != len(episode["action_ids"]) + 1:
        raise EvidenceError("state_action_count_mismatch")
    arc = cumulative_distance(states)
    last = len(states) - 1
    decisions_by_id = {d["decision_id"]: d for d in decision_spans(episode, states, settings)}
    spans = [(d["start"], d["end"], [d["decision_id"]], ["decision"])
             for d in decisions_by_id.values()]
    # Group nearby rotational actions; small corrective yaw is not an event.
    turns = [i for i, action in enumerate(episode["action_ids"]) if action in (2, 3)]
    clusters = []
    for index in turns:
        if clusters and arc[index] - arc[clusters[-1][-1]] <= settings["turn_cluster_m"]:
            clusters[-1].append(index)
        else:
            clusters.append([index])
    for cluster in clusters:
        first, end = cluster[0], cluster[-1] + 1
        heading = [yaw_degrees(s["rotation_xyzw"]) for s in states[first:end + 1]]
        excursion = np.ptp(np.unwrap(np.radians(heading))) * 180 / np.pi
        if excursion >= settings["turn_threshold_degrees"]:
            spans.append((before_distance(arc, first, settings["turn_approach_m"]),
                          after_distance(arc, end, settings["turn_exit_m"]), [], ["turn"]))
    spans.append((before_distance(arc, last, settings["arrival_approach_m"]), last, [], ["arrival"]))
    merged = []
    for start, end, decisions, kinds in sorted(spans):
        if merged and (start <= merged[-1][1] or
                       arc[start] - arc[merged[-1][1]] < settings["minimum_transit_m"]):
            old = merged[-1]
            merged[-1] = (old[0], max(end, old[1]), old[2] + decisions, old[3] + kinds)
        else:
            merged.append((start, end, decisions, kinds))
    cores = []
    cursor = 0
    for start, end, decisions, kinds in merged:
        if start > cursor:
            if arc[start] - arc[cursor] < settings["minimum_transit_m"]:
                start = cursor
            else:
                cores.append((cursor, start, [], ["transit"]))
        cores.append((start, end, decisions, kinds))
        cursor = end
    # Expanded approach/turn windows can join distinct, already completed choices.
    # Split those at the preceding choice's settled exit, while retaining the
    # shared approach as visual context. No decision itself is cut in half.
    natural_cores = []
    for start, end, decisions, kinds in cores:
        decisions = sorted(set(decisions), key=lambda d: int(d[1:]))
        cuts = [start]
        completed = start
        for left, right in zip(decisions, decisions[1:]):
            completed = max(completed, decisions_by_id[left]["end"])
            if cuts[-1] < completed < end and completed <= decisions_by_id[right]["event_index"]:
                cuts.append(completed)
        cuts.append(end)
        for lo, hi in zip(cuts, cuts[1:]):
            owned = [d for d in decisions if lo <= decisions_by_id[d]["event_index"] < hi]
            piece_kinds = ["decision"] if owned else ["transit"]
            headings = np.unwrap(np.radians([yaw_degrees(s["rotation_xyzw"]) for s in states[lo:hi+1]]))
            if np.degrees(np.ptp(headings)) >= settings["turn_threshold_degrees"]:
                piece_kinds.append("turn")
            if hi == last:
                piece_kinds.append("arrival")
            natural_cores.append((lo, hi, owned, piece_kinds))
    result = []
    for i, (start, end, decisions, kinds) in enumerate(natural_cores):
        result.append(Segment(
            f"s{i:03d}", start, end,
            min([before_distance(arc, start, settings["context_overlap_m"])] +
                [decisions_by_id[d]["start"] for d in decisions]),
            after_distance(arc, end, settings["context_overlap_m"]),
            sorted(set(decisions), key=lambda d: int(d[1:])), sorted(set(kinds)), end == last,
        ))
    validate_segments(result, episode, states, settings)
    return result


def validate_segments(segments, episode, states, settings):
    if not segments or segments[0].start != 0 or segments[-1].end != len(states)-1:
        raise EvidenceError("incomplete_segment_coverage")
    if [s.terminal for s in segments] != [False] * (len(segments)-1) + [True]:
        raise EvidenceError("terminal_segment_must_own_STOP")
    if any(a.end != b.start for a, b in zip(segments, segments[1:])):
        raise EvidenceError("gap_or_repeated_actions")
    owned = [d for s in segments for d in s.decision_ids]
    if sorted(owned, key=lambda x: int(x[1:])) != [f"d{i}" for i in range(len(episode["decision_events"]))]:
        raise EvidenceError("decision_ownership_mismatch")
    for span in decision_spans(episode, states, settings):
        owner = next(s for s in segments if span["decision_id"] in s.decision_ids)
        if (owner.context_start > span["start"] or owner.start > span["event_index"] or
                owner.end < span["end"]):
            raise EvidenceError("decision_cut_midway")


def sample_frames(segment, states, settings, required=()):
    arc = cumulative_distance(states)
    selected = {segment.context_start, segment.start, segment.end, segment.context_end, *required}
    last = segment.context_start
    for i in range(last + 1, segment.context_end + 1):
        if (arc[i] - arc[last] >= settings["frame_distance_m"] or
                abs(angle_difference(yaw_degrees(states[i]["rotation_xyzw"]),
                                     yaw_degrees(states[last]["rotation_xyzw"]))) >= settings["frame_turn_degrees"]):
            selected.add(i)
            last = i
    selected = sorted(i for i in selected if segment.context_start <= i <= segment.context_end)
    # A model budget must not silently erase turns or manufacture a fixed cut.
    if len(selected) > settings["max_video_frames"]:
        raise EvidenceError(f"video_budget_exceeded:{segment.segment_id}:{len(selected)}")
    return selected


def merge_for_repair(segments, segment_id):
    """Repair an unnatural cut by merging with one adjacent piece, preserving ownership."""
    index = next(i for i, s in enumerate(segments) if s.segment_id == segment_id)
    if len(segments) < 2:
        raise EvidenceError("no_adjacent_segment_for_repair")
    other = index - 1 if index > 0 else index + 1
    lo, hi = sorted([index, other])
    left, right = segments[lo], segments[hi]
    combined = Segment(left.segment_id + "m", left.start, right.end,
                       left.context_start, right.context_end,
                       left.decision_ids + right.decision_ids,
                       sorted(set(left.kinds + right.kinds)), right.terminal)
    return segments[:lo] + [combined] + segments[hi+1:], combined
