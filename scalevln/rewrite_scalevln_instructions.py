#!/usr/bin/env python3
"""Rewrite ScaleVLN instructions from trajectory panoramas.

The script keeps the training-facing JSONL schema unchanged:
{"episode_id": ..., "instruction": ..., "actions": [...]}

Generation metadata, raw model responses, contact sheets, and failures are
written under a progress directory next to the clean output JSONL by default.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
import os
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


DEFAULT_INPUT_JSONL = (
    "/workspace/code_dir/a_property/dataset/PanoVLN/sub_dataset/scalevln.jsonl"
)
DEFAULT_IMAGE_ROOT = "/workspace/code_dir/a_property/dataset/PanoVLN/images/scalevln"
DEFAULT_OUTPUT_JSONL = (
    "/workspace/code_dir/a_property/dataset/PanoVLN/sub_dataset/"
    "scalevln_qwen35_27b_r2rstyle.jsonl"
)
DEFAULT_QWEN_BASE_URL = "http://127.0.0.1:10426/v1"
DEFAULT_QWEN_MODEL = "Qwen3.5-27B"
DEFAULT_QWEN_API_KEY = "test"

ACTION_NAMES = {
    0: "stop",
    1: "forward",
    2: "left",
    3: "right",
}
TURN_DEGREES = 15.0

DATA_ARTIFACT_PATTERNS = (
    re.compile(r"\bimage\b", re.IGNORECASE),
    re.compile(r"\bpanorama\b", re.IGNORECASE),
    re.compile(r"contact\s+sheet", re.IGNORECASE),
    re.compile(r"\brow\s+\d+\b", re.IGNORECASE),
    re.compile(r"\bstep\s+\d+\b", re.IGNORECASE),
    re.compile(r"\bframe[_\s-]?\d+\b", re.IGNORECASE),
    re.compile(r"\baction\s+sequence\b", re.IGNORECASE),
)

OBJECT_POSITION_TERMS = (
    "stove",
    "refrigerator",
    "fridge",
    "table",
    "chair",
    "chairs",
    "couch",
    "sofa",
    "bed",
    "bookshelf",
    "bookshelves",
    "shelf",
    "shelves",
    "cabinet",
    "cabinets",
    "island",
    "bench",
    "piano",
    "dresser",
    "fireplace",
    "window",
    "mirror",
    "mirrors",
    "picture",
    "painting",
    "plant",
    "rug",
    "counter",
    "desk",
    "desks",
    "armchair",
    "armchairs",
    "furniture",
    "appliance",
    "appliances",
    "artwork",
    "wall art",
    "vase",
    "vases",
    "wardrobe",
    "vanity",
    "bathtub",
    "shower",
    "toilet",
    "staircase",
    "stairs",
)

GENERIC_ENDPOINT_LANDMARK_WORDS = {
    "floor",
    "floors",
    "wall",
    "walls",
    "ceiling",
    "ceilings",
    "room",
    "area",
    "space",
    "surface",
    "carpet",
    "tile",
    "tiles",
    "wooden",
    "hardwood",
    "grey",
    "gray",
    "white",
    "black",
    "large",
    "small",
    "open",
    "final",
    "visible",
}

_THREAD_LOCAL = threading.local()


@dataclass
class RenderedRoute:
    selected_frames: List[int]
    heading_by_frame: List[float]
    jpeg_bytes: bytes
    saved_path: Optional[str]
    start_frames: List[int]
    start_jpeg_bytes: bytes
    saved_start_path: Optional[str]
    endpoint_frames: List[int]
    endpoint_jpeg_bytes: bytes
    saved_endpoint_path: Optional[str]


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if "episode_id" not in row or "instruction" not in row or "actions" not in row:
                raise ValueError(f"{path}:{line_number} is not a ScaleVLN row")
            rows.append(row)
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    os.replace(tmp_path, output_path)
    return count


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def episode_id_key(row: Dict[str, Any]) -> str:
    return str(row["episode_id"])


def normalize_instruction(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace(" ,", ",").replace(" .", ".")
    text = text.replace(" ;", ";").replace(" :", ":")
    if text and text[-1] not in ".!?":
        text += "."
    if text:
        text = text[0].upper() + text[1:]
        text = re.sub(
            r"(^|[.!?]\s+)([a-z])",
            lambda match: match.group(1) + match.group(2).upper(),
            text,
        )
    return text


def strip_uncertain_object_positions(text: str) -> str:
    object_pattern = "|".join(re.escape(term) for term in OBJECT_POSITION_TERMS)
    side_suffix = r"(?:on|to) (?:the |your )?(?:left|right)(?: side)?(?: initially)?"

    def remove_suffix(match: re.Match[str]) -> str:
        return match.group(1).rstrip(" ,")

    cleaned = re.sub(
        rf"(\b(?:{object_pattern})\b[^.!?]{{0,70}}?)\s+{side_suffix}\b",
        remove_suffix,
        text,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        rf"\bon your (?:left|right),?\s+(?=(?:[^.!?]{{0,50}}\b(?:{object_pattern})\b))",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        rf"\bon the (?:left|right),?\s+(?=(?:[^.!?]{{0,50}}\b(?:{object_pattern})\b))",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        rf",?\s+(?:keeping|with)\s+(?:the\s+)?(?:{object_pattern})"
        rf"[^.!?]{{0,40}}?\s+to your (?:left|right)\b",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        rf"(\b(?:{object_pattern})\b[^.!?]{{0,70}}?)\s+to your (?:left|right)\b",
        remove_suffix,
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        rf"(\b(?:{object_pattern})\b[^.!?]{{0,70}}?)\s+to the (?:left|right)\b",
        remove_suffix,
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        rf"\bto your (?:left|right),?\s+(?=(?:[^.!?]{{0,50}}\b(?:{object_pattern})\b))",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        rf"\bto the (?:left|right),?\s+(?=(?:[^.!?]{{0,50}}\b(?:{object_pattern})\b))",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return normalize_instruction(cleaned)


def polish_instruction_artifacts(text: str) -> str:
    cleaned = re.sub(r"\bvisible\s+visible\b", "visible", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bthen\s+Continue\b", "then continue", cleaned)
    cleaned = re.sub(
        r"\bturn\s+(left|right)\s+to\s+face\s+the\s+([^,.!?]{3,60}?)(?:\s+directly)?"
        r"(?=[,.!?]| and| then)",
        lambda match: f"turn {match.group(1)} toward the {match.group(2).strip()}",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\bturn\s+(left|right)\s+to\s+align\s+with\s+the\s+([^,.!?]{3,70}?)(?:\s+directly)?"
        r"(?=[,.!?]| and| then)",
        lambda match: f"turn {match.group(1)} toward the {match.group(2).strip()}",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\bturn\s+(left|right)\s+toward\s+the\s+(?:open\s+)?living\s+(?:room|area),?\s+"
        r"then\s+continue\s+(?:forward\s+)?toward\b",
        lambda match: f"turn {match.group(1)} and continue toward",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^(?:walk|go|move|proceed)\s+past\s+the\s+([^,.!?]{3,70}?\bpainting\b)\s+"
        r"and\s+turn\s+(left|right)\b",
        lambda match: f"Start near the {match.group(1).strip()} and turn {match.group(2)}",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^turn\s+(left|right)\s+and\s+walk\s+past\s+the\s+([^,.!?]{3,70}?\bpainting\b)",
        lambda match: f"Start near the {match.group(2).strip()} and turn {match.group(1)}",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\bstop near the ([^,.!?]{2,60}?) visible\b",
        lambda match: f"stop near the {match.group(1).strip()}",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r",\s+with\s+([^,.!?]{3,70}?),\s+and\s+stop\b",
        lambda match: f" with {match.group(1).strip()} visible, and stop",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\bvisible\s+(?:on|to) (?:the |your )?(?:left|right)(?: side)?"
        r"(?: initially)?\b",
        "visible",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b(walk|continue|proceed|move) forward, with\b",
        lambda match: f"{match.group(1)} forward with",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\bwith\s+(the\s+[^,.!?]{3,60}?),\s+(toward|towards)\b",
        lambda match: f"with {match.group(1).strip()} visible, {match.group(2)}",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\bthrough the doorway leaving\b",
        "through the doorway, leaving",
        cleaned,
        flags=re.IGNORECASE,
    )
    return normalize_instruction(cleaned)


def fix_dangling_keeping(text: str) -> str:
    def replacement(match: re.Match[str]) -> str:
        article = match.group(1) or ""
        phrase = match.group(2).strip()
        phrase = re.sub(
            r"\s+(?:on|to) (?:the |your )?(?:left|right)(?: side)?"
            r"(?: initially)?\s*$",
            "",
            phrase,
            flags=re.IGNORECASE,
        ).strip()
        return f"with {article}{phrase} visible"

    cleaned = re.sub(
        r"\bkeeping\s+(the\s+)?([^.!?]{3,80}?)(?=[.!?])",
        replacement,
        text,
        flags=re.IGNORECASE,
    )
    return normalize_instruction(cleaned)


def strip_numeric_turn_angles(text: str) -> str:
    cleaned = re.sub(
        r"\bturn\s+(left|right)\s+(?:about|approximately|around)?\s*\d{1,3}\s*degrees\b",
        lambda match: f"turn {match.group(1)}",
        text,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\bmake\s+a\s+\d{1,3}\s*degree\s+(left|right)\s+turn\b",
        lambda match: f"turn {match.group(1)}",
        cleaned,
        flags=re.IGNORECASE,
    )
    return normalize_instruction(cleaned)


def clean_generated_instruction(text: str) -> str:
    cleaned = strip_numeric_turn_angles(text)
    cleaned = fix_dangling_keeping(cleaned)
    cleaned = strip_uncertain_object_positions(cleaned)
    cleaned = polish_instruction_artifacts(cleaned)
    return normalize_instruction(cleaned)


def extract_destination_hint(instruction: str) -> str:
    """Extract the most explicit original destination phrase, if one exists."""
    text = normalize_instruction(instruction)
    if not text:
        return ""

    clauses = re.split(r"(?<=[.!?])\s+|,\s+|;\s+|\s+then\s+", text, flags=re.IGNORECASE)
    destination_patterns = [
        r"\b(?:stop|stopping|wait|halt|finish|end|stand|remain)\b[^.!?;]*",
        r"\b(?:destination|goal)\b[^.!?;]*",
    ]
    matches: List[str] = []
    for clause in clauses:
        clause = clause.strip(" .")
        if not clause:
            continue
        for pattern in destination_patterns:
            match = re.search(pattern, clause, flags=re.IGNORECASE)
            if match:
                matches.append(match.group(0).strip(" ."))
                break
    if not matches:
        return ""

    hint = matches[-1]
    words = hint.split()
    if len(words) > 24:
        hint = " ".join(words[:24])
    return hint


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text))


def action_runs(actions: Sequence[int]) -> List[Tuple[int, int, int]]:
    runs = []
    run_action = None
    run_start = 0
    last_non_stop = -1
    for index, raw_action in enumerate(actions):
        action = int(raw_action)
        if action == 0:
            break
        last_non_stop = index
        if run_action is None:
            run_action = action
            run_start = index
            continue
        if action != run_action:
            runs.append((run_action, run_start, index - 1))
            run_action = action
            run_start = index
    if run_action is not None:
        runs.append((run_action, run_start, last_non_stop))
    return runs


def compact_action_summary(actions: Sequence[int], max_runs: int = 14) -> str:
    counts = Counter(int(action) for action in actions)
    runs = action_runs(actions)
    run_parts = []
    for action, start, end in runs[:max_runs]:
        length = end - start + 1
        name = ACTION_NAMES.get(action, str(action))
        if action == 1:
            run_parts.append(f"move forward for {length} step{'s' if length != 1 else ''}")
        elif action in (2, 3):
            direction = "left" if action == 2 else "right"
            degrees = int(length * TURN_DEGREES)
            run_parts.append(f"turn {direction} about {degrees} degrees")
        else:
            run_parts.append(f"{name} x{length}")
    if len(runs) > max_runs:
        run_parts.append(f"... plus {len(runs) - max_runs} more movement runs")

    totals = (
        f"forward={counts.get(1, 0)}, left_turns={counts.get(2, 0)}, "
        f"right_turns={counts.get(3, 0)}, stop={counts.get(0, 0)}"
    )
    return f"Action totals: {totals}. Movement outline: " + "; ".join(run_parts) + "."


def action_turn_requirements(
    actions: Sequence[int],
    min_degrees: int = 45,
    max_turns: int = 10,
) -> List[Dict[str, Any]]:
    requirements: List[Dict[str, Any]] = []
    runs = action_runs(actions)
    for run_index, (action, start, end) in enumerate(runs):
        if action not in {2, 3}:
            continue
        degrees = int((end - start + 1) * TURN_DEGREES)
        if degrees < min_degrees:
            continue
        direction = "left" if action == 2 else "right"
        if run_index == 0:
            route_phase = "initial turn"
        elif run_index >= max(0, len(runs) - 3):
            route_phase = "late route turn or final alignment"
        else:
            route_phase = "middle route turn"
        requirements.append(
            {
                "turn_steps": f"{start}-{end}",
                "direction": direction,
                "degrees": degrees,
                "route_phase": route_phase,
            }
        )
    return requirements[:max_turns]


def action_turn_requirements_to_prompt(actions: Sequence[int]) -> str:
    requirements = action_turn_requirements(actions)
    if not requirements:
        return (
            "No action-derived major turns of 45 degrees or more. Minor 15-30 "
            "degree turns are usually heading adjustments; mention them only when "
            "the images show a real route choice."
        )
    lines = [
        "Major action-derived turns from the ground-truth action list. These are "
        "higher priority than the original weak instruction. The planner should "
        "map them to visible route choices and the writer should include them "
        "unless they are only same-area heading alignment:"
    ]
    for index, item in enumerate(requirements, start=1):
        lines.append(
            f"{index}. steps {item['turn_steps']}: turn {item['direction']} "
            f"about {item['degrees']} degrees ({item['route_phase']})"
        )
    return "\n".join(lines)


def summarize_action_interval(actions: Sequence[int], start_frame: int, end_frame: int) -> str:
    interval = [int(action) for action in actions[start_frame:end_frame] if int(action) != 0]
    if not interval:
        return "observe without moving"
    parts = []
    for action, run_start, run_end in action_runs(interval):
        length = run_end - run_start + 1
        if action == 1:
            parts.append(f"move forward {length} step{'s' if length != 1 else ''}")
        elif action == 2:
            degrees = int(length * TURN_DEGREES)
            parts.append(f"turn left about {degrees} degrees")
        elif action == 3:
            degrees = int(length * TURN_DEGREES)
            parts.append(f"turn right about {degrees} degrees")
    return ", then ".join(parts)


def build_segmented_route_plan(
    actions: Sequence[int],
    selected_frames: Sequence[int],
) -> str:
    if len(selected_frames) < 2:
        return "1. Single waypoint only; describe the visible destination conservatively."
    lines = []
    for segment_index, (start_frame, end_frame) in enumerate(
        zip(selected_frames[:-1], selected_frames[1:]),
        start=1,
    ):
        action_text = summarize_action_interval(actions, start_frame, end_frame)
        lines.append(
            f"{segment_index}. waypoint step {start_frame} -> step {end_frame}: {action_text}"
        )
    return "\n".join(lines)


def build_route_constraints(actions: Sequence[int]) -> str:
    runs = action_runs(actions)
    if not runs:
        return "The route has no movement before stop."

    constraints = []
    first_action, _, first_end = runs[0]
    if first_action == 1:
        constraints.append(
            "The route starts by moving forward; do not begin the instruction with "
            "'turn around' or an immediate major turn."
        )
    elif first_action in (2, 3):
        direction = "left" if first_action == 2 else "right"
        degrees = int((first_end + 1) * TURN_DEGREES)
        if degrees >= 120:
            constraints.append(
                f"The route starts with a large {direction} turn of about {degrees} "
                "degrees; 'turn around' is acceptable only for this initial turn."
            )
        else:
            constraints.append(
                f"The route starts with a {direction} turn of about {degrees} degrees; "
                "do not call it a full turn around."
            )

    constraints.append(
        "Use left/right for route choices and turns, not for uncertain furniture or "
        "appliance positions."
    )
    constraints.append(
        "If the old instruction says a turn that conflicts with the action-derived "
        "start constraint or the visible trajectory, follow the trajectory."
    )
    major_turns = action_turn_requirements(actions)
    if major_turns:
        turn_text = "; ".join(
            f"steps {item['turn_steps']} turn {item['direction']} about {item['degrees']} degrees"
            for item in major_turns
        )
        constraints.append(
            "The action list contains these major route turns: "
            f"{turn_text}. Preserve them as natural left/right navigation cues "
            "unless visual evidence shows a turn is only same-area heading alignment."
        )
    return "\n".join(f"- {constraint}" for constraint in constraints)


def sorted_frame_paths(image_root: str, episode_id: str) -> List[Path]:
    episode_dir = Path(image_root) / str(episode_id)
    if not episode_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {episode_dir}")

    def frame_index(path: Path) -> int:
        match = re.search(r"frame_(\d+)\.", path.name)
        if not match:
            raise ValueError(f"Unexpected frame filename: {path}")
        return int(match.group(1))

    paths = sorted(episode_dir.glob("frame_*.jpg"), key=frame_index)
    if not paths:
        raise FileNotFoundError(f"No frame_*.jpg images under {episode_dir}")
    return paths


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


def select_route_frames(actions: Sequence[int], num_frames: int, max_waypoints: int) -> List[int]:
    max_waypoints = max(2, int(max_waypoints))
    last_frame = max(0, num_frames - 1)
    candidates = {0, last_frame}

    frame_index = 0
    previous_action = None
    forward_run_start = None
    forward_run_length = 0

    for action_index, raw_action in enumerate(actions):
        action = int(raw_action)
        if action == 0:
            break

        if previous_action is not None and action != previous_action:
            candidates.add(min(frame_index, last_frame))

        if action in (2, 3):
            candidates.add(min(frame_index, last_frame))
            candidates.add(min(frame_index + 1, last_frame))
            forward_run_start = None
            forward_run_length = 0
        elif action == 1:
            if forward_run_start is None:
                forward_run_start = frame_index
                forward_run_length = 0
            forward_run_length += 1
            if forward_run_length in {4, 8, 14, 22}:
                candidates.add(min(frame_index + 1, last_frame))

        frame_index += 1
        previous_action = action

    if len(candidates) < max_waypoints:
        evenly_spaced = np.linspace(0, last_frame, num=min(max_waypoints, last_frame + 1))
        candidates.update(int(round(value)) for value in evenly_spaced)

    ordered = sorted(index for index in candidates if 0 <= index <= last_frame)
    if len(ordered) <= max_waypoints:
        return ordered

    selected = {ordered[0], ordered[-1]}
    interior = ordered[1:-1]
    slots = max_waypoints - 2
    if slots > 0 and interior:
        positions = np.linspace(0, len(interior) - 1, num=slots)
        selected.update(interior[int(round(position))] for position in positions)
    return sorted(selected)


def select_endpoint_frames(num_frames: int, endpoint_window_frames: int) -> List[int]:
    last_frame = max(0, num_frames - 1)
    window = max(1, int(endpoint_window_frames))
    first_frame = max(0, last_frame - window + 1)
    return list(range(first_frame, last_frame + 1))


def select_start_frames(num_frames: int, start_window_frames: int) -> List[int]:
    if num_frames <= 0:
        return []
    window = max(1, int(start_window_frames))
    last_frame = min(num_frames - 1, window - 1)
    return list(range(0, last_frame + 1))


def equirect_to_perspective(
    image: Image.Image,
    yaw_degrees: float,
    pitch_degrees: float = 0.0,
    hfov_degrees: float = 90.0,
    width: int = 256,
    height: int = 192,
) -> Image.Image:
    if cv2 is None:
        return image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)

    source = np.asarray(image.convert("RGB"))
    source_h, source_w = source.shape[:2]

    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y = np.linspace(-height / width, height / width, height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x, -y)

    focal = 1.0 / math.tan(math.radians(hfov_degrees) / 2.0)
    dirs = np.stack([grid_x, grid_y, np.full_like(grid_x, focal)], axis=-1)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    yaw = math.radians(yaw_degrees)
    pitch = math.radians(pitch_degrees)
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    cos_pitch, sin_pitch = math.cos(pitch), math.sin(pitch)

    x_cam = dirs[..., 0]
    y_cam = dirs[..., 1]
    z_cam = dirs[..., 2]

    y_pitch = cos_pitch * y_cam - sin_pitch * z_cam
    z_pitch = sin_pitch * y_cam + cos_pitch * z_cam
    x_pitch = x_cam

    x_world = cos_yaw * x_pitch + sin_yaw * z_pitch
    z_world = -sin_yaw * x_pitch + cos_yaw * z_pitch
    y_world = y_pitch

    lon = np.arctan2(x_world, z_world)
    lat = np.arcsin(np.clip(y_world, -1.0, 1.0))

    map_x = ((lon / (2.0 * math.pi)) + 0.5) * source_w
    map_y = (0.5 - lat / math.pi) * source_h

    projected = cv2.remap(
        source,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )
    return Image.fromarray(projected, mode="RGB")


def draw_label(tile: Image.Image, label: str, label_height: int = 24) -> Image.Image:
    canvas = Image.new("RGB", (tile.width, tile.height + label_height), (245, 245, 245))
    canvas.paste(tile, (0, label_height))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, tile.width, label_height), fill=(31, 35, 40))
    draw.text((8, 5), label, fill=(255, 255, 255))
    return canvas


def render_view_sheet(
    frame_paths: Sequence[Path],
    frames: Sequence[int],
    heading_by_frame: Sequence[float],
    columns: Sequence[Tuple[str, float, float]],
    tile_width: int,
    tile_height: int,
    jpeg_quality: int,
    final_frame_prefix: str = "step",
) -> bytes:
    rows = []
    gap = 4
    last_selected_frame = frames[-1] if frames else -1
    for frame_index in frames:
        with Image.open(frame_paths[frame_index]) as panorama:
            heading = heading_by_frame[min(frame_index, len(heading_by_frame) - 1)]
            tiles = []
            for label, yaw_offset, pitch in columns:
                view = equirect_to_perspective(
                    panorama,
                    yaw_degrees=heading + yaw_offset,
                    pitch_degrees=pitch,
                    width=tile_width,
                    height=tile_height,
                )
                if frame_index == last_selected_frame and final_frame_prefix:
                    frame_label = f"FINAL {frame_index}"
                else:
                    frame_label = f"step {frame_index}"
                tiles.append(draw_label(view, f"{frame_label} - {label}"))
            row_width = sum(tile.width for tile in tiles) + gap * (len(tiles) - 1)
            row_height = max(tile.height for tile in tiles)
            row_canvas = Image.new("RGB", (row_width, row_height), (255, 255, 255))
            x_offset = 0
            for tile in tiles:
                row_canvas.paste(tile, (x_offset, 0))
                x_offset += tile.width + gap
            rows.append(row_canvas)

    sheet_width = max(row_image.width for row_image in rows)
    sheet_height = sum(row_image.height for row_image in rows) + gap * (len(rows) - 1)
    sheet = Image.new("RGB", (sheet_width, sheet_height), (255, 255, 255))
    y_offset = 0
    for row_image in rows:
        sheet.paste(row_image, (0, y_offset))
        y_offset += row_image.height + gap

    buffer = io.BytesIO()
    sheet.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    return buffer.getvalue()


def render_contact_sheet(
    row: Dict[str, Any],
    image_root: str,
    work_dir: str,
    max_waypoints: int,
    start_window_frames: int,
    endpoint_window_frames: int,
    tile_width: int,
    tile_height: int,
    jpeg_quality: int,
    save_contact_sheets: bool,
    use_action_heading: bool,
) -> RenderedRoute:
    episode_id = episode_id_key(row)
    actions = [int(action) for action in row["actions"]]
    frame_paths = sorted_frame_paths(image_root, episode_id)
    if use_action_heading:
        heading_by_frame = build_heading_by_frame(actions, len(frame_paths))
    else:
        heading_by_frame = [0.0 for _ in frame_paths]
    selected_frames = select_route_frames(actions, len(frame_paths), max_waypoints)
    start_frames = select_start_frames(len(frame_paths), start_window_frames)
    endpoint_frames = select_endpoint_frames(len(frame_paths), endpoint_window_frames)

    route_columns = [("left", -70.0, 0.0), ("forward", 0.0, 0.0), ("right", 70.0, 0.0)]
    start_columns = [
        ("left", -90.0, 0.0),
        ("forward", 0.0, 0.0),
        ("right", 90.0, 0.0),
        ("back", 180.0, 0.0),
        ("forward-down", 0.0, 25.0),
    ]
    endpoint_columns = [
        ("left", -90.0, 0.0),
        ("forward", 0.0, 0.0),
        ("right", 90.0, 0.0),
        ("forward-down", 0.0, 25.0),
    ]
    jpeg_bytes = render_view_sheet(
        frame_paths=frame_paths,
        frames=selected_frames,
        heading_by_frame=heading_by_frame,
        columns=route_columns,
        tile_width=tile_width,
        tile_height=tile_height,
        jpeg_quality=jpeg_quality,
    )
    start_jpeg_bytes = render_view_sheet(
        frame_paths=frame_paths,
        frames=start_frames,
        heading_by_frame=heading_by_frame,
        columns=start_columns,
        tile_width=tile_width,
        tile_height=tile_height,
        jpeg_quality=jpeg_quality,
        final_frame_prefix="start",
    )
    endpoint_jpeg_bytes = render_view_sheet(
        frame_paths=frame_paths,
        frames=endpoint_frames,
        heading_by_frame=heading_by_frame,
        columns=endpoint_columns,
        tile_width=tile_width,
        tile_height=tile_height,
        jpeg_quality=jpeg_quality,
    )

    saved_path = None
    saved_start_path = None
    saved_endpoint_path = None
    if save_contact_sheets:
        contact_dir = Path(work_dir) / "contact_sheets"
        contact_dir.mkdir(parents=True, exist_ok=True)
        saved_path = str(contact_dir / f"episode_{episode_id}.jpg")
        with open(saved_path, "wb") as handle:
            handle.write(jpeg_bytes)
        start_dir = Path(work_dir) / "start_sheets"
        start_dir.mkdir(parents=True, exist_ok=True)
        saved_start_path = str(start_dir / f"episode_{episode_id}.jpg")
        with open(saved_start_path, "wb") as handle:
            handle.write(start_jpeg_bytes)
        endpoint_dir = Path(work_dir) / "endpoint_sheets"
        endpoint_dir.mkdir(parents=True, exist_ok=True)
        saved_endpoint_path = str(endpoint_dir / f"episode_{episode_id}.jpg")
        with open(saved_endpoint_path, "wb") as handle:
            handle.write(endpoint_jpeg_bytes)

    return RenderedRoute(
        selected_frames=selected_frames,
        heading_by_frame=heading_by_frame,
        jpeg_bytes=jpeg_bytes,
        saved_path=saved_path,
        start_frames=start_frames,
        start_jpeg_bytes=start_jpeg_bytes,
        saved_start_path=saved_start_path,
        endpoint_frames=endpoint_frames,
        endpoint_jpeg_bytes=endpoint_jpeg_bytes,
        saved_endpoint_path=saved_endpoint_path,
    )


def save_contact_sheet_bytes(work_dir: str, episode_id: str, jpeg_bytes: bytes, subdir: str) -> str:
    output_dir = Path(work_dir) / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"episode_{episode_id}.jpg"
    with output_path.open("wb") as handle:
        handle.write(jpeg_bytes)
    return str(output_path)


def image_data_url(jpeg_bytes: bytes) -> str:
    encoded = base64.b64encode(jpeg_bytes).decode("ascii")
    return "data:image/jpeg;base64," + encoded


def build_endpoint_fact_messages(
    row: Dict[str, Any],
    route: RenderedRoute,
) -> List[Dict[str, Any]]:
    old_instruction = normalize_instruction(row["instruction"])
    destination_hint = extract_destination_hint(old_instruction) or "none"
    endpoint_frames = ", ".join(str(index) for index in route.endpoint_frames)
    endpoint_action_summary = summarize_action_interval(
        row["actions"],
        route.endpoint_frames[0],
        route.endpoint_frames[-1],
    )
    prompt = f"""
You are checking only the destination area of an indoor VLN trajectory.
The image is an endpoint contact sheet. Rows are chronological final approach frames;
the last row is the final observation. Each row shows left, forward, right, and
a slightly downward forward view. Use the last row most heavily for the destination.

Original weak instruction:
{old_instruction}

Original destination hint:
{destination_hint}

Endpoint frames: {endpoint_frames}
Final approach action summary: {endpoint_action_summary}

Return JSON only:
{{
  "visible_destination_area": "short conservative description of the final area",
  "certain_landmarks": ["landmarks clearly visible near the final stop"],
  "uncertain_or_avoid": ["object or spatial claims that should not be stated confidently"],
  "stop_surface": "floor|step|landing|threshold|unknown",
  "surface_evidence": "what the final forward-down view shows about the stopping surface",
  "recommended_stop_phrase": "a safe phrase for where the navigator should stop",
  "old_destination_supported": "yes|no|unclear",
  "preserve_original_destination": true
}}

Rules:
- Be conservative. If the endpoint is visually ambiguous, preserve only the
  generic area type from the original hint. Do not preserve exact doorway,
  closet, appliance, landing, top/bottom-of-stairs, or numbered-step wording
  unless the final observation clearly supports it.
- First decide whether the original destination hint is supported by the final
  observation. Do not let the old text override the final visual evidence.
- Set preserve_original_destination to true only when old_destination_supported is "yes".
- Use the forward-down view only to decide stopping surface and boundary facts:
  floor, step, threshold, landing/platform, or immediately near stairs. Do not
  use side views alone to decide "on the stairs", "in the doorway", "on a
  landing", or an exact step count.
- If the old hint says a doorway, closet, second step, landing, or a specific
  appliance but the final left/forward/right/forward-down views do not clearly
  support it, set old_destination_supported to "no" or "unclear" and recommend
  a safer generic stop phrase.
- For stairs, first decide whether the final observation is on visible steps, on
  a flat landing/platform, or merely near the stair entrance. Recommend
  "stop on the stairs" only when the forward/down view clearly shows the agent
  stopped on steps. If stair position is ambiguous, prefer "stop near the stairs"
  or "stop by the stairs"; do not force "on the stairs".
- Do not recommend "bottom of the stairs", "top of the stairs", "landing", or a
  numbered step unless the final observation makes that exact position clear.
- Do not name a closet, doorway, refrigerator, stove, second step, landing, or
  hallway end unless it is clearly visible or explicitly present in the original
  destination hint.
- For patio, balcony, porch, exterior, or doorway endpoints, distinguish
  "through the doorway into the outdoor/patio area" from "stop just inside/near
  the doorway with the outdoor area visible ahead." If the final forward-down
  view still shows the threshold/interior floor and the agent has not clearly
  moved outside, recommend a boundary-safe phrase such as "stop near the patio
  doorway" or "stop just inside the doorway."
- Avoid object left/right positions unless unmistakable in the final observation.
- The recommended stop phrase should be usable inside a human navigation instruction.
- If the final stop has clear nearby landmarks, make recommended_stop_phrase
  concrete enough to anchor the endpoint, for example "stop near the white
  wardrobe" or "stop near the sofa", not just "stop in the room". Use generic
  area wording only when no reliable endpoint landmark is visible.
""".strip()
    return [
        {
            "role": "system",
            "content": "You are a conservative VLN destination verifier. Return JSON only.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url(route.endpoint_jpeg_bytes)},
                },
            ],
        },
    ]


def endpoint_facts_to_prompt(endpoint_facts: Optional[Dict[str, Any]]) -> str:
    if not endpoint_facts:
        return "No separate endpoint facts are available."
    safe_facts = {
        "visible_destination_area": endpoint_facts.get("visible_destination_area", ""),
        "certain_landmarks": endpoint_facts.get("certain_landmarks", []),
        "uncertain_or_avoid": endpoint_facts.get("uncertain_or_avoid", []),
        "stop_surface": endpoint_facts.get("stop_surface", "unknown"),
        "surface_evidence": endpoint_facts.get("surface_evidence", ""),
        "recommended_stop_phrase": endpoint_facts.get("recommended_stop_phrase", ""),
        "old_destination_supported": endpoint_facts.get(
            "old_destination_supported", "unclear"
        ),
        "preserve_original_destination": endpoint_facts.get(
            "preserve_original_destination", True
        ),
    }
    return json.dumps(safe_facts, ensure_ascii=False, indent=2)


def landmark_significant_tokens(text: str) -> List[str]:
    tokens = []
    for token in re.findall(r"[a-z]+", str(text).lower()):
        if len(token) < 4:
            continue
        if token in GENERIC_ENDPOINT_LANDMARK_WORDS:
            continue
        tokens.append(token)
    return tokens


def concrete_endpoint_landmarks(
    endpoint_facts: Optional[Dict[str, Any]],
) -> List[str]:
    if not endpoint_facts:
        return []
    landmarks = endpoint_facts.get("certain_landmarks") or []
    if not isinstance(landmarks, list):
        return []
    concrete = []
    for item in landmarks:
        landmark = re.sub(r"\s+", " ", str(item).strip().lower())
        landmark = landmark.strip(" .,:;")
        if not landmark:
            continue
        if not landmark_significant_tokens(landmark):
            continue
        concrete.append(landmark)
    return concrete


def phrase_mentions_any_landmark(text: str, landmarks: Sequence[str]) -> bool:
    tokens = set(re.findall(r"[a-z]+", str(text).lower()))
    for landmark in landmarks:
        significant = landmark_significant_tokens(landmark)
        if significant and any(token in tokens for token in significant):
            return True
    return False


def generic_stop_phrase(text: str) -> bool:
    lower = str(text).strip().lower()
    return bool(
        re.search(
            r"\bstop\s+(?:in|at|by|near|around)?\s*(?:the\s+)?"
            r"(?:(?:center|middle)\s+of\s+the\s+)?"
            r"(?:room|area|space|open area|living room|family room)\b",
            lower,
        )
        or re.search(r"\b(?:center|middle)\s+of\s+the\s+room\b", lower)
    )


def preferred_endpoint_stop_phrase(
    endpoint_facts: Optional[Dict[str, Any]],
) -> str:
    landmarks = concrete_endpoint_landmarks(endpoint_facts)
    if not landmarks:
        return ""
    preferred_terms = (
        "wardrobe",
        "sofa",
        "couch",
        "armchair",
        "chair",
        "table",
        "island",
        "vanity",
        "mirror",
        "bathtub",
        "shower",
        "toilet",
        "shelves",
        "shelf",
        "cabinet",
        "television",
        "tv",
        "fireplace",
        "painting",
        "artwork",
        "staircase",
        "stairs",
        "window",
        "door",
    )
    ordered = sorted(
        enumerate(landmarks),
        key=lambda item: (
            0 if any(term in item[1] for term in preferred_terms) else 1,
            item[0],
        ),
    )
    landmark = ordered[0][1]
    words = landmark.split()
    if len(words) > 6:
        landmark = " ".join(words[:6])
    if landmark.startswith(("the ", "a ", "an ")):
        return f"stop near {landmark}"
    return f"stop near the {landmark}"


def sanitize_endpoint_facts(endpoint_facts: Dict[str, Any]) -> Dict[str, Any]:
    facts = dict(endpoint_facts)
    if not isinstance(facts.get("certain_landmarks"), list):
        facts["certain_landmarks"] = []
    if not isinstance(facts.get("uncertain_or_avoid"), list):
        facts["uncertain_or_avoid"] = []
    combined = " ".join(
        [
            str(facts.get("visible_destination_area", "")),
            " ".join(str(item) for item in (facts.get("certain_landmarks") or [])),
            " ".join(str(item) for item in (facts.get("uncertain_or_avoid") or [])),
        ]
    ).lower()
    recommended = str(facts.get("recommended_stop_phrase", "")).lower()
    support = str(facts.get("old_destination_supported", "")).lower()
    if "stair" in combined and any(
        term in recommended
        for term in ("bottom of the stairs", "top of the stairs", "landing", "second step")
    ):
        if support != "yes" or any(
            term in combined for term in ("second step", "landing", "hallway end")
        ):
            facts["recommended_stop_phrase"] = "stop near the stairs"
            facts["visible_destination_area"] = "stair area"
            if support != "yes":
                facts["preserve_original_destination"] = False
    recommended = str(facts.get("recommended_stop_phrase", "")).strip()
    preferred = preferred_endpoint_stop_phrase(facts)
    if (
        preferred
        and generic_stop_phrase(recommended)
        and not phrase_mentions_any_landmark(recommended, concrete_endpoint_landmarks(facts))
    ):
        facts["recommended_stop_phrase"] = preferred
    return facts


def build_start_fact_messages(
    row: Dict[str, Any],
    route: RenderedRoute,
) -> List[Dict[str, Any]]:
    old_instruction = normalize_instruction(row["instruction"])
    start_frames = ", ".join(str(index) for index in route.start_frames)
    start_action_summary = summarize_action_interval(
        row["actions"],
        route.start_frames[0] if route.start_frames else 0,
        route.start_frames[-1] if route.start_frames else 0,
    )
    route_constraints = build_route_constraints(row["actions"])
    prompt = f"""
You are the start-transition agent in a multi-agent VLN instruction generation
pipeline. Inspect only the beginning of the route and decide whether the final
instruction must mention an initial departure transition.

The image is a start contact sheet. Rows are chronological initial frames; each
row has left, forward, right, back, and forward-down views. The first row is the
initial observation. Use the first few rows to decide if the agent starts inside
a room, at a doorway/threshold, in a hallway, in an open area, or on/near stairs.

Original weak instruction:
{old_instruction}

Start frames: {start_frames}
Initial action summary: {start_action_summary}
Route start constraints:
{route_constraints}

Return JSON only:
{{
  "visible_start_area": "short description of the start area",
  "departure_transition": "what the agent does when leaving the start area, or 'none'",
  "certain_landmarks": ["start landmarks that are clearly visible"],
  "uncertain_or_avoid": ["start claims that are uncertain"],
  "recommended_start_phrase": "safe phrase for the beginning of the final instruction",
  "must_preserve_start_boundary": true
}}

Rules:
- If the initial view shows the agent inside a small room or threshold and the
  route leaves through a doorway/opening, set must_preserve_start_boundary=true.
- If the exact room type is uncertain, do not use a specific label like bathroom
  or bedroom. Use a safe generic phrase such as "Exit the room" or "Leave the
  room through the doorway".
- The recommended_start_phrase must obey the initial action summary and route
  start constraints. If the route starts with a right turn, do not copy a left
  turn from the original weak instruction; if it starts with a left turn, do not
  copy a right turn.
- If the start is already in an open living area or hallway with no meaningful
  room/door boundary, set must_preserve_start_boundary=false and recommend the
  first safe movement phrase.
- If the start is on stairs, at the top/bottom of stairs, or immediately
  descending/ascending stairs, say that directly. Do not call it a hallway just
  because a hall or door is visible ahead.
- A landmark visible on a wall is not automatically something the route passes.
  If the route turns before or beside a mirror, picture, console, or similar
  landmark, avoid "passing the ..." and prefer a neutral phrase such as "near"
  or omit the landmark.
- Do not delete a visible exit/leave/walk-out transition just because the old
  room name is uncertain.
""".strip()
    return [
        {
            "role": "system",
            "content": "You are a conservative VLN start-transition agent. Return JSON only.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url(route.start_jpeg_bytes)}},
            ],
        },
    ]


def start_facts_to_prompt(start_facts: Optional[Dict[str, Any]]) -> str:
    if not start_facts:
        return "No separate start facts are available."
    safe_facts = {
        "visible_start_area": start_facts.get("visible_start_area", ""),
        "departure_transition": start_facts.get("departure_transition", ""),
        "certain_landmarks": start_facts.get("certain_landmarks", []),
        "uncertain_or_avoid": start_facts.get("uncertain_or_avoid", []),
        "recommended_start_phrase": start_facts.get("recommended_start_phrase", ""),
        "must_preserve_start_boundary": start_facts.get(
            "must_preserve_start_boundary", False
        ),
    }
    return json.dumps(safe_facts, ensure_ascii=False, indent=2)


def sanitize_start_facts(start_facts: Dict[str, Any]) -> Dict[str, Any]:
    facts = dict(start_facts)
    if not isinstance(facts.get("certain_landmarks"), list):
        facts["certain_landmarks"] = []
    if not isinstance(facts.get("uncertain_or_avoid"), list):
        facts["uncertain_or_avoid"] = []
    value = facts.get("must_preserve_start_boundary", False)
    if isinstance(value, str):
        facts["must_preserve_start_boundary"] = value.strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
        }
    else:
        facts["must_preserve_start_boundary"] = bool(value)
    return facts


def align_start_facts_with_actions(
    start_facts: Dict[str, Any],
    actions: Sequence[int],
) -> Dict[str, Any]:
    facts = dict(start_facts)
    runs = action_runs(actions)
    if not runs:
        return facts
    first_action, first_start, first_end = runs[0]
    if first_action not in {2, 3}:
        return facts
    direction = "left" if first_action == 2 else "right"
    opposite = "right" if direction == "left" else "left"
    degrees = int((first_end - first_start + 1) * TURN_DEGREES)
    if degrees < TURN_DEGREES:
        return facts

    for key in ("recommended_start_phrase", "departure_transition"):
        text = str(facts.get(key, ""))
        if re.search(rf"\bturn {opposite}\b", text, flags=re.IGNORECASE):
            facts[key] = re.sub(
                rf"\bturn {opposite}\b",
                f"turn {direction}",
                text,
                flags=re.IGNORECASE,
            )
    avoid = facts.get("uncertain_or_avoid")
    if not isinstance(avoid, list):
        avoid = []
    avoid.append(
        f"initial turn direction conflicting with action summary: turn {opposite}"
    )
    facts["uncertain_or_avoid"] = avoid
    return facts


def extract_start_facts(
    args: argparse.Namespace,
    row: Dict[str, Any],
    route: RenderedRoute,
) -> Dict[str, Any]:
    messages = build_start_fact_messages(row=row, route=route)
    start_args = getattr(args, "start_args", args)
    raw_response = call_chat_completion(
        start_args,
        messages,
        temperature=args.fact_temperature,
        max_tokens=args.fact_max_tokens,
    )
    parsed = parse_json_response_with_repair(
        start_args,
        raw_response,
        '{"visible_start_area": "...", "departure_transition": "...", '
        '"certain_landmarks": [], "uncertain_or_avoid": [], '
        '"recommended_start_phrase": "...", "must_preserve_start_boundary": true}',
    )
    parsed = sanitize_start_facts(parsed)
    parsed = align_start_facts_with_actions(parsed, row["actions"])
    parsed["raw_response"] = raw_response
    return parsed


def extract_endpoint_facts(
    args: argparse.Namespace,
    row: Dict[str, Any],
    route: RenderedRoute,
) -> Dict[str, Any]:
    messages = build_endpoint_fact_messages(row=row, route=route)
    endpoint_args = getattr(args, "endpoint_args", args)
    raw_response = call_chat_completion(
        endpoint_args,
        messages,
        temperature=args.fact_temperature,
        max_tokens=args.fact_max_tokens,
    )
    parsed = parse_json_response_with_repair(
        endpoint_args,
        raw_response,
        '{"visible_destination_area": "...", "certain_landmarks": [], '
        '"uncertain_or_avoid": [], "stop_surface": "floor|step|landing|threshold|unknown", '
        '"surface_evidence": "...", "recommended_stop_phrase": "...", '
        '"old_destination_supported": "yes|no|unclear", '
        '"preserve_original_destination": true}',
    )
    parsed = sanitize_endpoint_facts(parsed)
    parsed["raw_response"] = raw_response
    return parsed


def route_plan_to_prompt(route_plan: Optional[Dict[str, Any]]) -> str:
    if not route_plan:
        return "No separate route plan is available."
    safe_plan = {
        "route_overview": route_plan.get("route_overview", ""),
        "major_segments": route_plan.get("major_segments", []),
        "action_turn_coverage": route_plan.get("action_turn_coverage", []),
        "safe_landmarks": route_plan.get("safe_landmarks", []),
        "passed_landmarks": route_plan.get("passed_landmarks", []),
        "near_not_passed_landmarks": route_plan.get("near_not_passed_landmarks", []),
        "avoid_claims": route_plan.get("avoid_claims", []),
        "start": route_plan.get("start", {}),
        "destination": route_plan.get("destination", {}),
        "writing_guidance": route_plan.get("writing_guidance", ""),
    }
    return json.dumps(safe_plan, ensure_ascii=False, indent=2)


def route_segment_checklist_to_prompt(route_plan: Optional[Dict[str, Any]]) -> str:
    if not route_plan:
        return "No separate route segment checklist is available."
    segments = route_plan.get("major_segments") or []
    if not isinstance(segments, list) or not segments:
        return "No separate route segment checklist is available."

    lines = []
    for index, segment in enumerate(segments, start=1):
        if not isinstance(segment, dict):
            continue
        movement = str(segment.get("movement", "")).strip()
        landmarks = segment.get("confirmed_landmarks") or []
        if isinstance(landmarks, list):
            landmark_text = ", ".join(str(item).strip() for item in landmarks[:4] if str(item).strip())
        else:
            landmark_text = str(landmarks).strip()
        if landmark_text:
            lines.append(f"{index}. {movement} Landmarks: {landmark_text}")
        else:
            lines.append(f"{index}. {movement}")
    return "\n".join(line for line in lines if line.strip()) or (
        "No separate route segment checklist is available."
    )


def build_route_plan_messages(
    row: Dict[str, Any],
    route: RenderedRoute,
    start_facts: Optional[Dict[str, Any]],
    endpoint_facts: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    old_instruction = normalize_instruction(row["instruction"])
    action_summary = compact_action_summary(row["actions"])
    action_turn_text = action_turn_requirements_to_prompt(row["actions"])
    route_constraints = build_route_constraints(row["actions"])
    start_fact_text = start_facts_to_prompt(start_facts)
    endpoint_fact_text = endpoint_facts_to_prompt(endpoint_facts)
    selected = ", ".join(str(index) for index in route.selected_frames)
    endpoint_selected = ", ".join(str(index) for index in route.endpoint_frames)
    prompt = f"""
You are the planning agent in a multi-agent VLN instruction generation pipeline.
Your job is not to write the final instruction. Your job is to visually inspect
the route and produce a structured, conservative route plan for the writer.

Image 1 is the route overview. Rows are chronological selected waypoints with
left, forward, and right views.
Image 2 is the endpoint evidence. Rows are the final approach; the final row is
the final observation.

Original weak instruction:
{old_instruction}

Trajectory support:
{action_summary}

Action-derived major turns:
{action_turn_text}

Route constraints:
{route_constraints}

Selected route frames: {selected}
Endpoint frames: {endpoint_selected}

Start facts from the start-transition agent:
{start_fact_text}

Endpoint facts from the endpoint agent:
{endpoint_fact_text}

Return JSON only:
{{
  "route_overview": "one sentence summary of the route",
  "major_segments": [
    {{
      "start_step": 0,
      "end_step": 0,
      "movement": "short human route command",
      "confirmed_landmarks": ["visible landmarks useful for navigation"],
      "uncertain": ["claims that are visually uncertain in this segment"],
      "confidence": "high|medium|low"
    }}
  ],
  "action_turn_coverage": [
    {{
      "turn_steps": "start-end action indices from Action-derived major turns",
      "direction": "left|right",
      "degrees": 45,
      "write_in_instruction": "yes|no|unclear",
      "covered_by_segment": "which major segment covers this turn",
      "reason": "why this turn should be written or omitted"
    }}
  ],
  "safe_landmarks": ["landmarks that are clearly visible and useful"],
  "passed_landmarks": ["landmarks the route clearly moves beyond"],
  "near_not_passed_landmarks": ["visible landmarks that are near the route but should not be described with pass/past"],
  "avoid_claims": ["claims the writer should avoid because they are uncertain or contradicted"],
  "start": {{
    "safe_start_phrase": "safe beginning phrase",
    "visual_evidence": "what the start sheet clearly shows",
    "avoid": ["unsafe start claims"]
  }},
  "destination": {{
    "safe_stop_phrase": "safe final stop phrase",
    "visual_evidence": "what the endpoint sheet clearly shows",
    "final_approach_direction": "left|right|straight|none|unclear",
    "final_direction_evidence": "which final-approach rows/views show the target area and whether the route actually turns into it",
    "old_final_turn_supported": "yes|no|unclear",
    "avoid": ["unsafe endpoint claims"]
  }},
  "writing_guidance": "brief guidance for the writer"
}}

Planning rules:
- Split the route into 3 to 6 major human-level segments. Do not describe every
  small turn or every selected frame.
- Do not put numeric degree values in movement text. Use natural turn language
  such as "turn right", "turn left", "turn slightly right", or "make a sharp
  right turn".
- The first segment must preserve the start facts. If the start facts say
  must_preserve_start_boundary=true, the first segment must include an explicit
  exit/leave/walk-out transition using the recommended safe start phrase. If the
  exact room type is uncertain, say "the room" rather than deleting the exit.
- If the start facts' recommended phrase conflicts with Route constraints or
  Trajectory support for the initial left/right direction, follow the trajectory
  and put the conflicting phrase in avoid_claims.
- Use the original weak instruction only as a hint. If it conflicts with the
  route images, action constraints, or endpoint facts, mark the old claim in
  avoid_claims.
- Treat the Action-derived major turns as the authoritative turn backbone. For
  each listed turn, fill action_turn_coverage. Use write_in_instruction="yes"
  when the turn changes which hallway, doorway, room, staircase, or open-area
  branch the route takes. Use "no" only when the images show it is merely
  same-area heading alignment, endpoint facing adjustment, or a small correction
  that should not be verbalized. Do not omit a major turn just because the
  original weak instruction omitted it or described a different turn.
- Same-area orientation changes inside one open room must not become a chain of
  human commands. If a turn only makes the view face or align with a sofa,
  artwork, television, balcony doors, or other landmark without entering a new
  doorway/room/stair/hall branch, set write_in_instruction="no" and express the
  segment as "continue toward..." or "move through the area toward..." instead
  of "turn ... to face/align".
- Major turns with write_in_instruction="yes" must appear naturally in
  major_segments, even if destination.final_approach_direction is straight. In
  that case, describe the turn into/onto the route area first, then describe the
  final straight approach to the stop separately.
- Be conservative with left/right for objects. Left/right is safe for route
  choices, not for furniture/appliance positions unless obvious.
- Be conservative with pass/go-past language. A visible landmark is not
  automatically passed. Use "pass", "passing", or "walk past" only when the
  chronological route clearly moves beyond that landmark. If the route turns
  before it or beside it, describe the turn or area without saying the landmark
  was passed.
- Fill passed_landmarks only for objects or room features that are along the
  movement path and become behind/side-behind in later views. Put side-wall
  landmarks, turn-adjacent landmarks, and endpoint-near landmarks in
  near_not_passed_landmarks instead.
- For turns into rooms or spaces, verify the direction from the chronological
  perspective views. If the destination room, couch, fireplace, or doorway is in
  the left view during the final approach, the route turns/branches left, not
  right; if it is in the right view, the route turns/branches right. Do not copy
  a left/right turn from the original weak instruction when the route views show
  the opposite.
- In destination.final_approach_direction, write a direction only when the route
  visually enters or branches into the target area. If the target is merely
  visible on one side while the route continues forward, use "straight", "none",
  or "unclear" and explain it in final_direction_evidence. If the old final turn
  disagrees with the final approach views, set old_final_turn_supported="no".
- Keep route_overview, major_segments, destination, and avoid_claims internally
  consistent. If destination.final_approach_direction is "straight", "none", or
  "unclear", do not write a major segment such as "turn left/right into the
  living room"; write "continue/walk forward into..." and put the directional
  entry phrase in avoid_claims.
- Use "hallway" or "corridor" only when the route passes through a visually
  narrow, linear passage for multiple frames. For open-plan transitions between
  patio, dining, kitchen, and living areas, use "open area", "dining area", or
  the visible landmarks instead of inventing a hallway.
- Do not encode object-relative side claims such as "furniture to your right" or
  "the sofa on your left" in the route plan. Use side words for turns, doorway
  choices, and corridor branches; describe objects without side labels unless
  the side is essential and visually unambiguous.
- The destination must follow the endpoint facts and final observation. If exact
  doorway/step/closet wording is uncertain, use the endpoint agent's safer stop
  phrase.
- Do not invent room names or object names; use generic terms when unsure.
""".strip()
    return [
        {
            "role": "system",
            "content": "You are a conservative VLN route planning agent. Return JSON only.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url(route.jpeg_bytes)}},
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url(route.endpoint_jpeg_bytes)},
                },
            ],
        },
    ]


def extract_route_plan(
    args: argparse.Namespace,
    row: Dict[str, Any],
    route: RenderedRoute,
    start_facts: Optional[Dict[str, Any]],
    endpoint_facts: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    messages = build_route_plan_messages(
        row=row,
        route=route,
        start_facts=start_facts,
        endpoint_facts=endpoint_facts,
    )
    planner_args = getattr(args, "planner_args", args)
    raw_response = call_chat_completion(
        planner_args,
        messages,
        temperature=args.planner_temperature,
        max_tokens=args.planner_max_tokens,
    )
    parsed = parse_json_response_with_repair(
        planner_args,
        raw_response,
        '{"route_overview": "...", "major_segments": [], "action_turn_coverage": [], '
        '"safe_landmarks": [], '
        '"passed_landmarks": [], "near_not_passed_landmarks": [], '
        '"avoid_claims": [], "destination": {"safe_stop_phrase": "...", '
        '"visual_evidence": "...", "final_approach_direction": "left|right|straight|none|unclear", '
        '"final_direction_evidence": "...", "old_final_turn_supported": "yes|no|unclear", '
        '"avoid": []}, "writing_guidance": "..."}',
    )
    parsed = align_route_plan_with_late_action_turns(
        route_plan=parsed,
        actions=row["actions"],
        endpoint_facts=endpoint_facts,
    )
    parsed = sanitize_route_plan_orientation_language(
        route_plan=parsed,
        endpoint_facts=endpoint_facts,
    )
    parsed["raw_response"] = raw_response
    return parsed


def build_generation_messages(
    row: Dict[str, Any],
    route: RenderedRoute,
    start_facts: Optional[Dict[str, Any]] = None,
    endpoint_facts: Optional[Dict[str, Any]] = None,
    route_plan: Optional[Dict[str, Any]] = None,
    repair_note: Optional[str] = None,
) -> List[Dict[str, Any]]:
    old_instruction = normalize_instruction(row["instruction"])
    destination_hint = extract_destination_hint(old_instruction) or "none"
    action_summary = compact_action_summary(row["actions"])
    action_turn_text = action_turn_requirements_to_prompt(row["actions"])
    selected = ", ".join(str(index) for index in route.selected_frames)
    endpoint_selected = ", ".join(str(index) for index in route.endpoint_frames)
    segmented_route_plan = build_segmented_route_plan(
        actions=row["actions"],
        selected_frames=route.selected_frames,
    )
    route_constraints = build_route_constraints(row["actions"])
    start_fact_text = start_facts_to_prompt(start_facts)
    endpoint_fact_text = endpoint_facts_to_prompt(endpoint_facts)
    route_plan_text = route_plan_to_prompt(route_plan)
    route_segment_checklist = route_segment_checklist_to_prompt(route_plan)

    system_prompt = (
        "You are an expert Room-to-Room navigation instruction writer. "
        "Write grounded indoor navigation instructions from visual trajectory evidence. "
        "Return valid JSON only."
    )
    user_prompt = f"""
You are given two contact sheets for a single indoor navigation route.
Image 1 is the route overview. Its rows are chronological waypoints; each row has
left, forward, and right views.
Image 2 is the endpoint evidence. Its rows are the final approach; the last row is
the final observation, with left, forward, right, and forward-down views. Use the
last row most heavily for the destination.
Use the route overview for the path and the endpoint evidence for the final stop.

Original weak instruction:
{old_instruction}

Original destination hint:
{destination_hint}

Trajectory support:
{action_summary}
Selected route frames: {selected}
Endpoint frames: {endpoint_selected}

Action-derived major turns:
{action_turn_text}

Route constraints:
{route_constraints}

Start facts:
{start_fact_text}

Endpoint facts:
{endpoint_fact_text}

Route plan from the planning agent:
{route_plan_text}

Required segment coverage checklist:
{route_segment_checklist}

Segmented route plan:
{segmented_route_plan}

Requirements:
- Output JSON only: {{"instruction": "..."}}
- The instruction must be one fluent English navigation instruction for a person.
- Treat the original weak instruction as a rough route skeleton, not as ground
  truth. Preserve its high-level route order only when it agrees with the action
  constraints and visual evidence. If the original says "turn around" but the
  route constraints say the path starts by moving forward, do not write "turn
  around".
- Use the model route plan as the main source for human-level segments and
  safe landmarks. Use the segmented action plan only to keep visual evidence in
  order. Do not
  verbalize every small action interval, and do not turn every interval into a new
  left/right command.
- The ground-truth action list is authoritative for actual turns. Include major
  action-derived turns as natural navigation cues when route_plan.action_turn_coverage
  marks write_in_instruction="yes". Do not delete a true major turn because the
  original weak instruction omitted it, and do not copy a turn from the original
  weak instruction when it conflicts with the action-derived turn sequence.
- If a major action turn leads into a doorway, hallway, room, staircase, or
  open-area branch, write it. If destination.final_approach_direction is
  "straight", you may still write the earlier route-choice turn, then describe
  the final approach as continuing straight to the stop.
- Do not write numeric degree measurements such as "turn right about 30 degrees",
  even if the route plan contains them. Write natural R2R turn language instead.
- Cover every item in the Required segment coverage checklist in order. Do not
  skip the first segment. If the first segment says to exit, leave, or walk out
  of a room, the final instruction must begin with that transition using a safe
  generic phrase such as "Exit the room" when the exact room type is uncertain.
- The Start facts are binding for the first sentence. If
  must_preserve_start_boundary=true, begin with the recommended start phrase or
  an equivalent explicit exit/leave/walk-out phrase. Do not remove the start
  boundary just because the original room type is uncertain.
- Smoothly merge adjacent segments into a natural route description. Avoid
  duplicated clauses, and do not write contradictory connectors such as exiting a
  room after already walking through the next room.
- Use Image 1 to add route landmarks and Image 2 to add only safe final stopping
  context around the original skeleton.
- Mention useful rooms, objects, doorways, stairs, turns, and the final stopping area.
- Use "hallway" or "corridor" only when the route clearly enters a narrow,
  linear passage. If the route moves through connected open patio/dining/kitchen/
  living spaces, describe those areas or their landmarks rather than inventing a
  hallway.
- Keep it grounded in visible evidence. Do not invent precise object names if unclear.
- Do not convert every visible landmark into "passing" language. Use "pass",
  "passing", or "walk past" only when the route actually moves beyond that
  landmark. If a mirror, picture, console, or similar object is just visible on
  the wall while the route turns before/beside it, use neutral wording such as
  "near the mirror" or omit it.
- The route plan's passed_landmarks and near_not_passed_landmarks are binding
  for pass/past wording. Use "pass" only for passed_landmarks. If a landmark is
  listed in near_not_passed_landmarks, use "near", "by", "toward", or omit it.
- The route plan's avoid_claims are binding too. Do not reuse an avoid_claim or
  a close paraphrase in the final instruction just because it appears in a route
  segment or the original weak instruction.
- If the route segment checklist conflicts with destination.final_approach_direction,
  passed_landmarks, near_not_passed_landmarks, or avoid_claims, follow the
  structured route-plan fields and use neutral wording.
- Verify the left/right direction for the final turn into the destination area
  from the chronological route views and action-derived route outline. If the
  living room, couch, fireplace, or target doorway is visible on the left during
  the final approach, write a left turn/branch, not a right turn; if it is on the
  right, write a right turn/branch.
- The route plan's destination.final_approach_direction is binding for the final
  approach wording. If it is "left" or "right", use that direction only when
  describing entry into the destination area. If it is "straight", "none", or
  "unclear", do not write "turn left/right into ..." for the final approach;
  use neutral wording such as "continue into", "approach", or "stop near".
- The final destination sentence is critical. If the original destination hint is
  explicit, preserve it only when Image 2 and the endpoint facts support it. If
  Image 2 contradicts it or the endpoint facts mark it as unsupported/unclear,
  use the recommended stop phrase or a conservative phrase such as "stop in this
  area" instead of guessing a doorway, closet, appliance, shelf, or step count.
- If endpoint_facts.certain_landmarks contains a clear final landmark, anchor the
  stop with one of those landmarks instead of writing only "stop in the room" or
  "stop in the living room".
- Do not make the destination more specific than the endpoint facts. In particular,
  do not end with "stop in the doorway", "stop at the closet", or "stop on the
  second step" unless the final observation clearly supports that exact phrase.
- For stair endpoints, follow the endpoint facts' recommended stop phrase closely.
  Do not change "near the stairs" into "on the stairs", and do not add secondary
  wall landmarks after the stop phrase if they could shift the stopping point.
- For patio, balcony, porch, exterior, and doorway endpoints, follow endpoint
  facts closely. If the endpoint facts recommend stopping near/just inside a
  doorway, do not change that into "walk through the doorway into the patio" or
  any wording that moves the stop outside the threshold.
- Use left/right confidently for route turns and doorway choices. For object positions,
  do not say "on your left" or "on your right" for furniture, appliances, wall art,
  windows, rugs, shelves, or small objects. Use safer wording such as "pass the
  stove and refrigerator" or "pass the table".
- Do not mention images, frames, rows, panels, panoramas, contact sheets, actions, or datasets.
- Do not mechanically list left/right/forward tokens; write like a human annotator.
- Do not mention the same passed landmark twice. If one long segment passes the
  kitchen island and sofa, the next sentence should continue from that point
  instead of saying "walk past the kitchen island" again.
- Do not write camera-alignment language such as "turn right to face..., then
  turn left to align with...". Compress same-room orientation adjustments into
  natural landmark guidance, for example "continue through the living area
  toward the glass balcony doors".
- Target length: 25 to 75 words. Longer is allowed only if the route is long.
""".strip()
    if repair_note:
        user_prompt += "\n\nYour previous answer failed validation:\n" + repair_note
        user_prompt += "\nRewrite it and obey the JSON-only format."

    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url(route.jpeg_bytes)},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url(route.endpoint_jpeg_bytes)},
                },
            ],
        },
    ]


def get_openai_client(base_url: str, api_key: str):
    cache_key = (base_url, api_key)
    client_cache = getattr(_THREAD_LOCAL, "client_cache", None)
    if client_cache is None:
        client_cache = {}
        _THREAD_LOCAL.client_cache = client_cache
    if cache_key not in client_cache:
        try:
            from openai import OpenAI
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "The openai package is required. Run from the vllm conda env, "
                "for example: source /opt/conda/bin/activate vllm"
            ) from error
        client_cache[cache_key] = OpenAI(base_url=base_url, api_key=api_key)
    return client_cache[cache_key]


def call_chat_completion(
    args: argparse.Namespace,
    messages: List[Dict[str, Any]],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    client = get_openai_client(args.base_url, args.api_key)
    kwargs = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature if temperature is None else temperature,
        "max_tokens": args.max_tokens if max_tokens is None else max_tokens,
        "timeout": args.request_timeout,
    }
    if args.provider == "qwen" and args.disable_thinking:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def build_self_check_messages(
    row: Dict[str, Any],
    route: RenderedRoute,
    instruction: str,
    endpoint_facts: Optional[Dict[str, Any]] = None,
    route_plan: Optional[Dict[str, Any]] = None,
    quality_hints: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    action_summary = compact_action_summary(row["actions"])
    action_turn_text = action_turn_requirements_to_prompt(row["actions"])
    destination_hint = extract_destination_hint(row.get("instruction", "")) or "none"
    endpoint_fact_text = endpoint_facts_to_prompt(endpoint_facts)
    route_plan_text = route_plan_to_prompt(route_plan)
    route_segment_checklist = route_segment_checklist_to_prompt(route_plan)
    segmented_route_plan = build_segmented_route_plan(
        actions=row["actions"],
        selected_frames=route.selected_frames,
    )
    hint_text = "\n".join(f"- {hint}" for hint in (quality_hints or [])) or "none"
    prompt = f"""
You are a strict human-style VLN dataset quality judge. Simulate a careful
manual annotator who visually checks whether the instruction matches the route
and endpoint. Do not judge by code rules alone; inspect the images.

Image 1 is the chronological route overview. Each row has left, forward,
and right perspective views at the agent's current position.
Image 2 is the endpoint evidence. Its final row is the final observation.

Original weak instruction:
{normalize_instruction(row.get("instruction", ""))}

Original destination hint:
{destination_hint}

Generated instruction:
{instruction}

Trajectory support:
{action_summary}

Action-derived major turns:
{action_turn_text}

Action-derived selected waypoint intervals:
{segmented_route_plan}

Endpoint facts from a separate visual pass:
{endpoint_fact_text}

Route plan from a separate visual planning pass:
{route_plan_text}

Required segment coverage checklist:
{route_segment_checklist}

Automatic risk hints. These can be wrong, so use them only as issues to inspect:
{hint_text}

Return JSON only:
{{
  "grounding_score": 1-5,
  "navigation_score": 1-5,
  "endpoint_score": 1-5,
  "r2r_style_score": 1-5,
  "hallucination_risk": "low|medium|high",
  "verdict": "pass|borderline|fail",
  "issues": [
    {{"severity": "minor|major", "type": "route|endpoint|landmark|direction|style", "text": "short issue"}}
  ],
  "corrected_instruction": "corrected instruction if verdict is borderline or fail, otherwise null",
  "reason": "short reason"
}}

Judging standard:
- pass: route order, major turns, landmarks, and final stop are all visually supported.
- borderline: mostly navigable but has one uncertain/specific phrase that should be
  made safer; provide a corrected_instruction.
- fail: wrong endpoint, wrong room, wrong major turn, invented landmark, or an
  instruction that would likely lead a human to the wrong place; provide a corrected_instruction.
- Segment coverage matters. Compare the generated instruction to every item in
  the Required segment coverage checklist. If it skips the first segment,
  especially an initial exit/leave/walk-out-from-room transition, mark it at
  least borderline and provide a corrected_instruction that restores the missing
  start transition without adding an uncertain room label.
- The corrected_instruction should smoothly connect adjacent route segments:
  no repeated clauses, no contradictory transitions, and no jump from a later
  room/landmark before describing how the agent got there.
- The corrected_instruction must be grammatically complete. Penalize dangling
  fragments such as "keeping the fireplace and television" with no relation.
- Penalize numeric degree wording such as "turn right about 30 degrees"; replace
  it with natural language such as "turn right" or "turn slightly right".
- Major action-derived turns are key navigation information. If the route plan's
  action_turn_coverage marks a major turn write_in_instruction="yes", the
  generated instruction should include that turn or an equivalent route-choice
  phrase. Do not remove a true major turn just because the original weak
  instruction omitted it. If a final straight approach follows an earlier true
  route-choice turn, keep both the turn and the straight approach.
- Do not reward same-room camera-orientation chains. If the instruction says
  "turn ... to face" and then "turn ... to align" while staying in one open
  living/dining/kitchen area, mark it borderline or fail for R2R style and
  correct it to a concise landmark-based route description.
- Penalize invented hallway/corridor labels when the route is visibly through
  open connected spaces; use area or landmark wording instead.
- Penalize incorrect traversal verbs. "Pass", "passing", and "walk past" require
  the route to move beyond the named landmark; a landmark merely visible on a
  wall before a turn is not enough.
- Treat the route plan's passed_landmarks and near_not_passed_landmarks as
  explicit evidence. If the instruction uses pass/past for a near_not_passed
  landmark, correct it to "near", "by", "toward", or remove that landmark.
- Penalize wrong turn direction into the destination area. Check the final
  approach views and the action-derived intervals: if the target room/couch/
  fireplace/doorway is on the left, a "turn right into ..." claim is a major
  direction error, and vice versa.
- Use destination.final_approach_direction and final_direction_evidence as a
  visual fact to inspect, not as style text. If it says "left" or "right" and
  the generated final approach uses the opposite direction, correct it. If it
  says "straight", "none", or "unclear", correct final "turn left/right into"
  wording to a neutral approach phrase unless the images clearly justify it.
- The endpoint is critical. If "doorway", "closet", "second step", "landing",
  or a specific appliance is not clearly supported by the endpoint sheet, correct
  it to a safer final stop phrase.
- Do not penalize the generated instruction for mistakes in the original weak
  instruction if the generated instruction has already corrected them.
- If your own corrected_instruction fixes the issue and is visually grounded, the
  next review should be able to pass it. Do not keep marking it borderline for
  the old instruction's error.
- If a corrected instruction has removed the phrase you objected to and the
  remaining concern is only wording specificity, return pass or borderline, not
  fail. Do not loop on tiny boundary wording differences after the route and
  endpoint are correct.
- If an object left/right position is uncertain, correct it by removing the object
  side while keeping useful landmarks.
- The corrected_instruction must be one natural R2R-style instruction, grounded
  in visible evidence, and must not mention images, rows, actions, or datasets.
""".strip()
    return [
        {
            "role": "system",
            "content": "You are a strict VLN dataset quality judge. Return JSON only.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url(route.jpeg_bytes)},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url(route.endpoint_jpeg_bytes)},
                },
            ],
        },
    ]


def self_check_instruction(
    args: argparse.Namespace,
    row: Dict[str, Any],
    route: RenderedRoute,
    instruction: str,
    endpoint_facts: Optional[Dict[str, Any]] = None,
    route_plan: Optional[Dict[str, Any]] = None,
    quality_hints: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    messages = build_self_check_messages(
        row=row,
        route=route,
        instruction=instruction,
        endpoint_facts=endpoint_facts,
        route_plan=route_plan,
        quality_hints=quality_hints,
    )
    review_args = getattr(args, "review_args", args)
    raw_response = call_chat_completion(
        review_args,
        messages,
        temperature=args.review_temperature,
        max_tokens=args.review_max_tokens,
    )
    parsed = parse_json_response_with_repair(
        review_args,
        raw_response,
        '{"grounding_score": 5, "navigation_score": 5, "endpoint_score": 5, '
        '"r2r_style_score": 5, "hallucination_risk": "low|medium|high", '
        '"verdict": "pass|borderline|fail", "issues": [], '
        '"corrected_instruction": null, "reason": "..."}',
    )
    parsed["raw_response"] = raw_response
    return parsed


def build_route_audit_messages(
    row: Dict[str, Any],
    route: RenderedRoute,
    instruction: str,
    start_facts: Optional[Dict[str, Any]] = None,
    endpoint_facts: Optional[Dict[str, Any]] = None,
    route_plan: Optional[Dict[str, Any]] = None,
    quality_hints: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    start_fact_text = start_facts_to_prompt(start_facts)
    endpoint_fact_text = endpoint_facts_to_prompt(endpoint_facts)
    route_plan_text = route_plan_to_prompt(route_plan)
    route_segment_checklist = route_segment_checklist_to_prompt(route_plan)
    action_summary = compact_action_summary(row["actions"])
    action_turn_text = action_turn_requirements_to_prompt(row["actions"])
    segmented_route_plan = build_segmented_route_plan(
        actions=row["actions"],
        selected_frames=route.selected_frames,
    )
    hint_text = "\n".join(f"- {hint}" for hint in (quality_hints or [])) or "none"
    prompt = f"""
You are the route-coherence auditor in a multi-agent VLN instruction generation
pipeline. Simulate a careful human R2R dataset reviewer. Your job is to check
the whole instruction, not just the start or endpoint.

Image 1 is the start evidence for the first few frames.
Image 2 is the chronological route overview.
Image 3 is the endpoint evidence.

Original weak instruction:
{normalize_instruction(row.get("instruction", ""))}

Generated instruction:
{instruction}

Start facts:
{start_fact_text}

Endpoint facts:
{endpoint_fact_text}

Trajectory support:
{action_summary}

Action-derived major turns:
{action_turn_text}

Action-derived selected waypoint intervals:
{segmented_route_plan}

Route plan:
{route_plan_text}

Required segment coverage checklist:
{route_segment_checklist}

Automatic risk hints. These can be wrong, but you must inspect them:
{hint_text}

Return JSON only:
{{
  "verdict": "pass|borderline|fail",
  "segment_issues": [
    {{"segment": 1, "severity": "minor|major", "text": "what is skipped, wrong, repeated, or contradictory"}}
  ],
  "style_issues": [
    {{"severity": "minor|major", "text": "R2R style issue"}}
  ],
  "corrected_instruction": "corrected instruction if verdict is borderline or fail, otherwise null",
  "reason": "short reason"
}}

Audit standard:
- The instruction must cover all major route segments in order: starting
  transition, intermediate space changes, important turns/door choices, reliable
  landmarks, and final stop.
- Major action-derived turns are authoritative route evidence. If
  route_plan.action_turn_coverage marks a turn write_in_instruction="yes", the
  instruction must preserve it as a natural left/right navigation cue. Do not
  drop true action-derived turns while making the language more conservative;
  separate them from final straight approach wording when needed.
- Do not let a visually important middle segment disappear. If the route plan
  says to enter a hallway, pass through a kitchen/dining/living area, go through
  a doorway, or turn into another corridor, the instruction should express that
  transition naturally unless the visual evidence contradicts it.
- If start facts say must_preserve_start_boundary=true, the first sentence must
  explicitly include an exit/leave/walk-out transition. If the exact room type is
  uncertain, use "room" rather than removing the boundary.
- Endpoint facts have priority over route-plan wording for the final stop. If
  endpoint facts or route_plan avoid_claims list a term as uncertain, do not put
  that term in corrected_instruction even when it looks like a useful landmark.
  Use the recommended safe stop phrase or a nearby unambiguous landmark instead.
- The instruction should sound like a high-quality R2R human instruction:
  concise but complete, natural sentence flow, no action-token list, no repeated
  clauses, no conflicting connectors, no image/frame/action terminology.
- Penalize camera-alignment prose such as "turn right to face..., then turn left
  to align with...". If the route stays inside one open room, correct it to
  natural landmark movement such as "continue through the living area toward the
  glass doors" instead of listing every heading adjustment.
- Penalize grammar fragments such as "keeping the fireplace and television" when
  the phrase does not say where or how the landmark is used.
- Penalize numeric angle text such as "about 30 degrees" or "45 degrees"; human
  R2R instructions should use natural turn language.
- Penalize invented hallway/corridor wording when the visual route is an open
  patio/dining/kitchen/living transition instead of a narrow linear passage.
- Penalize incorrect traversal verbs. Do not accept "pass", "passing", or "walk
  past" for a landmark that is merely visible on a wall or beside a turn. The
  route must actually move beyond that landmark.
- Treat route_plan.passed_landmarks and route_plan.near_not_passed_landmarks as
  explicit pass/past evidence. If the instruction says it passes a
  near_not_passed landmark, correct the wording.
- Penalize wrong left/right direction into a destination room or branch. Inspect
  the final approach views and the action-derived selected intervals; if the
  target room, couch, fireplace, or doorway is on the left during the final
  approach, "turn right into ..." is a major direction issue, and vice versa.
- Use destination.final_approach_direction and final_direction_evidence to
  decide whether final entry should be left, right, straight/continue, or
  direction-neutral. If the direction is unclear, prefer neutral wording rather
  than forcing a left/right turn.
- Penalize vague middle-route compression such as "make several turns",
  "navigate the winding path", or "continue for a while" when the route plan has
  specific human-level turns or space transitions that can be described.
- Penalize unstable object-side language such as "furniture to your right" or
  "the sofa on your left". Keep left/right for actual turns, doorway choices, or
  hallway branches; use object landmarks without side labels when possible.
- Use corrected_instruction to repair the whole instruction when needed. Keep it
  grounded, preserve segment order, and avoid over-specific uncertain room or
  endpoint labels.
""".strip()
    return [
        {
            "role": "system",
            "content": "You are a strict VLN route-coherence auditor. Return JSON only.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url(route.start_jpeg_bytes)}},
                {"type": "image_url", "image_url": {"url": image_data_url(route.jpeg_bytes)}},
                {"type": "image_url", "image_url": {"url": image_data_url(route.endpoint_jpeg_bytes)}},
            ],
        },
    ]


def route_audit_instruction(
    args: argparse.Namespace,
    row: Dict[str, Any],
    route: RenderedRoute,
    instruction: str,
    start_facts: Optional[Dict[str, Any]] = None,
    endpoint_facts: Optional[Dict[str, Any]] = None,
    route_plan: Optional[Dict[str, Any]] = None,
    quality_hints: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    messages = build_route_audit_messages(
        row=row,
        route=route,
        instruction=instruction,
        start_facts=start_facts,
        endpoint_facts=endpoint_facts,
        route_plan=route_plan,
        quality_hints=quality_hints,
    )
    review_args = getattr(args, "review_args", args)
    raw_response = call_chat_completion(
        review_args,
        messages,
        temperature=args.review_temperature,
        max_tokens=args.review_max_tokens,
    )
    parsed = parse_json_response_with_repair(
        review_args,
        raw_response,
        '{"verdict": "pass|borderline|fail", "segment_issues": [], '
        '"style_issues": [], "corrected_instruction": null, "reason": "..."}',
    )
    parsed["raw_response"] = raw_response
    return parsed


def route_audit_passed(audit: Dict[str, Any]) -> bool:
    return str(audit.get("verdict", "")).strip().lower() == "pass"


def route_audit_is_acceptable_borderline(audit: Dict[str, Any]) -> bool:
    if str(audit.get("verdict", "")).strip().lower() != "borderline":
        return False
    reason = str(audit.get("reason", "")).lower()
    if (
        ("route plan" in reason or "data integrity" in reason)
        and "instruction" in reason
        and (
            "instruction is correct" in reason
            or "generated instruction is correct" in reason
            or "instruction accurately" in reason
        )
    ):
        return True
    issues = []
    for key in ("segment_issues", "style_issues"):
        value = audit.get(key) or []
        if isinstance(value, list):
            issues.extend(item for item in value if isinstance(item, dict))
    if not issues:
        return True
    if all(str(issue.get("severity", "")).lower() != "major" for issue in issues):
        return True
    issue_text = " ".join(str(issue.get("text", "")) for issue in issues).lower()
    style_only_terms = ("degree", "numeric angle", "style", "repetitive", "robotic")
    if any(term in issue_text for term in style_only_terms):
        route_error_terms = (
            "wrong endpoint",
            "wrong destination",
            "wrong room",
            "wrong turn",
            "missing segment",
            "skips",
            "contradict",
            "hallucinat",
        )
        return not any(term in issue_text for term in route_error_terms)
    return False


def build_spatial_audit_messages(
    row: Dict[str, Any],
    route: RenderedRoute,
    instruction: str,
    endpoint_facts: Optional[Dict[str, Any]] = None,
    route_plan: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    prompt = f"""
You are the final spatial-boundary auditor for a VLN instruction.
The normal critic already checked general route quality. Your only job is to
look for over-specific or wrong spatial boundary wording that a human auditor
would mark as borderline.

Image 1 is the chronological route overview. Image 2 is the final approach and
endpoint. Inspect both images carefully.

Original weak instruction:
{normalize_instruction(row.get("instruction", ""))}

Generated instruction:
{instruction}

Endpoint facts:
{endpoint_facts_to_prompt(endpoint_facts)}

Route plan:
{route_plan_to_prompt(route_plan)}

Return JSON only:
{{
  "verdict": "pass|borderline|fail",
  "boundary_issues": [
    {{"phrase": "text from instruction", "issue": "why boundary wording is too strong or wrong"}}
  ],
  "corrected_instruction": "corrected instruction if verdict is borderline or fail, otherwise null",
  "reason": "short reason"
}}

Audit checklist:
- Audit only the Generated instruction. Every `boundary_issues[].phrase` must be
  copied from the Generated instruction. Do not mark a phrase that appears only
  in the Original weak instruction, endpoint facts, or route plan.
- When writing corrected_instruction, preserve route-plan pass/past constraints.
  Use "pass", "passing", or "walk past" only for route_plan.passed_landmarks.
  If a landmark is listed in route_plan.near_not_passed_landmarks, use "with ...
  visible", "near ...", "by ...", or omit it. Do not introduce a new pass/past
  phrase while correcting boundary wording.
- When route_plan.destination.final_approach_direction is "straight", "none", or
  "unclear", do not introduce "turn left/right into ..." for the final approach;
  use "continue into", "move into", or "stop near" wording.
- When a boundary is ambiguous, prefer neutral wording instead of oscillating
  between two specific labels. Good neutral phrases include "near the doorway",
  "near the entrance", "near the stairs", "by the stairs", "near the large doors",
  "in the area with ...", or "near the visible landmark".
- "hallway" vs open living/dining/kitchen edge: if the path or endpoint is an
  open area edge, do not force "hallway"; use "area", "open area", "near the
  wall", or the visible landmark.
- "doorway" vs in front of a door/near a doorway: use "in the doorway" only if
  the final observation clearly shows the agent stopped at/within a doorway.
  If it is unclear whether the agent is inside the room or exactly in the
  doorway, use "near the doorway" or "near the entrance to the room".
- Patio/balcony/porch/exterior transitions: use "go through the doorway into
  the patio/balcony/porch" only when the route clearly crosses the threshold
  and the final stop is outside or in that covered outdoor area. If the outdoor
  area is visible but the stop is at or just inside the threshold, use "near the
  patio doorway", "near the entrance", or "just inside the doorway".
- "inside the room" vs at threshold/near entrance: do not say inside if the final
  stop is at the threshold. If threshold vs inside is ambiguous, use "near the
  entrance" instead of choosing either side.
- "through the kitchen" vs past/along the kitchen: if the trajectory skirts the
  kitchen or crosses an open-plan boundary, use "past the kitchen area" or
  "through the open kitchen area" only when visually supported.
- "top/bottom/landing/second step" for stairs: use only if visually clear.
  Use "on the stairs" only if the final/downward view clearly places the agent
  on steps. If the exact stair position is uncertain, use "near the stairs" or
  "by the stairs" rather than top/bottom/landing/on the stairs.
- For foyer/open area/hallway labels, if the room type is debatable, anchor to
  visible landmarks instead, e.g. "near the large wooden double doors".
- If your corrected instruction already uses neutral wording and would be
  acceptable to a careful annotator, return pass. Do not keep alternating
  between specific labels.
- Always pass neutral boundary wording such as "near the stairs", "by the
  stairs", "near the doorway", "near the entrance", or "near the large doors"
  when the route and landmark are otherwise correct. Do not mark these phrases
  borderline just because a more specific top/bottom/inside/doorway label might
  also be possible.
- If the phrase is acceptable and would not bother a careful human annotator,
  return pass. If it is slightly too specific but still navigable, return
  borderline and correct it. If it would lead to the wrong place, return fail.
""".strip()
    return [
        {
            "role": "system",
            "content": "You are a strict spatial-boundary auditor for VLN data. Return JSON only.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url(route.jpeg_bytes)}},
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url(route.endpoint_jpeg_bytes)},
                },
            ],
        },
    ]


def spatial_audit_instruction(
    args: argparse.Namespace,
    row: Dict[str, Any],
    route: RenderedRoute,
    instruction: str,
    endpoint_facts: Optional[Dict[str, Any]] = None,
    route_plan: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    messages = build_spatial_audit_messages(
        row=row,
        route=route,
        instruction=instruction,
        endpoint_facts=endpoint_facts,
        route_plan=route_plan,
    )
    review_args = getattr(args, "review_args", args)
    raw_response = call_chat_completion(
        review_args,
        messages,
        temperature=args.review_temperature,
        max_tokens=args.review_max_tokens,
    )
    parsed = parse_json_response_with_repair(
        review_args,
        raw_response,
        '{"verdict": "pass|borderline|fail", "boundary_issues": [], '
        '"corrected_instruction": null, "reason": "..."}',
    )
    parsed["raw_response"] = raw_response
    return parsed


def accept_neutral_spatial_borderline(
    instruction: str,
    audit: Dict[str, Any],
) -> bool:
    verdict = str(audit.get("verdict", "")).strip().lower()
    if verdict != "borderline":
        return False
    lower = normalize_instruction(instruction).lower()
    neutral_patterns = (
        r"\bnear the (?:top of the |bottom of the )?(?:wooden |carpeted )?"
        r"(?:stairs|staircase|stairway)\b",
        r"\b(?:at|by|beside|next to) the (?:wooden |carpeted )?"
        r"(?:stairs|staircase|stairway)\b",
        r"\bon the landing(?: area)?(?: (?:near|by|beside|next to) the "
        r"(?:wooden |carpeted )?(?:stairs|staircase|stairway))?\b",
        r"\bnear the (?:large |double |wooden )?(?:door|doors|doorway|entrance)\b",
        r"\bat the (?:large |double |wooden )?(?:doorway|entrance)\b",
        r"\bin the area\b",
    )
    return any(re.search(pattern, lower) for pattern in neutral_patterns)


def accept_stale_spatial_audit(
    instruction: str,
    audit: Dict[str, Any],
) -> bool:
    verdict = str(audit.get("verdict", "")).strip().lower()
    if verdict not in {"borderline", "fail"}:
        return False
    normalized = normalize_instruction(instruction)
    corrected = normalize_instruction(str(audit.get("corrected_instruction") or ""))
    if corrected and corrected != "." and corrected == normalized:
        return True

    issues = audit.get("boundary_issues") or []
    phrases = []
    if isinstance(issues, list):
        for issue in issues:
            if isinstance(issue, dict):
                phrase = str(issue.get("phrase", "")).strip()
                if phrase:
                    phrases.append(phrase)
    if phrases and not any(phrase.lower() in normalized.lower() for phrase in phrases):
        return True
    return False


def self_check_passed(check: Dict[str, Any]) -> bool:
    verdict = str(check.get("verdict", "")).strip().lower()
    hallucination = str(check.get("hallucination_risk", "")).strip().lower()
    grounding = int(check.get("grounding_score", 0) or 0)
    navigation = int(check.get("navigation_score", 0) or 0)
    endpoint = int(check.get("endpoint_score", 0) or 0)
    if verdict != "pass":
        return False
    if hallucination == "high":
        return False
    if grounding < 4 or navigation < 4 or endpoint < 4:
        return False
    return True


def self_check_is_acceptable_borderline(check: Dict[str, Any]) -> bool:
    verdict = str(check.get("verdict", "")).strip().lower()
    hallucination = str(check.get("hallucination_risk", "")).strip().lower()
    grounding = int(check.get("grounding_score", 0) or 0)
    navigation = int(check.get("navigation_score", 0) or 0)
    endpoint = int(check.get("endpoint_score", 0) or 0)
    if verdict != "borderline" or hallucination == "high":
        return False
    if grounding < 4 or navigation < 4 or endpoint < 4:
        return False
    issues = check.get("issues") or []
    if not isinstance(issues, list):
        return False
    if not issues:
        return True
    for issue in issues:
        if not isinstance(issue, dict):
            return False
        if str(issue.get("severity", "")).lower() == "major":
            return False
        issue_type = str(issue.get("type", "")).lower()
        issue_text = str(issue.get("text", "")).lower()
        if issue_type not in {"style", "landmark", "route", "endpoint", "direction"}:
            return False
        serious_terms = (
            "wrong",
            "missing",
            "opposite",
            "unsupported",
            "not supported",
            "contradict",
            "hallucinat",
            "incorrect",
            "misaligned",
            "fails",
            "would lead",
        )
        if any(term in issue_text for term in serious_terms):
            return False
    return True


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


def parse_json_response_with_repair(
    args: argparse.Namespace,
    raw_response: str,
    expected_schema: str,
) -> Dict[str, Any]:
    try:
        return extract_json_object(raw_response)
    except Exception:
        repair_prompt = f"""
Repair the following malformed JSON into one valid JSON object.
Return JSON only. Do not add explanations.

Expected schema:
{expected_schema}

Malformed response:
{raw_response}
""".strip()
        messages = [
            {
                "role": "system",
                "content": "You repair malformed JSON. Return valid JSON only.",
            },
            {"role": "user", "content": repair_prompt},
        ]
        repair_max_tokens = min(
            max(
                int(getattr(args, "max_tokens", 512) or 512),
                int(getattr(args, "fact_max_tokens", 0) or 0),
                int(getattr(args, "planner_max_tokens", 0) or 0),
                int(getattr(args, "review_max_tokens", 0) or 0),
                len(raw_response) // 3 + 256,
                1024,
            ),
            4096,
        )
        fixed = call_chat_completion(
            args,
            messages,
            temperature=0.0,
            max_tokens=repair_max_tokens,
        )
        parsed = extract_json_object(fixed)
        parsed["_json_repair_raw_response"] = raw_response
        parsed["_json_repair_response"] = fixed
        return parsed


def validate_generated_instruction(
    instruction: str,
    old_instruction: str,
    min_words: int,
    max_words: int,
) -> Tuple[List[str], List[str]]:
    errors = []
    warnings = []
    normalized = normalize_instruction(instruction)
    count = word_count(normalized)
    if count < min_words:
        errors.append(f"too short: {count} words < {min_words}")
    if count > max_words:
        errors.append(f"too long: {count} words > {max_words}")
    if normalized.lower() == normalize_instruction(old_instruction).lower():
        errors.append("identical to the original instruction")
    for pattern in DATA_ARTIFACT_PATTERNS:
        if pattern.search(normalized):
            errors.append(f"contains data artifact phrase: {pattern.pattern}")
    if re.search(r"\b(left|right|forward|stop)\b(?:\s+\b(left|right|forward|stop)\b){3,}", normalized, re.IGNORECASE):
        errors.append("looks like an action-token list")
    if "stair" not in normalized.lower() and "stair" in old_instruction.lower():
        warnings.append("old instruction mentioned stairs but rewrite does not")
    return errors, warnings


def endpoint_consistency_hints(
    instruction: str,
    endpoint_facts: Optional[Dict[str, Any]],
) -> List[str]:
    if not endpoint_facts:
        return []
    errors = []
    destination = extract_destination_hint(instruction).lower()
    instruction_lower = instruction.lower()

    unsupported = str(endpoint_facts.get("old_destination_supported", "")).lower()
    preserve = endpoint_facts.get("preserve_original_destination", True)
    recommended = str(endpoint_facts.get("recommended_stop_phrase", "")).strip().lower()
    if unsupported in {"no", "unclear", "false"} or preserve is False:
        old_sensitive_terms = [
            "doorway",
            "closet",
            "second step",
            "landing",
            "hallway end",
            "top of the stairs",
            "bottom of the stairs",
            "counter",
            "service area",
            "kitchen",
            "patio",
            "balcony",
            "porch",
            "outside",
            "exterior",
            "appliance",
            "stove",
            "refrigerator",
        ]
        for term in old_sensitive_terms:
            if term in recommended:
                continue
            if (
                term == "top of the stairs"
                and re.search(r"\b(?:near|by|beside)\s+the\s+top\s+of\s+the\s+stairs\b", destination)
            ):
                continue
            if term in destination:
                errors.append(
                    f"endpoint fact pass did not support final destination term: {term}"
                )

    avoid_items = endpoint_facts.get("uncertain_or_avoid") or []
    if isinstance(avoid_items, list):
        for item in avoid_items:
            item_text = str(item).strip().lower()
            if not item_text:
                continue
            for sensitive in (
                "doorway",
                "closet",
                "second step",
                "landing",
                "hallway end",
                "top of the stairs",
                "bottom of the stairs",
                "counter",
                "service area",
                "kitchen",
                "patio",
                "balcony",
                "porch",
                "outside",
                "exterior",
                "appliance",
                "stove",
                "refrigerator",
            ):
                if sensitive in recommended:
                    continue
                if (
                    sensitive == "top of the stairs"
                    and re.search(r"\b(?:near|by|beside)\s+the\s+top\s+of\s+the\s+stairs\b", destination)
                ):
                    continue
                if sensitive in item_text and sensitive in destination:
                    errors.append(
                        f"endpoint fact pass marked final destination term uncertain: {sensitive}"
                    )
    if not destination:
        sensitive_stop = re.search(
            r"\b(?:stop|stopping|wait|halt|finish|end|stand|remain)\b[^.!?;]*",
            instruction_lower,
        )
        if sensitive_stop:
            destination = sensitive_stop.group(0)
        else:
            return errors

    if recommended:
        recommended_tokens = {
            token
            for token in re.findall(r"[a-z]+", recommended)
            if len(token) >= 4 and token not in {"stop", "wait", "near", "area", "with"}
        }
        destination_tokens = set(re.findall(r"[a-z]+", destination))
        if recommended_tokens and not (recommended_tokens & destination_tokens):
            if unsupported in {"no", "unclear", "false"} or preserve is False:
                errors.append("final destination may not use the recommended endpoint phrase")
        if "stair" in recommended and any(
            term in destination
            for term in ("bottom of the stairs", "top of the stairs", "second step", "landing")
        ):
            if not any(term in recommended for term in ("bottom", "top", "second", "landing")):
                errors.append("may add unsupported precise stair endpoint wording")
        if "stair" in recommended:
            landmark_terms = (
                "window",
                "picture",
                "pictures",
                "artwork",
                "artworks",
                "painting",
                "paintings",
                "framed",
            )
            if any(term in destination for term in landmark_terms) and "near" in destination:
                errors.append(
                    "final stair destination adds secondary wall landmarks that may shift "
                    "the stop point; prefer the recommended stair stop phrase"
                )
    return errors


def endpoint_specificity_hints(
    instruction: str,
    endpoint_facts: Optional[Dict[str, Any]],
) -> List[str]:
    landmarks = concrete_endpoint_landmarks(endpoint_facts)
    if not landmarks:
        return []

    normalized = normalize_instruction(instruction)
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", normalized)
        if sentence.strip()
    ]
    stop_index = None
    for index, sentence in enumerate(sentences):
        if re.search(
            r"\b(?:stop|stopping|wait|halt|finish|end|stand|remain)\b",
            sentence,
            flags=re.IGNORECASE,
        ):
            stop_index = index
    if stop_index is None:
        return []

    context = " ".join(sentences[max(0, stop_index - 1) : stop_index + 1])
    if phrase_mentions_any_landmark(context, landmarks):
        return []
    if not generic_stop_phrase(context):
        return []

    preferred = preferred_endpoint_stop_phrase(endpoint_facts)
    if not preferred:
        return []
    return [
        "final stop is too generic despite visible endpoint landmarks; "
        f"use a concrete endpoint anchor such as '{preferred}'"
    ]


def final_blocking_hints(
    instruction: str,
    actions: Sequence[int],
    start_facts: Optional[Dict[str, Any]],
    endpoint_facts: Optional[Dict[str, Any]],
    route_plan: Optional[Dict[str, Any]],
) -> List[str]:
    """Deterministic checks for constraints that should force a retry."""
    hints = []
    for hint in route_language_hints(instruction=instruction, actions=actions):
        if (
            "first sentence" in hint
            or "begins with turn around" in hint
        ):
            hints.append(hint)
    hints.extend(start_consistency_hints(instruction, start_facts))
    for hint in endpoint_consistency_hints(instruction, endpoint_facts):
        if (
            "did not support final destination term" in hint
            or "marked final destination term uncertain" in hint
            or "unsupported precise stair" in hint
        ):
            hints.append(hint)
    hints.extend(endpoint_specificity_hints(instruction, endpoint_facts))
    return sorted(set(hints))


def final_destination_turn_hints(
    instruction: str,
    actions: Sequence[int],
) -> List[str]:
    runs = action_runs(actions)
    turn_runs = []
    last_non_stop = -1
    for action, start, end in runs:
        last_non_stop = max(last_non_stop, end)
        if action in {2, 3}:
            direction = "left" if action == 2 else "right"
            degrees = int((end - start + 1) * TURN_DEGREES)
            turn_runs.append((direction, start, end, degrees))
    if not turn_runs or last_non_stop < 0:
        return []

    late_cutoff = int((last_non_stop + 1) * 0.45)
    late_turns = [item for item in turn_runs if item[2] >= late_cutoff]
    major_late_turns = [item for item in late_turns if item[3] >= 30]
    if major_late_turns:
        expected_direction, start, end, degrees = major_late_turns[-1]
    elif late_turns:
        expected_direction, start, end, degrees = late_turns[-1]
    else:
        expected_direction, start, end, degrees = turn_runs[-1]

    normalized = normalize_instruction(instruction)
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]
    if not sentences:
        return []
    stop_index = None
    for index, sentence in enumerate(sentences):
        if re.search(r"\b(?:stop|stopping|wait|halt|finish|end|remain|stand)\b", sentence, re.IGNORECASE):
            stop_index = index
    if stop_index is None:
        return []
    context = " ".join(sentences[max(0, stop_index - 1) : stop_index + 1]).lower()
    matches = list(re.finditer(r"\bturn\s+(left|right)\b", context))
    if not matches:
        return []
    stated_direction = matches[-1].group(1)
    if stated_direction == expected_direction:
        return []
    return [
        "final destination turn says "
        f"turn {stated_direction}, but the late action-derived turn before the "
        f"final approach is {expected_direction} (about {degrees} degrees); "
        "inspect the final route views before accepting the destination turn"
    ]


def last_late_major_turn(
    actions: Sequence[int],
    min_degrees: int = 45,
    start_fraction: float = 0.55,
) -> Optional[Dict[str, Any]]:
    runs = action_runs(actions)
    if not runs:
        return None
    last_non_stop = max(end for action, _start, end in runs if action != 0)
    cutoff = int((last_non_stop + 1) * start_fraction)
    candidates = []
    for action, start, end in runs:
        if action not in {2, 3} or end < cutoff:
            continue
        degrees = int((end - start + 1) * TURN_DEGREES)
        if degrees < min_degrees:
            continue
        candidates.append(
            {
                "direction": "left" if action == 2 else "right",
                "start": start,
                "end": end,
                "degrees": degrees,
            }
        )
    return candidates[-1] if candidates else None


def destination_entry_terms(
    endpoint_facts: Optional[Dict[str, Any]],
    route_plan: Optional[Dict[str, Any]],
) -> List[str]:
    texts = []
    if endpoint_facts:
        texts.extend(
            str(endpoint_facts.get(key, ""))
            for key in (
                "visible_destination_area",
                "recommended_stop_phrase",
                "surface_evidence",
            )
        )
        for item in endpoint_facts.get("certain_landmarks") or []:
            texts.append(str(item))
    if route_plan:
        destination = route_plan.get("destination") or {}
        if isinstance(destination, dict):
            texts.extend(
                str(destination.get(key, ""))
                for key in (
                    "safe_stop_phrase",
                    "visual_evidence",
                )
            )

    generic_terms = [
        "living room",
        "bedroom",
        "bathroom",
        "kitchen",
        "dining room",
        "foyer",
        "entryway",
        "patio",
        "balcony",
        "porch",
        "couch",
        "sofa",
        "fireplace",
        "bed",
        "sink",
        "counter",
        "table",
        "chair",
    ]
    joined = " ".join(texts).lower()
    terms = [term for term in generic_terms if term in joined]
    for phrase in re.findall(r"\b(?:purple|blue|white|wooden|round)\s+[a-z]+\b", joined):
        if any(
            skip in phrase
            for skip in (
                "door",
                "doorway",
                "stair",
                "stairs",
                "staircase",
                "floor",
                "wall",
                "ceiling",
            )
        ):
            continue
        terms.append(phrase)
    return sorted(set(terms), key=lambda term: (-len(term), term))


def endpoint_supports_directional_entry(
    endpoint_facts: Optional[Dict[str, Any]],
) -> bool:
    if not endpoint_facts:
        return True
    visible = str(endpoint_facts.get("visible_destination_area", "")).lower()
    recommended = str(endpoint_facts.get("recommended_stop_phrase", "")).lower()
    landmarks = " ".join(str(item).lower() for item in endpoint_facts.get("certain_landmarks") or [])
    text = f"{visible} {recommended} {landmarks}"

    if (
        re.search(r"\b(?:door|doors|doorway|entrance|threshold)\b", text)
        and re.search(r"\b(?:near|just before|just inside|inside|by|at)\b", recommended)
        and (
            re.search(r"\b(?:balcony|patio|porch|outdoor|outside|exterior)\b", text)
            or "threshold" in text
        )
    ):
        return False

    strong_entry_terms = (
        "living room",
        "bedroom",
        "bathroom",
        "kitchen",
        "dining room",
        "patio",
        "balcony",
        "porch",
        "outdoor",
        "exterior",
    )
    if any(term in text for term in strong_entry_terms):
        return True
    if re.search(r"\b(?:room|area)\b", visible) and not re.search(
        r"\b(?:hallway|corridor|stairs?|staircase|landing|step)\b",
        visible,
    ):
        return True
    if "doorway" in recommended and not re.search(
        r"\b(?:hallway|corridor|stairs?|staircase|landing|step)\b",
        recommended,
    ):
        return True
    return False


def endpoint_is_boundary_stop(endpoint_facts: Optional[Dict[str, Any]]) -> bool:
    if not endpoint_facts:
        return False
    visible = str(endpoint_facts.get("visible_destination_area", "")).lower()
    recommended = str(endpoint_facts.get("recommended_stop_phrase", "")).lower()
    surface = str(endpoint_facts.get("surface_evidence", "")).lower()
    landmarks = " ".join(str(item).lower() for item in endpoint_facts.get("certain_landmarks") or [])
    text = f"{visible} {recommended} {surface} {landmarks}"
    if not re.search(r"\b(?:door|doors|doorway|entrance|threshold)\b", text):
        return False
    if not re.search(r"\b(?:near|just before|just inside|inside|by|at)\b", recommended):
        return False
    return bool(
        re.search(r"\b(?:balcony|patio|porch|outdoor|outside|exterior)\b", text)
        or "threshold" in text
    )


def requires_directional_entry_mention(
    endpoint_facts: Optional[Dict[str, Any]],
) -> bool:
    if not endpoint_supports_directional_entry(endpoint_facts):
        return False
    if endpoint_is_boundary_stop(endpoint_facts):
        return False
    return True


def route_plan_treats_late_turn_as_alignment(
    route_plan: Optional[Dict[str, Any]],
    expected: Dict[str, Any],
) -> bool:
    if not route_plan:
        return False
    expected_steps = f"{expected['start']}-{expected['end']}"
    destination = route_plan.get("destination") or {}
    destination_text = ""
    if isinstance(destination, dict):
        destination_text = " ".join(
            str(destination.get(key, ""))
            for key in (
                "final_approach_direction",
                "final_direction_evidence",
                "visual_evidence",
            )
        )

    segment_texts = []
    for segment in route_plan.get("major_segments") or []:
        if isinstance(segment, dict):
            segment_texts.append(str(segment.get("movement", "")))

    coverage = route_plan.get("action_turn_coverage") or []
    if not isinstance(coverage, list):
        coverage = []
    for item in coverage:
        if not isinstance(item, dict):
            continue
        if str(item.get("turn_steps", "")).strip() != expected_steps:
            continue
        if str(item.get("direction", "")).strip().lower() != str(expected["direction"]):
            continue
        write_flag = str(item.get("write_in_instruction", "")).strip().lower()
        segment_index = covered_segment_index(item)
        movement = (
            segment_texts[segment_index]
            if segment_index is not None and 0 <= segment_index < len(segment_texts)
            else ""
        )
        combined = " ".join(
            [
                str(item.get("reason", "")),
                str(item.get("covered_by_segment", "")),
                movement,
                destination_text,
            ]
        )
        if write_flag in {"no", "false", "omit", "unclear"} and (
            is_orientation_alignment_text(combined)
            or re.search(r"\b(?:minor|slight)\b", combined, flags=re.IGNORECASE)
        ):
            return True

    if isinstance(destination, dict):
        final_direction = str(
            destination.get("final_approach_direction", "")
        ).strip().lower()
        if final_direction in {"straight", "none", "unclear"} and (
            is_orientation_alignment_text(destination_text)
            or re.search(r"\b(?:minor|slight)\b", destination_text, flags=re.IGNORECASE)
        ):
            return True
    return False


def covered_segment_index(item: Dict[str, Any]) -> Optional[int]:
    text = str(item.get("covered_by_segment", ""))
    match = re.search(r"major_segments\[(\d+)\]", text)
    if match:
        return int(match.group(1))
    return None


def is_orientation_alignment_text(text: str) -> bool:
    lower = text.lower()
    return bool(
        re.search(
            r"\b(?:to face|face the|align(?:ing)? with|heading adjustment|same-area|same area|"
            r"same open area|same room|view directly|reorientation)\b",
            lower,
        )
    )


def is_route_transition_text(text: str) -> bool:
    lower = text.lower()
    return bool(
        re.search(
            r"\b(?:exit|leave|out of|through|into|onto|enter|up|down|branch|"
            r"stair|stairs|staircase|hallway|corridor)\b",
            lower,
        )
    )


def naturalize_orientation_movement(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(
        r"\bTurn\s+(?:left|right)\s+to\s+face\s+the\s+([^,.!?]{3,70}?),\s*"
        r"then\s+turn\s+(?:left|right)\s+to\s+align\s+with\s+the\s+([^,.!?]{3,70}?)(?=[.!?]|$)",
        lambda match: (
            f"Continue through the area toward the {match.group(1).strip()}, "
            f"using the {match.group(2).strip()} as landmarks"
        ),
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\bTurn\s+(?:left|right)\s+to\s+face\s+the\s+([^,.!?]{3,70}?)(?:\s+directly)?\s+"
        r"and\s+walk\s+forward\s+to\s+the\s+([^,.!?]{3,70}?)(?=[.!?]|$)",
        lambda match: (
            f"Walk forward toward the {match.group(1).strip()} and continue to "
            f"the {match.group(2).strip()}"
        ),
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\bTurn\s+(?:left|right)\s+to\s+face\s+the\s+([^,.!?]{3,70}?)(?:\s+directly)?"
        r"(?=[.!?]|$)",
        lambda match: f"Continue toward the {match.group(1).strip()}",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r",?\s*then\s+turn\s+(?:left|right)\s+to\s+align\s+with\s+the\s+([^,.!?]{3,70}?)(?=[.!?]|$)",
        lambda match: f", using the {match.group(1).strip()} as landmarks",
        cleaned,
        flags=re.IGNORECASE,
    )
    return normalize_instruction(cleaned)


def sanitize_route_plan_orientation_language(
    route_plan: Dict[str, Any],
    endpoint_facts: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Keep planner output at human route-segment level.

    The VLM sometimes treats camera heading adjustments inside one open room as
    mandatory left/right commands. Those commands make unnatural R2R text and
    can conflict with the real branch/entry turns, so demote them before the
    writer and critics see the checklist.
    """
    warnings = route_plan.setdefault("_sanitizer_warnings", [])
    segments = route_plan.get("major_segments") or []
    segment_texts = []
    if isinstance(segments, list):
        for segment in segments:
            if not isinstance(segment, dict):
                segment_texts.append("")
                continue
            movement = str(segment.get("movement", ""))
            fixed = naturalize_orientation_movement(movement)
            if fixed != normalize_instruction(movement):
                segment["movement"] = fixed
                warning = "naturalized same-area face/align wording in route plan"
                if warning not in warnings:
                    warnings.append(warning)
            segment_texts.append(str(segment.get("movement", "")))

    coverage = route_plan.get("action_turn_coverage") or []
    if isinstance(coverage, list):
        boundary_stop = endpoint_is_boundary_stop(endpoint_facts)
        for item in coverage:
            if not isinstance(item, dict):
                continue
            reason = str(item.get("reason", ""))
            segment_index = covered_segment_index(item)
            movement = (
                segment_texts[segment_index]
                if segment_index is not None and 0 <= segment_index < len(segment_texts)
                else ""
            )
            combined = f"{reason} {movement}"
            is_final_boundary_alignment = (
                boundary_stop
                and "final destination entry" in str(item.get("covered_by_segment", "")).lower()
            )
            if (
                is_final_boundary_alignment
                or (
                    is_orientation_alignment_text(combined)
                    and not is_route_transition_text(combined)
                )
            ):
                if str(item.get("write_in_instruction", "")).strip().lower() in {
                    "yes",
                    "true",
                    "required",
                }:
                    item["write_in_instruction"] = "no"
                    item["reason"] = (
                        "Same-area orientation or endpoint-facing adjustment; "
                        "omit as a separate human command and describe the "
                        "landmark approach naturally."
                    )
                    warning = "demoted same-area orientation turn in action_turn_coverage"
                    if warning not in warnings:
                        warnings.append(warning)
    return route_plan


def align_route_plan_with_late_action_turns(
    route_plan: Dict[str, Any],
    actions: Sequence[int],
    endpoint_facts: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    expected = last_late_major_turn(actions)
    if not expected:
        return route_plan
    if not endpoint_supports_directional_entry(endpoint_facts):
        return route_plan

    terms = destination_entry_terms(endpoint_facts, route_plan)
    if not terms:
        return route_plan

    if route_plan_treats_late_turn_as_alignment(route_plan, expected):
        return route_plan

    must_write_entry_turn = requires_directional_entry_mention(endpoint_facts)
    expected_direction = str(expected["direction"])
    opposite = "right" if expected_direction == "left" else "left"
    term_pattern = "|".join(re.escape(term) for term in terms)
    opposite_entry_pattern = re.compile(
        rf"\bturn\s+{opposite}\b(?P<tail>[^.!?]{{0,100}}"
        rf"\b(?:into|toward|towards|onto|through|to enter|enter)\b"
        rf"[^.!?]{{0,100}}\b(?:{term_pattern})\b)",
        re.IGNORECASE,
    )
    expected_entry_pattern = re.compile(
        rf"\bturn\s+{expected_direction}\b[^.!?]{{0,100}}"
        rf"\b(?:into|toward|towards|onto|through|to enter|enter)\b"
        rf"[^.!?]{{0,100}}\b(?:{term_pattern})\b",
        re.IGNORECASE,
    )

    changed = False

    def replace_opposite_entry(text: str) -> str:
        nonlocal changed

        def replacement(match: re.Match[str]) -> str:
            nonlocal changed
            changed = True
            return f"turn {expected_direction}{match.group('tail')}"

        return opposite_entry_pattern.sub(replacement, text)

    for segment in route_plan.get("major_segments") or []:
        if not isinstance(segment, dict):
            continue
        movement = str(segment.get("movement", ""))
        fixed = replace_opposite_entry(movement)
        if fixed != movement:
            segment["movement"] = fixed
            uncertain = segment.get("uncertain")
            if not isinstance(uncertain, list):
                uncertain = []
            uncertain.append(
                f"opposite final destination entry turn ({opposite}) was corrected "
                f"from late action steps {expected['start']}-{expected['end']}"
            )
            segment["uncertain"] = uncertain

    avoid_claims = []
    expected_turn_avoid_pattern = re.compile(
        rf"\bturn\s+{expected_direction}\b",
        re.IGNORECASE,
    )
    for item in route_plan.get("avoid_claims") or []:
        text = str(item)
        lower = text.lower()
        if expected_entry_pattern.search(lower) or (
            expected_turn_avoid_pattern.search(lower)
            and (
                lower.strip(" .;") == f"turn {expected_direction}"
                or any(term in lower for term in terms)
                or "end of the hallway" in lower
                or "end of the hall" in lower
                or "final" in lower
                or "destination" in lower
            )
        ):
            changed = True
            continue
        avoid_claims.append(text)
    opposite_avoid = (
        f"turn {opposite} into/toward the final destination area; late action "
        f"steps {expected['start']}-{expected['end']} support {expected_direction}"
    )
    if opposite_avoid not in avoid_claims:
        avoid_claims.append(opposite_avoid)
        changed = True
    route_plan["avoid_claims"] = avoid_claims

    destination = route_plan.get("destination")
    if not isinstance(destination, dict):
        destination = {}
    destination_avoid = []
    expected_turn_pattern = re.compile(rf"\bturn\s+{expected_direction}\b", re.IGNORECASE)
    for item in destination.get("avoid") or []:
        text = str(item)
        lower = text.lower()
        if expected_turn_pattern.search(lower) and (
            lower.strip(" .;") == f"turn {expected_direction}"
            or any(term in lower for term in terms)
            or "final" in lower
            or "destination" in lower
        ):
            changed = True
            continue
        destination_avoid.append(text)
    destination["avoid"] = destination_avoid
    destination["final_approach_direction"] = expected_direction
    evidence = str(destination.get("final_direction_evidence", "")).strip()
    action_evidence = (
        f"Late action steps {expected['start']}-{expected['end']} form a major "
        f"{expected_direction} turn of about {expected['degrees']} degrees before "
        "the final approach."
    )
    if action_evidence not in evidence:
        destination["final_direction_evidence"] = (
            f"{evidence} {action_evidence}".strip()
        )
    destination["old_final_turn_supported"] = (
        "no" if opposite_entry_pattern.search(evidence) else destination.get(
            "old_final_turn_supported", "unclear"
        )
    )
    route_plan["destination"] = destination

    coverage = route_plan.get("action_turn_coverage")
    if not isinstance(coverage, list):
        coverage = []
    expected_steps = f"{expected['start']}-{expected['end']}"
    matched_coverage = False
    for item in coverage:
        if not isinstance(item, dict):
            continue
        if (
            str(item.get("direction", "")).strip().lower() == expected_direction
            and str(item.get("turn_steps", "")).strip() == expected_steps
        ):
            item["write_in_instruction"] = "yes" if must_write_entry_turn else "no"
            if must_write_entry_turn:
                item["reason"] = (
                    "Late major action turn before a destination area entry; "
                    "preserve it as the final destination entry direction."
                )
            else:
                item["reason"] = (
                    "Late major action turn before a boundary/threshold stop; "
                    "avoid the opposite turn but neutral continue/approach "
                    "wording is preferable."
                )
            matched_coverage = True
            changed = True
    if not matched_coverage:
        coverage.append(
            {
                "turn_steps": expected_steps,
                "direction": expected_direction,
                "degrees": int(expected["degrees"]),
                "write_in_instruction": "yes" if must_write_entry_turn else "no",
                "covered_by_segment": "final destination entry",
                "reason": (
                    "Late major action turn before a destination area entry; "
                    "preserve it as the final destination entry direction."
                    if must_write_entry_turn
                    else "Late major action turn before a boundary/threshold stop; "
                    "avoid the opposite turn but neutral continue/approach wording "
                    "is preferable."
                ),
            }
        )
        changed = True
    route_plan["action_turn_coverage"] = coverage

    guidance = str(route_plan.get("writing_guidance", "")).strip()
    action_guidance = (
        f"For the final destination entry, do not write turn {opposite}; use "
        f"turn {expected_direction} if visually supported, otherwise use neutral "
        "enter/continue wording."
    )
    if action_guidance not in guidance:
        route_plan["writing_guidance"] = f"{guidance} {action_guidance}".strip()
        changed = True

    if changed:
        warning = (
            f"aligned final destination entry with late major action turn: "
            f"{expected_direction} at steps {expected['start']}-{expected['end']}"
        )
        warnings = route_plan.setdefault("_sanitizer_warnings", [])
        if warning not in warnings:
            warnings.append(warning)
    return route_plan


def destination_entry_turn_hints(
    instruction: str,
    actions: Sequence[int],
    endpoint_facts: Optional[Dict[str, Any]],
    route_plan: Optional[Dict[str, Any]],
) -> List[str]:
    """Block final-room entry turns that contradict the late action turn.

    This is intentionally narrower than final_destination_turn_hints: it only
    fires when the same clause says turn left/right into/toward a destination
    room, doorway, or landmark. Earlier hallway turns should not be rejected just
    because a later alignment turn has the opposite direction.
    """
    expected = last_late_major_turn(actions)
    if not expected:
        return []
    if not endpoint_supports_directional_entry(endpoint_facts):
        return []
    expected_direction = str(expected["direction"])
    opposite = "right" if expected_direction == "left" else "left"

    normalized = normalize_instruction(instruction)
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]
    if not sentences:
        return []
    stop_index = None
    for index, sentence in enumerate(sentences):
        if re.search(r"\b(?:stop|stopping|wait|halt|finish|end|remain|stand)\b", sentence, re.IGNORECASE):
            stop_index = index
    if stop_index is None:
        context_sentences = sentences[-2:]
    else:
        context_sentences = sentences[max(0, stop_index - 1) : stop_index + 1]

    if route_plan_treats_late_turn_as_alignment(route_plan, expected):
        return []

    terms = destination_entry_terms(endpoint_facts, route_plan)
    if not terms:
        return []
    term_pattern = "|".join(re.escape(term) for term in terms)
    no_nested_turn = r"(?:(?!\bturn\s+(?:left|right)\b).)"
    entry_pattern = re.compile(
        rf"\bturn\s+(left|right)\b{no_nested_turn}{{0,100}}"
        rf"\b(?:into|toward|towards|onto|through|to enter|enter)\b"
        rf"{no_nested_turn}{{0,100}}\b(?:{term_pattern})\b",
        re.IGNORECASE,
    )
    hints = []
    for sentence in context_sentences:
        for match in entry_pattern.finditer(sentence):
            stated_direction = match.group(1).lower()
            if stated_direction != expected_direction:
                hints.append(
                    "final destination entry says "
                    f"turn {stated_direction}, but the last major late action turn "
                    f"is {expected_direction} at steps {expected['start']}-{expected['end']} "
                    f"(about {expected['degrees']} degrees). Rewrite the destination "
                    f"entry as turn {expected_direction} if visually supported, or use "
                    "neutral wording such as continue/enter into the destination area."
                )
    if hints:
        return sorted(set(hints))

    # If the final approach uses the opposite side as an entry without the verb
    # "turn", still flag it when it explicitly targets the destination.
    side_entry_pattern = re.compile(
        rf"\b(?:enter|go|walk|head|continue|move)\b[^.!?]{{0,80}}"
        rf"\b(?:on|to|toward|towards)\s+(?:the\s+)?{opposite}\b"
        rf"[^.!?]{{0,100}}\b(?:{term_pattern})\b",
        re.IGNORECASE,
    )
    for sentence in context_sentences:
        if side_entry_pattern.search(sentence):
            hints.append(
                "final destination entry uses the opposite side of the late "
                f"action turn; expected {expected_direction} or neutral wording, "
                f"not {opposite}."
            )
    return sorted(set(hints))


def unsupported_passing_hints(
    instruction: str,
    route_plan: Optional[Dict[str, Any]],
) -> List[str]:
    if not route_plan:
        return []
    lower = normalize_instruction(instruction).lower()
    pass_claims = []
    pass_patterns = (
        r"\bpassing\s+(?:the\s+|a\s+|an\s+)?([^.!?;,]{3,60})",
        r"\bwalk(?:ing)?\s+past\s+(?:the\s+|a\s+|an\s+)?([^.!?;,]{3,60})",
        r"\bgo(?:ing)?\s+past\s+(?:the\s+|a\s+|an\s+)?([^.!?;,]{3,60})",
        r"\bmov(?:e|ing)\s+past\s+(?:the\s+|a\s+|an\s+)?([^.!?;,]{3,60})",
        r"\bcontinue(?:ing)?(?:\s+straight)?\s+past\s+(?:the\s+|a\s+|an\s+)?([^.!?;,]{3,60})",
        r"\bproceed(?:ing)?(?:\s+straight)?\s+past\s+(?:the\s+|a\s+|an\s+)?([^.!?;,]{3,60})",
    )
    for pattern in pass_patterns:
        for match in re.finditer(pattern, lower):
            phrase = re.split(
                r"\b(?:then|before|after|until|toward|towards|into|through|and)\b",
                match.group(1),
                maxsplit=1,
            )[0].strip(" ,")
            if phrase:
                pass_claims.append(phrase)
    if not pass_claims:
        return []

    def token_set(values: Any) -> set[str]:
        tokens: set[str] = set()
        if isinstance(values, list):
            iterator = values
        else:
            iterator = [values]
        for value in iterator:
            for token in re.findall(r"[a-z]+", str(value).lower()):
                if len(token) >= 5:
                    tokens.add(token)
        return tokens

    passed_tokens = token_set(route_plan.get("passed_landmarks") or [])
    near_not_passed_tokens = token_set(route_plan.get("near_not_passed_landmarks") or [])
    confirmed_tokens = token_set(route_plan.get("safe_landmarks") or [])
    for segment in route_plan.get("major_segments") or []:
        if isinstance(segment, dict):
            confirmed_tokens.update(token_set(segment.get("confirmed_landmarks") or []))

    pass_movement_texts = []
    for segment in route_plan.get("major_segments") or []:
        if not isinstance(segment, dict):
            continue
        movement = str(segment.get("movement", "")).lower()
        if re.search(r"\b(?:pass|passing|past)\b", movement):
            pass_movement_texts.append(movement)
    pass_movement_text = " ".join(pass_movement_texts)

    hints = []
    stop_tokens = {
        "along",
        "around",
        "before",
        "after",
        "toward",
        "towards",
        "through",
        "into",
        "with",
        "near",
        "wall",
        "walls",
        "room",
        "area",
        "hall",
        "hallway",
        "corridor",
    }
    strict_pass_terms = {
        "fireplace",
        "television",
        "artwork",
        "painting",
        "picture",
        "mirror",
        "window",
    }
    for claim in pass_claims:
        tokens = [
            token
            for token in re.findall(r"[a-z]+", claim)
            if len(token) >= 5 and token not in stop_tokens
        ]
        strict_claim = any(term in tokens or term in claim for term in strict_pass_terms)
        if tokens and near_not_passed_tokens and any(
            token in near_not_passed_tokens for token in tokens
        ):
            hints.append(
                f"instruction says it passes '{claim}', but route_plan marks that "
                "landmark as near-not-passed; use near/by/toward wording instead"
            )
            continue
        if tokens and passed_tokens and not any(token in passed_tokens for token in tokens):
            if (
                not strict_claim
                and confirmed_tokens
                and any(token in confirmed_tokens for token in tokens)
            ):
                continue
            hints.append(
                f"instruction says it passes '{claim}', but that landmark is not "
                "listed in route_plan.passed_landmarks; inspect before accepting"
            )
            continue
        if not pass_movement_text:
            if strict_claim or not (
                confirmed_tokens and any(token in confirmed_tokens for token in tokens)
            ):
                hints.append(
                    f"instruction says it passes '{claim}', but the route plan does "
                    "not explicitly say to pass/go past that landmark; inspect whether "
                    "the route really moves beyond it or only turns near it"
                )
        elif tokens and not any(token in pass_movement_text for token in tokens):
            if (
                not strict_claim
                and confirmed_tokens
                and any(token in confirmed_tokens for token in tokens)
            ):
                continue
            hints.append(
                f"instruction says it passes '{claim}', but that passed landmark "
                "is not named in any route-plan pass/past movement; inspect before accepting"
            )
    return sorted(set(hints))


def route_plan_final_direction_hints(
    instruction: str,
    route_plan: Optional[Dict[str, Any]],
    endpoint_facts: Optional[Dict[str, Any]] = None,
) -> List[str]:
    if not route_plan:
        return []
    destination = route_plan.get("destination") or {}
    if not isinstance(destination, dict):
        return []
    expected = str(destination.get("final_approach_direction", "")).strip().lower()
    if expected not in {"left", "right", "straight", "none", "unclear"}:
        return []
    if expected in {"left", "right"} and not endpoint_supports_directional_entry(endpoint_facts):
        return []

    normalized = normalize_instruction(instruction)
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]
    if not sentences:
        return []
    destination_text = " ".join(
        str(destination.get(key, ""))
        for key in ("safe_stop_phrase", "visual_evidence", "final_direction_evidence")
    ).lower()
    destination_tokens = {
        token
        for token in re.findall(r"[a-z]+", destination_text)
        if len(token) >= 5
        and token
        not in {
            "visible",
            "final",
            "frames",
            "directly",
            "moving",
            "toward",
            "towards",
            "without",
            "turning",
            "approach",
            "observation",
            "agent",
            "shows",
            "ahead",
            "floor",
            "area",
            "open",
            "path",
            "leads",
            "room",
            "rooms",
            "living",
            "dining",
            "kitchen",
            "hallway",
            "corridor",
        }
    }
    destination_context = [
        sentence.lower()
        for sentence in sentences
        if not destination_tokens
        or any(token in sentence.lower() for token in destination_tokens)
    ]
    avoid_items = [
        str(item).lower()
        for item in (route_plan.get("avoid_claims") or []) + (destination.get("avoid") or [])
    ]
    segment_items = [
        str(segment.get("movement", "")).lower()
        for segment in (route_plan.get("major_segments") or [])
        if isinstance(segment, dict)
    ]

    def mentions_destination(text: str) -> bool:
        return not destination_tokens or any(token in text for token in destination_tokens)

    def route_plan_supports_entry_turn(direction: str) -> bool:
        for segment_text in segment_items:
            if not mentions_destination(segment_text):
                continue
            if re.search(
                rf"\bturn\s+{direction}\b[^.!?]{{0,90}}\b"
                r"(?:into|toward|towards|onto|to enter)\b",
                segment_text,
            ):
                return True
        return False

    def route_plan_avoids_entry_turn(direction: str) -> bool:
        for avoid_text in avoid_items:
            if not mentions_destination(avoid_text):
                continue
            if re.search(
                rf"\bturn\s+{direction}\b[^.!?]{{0,90}}\b"
                r"(?:into|toward|towards|onto|to enter)?\b",
                avoid_text,
            ):
                return True
        return False

    def route_plan_requires_action_turn(direction: str) -> bool:
        coverage = route_plan.get("action_turn_coverage") or []
        if not isinstance(coverage, list):
            return False
        for item in coverage:
            if not isinstance(item, dict):
                continue
            if str(item.get("direction", "")).strip().lower() != direction:
                continue
            if str(item.get("write_in_instruction", "")).strip().lower() in {
                "yes",
                "true",
                "required",
            }:
                return True
        return False

    stop_index = None
    for index, sentence in enumerate(sentences):
        if re.search(r"\b(?:stop|stopping|wait|halt|finish|end|remain|stand)\b", sentence, re.IGNORECASE):
            stop_index = index
    if stop_index is None:
        context = " ".join(destination_context)
    else:
        context = " ".join(sentences[max(0, stop_index - 1) : stop_index + 1]).lower()
    turn_matches = list(re.finditer(r"\bturn\s+(left|right)\b", context))
    if expected in {"left", "right"}:
        destination_turn_matches = []
        entry_terms = destination_entry_terms(endpoint_facts, route_plan)
        if entry_terms:
            term_pattern = "|".join(re.escape(term) for term in entry_terms)
            no_nested_turn = r"(?:(?!\bturn\s+(?:left|right)\b).)"
            entry_pattern = re.compile(
                rf"\bturn\s+(left|right)\b{no_nested_turn}{{0,100}}"
                rf"\b(?:into|toward|towards|onto|through|to enter|enter)\b"
                rf"{no_nested_turn}{{0,100}}\b(?:{term_pattern})\b",
                re.IGNORECASE,
            )
            for sentence in destination_context:
                destination_turn_matches.extend(entry_pattern.finditer(sentence))
        else:
            for sentence in destination_context:
                destination_turn_matches.extend(
                    re.finditer(
                        r"\bturn\s+(left|right)\b[^.!?]{0,90}\b"
                        r"(?:into|toward|towards|to enter)\b",
                        sentence,
                    )
                )
        relevant_matches = destination_turn_matches
        if not relevant_matches and len(turn_matches) == 1:
            relevant_matches = turn_matches
        if relevant_matches and relevant_matches[-1].group(1) != expected:
            return [
                "route plan final_approach_direction is "
                f"{expected}, but the final approach sentence says turn "
                f"{relevant_matches[-1].group(1)}; inspect final_direction_evidence"
            ]
    elif expected in {"straight", "none", "unclear"}:
        destination_turn_text = " ".join(destination_context) or context
        if destination_tokens:
            destination_term_pattern = "|".join(
                re.escape(token) for token in sorted(destination_tokens, key=len, reverse=True)
            )
            match = re.search(
                r"\bturn\s+(?:left|right)\b[^,.;!?]{0,60}\b"
                r"(?:into|toward|towards|to enter)\b"
                rf"[^,.;!?]{{0,60}}\b(?:{destination_term_pattern})\b",
                destination_turn_text,
            )
        else:
            match = re.search(
                r"\bturn\s+(?:left|right)\b[^.!?]{0,90}\b"
                r"(?:into|toward|towards|to enter)\b",
                destination_turn_text,
            )
        if match:
            direction_match = re.search(r"\bturn\s+(left|right)\b", match.group(0))
            direction = direction_match.group(1) if direction_match else ""
            if direction and route_plan_requires_action_turn(direction):
                return []
            context_has_turn = bool(re.search(
                r"\bturn\s+(?:left|right)\b[^.!?]{0,90}\b"
                r"(?:into|toward|towards|to enter)\b",
                context,
            ))
            if not (
                route_plan_avoids_entry_turn(direction)
                or (context_has_turn and not route_plan_supports_entry_turn(direction))
            ):
                return []
            return [
                "route plan does not support a directional final entry turn; "
                "use continue/approach/stop-near wording unless the images clearly justify it"
            ]
    return []


def route_plan_action_turn_coverage_hints(
    instruction: str,
    route_plan: Optional[Dict[str, Any]],
) -> List[str]:
    if not route_plan:
        return []
    coverage = route_plan.get("action_turn_coverage") or []
    if not isinstance(coverage, list):
        return []
    lower = normalize_instruction(instruction).lower()
    hints = []
    for item in coverage:
        if not isinstance(item, dict):
            continue
        write_flag = str(item.get("write_in_instruction", "")).strip().lower()
        if write_flag not in {"yes", "true", "required"}:
            continue
        direction = str(item.get("direction", "")).strip().lower()
        if direction not in {"left", "right"}:
            continue
        turn_steps = str(item.get("turn_steps", "")).strip()
        direction_pattern = (
            rf"\b(?:turn|bear|veer|go|head|take|enter|move)\b[^.!?]{{0,45}}\b{direction}\b"
            rf"|\b{direction}\b[^.!?]{{0,45}}\b(?:turn|doorway|door|hall|hallway|room|branch|stair|stairs)\b"
        )
        if not re.search(direction_pattern, lower):
            hints.append(
                "route_plan.action_turn_coverage requires the action-derived "
                f"{direction} turn at steps {turn_steps or '?'} to be written; "
                "include it as a natural route turn or doorway/branch choice"
            )
    return sorted(set(hints))


def route_plan_coverage_hints(
    instruction: str,
    route_plan: Optional[Dict[str, Any]],
) -> List[str]:
    if not route_plan:
        return []
    segments = route_plan.get("major_segments") or []
    if not isinstance(segments, list) or not segments:
        return []

    hints = []
    instruction_lower = normalize_instruction(instruction).lower()
    first_segment = segments[0] if isinstance(segments[0], dict) else {}
    first_movement = str(first_segment.get("movement", "")).lower()
    first_landmarks = first_segment.get("confirmed_landmarks") or []
    first_landmark_text = " ".join(str(item).lower() for item in first_landmarks)
    first_text = f"{first_movement} {first_landmark_text}"

    exit_terms = (
        "exit",
        "leave",
        "walk out",
        "go out",
        "step out",
        "head out",
        "out of the room",
        "out of the bedroom",
        "out of the bathroom",
        "through the door",
    )
    instruction_exit_terms = (
        "exit",
        "leave",
        "walk out",
        "go out",
        "step out",
        "head out",
        "out of the room",
        "out of the bedroom",
        "out of the bathroom",
        "through the door",
    )
    if any(term in first_text for term in exit_terms) and not any(
        term in instruction_lower for term in instruction_exit_terms
    ):
        hints.append(
            "route plan first segment includes exiting/leaving a room, but the "
            "generated instruction appears to start after that transition"
        )

    # Very coarse landmark coverage: require landmarks named in the segment
    # movement, not every object listed as visual evidence. Otherwise the writer
    # is pushed to mention side-wall landmarks as if the route passed them.
    for index, segment in enumerate(segments, start=1):
        if not isinstance(segment, dict):
            continue
        movement_text = str(segment.get("movement", "")).lower()
        if not movement_text:
            continue
        landmark_tokens = []
        for token in re.findall(r"[a-z]+", movement_text):
            if len(token) >= 5 and token not in {
                "white",
                "wooden",
                "walls",
                "floor",
                "floors",
                "room",
                "rooms",
                "area",
                "entry",
                "enter",
                "turn",
                "walk",
                "move",
                "forward",
                "through",
                "toward",
                "towards",
                "along",
                "continue",
            }:
                landmark_tokens.append(token)
        if landmark_tokens and not any(token in instruction_lower for token in landmark_tokens):
            hints.append(
                f"route plan segment {index} landmarks may be omitted: "
                + ", ".join(sorted(set(landmark_tokens))[:4])
            )
    return hints


def route_plan_avoidance_hints(
    instruction: str,
    route_plan: Optional[Dict[str, Any]],
) -> List[str]:
    if not route_plan:
        return []

    lower = normalize_instruction(instruction).lower()
    hints = []
    sensitive_terms = (
        "bedroom",
        "bathroom",
        "kitchen",
        "patio",
        "balcony",
        "porch",
        "outside",
        "exterior",
        "living room",
        "dining room",
        "closet",
        "foyer",
        "entryway",
        "alcove",
        "doorway",
        "threshold",
        "landing",
        "counter",
        "service area",
        "hallway end",
        "end of the hallway",
        "top of the stairs",
        "bottom of the stairs",
        "second step",
        "on the stairs",
        "in front of the door",
    )

    avoid_texts = []
    for item in route_plan.get("avoid_claims") or []:
        avoid_texts.append(str(item).lower())
    for section_name in ("start", "destination"):
        section = route_plan.get(section_name) or {}
        if isinstance(section, dict):
            for item in section.get("avoid") or []:
                avoid_texts.append(str(item).lower())
    for segment in route_plan.get("major_segments") or []:
        if isinstance(segment, dict):
            for item in segment.get("uncertain") or []:
                avoid_texts.append(str(item).lower())

    for avoid_text in avoid_texts:
        for term in sensitive_terms:
            if term in avoid_text and term in lower:
                hints.append(
                    f"route plan marked '{term}' as uncertain/avoid, but the "
                    "generated instruction uses it"
                )
    return sorted(set(hints))


def start_consistency_hints(
    instruction: str,
    start_facts: Optional[Dict[str, Any]],
) -> List[str]:
    if not start_facts:
        return []
    if not bool(start_facts.get("must_preserve_start_boundary", False)):
        return []
    lower = normalize_instruction(instruction).lower()
    first_words = " ".join(re.findall(r"[a-z]+", lower)[:22])
    exit_patterns = (
        r"\bexit\b",
        r"\bleave\b",
        r"\bwalk out\b",
        r"\bgo out\b",
        r"\bstep out\b",
        r"\bhead out\b",
        r"\bout of\b",
        r"\bthrough the (?:door|doorway|opening)\b",
        r"\binto the (?:hallway|corridor|hall|next room)\b",
    )
    recommended = str(start_facts.get("recommended_start_phrase", "")).strip()
    recommended_words = " ".join(re.findall(r"[a-z]+", recommended.lower())[:22])
    if recommended_words and not any(
        re.search(pattern, recommended_words) for pattern in exit_patterns
    ):
        return []
    if not any(re.search(pattern, first_words) for pattern in exit_patterns):
        return [
            "start facts require preserving an initial room/doorway exit, but "
            f"the generated instruction does not clearly begin with it; recommended: {recommended}"
        ]
    return []


def style_quality_hints(instruction: str) -> List[str]:
    lower = normalize_instruction(instruction).lower()
    hints = []
    vague_patterns = (
        r"\bmake several (?:left |right )?turns\b",
        r"\bmaking several (?:left |right )?turns\b",
        r"\bnavigate (?:the )?(?:winding )?path\b",
        r"\bcontinue for a while\b",
        r"\bwalk for a while\b",
    )
    for pattern in vague_patterns:
        if re.search(pattern, lower):
            hints.append(
                "uses vague middle-route compression; describe the route-plan "
                "segments more concretely"
            )
            break
    if lower.count("walk forward") >= 3:
        hints.append("repeats 'walk forward' too often; improve R2R style")
    if re.search(r"\bstop there\.?$", lower):
        hints.append("uses vague final phrase 'stop there'; prefer an explicit landmark stop phrase")
    if not re.search(r"\b(?:stop|stopping|wait|halt|finish|end|stand|remain)\b", lower):
        hints.append("does not explicitly tell the navigator where to stop")
    if re.search(r"\b(?:about|approximately|around)?\s*\d{2,3}\s*degrees\b", lower):
        hints.append("uses numeric degree wording; replace it with natural R2R turn language")
    for sentence in re.split(r"[.!?]+", lower):
        stripped = sentence.strip(" ,;")
        if re.search(r"\bkeeping\s+(?:the\s+)?[a-z0-9 ,'-]+$", stripped) and not re.search(
            r"\bkeeping\b.*\b(?:on|to|along|near|beside|next to|toward|towards|ahead|behind)\b",
            stripped,
        ):
            hints.append("contains a dangling 'keeping ...' phrase; complete or remove it")
            break
    if re.search(r"\bstop on the flight\b", lower):
        hints.append("uses unnatural stair endpoint phrase 'stop on the flight'")
    if re.search(r"\b(?:and|or|to|toward|towards|with|where|near|by)\s*[.!?]", lower):
        hints.append("contains a dangling connector or preposition near sentence end")
    if re.search(r"\bstop near the [^,.!?]{2,60}? visible\b", lower):
        hints.append("uses unnatural final phrase 'stop near the ... visible'")
    if re.search(r",\s+with\s+[^,.!?]{3,70}?,\s+and\s+stop\b", lower):
        hints.append("uses awkward ', with ..., and stop' phrasing")
    if re.search(r"\bturn\s+(?:left|right)\s+to\s+align\b", lower):
        hints.append(
            "uses camera-alignment wording ('turn ... to align'); rewrite as "
            "natural landmark guidance"
        )
    face_turn_count = len(re.findall(r"\bturn\s+(?:left|right)\s+(?:to\s+)?face\b", lower))
    if face_turn_count >= 2 or re.search(
        r"\bturn\s+(?:left|right)\s+(?:to\s+)?face\b[^.!?]{0,100}\bthen\s+turn\b",
        lower,
    ):
        hints.append(
            "lists repeated face/align orientation turns; compress same-area "
            "heading changes into a natural route phrase"
        )
    target_counts = Counter(
        match.strip()
        for match in re.findall(
            r"\bwalk (?:toward|towards) the ([a-z][a-z ]{2,28}?)(?:[,.]| and| then|$)",
            lower,
        )
    )
    repeated_targets = [target for target, count in target_counts.items() if count >= 2]
    if repeated_targets:
        hints.append(
            "repeats the same target landmark instead of merging the movement: "
            + ", ".join(repeated_targets[:3])
        )
    pass_targets = Counter(
        re.sub(r"\b(?:the|a|an)\b", "", match.strip()).strip()
        for match in re.findall(
            r"\b(?:walk\s+past|go\s+past|move\s+past|moving\s+past|"
            r"passing|pass)\s+(?:the\s+)?"
            r"([a-z][a-z ]{2,35}?)(?:[,.]| and| then|$)",
            lower,
        )
    )
    repeated_pass_targets = [
        target for target, count in pass_targets.items() if target and count >= 2
    ]
    if repeated_pass_targets:
        hints.append(
            "repeats passing the same landmark; merge the repeated clause: "
            + ", ".join(repeated_pass_targets[:3])
        )
    return hints


def final_style_blocking_hints(instruction: str) -> List[str]:
    lower = normalize_instruction(instruction).lower()
    blockers = []
    if not re.search(r"\b(?:stop|stopping|wait|halt|finish|end|stand|remain)\b", lower):
        blockers.append("final instruction does not explicitly tell the navigator where to stop")
    if re.search(r"\b(?:about|approximately|around)?\s*\d{2,3}\s*degrees\b", lower):
        blockers.append("final instruction still uses numeric degree wording")
    if re.search(r"\bstop on the flight\b", lower):
        blockers.append("final instruction uses unnatural stair endpoint phrase 'stop on the flight'")
    dangling_patterns = (
        r"\b(?:and|or|to|toward|towards|with|where|near|by)\s*[.!?]",
        r"\bvisible\s+ahead\s+and\s*[.!?]",
        r"\bwhere\s+(?:a|an|the)?\s*[.!?]",
        r"\bstop near the [^,.!?]{2,60}? visible\b",
        r",\s+with\s+[^,.!?]{3,70}?,\s+and\s+stop\b",
        r"\bturn\s+(?:left|right)\s+to\s+align\b",
        r"\bturn\s+(?:left|right)\s+(?:to\s+)?face\b[^.!?]{0,100}\bthen\s+turn\b",
    )
    for pattern in dangling_patterns:
        if re.search(pattern, lower):
            blockers.append("final instruction contains a dangling connector/preposition fragment")
            break
    if len(re.findall(r"\bturn\s+(?:left|right)\s+(?:to\s+)?face\b", lower)) >= 2:
        blockers.append("final instruction lists repeated face/align orientation turns")
    pass_targets = Counter(
        re.sub(r"\b(?:the|a|an)\b", "", match.strip()).strip()
        for match in re.findall(
            r"\b(?:walk\s+past|go\s+past|move\s+past|moving\s+past|"
            r"passing|pass)\s+(?:the\s+)?"
            r"([a-z][a-z ]{2,35}?)(?:[,.]| and| then|$)",
            lower,
        )
    )
    if any(target and count >= 2 for target, count in pass_targets.items()):
        blockers.append("final instruction repeats passing the same landmark")
    for sentence in re.split(r"[.!?]+", lower):
        stripped = sentence.strip(" ,;")
        if re.search(r"\bkeeping\s+(?:the\s+)?[a-z0-9 ,'-]+$", stripped) and not re.search(
            r"\bkeeping\b.*\b(?:on|to|along|near|beside|next to|toward|towards|ahead|behind)\b",
            stripped,
        ):
            blockers.append("final instruction contains a dangling 'keeping ...' phrase")
            break
    return blockers


def route_language_hints(instruction: str, actions: Sequence[int]) -> List[str]:
    errors = []
    normalized = normalize_instruction(instruction)
    lower = normalized.lower()
    runs = action_runs(actions)
    if runs:
        first_action, first_start, first_end = runs[0]
        first_turn_degrees = 0
        if first_action in (2, 3):
            first_turn_degrees = int((first_end - first_start + 1) * TURN_DEGREES)
        if re.match(r"^(?:please\s+)?(?:turn around|turn back|make a u-turn)\b", lower):
            if first_action == 1 or first_turn_degrees < 120:
                errors.append(
                    "instruction begins with turn around but the action sequence does not"
                )
        if first_action in {2, 3} and first_turn_degrees >= 30:
            expected = "left" if first_action == 2 else "right"
            opposite = "right" if expected == "left" else "left"
            first_sentence = re.split(r"[.!?]", lower, maxsplit=1)[0]
            if re.search(rf"\bturn {opposite}\b", first_sentence):
                errors.append(
                    f"instruction first sentence says turn {opposite}, but the "
                    f"action sequence starts with a {expected} turn"
                )

    object_terms = (
        "stove",
        "refrigerator",
        "fridge",
        "table",
        "chair",
        "couch",
        "sofa",
        "bed",
        "bookshelf",
        "bookshelves",
        "shelf",
        "shelves",
        "cabinet",
        "cabinets",
        "island",
        "bench",
        "piano",
        "dresser",
        "fireplace",
        "window",
        "picture",
        "painting",
        "plant",
        "rug",
        "counter",
        "desk",
        "door",
        "doors",
        "doorway",
        "furniture",
        "appliance",
        "appliances",
        "artwork",
    )
    object_pattern = "|".join(re.escape(term) for term in object_terms)
    if re.search(
        rf"\b(?:{object_pattern})\b[^.!?]{{0,55}}\b(?:on|to) (?:the |your )?(?:left|right)\b",
        lower,
    ) or re.search(
        rf"\b(?:on|to) (?:the |your )?(?:left|right)\b[^.!?]{{0,55}}\b(?:{object_pattern})\b",
        lower,
    ) or re.search(
        rf"\bkeeping\b[^.!?]{{0,65}}\b(?:on|to) (?:the |your )?(?:left|right)\b",
        lower,
    ):
        errors.append("uses left/right for an object position; use route turns only")
    return errors


def parse_and_validate_response(
    raw_response: str,
    old_instruction: str,
    min_words: int,
    max_words: int,
) -> Tuple[str, List[str]]:
    try:
        parsed = extract_json_object(raw_response)
        instruction = parsed.get("instruction")
    except json.JSONDecodeError:
        instruction = raw_response.strip()
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("JSON object is missing non-empty string field 'instruction'")
    instruction = clean_generated_instruction(instruction)
    errors, warnings = validate_generated_instruction(
        instruction=instruction,
        old_instruction=old_instruction,
        min_words=min_words,
        max_words=max_words,
    )
    if errors:
        raise ValueError("; ".join(errors))
    return instruction, warnings


@dataclass
class EpisodeState:
    """Mutable state passed between the rewrite agents for one episode."""

    row: Dict[str, Any]
    args: argparse.Namespace
    episode_id: str
    started_at: float
    candidate: Dict[str, Any]
    route: Optional[RenderedRoute] = None

    @classmethod
    def create(cls, row: Dict[str, Any], args: argparse.Namespace) -> "EpisodeState":
        episode_id = episode_id_key(row)
        candidate: Dict[str, Any] = {
            "episode_id": episode_id,
            "status": "failed",
            "old_instruction": row["instruction"],
            "actions": [int(action) for action in row["actions"]],
            "provider": args.provider,
            "model": args.model,
            "base_url": args.base_url,
            "stage_models": getattr(args, "stage_models", None),
            "attempts": 0,
            "selected_frames": [],
            "start_frames": [],
            "endpoint_frames": [],
            "contact_sheet": None,
            "start_sheet": None,
            "endpoint_sheet": None,
            "destination_hint": extract_destination_hint(row["instruction"]),
            "start_facts": None,
            "endpoint_facts": None,
            "route_plan": None,
            "start_error": None,
            "endpoint_error": None,
            "route_plan_error": None,
            "warnings": [],
            "error": None,
            "raw_response": None,
            "self_checks": [],
            "route_audits": [],
            "spatial_audits": [],
        }
        return cls(
            row=row,
            args=args,
            episode_id=episode_id,
            started_at=time.time(),
            candidate=candidate,
        )

    def fail(self, error: str) -> Dict[str, Any]:
        self.candidate["error"] = error
        self.candidate["elapsed_seconds"] = round(time.time() - self.started_at, 3)
        return self.candidate

    def finish(self) -> Dict[str, Any]:
        self.candidate["elapsed_seconds"] = round(time.time() - self.started_at, 3)
        return self.candidate


class RouteEvidenceAgent:
    """Builds visual evidence sheets shared by all later agents."""

    name = "route_evidence"

    def run(self, state: EpisodeState) -> bool:
        args = state.args
        try:
            route = render_contact_sheet(
                row=state.row,
                image_root=args.image_root,
                work_dir=args.work_dir,
                max_waypoints=args.max_waypoints,
                start_window_frames=args.start_window_frames,
                endpoint_window_frames=args.endpoint_window_frames,
                tile_width=args.tile_width,
                tile_height=args.tile_height,
                jpeg_quality=args.jpeg_quality,
                save_contact_sheets=args.save_contact_sheets,
                use_action_heading=args.use_action_heading,
            )
        except Exception as error:
            state.fail(f"render_failed: {type(error).__name__}: {error}")
            return False

        state.route = route
        candidate = state.candidate
        candidate["selected_frames"] = route.selected_frames
        candidate["start_frames"] = route.start_frames
        candidate["endpoint_frames"] = route.endpoint_frames
        candidate["contact_sheet"] = route.saved_path
        candidate["start_sheet"] = route.saved_start_path
        candidate["endpoint_sheet"] = route.saved_endpoint_path
        candidate["action_summary"] = compact_action_summary(state.row["actions"])
        return True


class StartFactAgent:
    """Extracts start-location facts and the required first transition."""

    name = "start_facts"

    def run(self, state: EpisodeState) -> bool:
        args = state.args
        if not args.start_fact_pass:
            return self._check_required(state)
        assert state.route is not None
        last_stage_error = None
        for stage_attempt in range(1, args.stage_retries + 2):
            try:
                state.candidate["start_facts"] = extract_start_facts(
                    args,
                    state.row,
                    state.route,
                )
                state.candidate["start_attempts"] = stage_attempt
                break
            except Exception as error:
                last_stage_error = f"{type(error).__name__}: {error}"
                state.candidate["start_error"] = last_stage_error
                if args.sleep_between_retries > 0:
                    time.sleep(args.sleep_between_retries)
        if state.candidate.get("start_facts") is None:
            state.candidate["warnings"].append(
                f"start fact pass failed; continuing without start facts: {last_stage_error}"
            )
        return self._check_required(state)

    def _check_required(self, state: EpisodeState) -> bool:
        if state.args.require_start_facts and state.candidate.get("start_facts") is None:
            state.fail("start_facts_required_but_missing")
            return False
        return True


class EndpointFactAgent:
    """Extracts conservative final-stop facts from the endpoint sheet."""

    name = "endpoint_facts"

    def run(self, state: EpisodeState) -> bool:
        args = state.args
        if not args.endpoint_fact_pass:
            return self._check_required(state)
        assert state.route is not None
        last_stage_error = None
        for stage_attempt in range(1, args.stage_retries + 2):
            try:
                state.candidate["endpoint_facts"] = extract_endpoint_facts(
                    args,
                    state.row,
                    state.route,
                )
                state.candidate["endpoint_attempts"] = stage_attempt
                break
            except Exception as error:
                last_stage_error = f"{type(error).__name__}: {error}"
                state.candidate["endpoint_error"] = last_stage_error
                if args.sleep_between_retries > 0:
                    time.sleep(args.sleep_between_retries)
        if state.candidate.get("endpoint_facts") is None:
            state.candidate["warnings"].append(
                f"endpoint fact pass failed; continuing without endpoint facts: {last_stage_error}"
            )
        return self._check_required(state)

    def _check_required(self, state: EpisodeState) -> bool:
        if state.args.require_endpoint_facts and state.candidate.get("endpoint_facts") is None:
            state.fail("endpoint_facts_required_but_missing")
            return False
        return True


class RoutePlannerAgent:
    """Creates ordered human-level route segments for the writer."""

    name = "route_planner"

    def run(self, state: EpisodeState) -> bool:
        args = state.args
        if not args.route_plan_pass:
            return self._check_required(state)
        assert state.route is not None
        last_stage_error = None
        for stage_attempt in range(1, args.stage_retries + 2):
            try:
                state.candidate["route_plan"] = extract_route_plan(
                    args=args,
                    row=state.row,
                    route=state.route,
                    start_facts=state.candidate.get("start_facts"),
                    endpoint_facts=state.candidate.get("endpoint_facts"),
                )
                state.candidate["route_plan_attempts"] = stage_attempt
                break
            except Exception as error:
                last_stage_error = f"{type(error).__name__}: {error}"
                state.candidate["route_plan_error"] = last_stage_error
                if args.sleep_between_retries > 0:
                    time.sleep(args.sleep_between_retries)
        if state.candidate.get("route_plan") is None:
            state.candidate["warnings"].append(
                f"route plan pass failed; continuing without route plan: {last_stage_error}"
            )
        return self._check_required(state)

    def _check_required(self, state: EpisodeState) -> bool:
        if state.args.require_route_plan and state.candidate.get("route_plan") is None:
            state.fail("route_plan_required_but_missing")
            return False
        return True


def build_quality_hints(state: EpisodeState, instruction: str) -> List[str]:
    candidate = state.candidate
    hints = route_language_hints(
        instruction=instruction,
        actions=state.row["actions"],
    )
    hints.extend(style_quality_hints(instruction))
    hints.extend(endpoint_consistency_hints(
        instruction=instruction,
        endpoint_facts=candidate.get("endpoint_facts"),
    ))
    hints.extend(endpoint_specificity_hints(
        instruction=instruction,
        endpoint_facts=candidate.get("endpoint_facts"),
    ))
    hints.extend(start_consistency_hints(
        instruction=instruction,
        start_facts=candidate.get("start_facts"),
    ))
    hints.extend(route_plan_coverage_hints(
        instruction=instruction,
        route_plan=candidate.get("route_plan"),
    ))
    hints.extend(route_plan_avoidance_hints(
        instruction=instruction,
        route_plan=candidate.get("route_plan"),
    ))
    hints.extend(unsupported_passing_hints(
        instruction=instruction,
        route_plan=candidate.get("route_plan"),
    ))
    hints.extend(route_plan_final_direction_hints(
        instruction=instruction,
        route_plan=candidate.get("route_plan"),
        endpoint_facts=candidate.get("endpoint_facts"),
    ))
    hints.extend(destination_entry_turn_hints(
        instruction=instruction,
        actions=state.row["actions"],
        endpoint_facts=candidate.get("endpoint_facts"),
        route_plan=candidate.get("route_plan"),
    ))
    hints.extend(route_plan_action_turn_coverage_hints(
        instruction=instruction,
        route_plan=candidate.get("route_plan"),
    ))
    hints.extend(final_destination_turn_hints(
        instruction=instruction,
        actions=state.row["actions"],
    ))
    return hints


class InstructionWriterAgent:
    """Writes the final R2R-style instruction from agent facts and route plan."""

    name = "writer"

    def run(
        self,
        state: EpisodeState,
        repair_note: Optional[str],
    ) -> Tuple[str, List[str]]:
        args = state.args
        assert state.route is not None
        messages = build_generation_messages(
            row=state.row,
            route=state.route,
            start_facts=state.candidate.get("start_facts"),
            endpoint_facts=state.candidate.get("endpoint_facts"),
            route_plan=state.candidate.get("route_plan"),
            repair_note=repair_note,
        )
        raw_response = call_chat_completion(args, messages)
        state.candidate["raw_response"] = raw_response
        return parse_and_validate_response(
            raw_response=raw_response,
            old_instruction=state.row["instruction"],
            min_words=args.min_words,
            max_words=args.max_words,
        )


class SelfCheckCriticAgent:
    """Reviews grounding, navigation, endpoint, and R2R style."""

    name = "self_check_critic"

    def run(
        self,
        state: EpisodeState,
        instruction: str,
        warnings: List[str],
    ) -> Tuple[str, List[str]]:
        args = state.args
        if not args.self_check:
            return instruction, warnings
        assert state.route is not None

        check = self_check_instruction(
            args=args,
            row=state.row,
            route=state.route,
            instruction=instruction,
            endpoint_facts=state.candidate.get("endpoint_facts"),
            route_plan=state.candidate.get("route_plan"),
            quality_hints=build_quality_hints(state, instruction),
        )
        state.candidate["self_checks"].append(check)
        if self_check_passed(check):
            return instruction, warnings

        corrected = clean_generated_instruction(
            str(check.get("corrected_instruction") or "")
        )
        if corrected and corrected != ".":
            instruction = corrected
            final_check = self_check_instruction(
                args=args,
                row=state.row,
                route=state.route,
                instruction=instruction,
                endpoint_facts=state.candidate.get("endpoint_facts"),
                route_plan=state.candidate.get("route_plan"),
                quality_hints=build_quality_hints(state, instruction),
            )
            state.candidate["self_checks"].append(final_check)
            if self_check_passed(final_check):
                warnings.append("model critic revised the instruction")
                return instruction, warnings
            if self_check_is_acceptable_borderline(final_check):
                warnings.append("accepted minor self-check borderline after repair")
                return instruction, warnings
            final_corrected = clean_generated_instruction(
                str(final_check.get("corrected_instruction") or "")
            )
            if final_corrected and final_corrected != "." and final_corrected != instruction:
                warnings.append("accepted self-check final correction after non-pass")
                return final_corrected, warnings
            raise ValueError(
                "review_repair_failed: "
                f"verdict={final_check.get('verdict')} "
                f"reason={final_check.get('reason')}"
            )

        if self_check_is_acceptable_borderline(check):
            warnings.append("accepted minor self-check borderline")
            return instruction, warnings
        raise ValueError(
            "self_check_failed_without_correction: "
            f"verdict={check.get('verdict')} reason={check.get('reason')}"
        )


class RouteCoherenceAuditorAgent:
    """Audits full-route segment coverage and natural R2R wording."""

    name = "route_coherence_auditor"

    def run(
        self,
        state: EpisodeState,
        instruction: str,
        warnings: List[str],
    ) -> Tuple[str, List[str]]:
        args = state.args
        if not args.route_audit:
            return instruction, warnings
        assert state.route is not None

        route_audit = route_audit_instruction(
            args=args,
            row=state.row,
            route=state.route,
            instruction=instruction,
            start_facts=state.candidate.get("start_facts"),
            endpoint_facts=state.candidate.get("endpoint_facts"),
            route_plan=state.candidate.get("route_plan"),
            quality_hints=build_quality_hints(state, instruction),
        )
        state.candidate["route_audits"].append(route_audit)
        if route_audit_passed(route_audit):
            return instruction, warnings

        corrected = clean_generated_instruction(
            str(route_audit.get("corrected_instruction") or "")
        )
        if not corrected or corrected == ".":
            if route_audit_is_acceptable_borderline(route_audit):
                warnings.append("accepted minor route-audit borderline")
                return instruction, warnings
            raise ValueError(
                "route_audit_failed_without_correction: "
                f"verdict={route_audit.get('verdict')} "
                f"reason={route_audit.get('reason')}"
            )

        instruction = corrected
        final_route_audit = route_audit_instruction(
            args=args,
            row=state.row,
            route=state.route,
            instruction=instruction,
            start_facts=state.candidate.get("start_facts"),
            endpoint_facts=state.candidate.get("endpoint_facts"),
            route_plan=state.candidate.get("route_plan"),
            quality_hints=build_quality_hints(state, instruction),
        )
        state.candidate["route_audits"].append(final_route_audit)
        if route_audit_passed(final_route_audit):
            warnings.append("route auditor revised the instruction")
            return instruction, warnings

        final_corrected = clean_generated_instruction(
            str(final_route_audit.get("corrected_instruction") or "")
        )
        if final_corrected and final_corrected != "." and final_corrected != instruction:
            warnings.append("accepted route auditor final correction after non-pass")
            return final_corrected, warnings
        if route_audit_is_acceptable_borderline(final_route_audit):
            warnings.append("accepted minor route-audit borderline after repair")
            return instruction, warnings
        raise ValueError(
            "route_audit_repair_failed: "
            f"verdict={final_route_audit.get('verdict')} "
            f"reason={final_route_audit.get('reason')}"
        )


class SpatialBoundaryAuditorAgent:
    """Audits final spatial boundary wording such as doorway/stairs/inside."""

    name = "spatial_boundary_auditor"

    def run(
        self,
        state: EpisodeState,
        instruction: str,
        warnings: List[str],
    ) -> Tuple[str, List[str]]:
        args = state.args
        if not args.spatial_audit:
            return instruction, warnings
        assert state.route is not None

        audit = spatial_audit_instruction(
            args=args,
            row=state.row,
            route=state.route,
            instruction=instruction,
            endpoint_facts=state.candidate.get("endpoint_facts"),
            route_plan=state.candidate.get("route_plan"),
        )
        state.candidate["spatial_audits"].append(audit)
        if str(audit.get("verdict", "")).lower() == "pass":
            return instruction, warnings

        corrected = clean_generated_instruction(
            str(audit.get("corrected_instruction") or "")
        )
        if not corrected or corrected == ".":
            raise ValueError(
                "spatial_audit_failed_without_correction: "
                f"verdict={audit.get('verdict')} reason={audit.get('reason')}"
            )

        instruction = corrected
        final_audit = spatial_audit_instruction(
            args=args,
            row=state.row,
            route=state.route,
            instruction=instruction,
            endpoint_facts=state.candidate.get("endpoint_facts"),
            route_plan=state.candidate.get("route_plan"),
        )
        state.candidate["spatial_audits"].append(final_audit)
        if str(final_audit.get("verdict", "")).lower() == "pass":
            warnings.append("spatial auditor revised the instruction")
            return instruction, warnings
        if accept_stale_spatial_audit(instruction, final_audit):
            warnings.append("accepted stale spatial audit that targeted old wording")
            return instruction, warnings
        if accept_neutral_spatial_borderline(instruction, final_audit):
            warnings.append("accepted neutral spatial wording after borderline audit")
            return instruction, warnings

        final_corrected = clean_generated_instruction(
            str(final_audit.get("corrected_instruction") or "")
        )
        if (
            final_corrected
            and final_corrected != "."
            and accept_neutral_spatial_borderline(final_corrected, final_audit)
        ):
            warnings.append("accepted neutral spatial wording from borderline audit")
            return final_corrected, warnings
        if final_corrected and final_corrected != ".":
            hygiene_hints = []
            hygiene_hints.extend(unsupported_passing_hints(
                instruction=final_corrected,
                route_plan=state.candidate.get("route_plan"),
            ))
            hygiene_hints.extend(route_plan_final_direction_hints(
                instruction=final_corrected,
                route_plan=state.candidate.get("route_plan"),
                endpoint_facts=state.candidate.get("endpoint_facts"),
            ))
            if (
                str(final_audit.get("verdict", "")).lower() == "borderline"
                and not hygiene_hints
            ):
                warnings.append(
                    "accepted spatial auditor final correction after borderline audit"
                )
                return final_corrected, warnings
        raise ValueError(
            "spatial_audit_repair_failed: "
            f"verdict={final_audit.get('verdict')} reason={final_audit.get('reason')}"
        )


class FinalQualityGateAgent:
    """Applies deterministic blockers for errors that should trigger retry/drop."""

    name = "final_quality_gate"

    def run(self, state: EpisodeState, instruction: str) -> None:
        final_blockers = final_blocking_hints(
            instruction=instruction,
            actions=state.row["actions"],
            start_facts=state.candidate.get("start_facts"),
            endpoint_facts=state.candidate.get("endpoint_facts"),
            route_plan=state.candidate.get("route_plan"),
        )
        final_blockers.extend(route_plan_final_direction_hints(
            instruction=instruction,
            route_plan=state.candidate.get("route_plan"),
            endpoint_facts=state.candidate.get("endpoint_facts"),
        ))
        final_blockers.extend(destination_entry_turn_hints(
            instruction=instruction,
            actions=state.row["actions"],
            endpoint_facts=state.candidate.get("endpoint_facts"),
            route_plan=state.candidate.get("route_plan"),
        ))
        final_blockers.extend(unsupported_passing_hints(
            instruction=instruction,
            route_plan=state.candidate.get("route_plan"),
        ))
        final_blockers.extend(route_plan_action_turn_coverage_hints(
            instruction=instruction,
            route_plan=state.candidate.get("route_plan"),
        ))
        final_blockers.extend(final_style_blocking_hints(instruction))
        if final_blockers:
            raise ValueError("final_quality_blockers: " + "; ".join(final_blockers))


class ScaleVLNRewritePipeline:
    """Agent graph for one ScaleVLN episode."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.evidence_agent = RouteEvidenceAgent()
        self.start_agent = StartFactAgent()
        self.endpoint_agent = EndpointFactAgent()
        self.planner_agent = RoutePlannerAgent()
        self.writer_agent = InstructionWriterAgent()
        self.self_check_agent = SelfCheckCriticAgent()
        self.route_auditor_agent = RouteCoherenceAuditorAgent()
        self.spatial_auditor_agent = SpatialBoundaryAuditorAgent()
        self.final_gate_agent = FinalQualityGateAgent()

    def run_episode(self, row: Dict[str, Any]) -> Dict[str, Any]:
        state = EpisodeState.create(row, self.args)
        if not self.evidence_agent.run(state):
            return state.candidate
        if not self.start_agent.run(state):
            return state.candidate
        if not self.endpoint_agent.run(state):
            return state.candidate
        if not self.planner_agent.run(state):
            return state.candidate

        repair_note = None
        for attempt in range(1, self.args.retries + 2):
            state.candidate["attempts"] = attempt
            try:
                instruction, warnings = self.writer_agent.run(state, repair_note)
                instruction, warnings = self.self_check_agent.run(
                    state,
                    instruction,
                    warnings,
                )
                instruction, warnings = self.route_auditor_agent.run(
                    state,
                    instruction,
                    warnings,
                )
                instruction, warnings = self.spatial_auditor_agent.run(
                    state,
                    instruction,
                    warnings,
                )
                self.final_gate_agent.run(state, instruction)
                state.candidate["status"] = "success"
                state.candidate["instruction"] = instruction
                state.candidate["warnings"] = warnings
                state.candidate["error"] = None
                break
            except Exception as error:
                repair_note = f"{type(error).__name__}: {error}"
                state.candidate["error"] = repair_note
                if self.args.sleep_between_retries > 0:
                    time.sleep(self.args.sleep_between_retries)

        if (
            state.candidate["status"] not in {"success"}
            and self.args.save_failed_contact_sheets
            and not state.candidate.get("contact_sheet")
            and state.route is not None
        ):
            state.candidate["contact_sheet"] = save_contact_sheet_bytes(
                work_dir=self.args.work_dir,
                episode_id=state.episode_id,
                jpeg_bytes=state.route.jpeg_bytes,
                subdir="failed_contact_sheets",
            )
        return state.finish()


def process_episode(row: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    return ScaleVLNRewritePipeline(args).run_episode(row)


def candidate_paths(work_dir: str) -> List[Path]:
    root = Path(work_dir)
    paths = []
    merged_path = root / "candidates.jsonl"
    if merged_path.exists():
        paths.append(merged_path)
    paths.extend(sorted(root.glob("candidates_rank*.jsonl")))
    return paths


def failed_paths(work_dir: str) -> List[Path]:
    root = Path(work_dir)
    paths = []
    merged_path = root / "failed.jsonl"
    if merged_path.exists():
        paths.append(merged_path)
    paths.extend(sorted(root.glob("failed_rank*.jsonl")))
    return paths


def clear_progress_files(work_dir: str) -> None:
    root = Path(work_dir)
    for pattern in ("candidates.jsonl", "failed.jsonl", "candidates_rank*.jsonl", "failed_rank*.jsonl"):
        for path in root.glob(pattern):
            path.unlink()


def load_existing_candidates(work_dir: str) -> Dict[str, Dict[str, Any]]:
    existing = {}
    for path in candidate_paths(work_dir):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                episode_id = str(row.get("episode_id", ""))
                if not episode_id:
                    continue
                existing[episode_id] = row
    return existing


def merge_progress_files(
    work_dir: str,
    selected_rows: Sequence[Dict[str, Any]],
    candidates_by_episode: Dict[str, Dict[str, Any]],
) -> Tuple[int, int]:
    candidate_rows = []
    failed_rows = []
    for row in selected_rows:
        candidate = candidates_by_episode.get(episode_id_key(row))
        if not candidate:
            continue
        candidate_rows.append(candidate)
        if not candidate_is_usable(candidate):
            failed_rows.append(candidate)

    root = Path(work_dir)
    write_jsonl(str(root / "candidates.jsonl"), candidate_rows)
    write_jsonl(str(root / "failed.jsonl"), failed_rows)
    for path in sorted(root.glob("candidates_rank*.jsonl")) + sorted(root.glob("failed_rank*.jsonl")):
        path.unlink()
    return len(candidate_rows), len(failed_rows)


def choose_rows(
    rows: List[Dict[str, Any]],
    max_episodes: Optional[int],
    sample_mode: str,
    seed: int,
    episode_ids: Optional[List[str]],
) -> List[Dict[str, Any]]:
    if episode_ids:
        wanted = {str(episode_id) for episode_id in episode_ids}
        selected = [row for row in rows if episode_id_key(row) in wanted]
        missing = sorted(wanted - {episode_id_key(row) for row in selected})
        if missing:
            raise ValueError(f"Missing episode ids in input JSONL: {missing}")
        return selected

    if max_episodes is None or max_episodes <= 0 or max_episodes >= len(rows):
        return rows

    if sample_mode == "first":
        return rows[:max_episodes]

    rng = random.Random(seed)
    if sample_mode == "random":
        indices = sorted(rng.sample(range(len(rows)), max_episodes))
        return [rows[index] for index in indices]

    if sample_mode == "stratified":
        indexed = list(enumerate(rows))
        indexed.sort(key=lambda item: len(item[1]["actions"]))
        buckets = [indexed[0::3], indexed[1::3], indexed[2::3]]
        selected_indices = []
        base = max_episodes // 3
        remainder = max_episodes % 3
        for bucket_index, bucket in enumerate(buckets):
            take = base + (1 if bucket_index < remainder else 0)
            if take <= 0:
                continue
            take = min(take, len(bucket))
            selected_indices.extend(index for index, _ in rng.sample(bucket, take))
        while len(selected_indices) < max_episodes:
            candidate = rng.randrange(len(rows))
            if candidate not in selected_indices:
                selected_indices.append(candidate)
        return [rows[index] for index in sorted(selected_indices[:max_episodes])]

    raise ValueError(f"Unknown sample mode: {sample_mode}")


def candidate_is_usable(candidate: Dict[str, Any]) -> bool:
    return candidate.get("status") == "success" and bool(candidate.get("instruction"))


def build_clean_rows(
    selected_rows: List[Dict[str, Any]],
    candidates_by_episode: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    clean_rows = []
    missing = []
    for row in selected_rows:
        episode_id = episode_id_key(row)
        candidate = candidates_by_episode.get(episode_id)
        if not candidate_is_usable(candidate or {}):
            missing.append(episode_id)
            continue
        clean_rows.append(
            {
                "episode_id": row["episode_id"],
                "instruction": candidate["instruction"],
                "actions": row["actions"],
            }
        )
    return clean_rows, missing


def write_gallery(
    selected_rows: List[Dict[str, Any]],
    candidates_by_episode: Dict[str, Dict[str, Any]],
    gallery_path: str,
) -> None:
    output_path = Path(gallery_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>ScaleVLN Instruction Rewrite Gallery</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:24px;line-height:1.35;color:#202124}",
        ".item{border:1px solid #ddd;border-radius:6px;padding:16px;margin:18px 0}",
        ".ok{color:#137333}.bad{color:#b3261e}.warn{color:#b06000}",
        "img{max-width:100%;height:auto;border:1px solid #ddd}",
        "pre{white-space:pre-wrap;background:#f6f8fa;padding:8px;border-radius:4px}",
        "</style></head><body>",
        "<h1>ScaleVLN Instruction Rewrite Gallery</h1>",
    ]
    for row in selected_rows:
        episode_id = episode_id_key(row)
        candidate = candidates_by_episode.get(episode_id, {})
        status = candidate.get("status", "missing")
        status_class = "ok" if candidate_is_usable(candidate) else "bad"
        parts.append("<div class='item'>")
        parts.append(
            f"<h2>Episode {html.escape(episode_id)} "
            f"<span class='{status_class}'>[{html.escape(status)}]</span></h2>"
        )
        contact_sheet = candidate.get("contact_sheet")
        if contact_sheet and os.path.exists(contact_sheet):
            rel = os.path.relpath(contact_sheet, output_path.parent)
            parts.append("<h3>Route Evidence</h3>")
            parts.append(f"<img src='{html.escape(rel)}'>")
        start_sheet = candidate.get("start_sheet")
        if start_sheet and os.path.exists(start_sheet):
            rel = os.path.relpath(start_sheet, output_path.parent)
            parts.append("<h3>Start Evidence</h3>")
            parts.append(f"<img src='{html.escape(rel)}'>")
        endpoint_sheet = candidate.get("endpoint_sheet")
        if endpoint_sheet and os.path.exists(endpoint_sheet):
            rel = os.path.relpath(endpoint_sheet, output_path.parent)
            parts.append("<h3>Endpoint Evidence</h3>")
            parts.append(f"<img src='{html.escape(rel)}'>")
        parts.append("<h3>Old</h3>")
        parts.append(f"<p>{html.escape(str(row.get('instruction', '')))}</p>")
        parts.append("<h3>New</h3>")
        parts.append(f"<p>{html.escape(str(candidate.get('instruction', '')))}</p>")
        warnings = candidate.get("warnings") or []
        if warnings:
            parts.append("<h3>Warnings</h3>")
            parts.append(f"<pre class='warn'>{html.escape(json.dumps(warnings, ensure_ascii=False, indent=2))}</pre>")
        if candidate.get("error"):
            parts.append("<h3>Error</h3>")
            parts.append(f"<pre class='bad'>{html.escape(str(candidate['error']))}</pre>")
        if candidate.get("raw_response"):
            parts.append("<details><summary>Raw response</summary>")
            parts.append(f"<pre>{html.escape(str(candidate['raw_response']))}</pre>")
            parts.append("</details>")
        parts.append("</div>")
    parts.append("</body></html>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def provider_defaults(args: argparse.Namespace) -> None:
    if args.provider != "qwen":
        raise ValueError(f"Unknown provider: {args.provider}")
    args.base_url = args.base_url or DEFAULT_QWEN_BASE_URL
    args.model = args.model or DEFAULT_QWEN_MODEL
    args.api_key = args.api_key or DEFAULT_QWEN_API_KEY


def default_progress_dir(output_jsonl: str) -> str:
    output_path = Path(output_jsonl)
    return str(output_path.parent / f"{output_path.stem}_progress")


def parse_episode_ids(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    items = []
    for part in value.split(","):
        stripped = part.strip()
        if stripped:
            items.append(stripped)
    return items or None


def run(args: argparse.Namespace) -> int:
    provider_defaults(args)
    if not args.work_dir:
        args.work_dir = default_progress_dir(args.output_jsonl)
    args.stage_models = {
        "writer": {
            "provider": args.provider,
            "model": args.model,
            "base_url": args.base_url,
        },
        "start": {
            "provider": args.provider,
            "model": args.model,
            "base_url": args.base_url,
        },
        "endpoint": {
            "provider": args.provider,
            "model": args.model,
            "base_url": args.base_url,
        },
        "planner": {
            "provider": args.provider,
            "model": args.model,
            "base_url": args.base_url,
        },
        "review": {
            "provider": args.provider,
            "model": args.model,
            "base_url": args.base_url,
        },
    }
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    gallery_path = args.gallery_path or str(Path(args.work_dir) / "gallery.html")

    if not args.resume:
        clear_progress_files(args.work_dir)

    all_rows = read_jsonl(args.input_jsonl)
    selected_rows = choose_rows(
        rows=all_rows,
        max_episodes=args.max_episodes,
        sample_mode=args.sample_mode,
        seed=args.seed,
        episode_ids=parse_episode_ids(args.episode_ids),
    )
    existing = load_existing_candidates(args.work_dir)
    candidates_by_episode = dict(existing)

    pending_rows = []
    resumed_success_count = 0
    for row in selected_rows:
        episode_id = episode_id_key(row)
        existing_candidate = candidates_by_episode.get(episode_id)
        if args.resume and candidate_is_usable(existing_candidate or {}):
            resumed_success_count += 1
            continue
        pending_rows.append(row)

    print(
        f"input_rows={len(all_rows)} selected={len(selected_rows)} "
        f"existing={len(existing)} resumed_success={resumed_success_count} "
        f"pending={len(pending_rows)} provider={args.provider} "
        f"model={args.model} workers={args.num_workers}"
    )

    progress = None
    if tqdm is not None:
        progress = tqdm(
            total=len(selected_rows),
            initial=resumed_success_count,
            desc="rewrite",
            dynamic_ncols=True,
        )
        if resumed_success_count:
            progress.set_postfix_str(
                f"resumed={resumed_success_count} pending={len(pending_rows)}"
            )

    rank_count = max(1, int(args.num_workers))
    rank_locks = [threading.Lock() for _ in range(rank_count)]

    def record_candidate(rank: int, candidate: Dict[str, Any]) -> None:
        rank = rank % rank_count
        candidates_by_episode[str(candidate["episode_id"])] = candidate
        with rank_locks[rank]:
            append_jsonl(
                str(Path(args.work_dir) / f"candidates_rank{rank}.jsonl"),
                candidate,
            )
            if not candidate_is_usable(candidate):
                append_jsonl(
                    str(Path(args.work_dir) / f"failed_rank{rank}.jsonl"),
                    candidate,
                )

    if args.num_workers <= 1:
        for row in pending_rows:
            candidate = process_episode(row, args)
            record_candidate(0, candidate)
            if args.sleep_between_requests > 0:
                time.sleep(args.sleep_between_requests)
            if progress is not None:
                progress.update(1)
                progress.set_postfix_str(
                    f"ep={candidate['episode_id']} status={candidate['status']}"
                )
    else:
        with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
            future_to_info = {
                pool.submit(process_episode, row, args): (row, index % rank_count)
                for index, row in enumerate(pending_rows)
            }
            for future in as_completed(future_to_info):
                row, rank = future_to_info[future]
                try:
                    candidate = future.result()
                except Exception as error:
                    candidate = {
                        "episode_id": episode_id_key(row),
                        "status": "failed",
                        "old_instruction": row["instruction"],
                        "actions": row["actions"],
                        "provider": args.provider,
                        "model": args.model,
                        "error": f"worker_failed: {type(error).__name__}: {error}",
                    }
                record_candidate(rank, candidate)
                if progress is not None:
                    progress.update(1)
                    progress.set_postfix_str(
                        f"ep={candidate['episode_id']} status={candidate['status']}"
                    )

    if progress is not None:
        progress.close()

    merged_count, failed_count = merge_progress_files(
        args.work_dir,
        selected_rows,
        candidates_by_episode,
    )
    clean_rows, missing_episode_ids = build_clean_rows(selected_rows, candidates_by_episode)
    summary = {
        "input_jsonl": args.input_jsonl,
        "output_jsonl": args.output_jsonl,
        "work_dir": args.work_dir,
        "provider": args.provider,
        "model": args.model,
        "stage_models": args.stage_models,
        "selected": len(selected_rows),
        "merged_candidates": merged_count,
        "clean_rows": len(clean_rows),
        "failed_candidates": failed_count,
        "missing_or_failed": len(missing_episode_ids),
        "missing_episode_ids_preview": missing_episode_ids[:50],
    }
    summary_path = Path(args.work_dir) / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    output_path = args.output_jsonl
    if missing_episode_ids:
        if args.drop_failed:
            written = write_jsonl(output_path, clean_rows)
            print(
                f"wrote clean output rows={written} after dropping "
                f"{len(missing_episode_ids)} failed/missing episodes: {output_path}"
            )
            if args.write_gallery:
                write_gallery(selected_rows, candidates_by_episode, gallery_path)
                print(f"wrote gallery: {gallery_path}")
            print(f"wrote summary: {summary_path}")
            return 0
        partial_path = output_path + ".partial"
        if args.write_partial_output:
            written = write_jsonl(partial_path, clean_rows)
            print(f"wrote partial clean output rows={written}: {partial_path}")
        print(
            f"missing_or_failed={len(missing_episode_ids)}. "
            f"Clean output was not finalized: {output_path}"
        )
        if args.write_gallery:
            write_gallery(selected_rows, candidates_by_episode, gallery_path)
            print(f"wrote gallery: {gallery_path}")
        return 2

    written = write_jsonl(output_path, clean_rows)
    print(f"wrote clean output rows={written}: {output_path}")
    if args.write_gallery:
        write_gallery(selected_rows, candidates_by_episode, gallery_path)
        print(f"wrote gallery: {gallery_path}")
    print(f"wrote summary: {summary_path}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rewrite ScaleVLN instructions using trajectory panoramas and a VLM."
    )
    parser.add_argument("--input-jsonl", default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-jsonl", default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument(
        "--work-dir",
        default=None,
        help=(
            "Progress directory for candidates_rank*.jsonl and diagnostics. "
            "Defaults to <output_jsonl_stem>_progress next to --output-jsonl."
        ),
    )

    parser.add_argument("--provider", choices=("qwen",), default="qwen")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--disable-thinking", type=str2bool, default=True)

    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--episode-ids", default=None, help="Comma-separated episode ids.")
    parser.add_argument(
        "--sample-mode",
        choices=("first", "random", "stratified"),
        default="first",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str2bool, default=True)
    parser.add_argument("--num-workers", type=int, default=1)

    parser.add_argument("--max-waypoints", type=int, default=10)
    parser.add_argument("--start-window-frames", type=int, default=6)
    parser.add_argument("--endpoint-window-frames", type=int, default=6)
    parser.add_argument("--tile-width", type=int, default=256)
    parser.add_argument("--tile-height", type=int, default=192)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument(
        "--use-action-heading",
        type=str2bool,
        default=False,
        help=(
            "Apply cumulative left/right actions to rotate each panorama before "
            "cropping. Keep false for Habitat agent-local equirect frames."
        ),
    )
    parser.add_argument("--save-contact-sheets", action="store_true")
    parser.add_argument("--save-failed-contact-sheets", type=str2bool, default=True)
    parser.add_argument("--write-gallery", action="store_true")
    parser.add_argument("--gallery-path", default=None)

    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=240)
    parser.add_argument("--start-fact-pass", type=str2bool, default=True)
    parser.add_argument("--endpoint-fact-pass", type=str2bool, default=True)
    parser.add_argument("--require-start-facts", type=str2bool, default=True)
    parser.add_argument("--require-endpoint-facts", type=str2bool, default=True)
    parser.add_argument("--require-route-plan", type=str2bool, default=True)
    parser.add_argument("--fact-temperature", type=float, default=0.0)
    parser.add_argument("--fact-max-tokens", type=int, default=220)
    parser.add_argument("--route-plan-pass", type=str2bool, default=True)
    parser.add_argument("--planner-temperature", type=float, default=0.0)
    parser.add_argument("--planner-max-tokens", type=int, default=760)
    parser.add_argument("--review-temperature", type=float, default=0.0)
    parser.add_argument("--review-max-tokens", type=int, default=650)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--stage-retries", type=int, default=1)
    parser.add_argument("--self-check", type=str2bool, default=True)
    parser.add_argument("--route-audit", type=str2bool, default=True)
    parser.add_argument("--spatial-audit", type=str2bool, default=True)
    parser.add_argument("--sleep-between-retries", type=float, default=0.5)
    parser.add_argument("--sleep-between-requests", type=float, default=0.0)

    parser.add_argument("--min-words", type=int, default=12)
    parser.add_argument("--max-words", type=int, default=120)
    parser.add_argument("--drop-failed", type=str2bool, default=False)
    parser.add_argument("--write-partial-output", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        raise SystemExit(run(args))
    except KeyboardInterrupt:
        raise
    except Exception as error:
        print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
