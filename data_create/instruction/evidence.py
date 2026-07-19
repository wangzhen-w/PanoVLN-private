"""Panorama evidence selection and contact-sheet rendering."""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .actions import (
    build_heading_by_frame,
    major_turns,
    segment_action_summary,
    select_endpoint_frames,
    select_route_frames,
    select_start_frames,
)
from .io_utils import hash_file


@dataclass(frozen=True)
class SegmentEvidence:
    segment_id: int
    frames: List[int]
    action_summary: Dict[str, Any]
    sheet: bytes
    sheet_path: Optional[str]


@dataclass(frozen=True)
class EvidencePacket:
    image_key: str
    selected_frames: List[int]
    start_frames: List[int]
    endpoint_frames: List[int]
    frame_paths: List[Path]
    route_sheet: bytes
    start_sheet: bytes
    endpoint_sheet: bytes
    final_sheet: bytes
    final_view_tiles: List[Tuple[str, bytes]]
    route_sheet_path: Optional[str]
    start_sheet_path: Optional[str]
    endpoint_sheet_path: Optional[str]
    final_sheet_path: Optional[str]
    segment_sheets: List[SegmentEvidence]
    evidence_fingerprint: str


def image_key_candidates(row: Mapping[str, Any]) -> List[str]:
    candidates = []
    for key in ("trajectory_id", "source_trajectory_id", "episode_id"):
        value = row.get(key)
        if value is None:
            continue
        text = str(value)
        if text not in candidates:
            candidates.append(text)
    return candidates


def sorted_frame_paths(image_root: str, row: Mapping[str, Any]) -> Tuple[str, List[Path]]:
    root = Path(image_root)
    missing = []
    for key in image_key_candidates(row):
        directory = root / key
        if not directory.is_dir():
            missing.append(str(directory))
            continue

        def frame_index(path: Path) -> int:
            match = re.fullmatch(r"frame_(0|[1-9][0-9]*)\.jpg", path.name)
            if match is None:
                raise ValueError(f"Unexpected frame filename: {path}")
            return int(match.group(1))

        paths = sorted(directory.glob("frame_*.jpg"), key=frame_index)
        indices = [frame_index(path) for path in paths]
        if not paths:
            raise FileNotFoundError(f"No frame_*.jpg files under {directory}")
        if indices != list(range(len(indices))):
            raise ValueError(
                f"Non-contiguous frame sequence under {directory}: got {indices[:20]}"
            )
        return key, paths
    raise FileNotFoundError(
        "Missing image directory for row "
        f"episode_id={row.get('episode_id')}; tried {missing}"
    )


def equirect_to_perspective(
    image: Image.Image,
    *,
    yaw_degrees: float,
    pitch_degrees: float = 0.0,
    hfov_degrees: float = 90.0,
    width: int,
    height: int,
) -> Image.Image:
    source = np.asarray(image.convert("RGB"))
    source_h, source_w = source.shape[:2]
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y = np.linspace(-height / width, height / width, height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x, -y)
    focal = 1.0 / np.tan(np.radians(hfov_degrees) / 2.0)
    dirs = np.stack([grid_x, grid_y, np.full_like(grid_x, focal)], axis=-1)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    yaw = np.radians(yaw_degrees)
    pitch = np.radians(pitch_degrees)
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)

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
    map_x = ((lon / (2.0 * np.pi)) + 0.5) * source_w
    map_y = (0.5 - lat / np.pi) * source_h
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
    *,
    tile_width: int,
    tile_height: int,
    jpeg_quality: int,
    final_label: bool,
) -> bytes:
    rows: List[Image.Image] = []
    gap = 4
    final_frame = frames[-1] if frames else -1
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
                prefix = "FINAL" if final_label and frame_index == final_frame else "step"
                tiles.append(draw_label(view, f"{prefix} {frame_index} - {label}"))
        row_width = sum(tile.width for tile in tiles) + gap * (len(tiles) - 1)
        row_height = max(tile.height for tile in tiles)
        row = Image.new("RGB", (row_width, row_height), (255, 255, 255))
        x = 0
        for tile in tiles:
            row.paste(tile, (x, 0))
            x += tile.width + gap
        rows.append(row)
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows) + gap * (len(rows) - 1)
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height + gap
    buffer = io.BytesIO()
    sheet.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    return buffer.getvalue()


def render_single_view(
    frame_paths: Sequence[Path],
    *,
    frame_index: int,
    heading_by_frame: Sequence[float],
    label: str,
    yaw_offset: float,
    pitch: float,
    tile_width: int,
    tile_height: int,
    jpeg_quality: int,
) -> bytes:
    with Image.open(frame_paths[frame_index]) as panorama:
        heading = heading_by_frame[min(frame_index, len(heading_by_frame) - 1)]
        view = equirect_to_perspective(
            panorama,
            yaw_degrees=heading + yaw_offset,
            pitch_degrees=pitch,
            width=tile_width,
            height=tile_height,
        )
    tile = draw_label(view, label, label_height=32)
    buffer = io.BytesIO()
    tile.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    return buffer.getvalue()


def split_frame_segments(
    frames: Sequence[int],
    *,
    max_rows: int,
    overlap: int,
) -> List[List[int]]:
    unique = list(dict.fromkeys(int(frame) for frame in frames))
    if not unique:
        return []
    max_rows = max(2, int(max_rows))
    overlap = max(0, min(int(overlap), max_rows - 1))
    if len(unique) <= max_rows:
        return [unique]
    segments: List[List[int]] = []
    start = 0
    while start < len(unique):
        end = min(len(unique), start + max_rows)
        segments.append(unique[start:end])
        if end == len(unique):
            break
        start = max(0, end - overlap)
    return segments


def segmented_route_enabled(actions: Sequence[int], args: Any) -> bool:
    mode = str(getattr(args, "route_evidence_mode", "auto"))
    if mode == "segmented":
        return True
    if mode == "auto":
        return route_needs_segmented_evidence(
            actions,
            action_threshold=int(getattr(args, "segmented_min_actions", 80)),
            max_waypoints=int(getattr(args, "max_waypoints", 14)),
        )
    return False


def route_needs_segmented_evidence(
    actions: Sequence[int],
    *,
    action_threshold: int,
    max_waypoints: int,
) -> bool:
    """Decide whether one overview sheet would over-compress a route.

    Action count alone is a poor proxy for route difficulty because in-place
    rotations and translated waypoints have very different visual impact.  In
    auto mode we therefore segment either conventionally long routes or routes
    that contain both substantial translation and several major heading
    changes.  The latter identifies multi-space routes that need local evidence
    even when their low-level action list happens to fall below the length
    threshold.
    """

    normalized_actions = [int(action) for action in actions]
    non_stop_actions = [action for action in normalized_actions if action != 0]
    if len(normalized_actions) >= max(1, int(action_threshold)):
        return True
    translated_positions = 1 + sum(action == 1 for action in non_stop_actions)
    route_choices = len(major_turns(non_stop_actions))
    evidence_capacity = max(2, int(max_waypoints))
    return translated_positions > 2 * evidence_capacity and route_choices >= 3


def evidence_frame_fingerprint(
    row: Mapping[str, Any],
    *,
    image_root: str,
    max_waypoints: int,
    start_window_frames: int,
    endpoint_window_frames: int,
    route_evidence_mode: str = "auto",
    segmented_min_actions: int = 80,
    segment_max_waypoints: int = 0,
    mode: str = "content",
) -> str:
    _, frame_paths = sorted_frame_paths(image_root, row)
    actions = [int(action) for action in row["actions"]]
    expected = sum(action != 0 for action in actions) + 1
    if len(frame_paths) != expected:
        raise ValueError(
            f"Action/frame mismatch for episode {row.get('episode_id')}: "
            f"expected {expected}, found {len(frame_paths)}"
        )
    route_selected = select_route_frames(actions, len(frame_paths), max_waypoints)
    extra_segment_frames: List[int] = []
    use_segments = route_evidence_mode == "segmented" or (
        route_evidence_mode == "auto"
        and route_needs_segmented_evidence(
            actions,
            action_threshold=int(segmented_min_actions),
            max_waypoints=int(max_waypoints),
        )
    )
    if use_segments:
        if int(segment_max_waypoints) <= 0:
            segment_max_waypoints = max_waypoints
        extra_segment_frames = select_route_frames(
            actions,
            len(frame_paths),
            segment_max_waypoints,
        )
    selected = sorted(
        set(
            route_selected
            + extra_segment_frames
            + select_start_frames(actions, len(frame_paths), start_window_frames)
            + select_endpoint_frames(actions, len(frame_paths), endpoint_window_frames)
        )
    )
    digest = hashlib.sha256()
    digest.update(json_bytes({"selected_frames": selected, "image_keys": image_key_candidates(row)}))
    for index in selected:
        hash_file(frame_paths[index], digest, mode)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def build_evidence(row: Mapping[str, Any], args: Any) -> EvidencePacket:
    image_key, frame_paths = sorted_frame_paths(args.image_root, row)
    actions = [int(action) for action in row["actions"]]
    expected = sum(action != 0 for action in actions) + 1
    if len(frame_paths) != expected:
        raise ValueError(
            f"Action/frame mismatch for episode {row.get('episode_id')}: "
            f"expected {expected} frames, found {len(frame_paths)}"
        )
    if args.use_action_heading:
        headings = build_heading_by_frame(actions, len(frame_paths))
    else:
        headings = [0.0 for _ in frame_paths]
    route_columns = [("left", -90.0, 0.0), ("forward", 0.0, 0.0), ("right", 90.0, 0.0), ("back", 180.0, 0.0)]
    endpoint_columns = route_columns + [("forward-down", 0.0, 25.0)]
    final_columns = [
        ("FORWARD", 0.0, 0.0),
        ("FORWARD-DOWN", 0.0, 25.0),
        ("LEFT", -90.0, 0.0),
        ("RIGHT", 90.0, 0.0),
        ("BACK", 180.0, 0.0),
    ]
    selected = select_route_frames(actions, len(frame_paths), args.max_waypoints)
    segment_sheets: List[SegmentEvidence] = []
    if segmented_route_enabled(actions, args):
        segment_max_waypoints = getattr(args, "segment_max_waypoints", 0)
        if int(segment_max_waypoints) <= 0:
            segment_max_waypoints = args.max_waypoints
        segment_frames = select_route_frames(
            actions,
            len(frame_paths),
            segment_max_waypoints,
        )
        for segment_id, frames in enumerate(
            split_frame_segments(
                segment_frames,
                max_rows=getattr(args, "segment_rows", 5),
                overlap=getattr(args, "segment_overlap", 1),
            ),
            start=1,
        ):
            segment_sheet = render_view_sheet(
                frame_paths,
                frames,
                headings,
                route_columns,
                tile_width=args.tile_width,
                tile_height=args.tile_height,
                jpeg_quality=args.jpeg_quality,
                final_label=False,
            )
            segment_sheets.append(
                SegmentEvidence(
                    segment_id=segment_id,
                    frames=list(frames),
                    action_summary=segment_action_summary(actions, frames),
                    sheet=segment_sheet,
                    sheet_path=None,
                )
            )
    start = select_start_frames(actions, len(frame_paths), args.start_window_frames)
    endpoint = select_endpoint_frames(actions, len(frame_paths), args.endpoint_window_frames)
    route_sheet = render_view_sheet(
        frame_paths,
        selected,
        headings,
        route_columns,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        jpeg_quality=args.jpeg_quality,
        final_label=True,
    )
    start_sheet = render_view_sheet(
        frame_paths,
        start,
        headings,
        endpoint_columns,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        jpeg_quality=args.jpeg_quality,
        final_label=False,
    )
    endpoint_sheet = render_view_sheet(
        frame_paths,
        endpoint,
        headings,
        endpoint_columns,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        jpeg_quality=args.jpeg_quality,
        final_label=True,
    )
    final_sheet = render_view_sheet(
        frame_paths,
        [len(frame_paths) - 1],
        headings,
        final_columns,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        jpeg_quality=args.jpeg_quality,
        final_label=True,
    )
    final_frame = len(frame_paths) - 1
    final_view_tiles = [
        (
            f"FINAL {final_frame} {label}",
            render_single_view(
                frame_paths,
                frame_index=final_frame,
                heading_by_frame=headings,
                label=f"FINAL {final_frame} - {label}",
                yaw_offset=yaw_offset,
                pitch=pitch,
                tile_width=args.tile_width,
                tile_height=args.tile_height,
                jpeg_quality=args.jpeg_quality,
            ),
        )
        for label, yaw_offset, pitch in final_columns
    ]
    route_path = start_path = endpoint_path = final_path = None
    if args.save_contact_sheets or args.write_gallery:
        contact_root = Path(args.work_dir) / "contact_sheets"
        route_dir = contact_root / "route"
        start_dir = contact_root / "start"
        endpoint_dir = contact_root / "endpoint"
        final_dir = contact_root / "final"
        segment_dir = contact_root / "segments"
        for directory in (route_dir, start_dir, endpoint_dir, final_dir, segment_dir):
            directory.mkdir(parents=True, exist_ok=True)
        episode_id = str(row["episode_id"])
        route_path = str(route_dir / f"{episode_id}.jpg")
        start_path = str(start_dir / f"{episode_id}.jpg")
        endpoint_path = str(endpoint_dir / f"{episode_id}.jpg")
        final_path = str(final_dir / f"{episode_id}.jpg")
        Path(route_path).write_bytes(route_sheet)
        Path(start_path).write_bytes(start_sheet)
        Path(endpoint_path).write_bytes(endpoint_sheet)
        Path(final_path).write_bytes(final_sheet)
        saved_segments: List[SegmentEvidence] = []
        for segment in segment_sheets:
            segment_path = str(segment_dir / f"{episode_id}_seg{segment.segment_id:02d}.jpg")
            Path(segment_path).write_bytes(segment.sheet)
            saved_segments.append(
                SegmentEvidence(
                    segment_id=segment.segment_id,
                    frames=segment.frames,
                    action_summary=segment.action_summary,
                    sheet=segment.sheet,
                    sheet_path=segment_path,
                )
            )
        segment_sheets = saved_segments
    fingerprint = evidence_frame_fingerprint(
        row,
        image_root=args.image_root,
        max_waypoints=args.max_waypoints,
        start_window_frames=args.start_window_frames,
        endpoint_window_frames=args.endpoint_window_frames,
        route_evidence_mode=getattr(args, "route_evidence_mode", "auto"),
        segmented_min_actions=getattr(args, "segmented_min_actions", 80),
        segment_max_waypoints=getattr(args, "segment_max_waypoints", 0),
        mode=args.evidence_fingerprint_mode,
    )
    return EvidencePacket(
        image_key=image_key,
        selected_frames=selected,
        start_frames=start,
        endpoint_frames=endpoint,
        frame_paths=frame_paths,
        route_sheet=route_sheet,
        start_sheet=start_sheet,
        endpoint_sheet=endpoint_sheet,
        final_sheet=final_sheet,
        final_view_tiles=final_view_tiles,
        route_sheet_path=route_path,
        start_sheet_path=start_path,
        endpoint_sheet_path=endpoint_path,
        final_sheet_path=final_path,
        segment_sheets=segment_sheets,
        evidence_fingerprint=fingerprint,
    )
