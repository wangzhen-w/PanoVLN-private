#!/usr/bin/env python3
"""Export paired instructions and expert actions to the ScaleVLN-style VLN-CE contract.

The exporter is deliberately a strict publication gate.  It does not try to
repair incomplete instruction runs or malformed trajectories: both instruction
styles must describe the same source episodes and carry matching trajectory
provenance before any output is published.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from data_create.trajectory.scene_paths import validate_public_scene_id


REQUIRED_STYLES = ("concise", "dense")
QUATERNION_NORM_TOLERANCE = 1e-3
POSITION_TOLERANCE = 1e-3
EMPTY_INSTRUCTION_VOCAB = {
    "word_list": [],
    "word2idx_dict": {},
    "stoi": {},
    "itos": [],
    "num_vocab": 0,
    "UNK_INDEX": 1,
    "PAD_INDEX": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Collected VLN-CE JSON/JSON.GZ")
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        metavar="STYLE=JSONL",
        help="Verified instruction JSONL; provide concise and dense exactly once.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--gzip-output",
        required=True,
        help="Compressed copy of --output with identical JSON content.",
    )
    parser.add_argument(
        "--source-gt",
        required=True,
        help="Source trajectory GT JSON/JSON.GZ without terminal STOP.",
    )
    parser.add_argument(
        "--image-root",
        required=True,
        help="Panorama root containing <trajectory_id>/frame_<index>.jpg.",
    )
    parser.add_argument(
        "--gt-output",
        required=True,
        help="R2R-compatible train_gt.json.gz output with terminal STOP.",
    )
    parser.add_argument("--goal-radius", type=float, default=0.3)
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help=(
            "Allow the paired variant files to cover a strict subset of the "
            "dataset. By default, complete episode coverage is required."
        ),
    )
    return parser.parse_args()


def read_json(path: str) -> Dict[str, Any]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def canonical_episode_id(value: Any, context: str) -> int:
    """Accept only the integer episode IDs used by the target schema."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} episode_id must be a non-negative integer")
    return value


def read_jsonl(path: str) -> Dict[int, Dict[str, Any]]:
    rows: Dict[int, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} invalid JSON: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            episode_id = canonical_episode_id(
                row.get("episode_id"), f"{path}:{line_number}"
            )
            if episode_id in rows:
                raise ValueError(f"{path}:{line_number} duplicate episode_id {episode_id}")
            instruction = row.get("instruction")
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(f"{path}:{line_number} has no instruction")
            if "status" in row and row["status"] != "success":
                raise ValueError(
                    f"{path}:{line_number} has non-success status {row['status']!r}"
                )
            validate_finite_numbers(row, f"{path}:{line_number}")
            rows[episode_id] = row
    if not rows:
        raise ValueError(f"{path} contains no instruction rows")
    return rows


def parse_variants(values: Iterable[str]) -> List[Tuple[str, str]]:
    parsed: Dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected STYLE=JSONL, got {value!r}")
        style, path = value.split("=", 1)
        style = style.strip()
        path = path.strip()
        if style not in REQUIRED_STYLES:
            raise ValueError(f"Unknown instruction style: {style}")
        if style in parsed:
            raise ValueError(f"Duplicate style: {style}")
        if not path:
            raise ValueError(f"Variant {style} has an empty path")
        parsed[style] = path
    missing = sorted(set(REQUIRED_STYLES) - set(parsed))
    if missing:
        raise ValueError(f"Both concise and dense variants are required; missing={missing}")
    return [(style, parsed[style]) for style in REQUIRED_STYLES]


def validate_finite_numbers(value: Any, context: str) -> None:
    """Reject JSON NaN/Infinity anywhere in a published source or row."""

    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{context} contains a non-finite number")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            validate_finite_numbers(child, f"{context}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            validate_finite_numbers(child, f"{context}[{index}]")


def validate_vector(value: Any, length: int, context: str) -> List[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{context} must be a length-{length} list")
    result = []
    for index, coordinate in enumerate(value):
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            raise ValueError(f"{context}[{index}] must be numeric")
        coordinate = float(coordinate)
        if not math.isfinite(coordinate):
            raise ValueError(f"{context}[{index}] must be finite")
        result.append(coordinate)
    return result


def validate_source_episode(episode: Any, context: str) -> int:
    if not isinstance(episode, dict):
        raise ValueError(f"{context} must be an object")
    episode_id = canonical_episode_id(episode.get("episode_id"), context)
    validate_finite_numbers(episode, context)
    if not isinstance(episode.get("scene_id"), str) or not episode["scene_id"].strip():
        raise ValueError(f"{context}.scene_id must be a non-empty string")
    try:
        validate_public_scene_id(episode["scene_id"])
    except ValueError as error:
        raise ValueError(f"{context}.scene_id is not a public HM3D path: {error}") from error
    trajectory_id = episode.get("trajectory_id")
    if isinstance(trajectory_id, bool) or not isinstance(trajectory_id, (str, int)):
        raise ValueError(f"{context}.trajectory_id must be a string or integer")
    if isinstance(trajectory_id, str) and not trajectory_id.strip():
        raise ValueError(f"{context}.trajectory_id cannot be empty")
    if isinstance(trajectory_id, int) and trajectory_id < 0:
        raise ValueError(f"{context}.trajectory_id cannot be negative")
    if not isinstance(episode.get("info"), dict):
        raise ValueError(f"{context}.info must be an object")

    start = validate_vector(episode.get("start_position"), 3, f"{context}.start_position")
    rotation = validate_vector(
        episode.get("start_rotation"), 4, f"{context}.start_rotation"
    )
    rotation_norm = math.sqrt(sum(value * value for value in rotation))
    if abs(rotation_norm - 1.0) > QUATERNION_NORM_TOLERANCE:
        raise ValueError(
            f"{context}.start_rotation must be a unit quaternion; norm={rotation_norm:.8f}"
        )

    goals = episode.get("goals")
    if not isinstance(goals, list) or len(goals) != 1:
        raise ValueError(f"{context}.goals must contain exactly one goal")
    goal_positions = []
    for index, goal in enumerate(goals):
        if not isinstance(goal, dict):
            raise ValueError(f"{context}.goals[{index}] must be an object")
        goal_positions.append(
            validate_vector(goal.get("position"), 3, f"{context}.goals[{index}].position")
        )
        radius = goal.get("radius")
        if (
            isinstance(radius, bool)
            or not isinstance(radius, (int, float))
            or not math.isfinite(float(radius))
            or float(radius) <= 0
        ):
            raise ValueError(f"{context}.goals[{index}].radius must be finite and positive")

    reference_path = episode.get("reference_path")
    if not isinstance(reference_path, list) or len(reference_path) < 2:
        raise ValueError(f"{context}.reference_path must contain at least two points")
    reference = [
        validate_vector(point, 3, f"{context}.reference_path[{index}]")
        for index, point in enumerate(reference_path)
    ]
    if math.dist(start, reference[0]) > POSITION_TOLERANCE:
        raise ValueError(f"{context}.reference_path does not start at start_position")
    if min(math.dist(reference[-1], goal) for goal in goal_positions) > POSITION_TOLERANCE:
        raise ValueError(f"{context}.reference_path does not end at a goal position")
    return episode_id


def validate_actions(actions: Any, context: str) -> Tuple[int, ...]:
    if not isinstance(actions, list) or not actions:
        raise ValueError(f"{context} actions must be a non-empty list")
    if any(isinstance(action, bool) or not isinstance(action, int) for action in actions):
        raise ValueError(f"{context} actions must contain integer action IDs")
    invalid = sorted(set(actions) - {0, 1, 2, 3})
    if invalid:
        raise ValueError(f"{context} actions contain invalid IDs: {invalid}")
    if actions[-1] != 0 or actions.count(0) != 1:
        raise ValueError(f"{context} actions must contain exactly one terminal STOP (0)")
    return tuple(actions)


def index_source_gt(payload: Mapping[str, Any], path: str) -> Dict[int, Dict[str, Any]]:
    indexed: Dict[int, Dict[str, Any]] = {}
    for raw_key, record in payload.items():
        if not isinstance(raw_key, str) or re.fullmatch(r"0|[1-9][0-9]*", raw_key) is None:
            raise ValueError(f"{path} has non-canonical episode key {raw_key!r}")
        episode_id = int(raw_key)
        if episode_id in indexed:
            raise ValueError(f"{path} has duplicate normalized episode key {episode_id}")
        if not isinstance(record, dict):
            raise ValueError(f"{path}[{raw_key!r}] must be an object")
        indexed[episode_id] = record
    if not indexed:
        raise ValueError(f"{path} contains no GT records")
    return indexed


def validate_source_gt_record(
    record: Mapping[str, Any],
    source: Mapping[str, Any],
    episode_id: int,
    output_goal_radius: float,
) -> Dict[str, Any]:
    context = f"source GT episode {episode_id}"
    validate_finite_numbers(record, context)

    actions = record.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError(f"{context}.actions must be a non-empty list")
    if any(isinstance(action, bool) or not isinstance(action, int) for action in actions):
        raise ValueError(f"{context}.actions must contain integer action IDs")
    if 0 in actions:
        raise ValueError(f"{context}.actions must not contain STOP (0)")
    invalid = sorted(set(actions) - {1, 2, 3})
    if invalid:
        raise ValueError(f"{context}.actions contain invalid IDs: {invalid}")

    forward_steps = record.get("forward_steps")
    if (
        isinstance(forward_steps, bool)
        or not isinstance(forward_steps, int)
        or forward_steps <= 0
    ):
        raise ValueError(f"{context}.forward_steps must be a positive integer")
    if actions.count(1) != forward_steps:
        raise ValueError(
            f"{context}.forward_steps={forward_steps} but actions contain "
            f"{actions.count(1)} MOVE_FORWARD actions"
        )

    raw_locations = record.get("locations")
    if not isinstance(raw_locations, list) or len(raw_locations) != forward_steps + 1:
        raise ValueError(
            f"{context}.locations must contain forward_steps + 1 positions"
        )
    locations = [
        validate_vector(location, 3, f"{context}.locations[{index}]")
        for index, location in enumerate(raw_locations)
    ]
    start = validate_vector(source["start_position"], 3, f"dataset episode {episode_id}.start")
    if math.dist(locations[0], start) > POSITION_TOLERANCE:
        raise ValueError(f"{context}.locations does not start at dataset start_position")

    goal = source["goals"][0]
    goal_position = validate_vector(
        goal["position"], 3, f"dataset episode {episode_id}.goal"
    )
    source_radius = float(goal["radius"])
    terminal_distance = math.dist(locations[-1], goal_position)
    if terminal_distance > source_radius + POSITION_TOLERANCE:
        raise ValueError(
            f"{context} terminal location is {terminal_distance:.4f}m from goal, "
            f"outside source radius {source_radius:.4f}m"
        )
    if terminal_distance > output_goal_radius + POSITION_TOLERANCE:
        raise ValueError(
            f"{context} terminal location is outside output goal radius "
            f"{output_goal_radius:.4f}m"
        )

    return {
        "locations": copy.deepcopy(raw_locations),
        "actions": list(actions),
        "forward_steps": forward_steps,
    }


def validate_image_sequence(
    image_root: Path,
    trajectory_id: Any,
    actions_without_stop: Sequence[int],
) -> int:
    """Require one initial frame plus one frame after every non-STOP action."""

    directory = image_root / str(trajectory_id)
    if not directory.is_dir():
        raise ValueError(
            f"Missing image directory for trajectory {trajectory_id}: {directory}"
        )
    indices: List[int] = []
    for path in directory.glob("frame_*.jpg"):
        match = re.fullmatch(r"frame_(0|[1-9][0-9]*)\.jpg", path.name)
        if match is None:
            raise ValueError(f"Invalid panorama filename: {path}")
        indices.append(int(match.group(1)))
    expected_count = len(actions_without_stop) + 1
    expected_indices = list(range(expected_count))
    if sorted(indices) != expected_indices:
        raise ValueError(
            f"Trajectory {trajectory_id} image/action mismatch: expected frame indices "
            f"0..{expected_count - 1}, got {sorted(indices)[:20]}"
        )
    return expected_count


def row_trajectory_id(row: Mapping[str, Any]) -> Any:
    for key in ("trajectory_id", "source_trajectory_id"):
        if key in row:
            return row[key]
    return None


def validate_row_provenance(
    row: Mapping[str, Any], source: Mapping[str, Any], style: str, episode_id: int
) -> Dict[str, Any]:
    """Validate any available trajectory keys and return normalized evidence."""

    context = f"Variant {style} episode {episode_id}"
    if row.get("instruction_profile") != style:
        raise ValueError(
            f"{context} instruction_profile={row.get('instruction_profile')!r}"
        )

    evidence: Dict[str, Any] = {}
    if "actions" in row:
        evidence["actions"] = validate_actions(row["actions"], context)

    trajectory_id = row_trajectory_id(row)
    if isinstance(trajectory_id, bool) or not isinstance(trajectory_id, (str, int)):
        raise ValueError(f"{context} trajectory_id must be a string or integer")
    if str(trajectory_id) != str(source["trajectory_id"]):
        raise ValueError(
            f"{context} trajectory_id {trajectory_id!r} does not match "
            f"source {source['trajectory_id']!r}"
        )
    evidence["trajectory_id"] = str(trajectory_id)

    input_fingerprint = row.get("input_fingerprint")
    if not isinstance(input_fingerprint, str) or not input_fingerprint:
        raise ValueError(f"{context} must carry a non-empty input_fingerprint")
    evidence["input_fingerprint"] = input_fingerprint
    pipeline_fingerprint = row.get("pipeline_fingerprint")
    if not isinstance(pipeline_fingerprint, str) or not pipeline_fingerprint:
        raise ValueError(f"{context} must carry a non-empty pipeline_fingerprint")

    route_hash = row.get("source_route_hash", row.get("route_hash"))
    if route_hash is not None:
        source_hash = (source.get("info") or {}).get("route_hash")
        if not isinstance(route_hash, str) or route_hash != source_hash:
            raise ValueError(f"{context} route_hash does not match source trajectory")
        evidence["route_hash"] = route_hash

    if not evidence:
        raise ValueError(
            f"{context} has no provenance; provide actions, trajectory_id, or route_hash"
        )
    return evidence


def validate_paired_provenance(
    evidence_by_style: Mapping[str, Mapping[str, Any]], episode_id: int
) -> Tuple[int, ...] | None:
    concise = evidence_by_style["concise"]
    dense = evidence_by_style["dense"]
    for key in ("trajectory_id", "input_fingerprint"):
        if key not in concise or key not in dense or concise[key] != dense[key]:
            raise ValueError(f"Episode {episode_id} concise/dense {key} does not match")
    if ("actions" in concise) != ("actions" in dense):
        raise ValueError(
            f"Episode {episode_id} paired variants must both carry actions or both omit them"
        )
    if "actions" in concise:
        if concise["actions"] != dense["actions"]:
            raise ValueError(f"Episode {episode_id} concise/dense actions do not match")
        return concise["actions"]

    common_keys = set(concise) & set(dense) & {"trajectory_id", "route_hash"}
    if not common_keys:
        raise ValueError(
            f"Episode {episode_id} concise/dense rows have no common provenance field"
        )
    for key in common_keys:
        if concise[key] != dense[key]:
            raise ValueError(f"Episode {episode_id} concise/dense {key} does not match")
    return None


def source_id_sha256(episode_ids: Sequence[int]) -> str:
    payload = json.dumps(list(episode_ids), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def geodesic_distance(info: Mapping[str, Any], reference_path: Sequence[Sequence[float]]) -> float:
    for key in ("geodesic_distance", "shortest_distance", "reference_length"):
        value = info.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        value = float(value)
        if math.isfinite(value) and value > 0:
            return value
    length = sum(
        math.dist(reference_path[index - 1], reference_path[index])
        for index in range(1, len(reference_path))
    )
    if not math.isfinite(length) or length <= 0:
        raise ValueError("Cannot derive a positive geodesic_distance")
    return length


def stage_json(path: Path, payload: Dict[str, Any]) -> Path:
    """Fully serialize and fsync a JSON artifact beside its final path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        opener = gzip.open if path.name.endswith(".gz") else open
        with opener(temporary, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, allow_nan=False)
            handle.flush()
        # Ensure the staged bytes reach the filesystem before publication.
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def publish_json_artifacts(
    artifacts: Sequence[Tuple[Path, Dict[str, Any]]]
) -> None:
    """Stage every artifact before atomically replacing any individual file."""

    staged: List[Tuple[Path, Path]] = []
    try:
        for path, payload in artifacts:
            staged.append((path, stage_json(path, payload)))
        for path, temporary in staged:
            os.replace(temporary, path)
    finally:
        for _, temporary in staged:
            temporary.unlink(missing_ok=True)


def validate_export_arguments(args: argparse.Namespace) -> None:
    if not math.isfinite(args.goal_radius) or args.goal_radius <= 0:
        raise ValueError("--goal-radius must be finite and positive")


def export(args: argparse.Namespace) -> Dict[str, Any]:
    validate_export_arguments(args)
    variants = parse_variants(args.variant)
    output_path = Path(args.output).resolve()
    gzip_output_path = Path(args.gzip_output).resolve()
    source_gt_path = Path(args.source_gt).resolve()
    image_root = Path(args.image_root).resolve()
    gt_output_path = Path(args.gt_output).resolve()
    if not image_root.is_dir():
        raise ValueError(f"--image-root is not a directory: {image_root}")
    if not gt_output_path.name.endswith(".json.gz"):
        raise ValueError("--gt-output must end with .json.gz")
    inputs = {
        Path(args.dataset).resolve(),
        *(Path(path).resolve() for _, path in variants),
    }
    inputs.add(source_gt_path)
    outputs = {output_path, gzip_output_path, gt_output_path}
    if len(outputs) != 3:
        raise ValueError("Dataset JSON, dataset JSON.GZ, and GT outputs must be distinct")
    collisions = sorted(str(path) for path in outputs & inputs)
    if collisions:
        raise ValueError(f"Output artifacts must not overwrite inputs: {collisions}")

    dataset = read_json(args.dataset)
    episodes = dataset.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Collected dataset must contain a non-empty episodes list")
    source_episodes: Dict[int, Dict[str, Any]] = {}
    for index, episode in enumerate(episodes):
        episode_id = validate_source_episode(episode, f"dataset.episodes[{index}]")
        if episode_id in source_episodes:
            raise ValueError(f"Collected dataset has duplicate episode_id {episode_id}")
        source_episodes[episode_id] = episode

    rows_by_style = {style: read_jsonl(path) for style, path in variants}
    concise_ids = set(rows_by_style["concise"])
    dense_ids = set(rows_by_style["dense"])
    if concise_ids != dense_ids:
        concise_only = sorted(concise_ids - dense_ids)
        dense_only = sorted(dense_ids - concise_ids)
        raise ValueError(
            "Concise/dense episode ID sets must be identical: "
            f"concise_only={concise_only[:10]}, dense_only={dense_only[:10]}"
        )
    selected_ids = sorted(concise_ids)
    source_ids = set(source_episodes)
    extra = sorted(concise_ids - source_ids)
    missing = sorted(source_ids - concise_ids)
    if extra:
        raise ValueError(f"Variant episode IDs are absent from dataset: {extra[:10]}")
    allow_subset = bool(getattr(args, "allow_subset", False))
    if missing and not allow_subset:
        raise ValueError(
            f"Variants do not completely cover dataset; missing={missing[:10]}. "
            "Use --allow-subset only for an explicit partial/ablation export."
        )

    source_gt_by_id: Dict[int, Dict[str, Any]] = {}
    selected_source_gt: Dict[int, Dict[str, Any]] = {}
    source_gt_by_id = index_source_gt(
        read_json(str(source_gt_path)), str(source_gt_path)
    )
    unknown_gt = sorted(set(source_gt_by_id) - source_ids)
    if unknown_gt:
        raise ValueError(
            f"Source GT contains IDs absent from source dataset: {unknown_gt[:10]}"
        )
    missing_gt = sorted(set(selected_ids) - set(source_gt_by_id))
    if missing_gt:
        raise ValueError(
            f"Source GT does not cover selected episode IDs: {missing_gt[:10]}"
        )
    for episode_id in selected_ids:
        selected_source_gt[episode_id] = validate_source_gt_record(
            source_gt_by_id[episode_id],
            source_episodes[episode_id],
            episode_id,
            float(args.goal_radius),
        )

    # Panorama folders are keyed by the source episode ID. Preserve that value
    # as trajectory_id so every exported language variant can resolve the same
    # visual trajectory without an additional mapping sidecar.
    trajectory_ids = {
        episode_id: source_episodes[episode_id]["trajectory_id"]
        for episode_id in selected_ids
    }
    image_frame_counts = {
        episode_id: validate_image_sequence(
            image_root,
            trajectory_ids[episode_id],
            selected_source_gt[episode_id]["actions"],
        )
        for episode_id in selected_ids
    }
    if not allow_subset:
        actual_image_directories = {
            path.name
            for path in image_root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        }
        expected_image_directories = {
            str(trajectory_ids[episode_id]) for episode_id in selected_ids
        }
        if actual_image_directories != expected_image_directories:
            raise ValueError(
                "Image directory set does not match trajectories: "
                f"missing={sorted(expected_image_directories - actual_image_directories)[:10]}, "
                f"extra={sorted(actual_image_directories - expected_image_directories)[:10]}"
            )
    pair_actions: Dict[int, Tuple[int, ...] | None] = {}
    for episode_id in selected_ids:
        evidence = {
            style: validate_row_provenance(
                rows_by_style[style][episode_id],
                source_episodes[episode_id],
                style,
                episode_id,
            )
            for style in REQUIRED_STYLES
        }
        pair_actions[episode_id] = validate_paired_provenance(evidence, episode_id)
        if pair_actions[episode_id] is None:
            raise ValueError(
                f"Episode {episode_id} variants must carry actions when publishing GT"
            )
        expected_actions = tuple(selected_source_gt[episode_id]["actions"] + [0])
        if pair_actions[episode_id] != expected_actions:
            raise ValueError(
                f"Episode {episode_id} variant actions do not equal source GT actions + STOP"
            )

    output_episodes: List[Dict[str, Any]] = []
    per_style = {style: 0 for style in REQUIRED_STYLES}
    output_gt: Dict[str, Dict[str, Any]] = {}
    # Interleave concise/dense so every adjacent pair shares trajectory_id.
    for source_id in selected_ids:
        source = source_episodes[source_id]
        source_info = dict(source.get("info") or {})
        for style in REQUIRED_STYLES:
            row = rows_by_style[style][source_id]
            per_style[style] += 1

            output_episode_id = len(output_episodes) + 1
            distance = geodesic_distance(source_info, source["reference_path"])
            goals = [
                {
                    "position": copy.deepcopy(source["goals"][0]["position"]),
                    "radius": float(args.goal_radius),
                }
            ]
            # Construct the public episode from an explicit R2R/ScaleVLN
            # whitelist.  Source-only actions, locations, sampling metadata,
            # or future auxiliary fields can never leak into train.json.
            episode = {
                "episode_id": output_episode_id,
                "trajectory_id": trajectory_ids[source_id],
                "scene_id": source["scene_id"],
                "start_position": copy.deepcopy(source["start_position"]),
                "start_rotation": copy.deepcopy(source["start_rotation"]),
                "info": {"geodesic_distance": distance},
                "goals": goals,
                "instruction": {
                    "instruction_text": row["instruction"].strip(),
                    "instruction_tokens": None,
                },
                "reference_path": copy.deepcopy(source["reference_path"]),
            }
            validate_finite_numbers(episode, f"output episode {episode['episode_id']}")
            output_episodes.append(episode)

            source_gt_record = selected_source_gt[source_id]
            published_actions = list(source_gt_record["actions"]) + [0]
            if len(published_actions) != image_frame_counts[source_id]:
                raise ValueError(
                    f"Trajectory {source_id} GT actions including STOP do not match frames"
                )
            output_gt[str(output_episode_id)] = {
                "locations": copy.deepcopy(source_gt_record["locations"]),
                "actions": published_actions,
                "forward_steps": source_gt_record["forward_steps"],
            }

    expected_gt_keys = {str(episode["episode_id"]) for episode in output_episodes}
    if set(output_gt) != expected_gt_keys:
        raise ValueError("GT keys do not exactly match published episode IDs")
    for key, record in output_gt.items():
        actions = record["actions"]
        if actions[-1] != 0 or actions.count(0) != 1:
            raise ValueError(f"GT episode {key} must end with exactly one STOP")
        if actions.count(1) != record["forward_steps"]:
            raise ValueError(f"GT episode {key} forward_steps mismatch")
        if len(record["locations"]) != record["forward_steps"] + 1:
            raise ValueError(f"GT episode {key} locations mismatch")
    for index in range(0, len(output_episodes), len(REQUIRED_STYLES)):
        pair = output_episodes[index : index + len(REQUIRED_STYLES)]
        if len(pair) != len(REQUIRED_STYLES):
            raise ValueError("Published episode list ends with an incomplete style pair")
        if len({episode["trajectory_id"] for episode in pair}) != 1:
            raise ValueError("Adjacent concise/dense episodes do not share trajectory_id")
        pair_gt = [output_gt[str(episode["episode_id"])] for episode in pair]
        if pair_gt[1:] != pair_gt[:-1]:
            raise ValueError("Adjacent concise/dense episodes do not share identical GT")

    output = {
        "episodes": output_episodes,
        "instruction_vocab": copy.deepcopy(EMPTY_INSTRUCTION_VOCAB),
    }
    summary = {
        "dataset_trajectories": len(source_episodes),
        "paired_trajectories": len(selected_ids),
        "exported_episodes": len(output_episodes),
        "allow_subset": allow_subset,
        "is_subset": len(selected_ids) != len(source_episodes),
        "selected_source_ids_sha256": source_id_sha256(selected_ids),
        "per_style": per_style,
        "goal_radius": args.goal_radius,
        "source_gt": str(source_gt_path) if source_gt_path is not None else None,
        "source_gt_records": len(source_gt_by_id),
        "gt_records": len(output_gt),
        "image_root": str(image_root),
        "output": str(output_path),
        "gzip_output": str(gzip_output_path),
        "gt_output": str(gt_output_path),
    }

    # Complete every check before atomically replacing any formal dataset file.
    publish_json_artifacts(
        (
            (output_path, output),
            (gzip_output_path, output),
            (gt_output_path, output_gt),
        )
    )
    return summary


def main() -> None:
    args = parse_args()
    print(json.dumps(export(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
