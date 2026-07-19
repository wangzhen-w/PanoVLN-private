"""I/O, normalization, and fingerprint helpers for instruction generation."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .actions import validate_actions


EMPTY_INSTRUCTION_VOCAB = {
    "word_list": [],
    "word2idx_dict": {},
    "stoi": {},
    "itos": [],
    "num_vocab": 0,
    "UNK_INDEX": 1,
    "PAD_INDEX": 0,
}


def episode_key(row: Mapping[str, Any]) -> str:
    return str(row["episode_id"])


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def normalize_instruction(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^final instruction\s*:\s*", "", text, flags=re.IGNORECASE)
    text = text.strip("`\"' \n\t")
    text = re.sub(r"\s+", " ", text)
    text = text.replace(" ,", ",").replace(" .", ".").replace(" ;", ";")
    text = re.sub(r"\s+([.!?])", r"\1", text)
    text = re.sub(r"([.!?])(?=[A-Za-z])", r"\1 ", text)
    if text and text[-1] not in ".!?":
        text += "."
    if text:
        text = text[0].upper() + text[1:]
    return text


def read_jsonl(path: str, *, mode: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            if "episode_id" not in row or "actions" not in row:
                raise ValueError(f"{path}:{line_number} missing episode_id/actions")
            row["actions"] = validate_actions(row["actions"], f"{path}:{line_number}.actions")
            if mode == "generate":
                row["instruction"] = ""
            else:
                row["instruction"] = str(row.get("instruction") or "")
            key = episode_key(row)
            if key in seen:
                raise ValueError(f"{path}:{line_number} duplicate episode_id={key}")
            seen.add(key)
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no usable rows")
    return rows


def atomic_write_jsonl(path: str, rows: Iterable[Mapping[str, Any]]) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(fd)
    count = 0
    tmp = Path(tmp_name)
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, output)
        return count
    finally:
        tmp.unlink(missing_ok=True)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl_mapping(paths: Sequence[Path]) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or "episode_id" not in row:
                    continue
                rows[episode_key(row)] = row
    return rows


def existing_candidates(work_dir: Path) -> Dict[str, Dict[str, Any]]:
    return read_jsonl_mapping(sorted(work_dir.glob("candidates_rank*.jsonl")))


def source_row_payload(row: Mapping[str, Any], *, mode: str) -> Dict[str, Any]:
    return {
        "episode_id": str(row.get("episode_id", "")),
        "trajectory_id": row.get("trajectory_id"),
        "actions": [int(action) for action in row.get("actions", [])],
        "trajectory_metadata": row.get("trajectory_metadata") or {},
        "instruction": "" if mode == "generate" else str(row.get("instruction") or ""),
    }


def hash_file(path: Path, digest: "hashlib._Hash", mode: str) -> None:
    if mode == "content":
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    elif mode == "metadata":
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8"))
    else:
        raise ValueError(f"Unknown fingerprint mode: {mode}")


def pipeline_source_files_sha256() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
    return digest.hexdigest()


def build_pipeline_fingerprint(args: argparse.Namespace) -> str:
    payload = {
        "schema_version": "panovln-instruction-v1",
        "source_files_sha256": pipeline_source_files_sha256(),
        "mode": args.mode,
        "instruction_profile": args.instruction_profile,
        "model": args.model,
        "max_waypoints": args.max_waypoints,
        "route_evidence_mode": getattr(args, "route_evidence_mode", "auto"),
        "segmented_min_actions": getattr(args, "segmented_min_actions", 80),
        "segment_max_waypoints": getattr(args, "segment_max_waypoints", 0),
        "segment_rows": getattr(args, "segment_rows", 5),
        "segment_overlap": getattr(args, "segment_overlap", 1),
        "segment_fact_max_tokens": getattr(args, "segment_fact_max_tokens", 280),
        "start_window_frames": args.start_window_frames,
        "endpoint_window_frames": args.endpoint_window_frames,
        "tile_width": args.tile_width,
        "tile_height": args.tile_height,
        "jpeg_quality": args.jpeg_quality,
        "use_action_heading": args.use_action_heading,
        "temperature": args.temperature,
        "planner_temperature": args.planner_temperature,
        "review_temperature": args.review_temperature,
        "candidate_count": getattr(args, "candidate_count", 1),
        "candidate_temperature": getattr(args, "candidate_temperature", 0.4),
        "prompt_family": "segmented-evidence-labeled-final-views-endpoint-aware-boundaries-side-neutral-final-face-v5",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def clean_row_from_candidate(row: Mapping[str, Any], candidate: Mapping[str, Any]) -> Dict[str, Any]:
    clean = {
        "episode_id": row["episode_id"],
        "instruction": normalize_instruction(str(candidate["instruction"])),
        "actions": [int(action) for action in row["actions"]],
        "instruction_profile": candidate.get("instruction_profile"),
        "pipeline_fingerprint": candidate.get("pipeline_fingerprint"),
        "input_fingerprint": candidate.get("input_fingerprint"),
    }
    if row.get("trajectory_id") is not None:
        clean["trajectory_id"] = row["trajectory_id"]
    if row.get("source_trajectory_id") is not None:
        clean["source_trajectory_id"] = row["source_trajectory_id"]
    return clean


def candidate_is_current(
    candidate: Optional[Mapping[str, Any]],
    *,
    input_fingerprint: str,
    pipeline_fingerprint: str,
    profile: str,
) -> bool:
    if not candidate:
        return False
    return (
        candidate.get("status") == "success"
        and candidate.get("input_fingerprint") == input_fingerprint
        and candidate.get("pipeline_fingerprint") == pipeline_fingerprint
        and candidate.get("instruction_profile") == profile
        and bool(str(candidate.get("instruction") or "").strip())
    )


def candidate_paths(work_dir: Path, num_workers: int) -> List[Path]:
    if num_workers <= 0:
        num_workers = 1
    return [work_dir / f"candidates_rank{index}.jsonl" for index in range(num_workers)]


def stale_outputs_for(output_path: str) -> List[str]:
    return sorted(glob.glob(f"{output_path}.partial*"))
