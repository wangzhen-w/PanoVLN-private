"""I/O and normalization helpers for instruction generation."""

from __future__ import annotations

import argparse
import glob
import json
import math
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
    if path.exists() and path.stat().st_size:
        with path.open("rb") as handle:
            handle.seek(-1, os.SEEK_END)
            needs_newline = handle.read(1) != b"\n"
        if needs_newline:
            # Isolate a truncated final journal record before appending. The
            # reader will ignore that final malformed line and retain all
            # complete records written afterward.
            with path.open("ab") as handle:
                handle.write(b"\n")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl_mapping(paths: Sequence[Path]) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    versions: Dict[str, tuple[int, int, int, int]] = {}
    for path_order, path in enumerate(paths):
        if not path.exists():
            continue
        file_version = path.stat().st_mtime_ns
        with path.open("r", encoding="utf-8") as handle:
            lines = handle.readlines()
            for line_number, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # Journals are resumable progress, not published data. A
                    # process may be interrupted midway through an append; skip
                    # only that corrupt record and preserve later valid work.
                    continue
                if not isinstance(row, dict) or "episode_id" not in row:
                    continue
                key = episode_key(row)
                try:
                    explicit_time = float(row["completed_at_unix"])
                    if not math.isfinite(explicit_time):
                        raise ValueError("non-finite completion timestamp")
                    record_version = int(explicit_time * 1_000_000_000)
                    has_explicit_timestamp = 1
                except (KeyError, TypeError, ValueError, OverflowError):
                    record_version = file_version
                    has_explicit_timestamp = 0
                # New-format records with their own completion timestamp must
                # always outrank legacy rows. A journal file's mtime changes on
                # every append and therefore cannot date an individual old row.
                version = (
                    has_explicit_timestamp,
                    record_version,
                    path_order,
                    line_number,
                )
                if version >= versions.get(key, (-1, -1, -1, -1)):
                    rows[key] = row
                    versions[key] = version
    return rows


def existing_candidates(work_dir: Path) -> Dict[str, Dict[str, Any]]:
    return read_jsonl_mapping(sorted(work_dir.glob("candidates_rank*.jsonl")))


def clean_row_from_candidate(row: Mapping[str, Any], candidate: Mapping[str, Any]) -> Dict[str, Any]:
    clean = {
        "episode_id": row["episode_id"],
        "instruction": normalize_instruction(str(candidate["instruction"])),
        "actions": [int(action) for action in row["actions"]],
        "instruction_profile": candidate.get("instruction_profile"),
    }
    if row.get("trajectory_id") is not None:
        clean["trajectory_id"] = row["trajectory_id"]
    if row.get("source_trajectory_id") is not None:
        clean["source_trajectory_id"] = row["source_trajectory_id"]
    return clean


def candidate_is_complete(
    candidate: Optional[Mapping[str, Any]],
    *,
    profile: str,
) -> bool:
    if not candidate:
        return False
    return (
        candidate.get("status") == "success"
        and candidate.get("instruction_profile") == profile
        and bool(str(candidate.get("instruction") or "").strip())
    )


def candidate_is_terminal_failure(
    candidate: Optional[Mapping[str, Any]],
    *,
    profile: str,
) -> bool:
    """Return whether a quality-gated failure should be dropped, not retried.

    Older journals predate the explicit marker.  Their structured QA/audit error
    identifies the same exhausted quality path, while runtime exceptions carry
    ``type/message/traceback`` instead and remain retryable.
    """

    if not candidate or candidate.get("status") != "failed":
        return False
    if candidate.get("instruction_profile") != profile:
        return False
    marker = candidate.get("terminal_failure")
    if isinstance(marker, bool):
        return marker
    error = candidate.get("error")
    return isinstance(error, Mapping) and (
        "deterministic_qa" in error or "blind_grounding_audit" in error
    )


def candidate_paths(work_dir: Path, num_workers: int) -> List[Path]:
    if num_workers <= 0:
        num_workers = 1
    return [work_dir / f"candidates_rank{index}.jsonl" for index in range(num_workers)]


def stale_outputs_for(output_path: str) -> List[str]:
    return sorted(glob.glob(f"{output_path}.partial*"))
