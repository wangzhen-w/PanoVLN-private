#!/usr/bin/env python3
"""Reproducible text/action audit for VLN instruction JSONL files.

The audit deliberately separates observable text/action integrity from visual
grounding. It produces a risk-enriched manifest for blind panorama review; none
of the text heuristics is presented as proof that an instruction is grounded.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?", re.IGNORECASE)
SENTENCE_RE = re.compile(r"[.!?]+")
STOP_RE = re.compile(
    r"(?:\b(?:stop|stopping|wait|halt|finish|stand|remain)\b|"
    r"\b(?:route|path|navigation)\s+ends?\b|"
    r"\b(?:this|that)\s+(?:is|marks)\s+(?:the\s+)?end\b)",
    re.IGNORECASE,
)
LANDMARK_RE = re.compile(
    r"\b(?:bed|table|chair|sofa|couch|stairs?|staircase|door(?:way)?|window|"
    r"sink|counter|island|fireplace|cabinet|refrigerator|fridge|toilet|shower|"
    r"bathtub|mirror|rug|plant|desk|piano|bookshel(?:f|ves)|wardrobe|vanity)\b",
    re.IGNORECASE,
)
ROOM_RE = re.compile(
    r"\b(?:bedroom|bathroom|kitchen|living room|dining room|hallway|corridor|"
    r"foyer|entryway|closet|office|lobby|room|area|patio|balcony|porch)\b",
    re.IGNORECASE,
)
ARTIFACT_PATTERNS = (
    re.compile(r"\bimage\b", re.IGNORECASE),
    re.compile(r"\bpanorama\b", re.IGNORECASE),
    re.compile(r"contact\s+sheet", re.IGNORECASE),
    re.compile(r"\brow\s+\d+\b", re.IGNORECASE),
    re.compile(r"\bstep\s+\d+\b", re.IGNORECASE),
    re.compile(r"\bframe[_\s-]?\d+\b", re.IGNORECASE),
    re.compile(r"\baction\s+sequence\b", re.IGNORECASE),
)

PHRASES = {
    "starts_turn_left_or_right": r"^turn\s+(?:left|right)\b",
    "turn_left_and_walk": r"^turn\s+left\s+and\s+walk\b",
    "turn_right_and_walk": r"^turn\s+right\s+and\s+walk\b",
    "walk_forward": r"\bwalk forward\b",
    "continue_straight": r"\bcontinue straight\b",
    "proceed_forward": r"\bproceed forward\b",
    "pass_or_past": r"\b(?:pass|passing|past)\b",
    "stop_near_the": r"\bstop(?:ping)? near the\b",
    "turn_around": r"\b(?:turn around|turn back|make a u-turn)\b",
    "numeric_degrees": r"\b\d{2,3}\s*degrees?\b",
    "boundary_specific": (
        r"\b(?:in the doorway|on the landing|top of the stairs|bottom of the stairs|"
        r"second step|inside the room|on the stairs)\b"
    ),
}


def parse_args() -> argparse.Namespace:
    root = "/workspace/data1/dataset/PanoVLN/sub_dataset"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        default=f"{root}/scalevln_qwen36_27b_panovln.jsonl",
    )
    parser.add_argument("--source", default=f"{root}/scalevln.jsonl")
    parser.add_argument("--r2r", default=f"{root}/r2r.jsonl")
    parser.add_argument("--rxr", default=f"{root}/rxr.jsonl")
    parser.add_argument(
        "--image-root",
        default="/workspace/data1/dataset/PanoVLN/images/scalevln",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--review-sample-size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-words", type=int, default=12)
    parser.add_argument("--max-words", type=int, default=120)
    parser.add_argument("--distinct-token-budget", type=int, default=250_000)
    return parser.parse_args()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    seen_ids = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            episode_id = row.get("episode_id") if isinstance(row, dict) else None
            if episode_id is None or isinstance(episode_id, bool):
                raise ValueError(f"{path}:{line_number}: missing valid episode_id")
            episode_key = str(episode_id)
            if episode_key in seen_ids:
                raise ValueError(
                    f"{path}:{line_number}: duplicate episode_id {episode_id!r}"
                )
            seen_ids.add(episode_key)
            rows.append(row)
    return rows


def tokens(text: str) -> List[str]:
    return TOKEN_RE.findall(str(text).lower())


def normalize(text: str) -> str:
    return " ".join(tokens(text))


def quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * probability))
    return float(ordered[index])


def action_runs(actions: Sequence[int]) -> List[Tuple[int, int, int]]:
    runs: List[Tuple[int, int, int]] = []
    active: Optional[int] = None
    start = 0
    last = -1
    for index, raw_action in enumerate(actions):
        action = int(raw_action)
        if action == 0:
            break
        last = index
        if active is None:
            active = action
            start = index
        elif action != active:
            runs.append((active, start, index - 1))
            active = action
            start = index
    if active is not None:
        runs.append((active, start, last))
    return runs


def pearson_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    left_scale = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_scale = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if not left_scale or not right_scale:
        return 0.0
    return numerator / (left_scale * right_scale)


def old_late_turn_heuristic_trigger(actions: Sequence[int]) -> bool:
    runs = action_runs(actions)
    if not runs:
        return False
    last_non_stop = max(end for action, _start, end in runs if action != 0)
    cutoff = int((last_non_stop + 1) * 0.55)
    return any(
        action in {2, 3}
        and end >= cutoff
        and (end - start + 1) * 15 >= 45
        for action, start, end in runs
    )


def fixed_budget_distinct(
    rows: Sequence[Dict[str, Any]],
    n: int,
    token_budget: int,
    seed: int,
) -> float:
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    stream: List[str] = []
    for index in order:
        stream.extend(tokens(rows[index].get("instruction", "")))
        if len(stream) >= token_budget:
            break
    stream = stream[:token_budget]
    if len(stream) < n:
        return 0.0
    grams = {tuple(stream[index : index + n]) for index in range(len(stream) - n + 1)}
    return len(grams) / (len(stream) - n + 1)


def corpus_metrics(
    rows: Sequence[Dict[str, Any]],
    token_budget: int,
    seed: int,
) -> Dict[str, Any]:
    word_lengths = [len(tokens(row.get("instruction", ""))) for row in rows]
    sentence_lengths = [
        len([part for part in SENTENCE_RE.split(str(row.get("instruction", ""))) if part.strip()])
        for row in rows
    ]
    action_lengths = [len(row.get("actions", [])) for row in rows]
    forward_counts = [list(map(int, row.get("actions", []))).count(1) for row in rows]
    run_counts = [len(action_runs(row.get("actions", []))) for row in rows]
    late_turn_trigger_count = sum(
        old_late_turn_heuristic_trigger(row.get("actions", [])) for row in rows
    )
    normalized = [normalize(row.get("instruction", "")) for row in rows]
    vocabulary = Counter(token for row in rows for token in tokens(row.get("instruction", "")))
    prefix_counts = Counter(
        " ".join(tokens(row.get("instruction", ""))[:4]) for row in rows
    )
    total_tokens = sum(vocabulary.values())
    entropy = 0.0
    if total_tokens:
        for count in vocabulary.values():
            probability = count / total_tokens
            entropy -= probability * math.log2(probability)
    phrase_rates = {
        name: round(
            100.0
            * sum(bool(re.search(pattern, str(row.get("instruction", "")), re.IGNORECASE)) for row in rows)
            / max(1, len(rows)),
            4,
        )
        for name, pattern in PHRASES.items()
    }
    return {
        "rows": len(rows),
        "unique_normalized_instructions": len(set(normalized)),
        "unique_rate": round(len(set(normalized)) / max(1, len(rows)), 6),
        "words": distribution(word_lengths),
        "sentences": distribution(sentence_lengths),
        "actions": distribution(action_lengths),
        "estimated_forward_meters": distribution([count * 0.25 for count in forward_counts]),
        "words_per_forward_action": round(sum(word_lengths) / max(1, sum(forward_counts)), 6),
        "word_count_forward_count_pearson": round(
            pearson_correlation(word_lengths, forward_counts), 6
        ),
        "action_run_count": distribution(run_counts),
        "action_runs_over_14_count": sum(count > 14 for count in run_counts),
        "action_runs_over_14_rate_percent": round(
            100.0 * sum(count > 14 for count in run_counts) / max(1, len(rows)), 4
        ),
        "old_late_turn_heuristic_trigger_count": late_turn_trigger_count,
        "old_late_turn_heuristic_trigger_rate_percent": round(
            100.0 * late_turn_trigger_count / max(1, len(rows)), 4
        ),
        "vocabulary_size": len(vocabulary),
        "vocabulary_entropy_bits": round(entropy, 6),
        "hapax_types": sum(count == 1 for count in vocabulary.values()),
        "distinct_2_fixed_budget": round(
            fixed_budget_distinct(rows, 2, token_budget, seed), 6
        ),
        "distinct_3_fixed_budget": round(
            fixed_budget_distinct(rows, 3, token_budget, seed), 6
        ),
        "top_four_word_prefixes": prefix_counts.most_common(12),
        "phrase_rates_percent": phrase_rates,
        "landmark_mention_rate_percent": round(
            100.0
            * sum(bool(LANDMARK_RE.search(str(row.get("instruction", "")))) for row in rows)
            / max(1, len(rows)),
            4,
        ),
        "room_mention_rate_percent": round(
            100.0
            * sum(bool(ROOM_RE.search(str(row.get("instruction", "")))) for row in rows)
            / max(1, len(rows)),
            4,
        ),
    }


def distribution(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {key: 0.0 for key in ("mean", "p05", "p25", "p50", "p75", "p95", "max")}
    return {
        "mean": round(float(statistics.mean(values)), 6),
        "p05": round(quantile(values, 0.05), 6),
        "p25": round(quantile(values, 0.25), 6),
        "p50": round(quantile(values, 0.50), 6),
        "p75": round(quantile(values, 0.75), 6),
        "p95": round(quantile(values, 0.95), 6),
        "max": round(float(max(values)), 6),
    }


def validate_schema(row: Dict[str, Any]) -> List[str]:
    flags = []
    for key in ("episode_id", "instruction", "actions"):
        if key not in row:
            flags.append(f"schema_missing_{key}")
    if not isinstance(row.get("instruction"), str):
        flags.append("schema_instruction_not_string")
    actions = row.get("actions")
    if not isinstance(actions, list) or not actions:
        flags.append("schema_actions_invalid")
        return flags
    try:
        action_values = [int(action) for action in actions]
    except (TypeError, ValueError):
        flags.append("schema_actions_not_integer")
        return flags
    if set(action_values) - {0, 1, 2, 3}:
        flags.append("schema_unknown_action")
    if action_values[-1:] != [0] or action_values.count(0) != 1:
        flags.append("schema_stop_not_unique_terminal")
    return flags


def instruction_flags(
    row: Dict[str, Any],
    old_instruction: str,
    min_words: int,
    max_words: int,
) -> List[str]:
    flags = validate_schema(row)
    instruction = str(row.get("instruction", ""))
    lower = normalize(instruction)
    word_count = len(tokens(instruction))
    if word_count < min_words:
        flags.append("too_short")
    if word_count > max_words:
        flags.append("too_long")
    if normalize(old_instruction) == lower:
        flags.append("identical_to_source")
    if not STOP_RE.search(instruction):
        flags.append("missing_explicit_stop")
    if any(pattern.search(instruction) for pattern in ARTIFACT_PATTERNS):
        flags.append("data_artifact")
    if len(re.findall(r"\bwalk forward\b", lower)) >= 3:
        flags.append("repeated_walk_forward")
    if re.search(r"\bstop there\.?$", lower):
        flags.append("vague_stop_there")
    if re.search(r"\b(?:and|or|to|toward|towards|with|where|near|by)\s*[.!?]", instruction, re.I):
        flags.append("dangling_connector")
    if re.search(PHRASES["numeric_degrees"], instruction, re.I):
        flags.append("numeric_turn_angle")

    actions = [int(action) for action in row.get("actions", []) if str(action).lstrip("-").isdigit()]
    runs = action_runs(actions)
    if runs:
        first_action, start, end = runs[0]
        first_sentence = SENTENCE_RE.split(instruction.lower(), maxsplit=1)[0]
        degrees = (end - start + 1) * 15
        if re.match(r"^(?:please\s+)?(?:turn around|turn back|make a u-turn)\b", first_sentence):
            if first_action == 1 or degrees < 120:
                flags.append("unsupported_initial_turn_around")
        if first_action in {2, 3} and degrees >= 30:
            expected = "left" if first_action == 2 else "right"
            opposite = "right" if expected == "left" else "left"
            if re.search(rf"\bturn\s+{opposite}\b", first_sentence):
                flags.append("opposite_initial_turn")
    return sorted(set(flags))


def compare_integrity(
    source_rows: Sequence[Dict[str, Any]],
    candidate_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    source = {str(row.get("episode_id")): row for row in source_rows}
    candidate = {str(row.get("episode_id")): row for row in candidate_rows}
    source_ids = set(source)
    candidate_ids = set(candidate)
    source_order = [str(row["episode_id"]) for row in source_rows]
    candidate_order = [str(row["episode_id"]) for row in candidate_rows]
    first_order_mismatch = next(
        (
            index
            for index, (source_id, candidate_id) in enumerate(
                zip(source_order, candidate_order)
            )
            if source_id != candidate_id
        ),
        None,
    )
    if first_order_mismatch is None and len(source_order) != len(candidate_order):
        first_order_mismatch = min(len(source_order), len(candidate_order))
    mismatches = [
        episode_id
        for episode_id in sorted(source_ids & candidate_ids)
        if source[episode_id].get("actions") != candidate[episode_id].get("actions")
    ]
    missing_ids = sorted(source_ids - candidate_ids)
    missing_rows = [source[episode_id] for episode_id in missing_ids]
    full_action_lengths = [len(row.get("actions", [])) for row in source_rows]
    missing_action_lengths = [len(row.get("actions", [])) for row in missing_rows]
    turn_around_rate = lambda rows: (
        sum("turn around" in str(row.get("instruction", "")).lower() for row in rows)
        / max(1, len(rows))
    )
    return {
        "source_rows": len(source_rows),
        "candidate_rows": len(candidate_rows),
        "missing_count": len(missing_ids),
        "extra_count": len(candidate_ids - source_ids),
        "action_mismatch_count": len(mismatches),
        "order_match": source_order == candidate_order,
        "first_order_mismatch_index": first_order_mismatch,
        "missing_ids": missing_ids,
        "extra_ids": sorted(candidate_ids - source_ids),
        "action_mismatch_ids": mismatches,
        "full_action_length_mean": round(statistics.mean(full_action_lengths), 6),
        "missing_action_length_mean": (
            round(statistics.mean(missing_action_lengths), 6)
            if missing_action_lengths
            else None
        ),
        "full_turn_around_rate": round(turn_around_rate(source_rows), 6),
        "missing_turn_around_rate": (
            round(turn_around_rate(missing_rows), 6) if missing_rows else None
        ),
    }


def select_review_manifest(
    annotated: Sequence[Dict[str, Any]],
    sample_size: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if sample_size <= 0:
        return []
    rng = random.Random(seed)
    selected: Dict[str, Dict[str, Any]] = {}

    by_flag: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in annotated:
        for flag in row["audit_flags"]:
            by_flag[flag].append(row)
    # Round-robin across failure modes so one common regex cannot dominate the
    # visual audit sample.
    flag_names = sorted(by_flag)
    rng.shuffle(flag_names)
    target_risk = min(sample_size // 2, len(annotated))
    while len(selected) < target_risk and flag_names:
        progressed = False
        for flag in list(flag_names):
            candidates = by_flag[flag]
            rng.shuffle(candidates)
            row = next(
                (item for item in candidates if str(item["episode_id"]) not in selected),
                None,
            )
            if row is None:
                flag_names.remove(flag)
                continue
            selected[str(row["episode_id"])] = row
            progressed = True
            if len(selected) >= target_risk:
                break
        if not progressed:
            break

    # Fill with action-length deciles to estimate ordinary, not only risk-enriched,
    # quality. The manifest records the sampling stratum for downstream weighting.
    ordered = sorted(annotated, key=lambda row: row["action_count"])
    buckets = [ordered[index::10] for index in range(10)]
    bucket_index = 0
    while len(selected) < min(sample_size, len(annotated)):
        bucket = buckets[bucket_index % len(buckets)]
        bucket_index += 1
        available = [row for row in bucket if str(row["episode_id"]) not in selected]
        if not available:
            if bucket_index > len(buckets) * 3:
                break
            continue
        row = rng.choice(available)
        selected[str(row["episode_id"])] = row
    return sorted(selected.values(), key=lambda row: int(row["episode_id"]))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def markdown_report(summary: Dict[str, Any]) -> str:
    corpora = summary["corpora"]
    candidate = corpora["candidate"]
    source = corpora["source"]
    r2r = corpora["r2r"]
    rxr = corpora["rxr"]
    integrity = summary["integrity"]
    flags = summary["flag_counts"]
    lines = [
        "# ScaleVLN instruction quality audit",
        "",
        "This report separates text/action checks from visual grounding. Text flags are",
        "risk indicators, not evidence that a panorama claim is true or false.",
        "",
        "## Executive findings",
        "",
        f"- Candidate rows: **{candidate['rows']:,}**; source rows: **{source['rows']:,}**; "
        f"missing: **{integrity['missing_count']:,}**; action mismatches: "
        f"**{integrity['action_mismatch_count']:,}**.",
        f"- Mean words: candidate **{candidate['words']['mean']:.2f}**, R2R "
        f"**{r2r['words']['mean']:.2f}**, RxR **{rxr['words']['mean']:.2f}**.",
        f"- Words per forward action: candidate **{candidate['words_per_forward_action']:.2f}**, "
        f"R2R **{r2r['words_per_forward_action']:.2f}**, RxR "
        f"**{rxr['words_per_forward_action']:.2f}**.",
        f"- Candidate word-count/forward-count Pearson correlation: "
        f"**{candidate['word_count_forward_count_pearson']:.3f}**.",
        f"- Source routes with >14 action runs: "
        f"**{source['action_runs_over_14_count']:,}/{source['rows']:,} "
        f"({source['action_runs_over_14_rate_percent']:.2f}%)**; source routes that trigger "
        f"the old >=45-degree late-turn heuristic: "
        f"**{source['old_late_turn_heuristic_trigger_count']:,}/{source['rows']:,} "
        f"({source['old_late_turn_heuristic_trigger_rate_percent']:.2f}%)**. "
        "The latter is a trigger rate, not an instruction-error rate.",
        f"- Four-word prefix `turn left/right ...` family starts "
        f"**{candidate['phrase_rates_percent']['starts_turn_left_or_right']:.2f}%** of candidate "
        f"instructions versus **{r2r['phrase_rates_percent']['starts_turn_left_or_right']:.2f}%** "
        f"for R2R and **{rxr['phrase_rates_percent']['starts_turn_left_or_right']:.2f}%** for RxR.",
        f"- Fixed-budget distinct-2: candidate **{candidate['distinct_2_fixed_budget']:.4f}**, "
        f"R2R **{r2r['distinct_2_fixed_budget']:.4f}**, RxR "
        f"**{rxr['distinct_2_fixed_budget']:.4f}**.",
        f"- Fixed-budget distinct-3: candidate **{candidate['distinct_3_fixed_budget']:.4f}**, "
        f"R2R **{r2r['distinct_3_fixed_budget']:.4f}**, RxR "
        f"**{rxr['distinct_3_fixed_budget']:.4f}**.",
        "",
        "The rewrite is therefore much more verbose than original ScaleVLN and has RxR-English-like",
        "surface words/forward-action density under this conversion. That statistic does not prove",
        "grounding or functional quality, and the text remains more templated than either human",
        "benchmark. High landmark/room mention rates require a separate visual precision",
        "audit; they must not be treated as automatic quality gains.",
        "",
        "## Deterministic risk flags",
        "",
        "| Flag | Count | Rate |",
        "|---|---:|---:|",
    ]
    for flag, count in sorted(flags.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {flag} | {count:,} | {100.0 * count / max(1, candidate['rows']):.3f}% |")
    lines.extend(
        [
            "",
            "## Failure-selection bias",
            "",
            f"Missing rows have mean action length **{integrity['missing_action_length_mean']}** "
            f"versus **{integrity['full_action_length_mean']}** overall. Their original "
            f"`turn around` rate is **{integrity['missing_turn_around_rate']}** versus "
            f"**{integrity['full_turn_around_rate']}** overall. The production source and "
            "rewrite inputs must already contain the same episode ID set.",
            "",
            "## Outputs",
            "",
            "- `summary.json`: all corpus and integrity statistics.",
            "- `flagged.jsonl`: every candidate with one or more deterministic risk flags.",
            "- `review_manifest.jsonl`: risk-balanced plus action-length-stratified panorama review sample.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else (
        Path("/workspace/data1/dataset/PanoVLN/audits")
        / Path(args.candidate).stem
    )
    source_rows = read_jsonl(args.source)
    candidate_rows = read_jsonl(args.candidate)
    r2r_rows = read_jsonl(args.r2r)
    rxr_rows = read_jsonl(args.rxr)
    source = {str(row.get("episode_id")): row for row in source_rows}

    annotated = []
    flag_counts: Counter[str] = Counter()
    for row in candidate_rows:
        episode_id = str(row.get("episode_id"))
        old = str(source.get(episode_id, {}).get("instruction", ""))
        flags = instruction_flags(row, old, args.min_words, args.max_words)
        flag_counts.update(flags)
        annotated.append(
            {
                "episode_id": row.get("episode_id"),
                "instruction": row.get("instruction", ""),
                "source_instruction": old,
                "actions": row.get("actions", []),
                "action_count": len(row.get("actions", [])),
                "forward_count": list(map(int, row.get("actions", []))).count(1),
                "estimated_forward_meters": 0.25
                * list(map(int, row.get("actions", []))).count(1),
                "audit_flags": flags,
                "image_dir": str(Path(args.image_root) / episode_id),
            }
        )

    summary = {
        "candidate_path": args.candidate,
        "source_path": args.source,
        "r2r_path": args.r2r,
        "rxr_path": args.rxr,
        "seed": args.seed,
        "distinct_token_budget": args.distinct_token_budget,
        "corpora": {
            "candidate": corpus_metrics(
                candidate_rows, args.distinct_token_budget, args.seed
            ),
            "source": corpus_metrics(source_rows, args.distinct_token_budget, args.seed),
            "r2r": corpus_metrics(r2r_rows, args.distinct_token_budget, args.seed),
            "rxr": corpus_metrics(rxr_rows, args.distinct_token_budget, args.seed),
        },
        "integrity": compare_integrity(source_rows, candidate_rows),
        "flag_counts": dict(flag_counts),
    }
    manifest = select_review_manifest(
        annotated,
        sample_size=args.review_sample_size,
        seed=args.seed,
    )
    summary["review_manifest_rows"] = len(manifest)

    write_json(output_dir / "summary.json", summary)
    write_jsonl(
        output_dir / "flagged.jsonl",
        (row for row in annotated if row["audit_flags"]),
    )
    write_jsonl(output_dir / "review_manifest.jsonl", manifest)
    (output_dir / "report.md").write_text(
        markdown_report(summary) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
