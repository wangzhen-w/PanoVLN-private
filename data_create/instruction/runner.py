"""Concurrent instruction generation runner."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from tqdm.auto import tqdm

from .actions import compact_action_summary
from .evidence import build_evidence, evidence_frame_fingerprint
from .io_utils import (
    append_jsonl,
    atomic_write_jsonl,
    build_pipeline_fingerprint,
    candidate_is_current,
    candidate_paths,
    clean_row_from_candidate,
    episode_key,
    existing_candidates,
    normalize_instruction,
    read_jsonl,
    source_row_payload,
)
from .llm import QwenClient
from .prompts import (
    SYSTEM_JSON,
    audit_prompt,
    endpoint_fact_audit_prompt,
    candidate_judge_prompt,
    endpoint_fact_prompt,
    plan_write_prompt,
    repair_prompt,
    segment_fact_prompt,
    segmented_merge_prompt,
)
from .qa import audit_contract_complete, audit_passed, validate_instruction
from .trajectory_metadata import enrich_trajectory_metadata


def select_rows(rows: Sequence[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    selected = list(rows)
    if args.episode_ids:
        wanted = {item.strip() for item in args.episode_ids.split(",") if item.strip()}
        selected = [row for row in selected if episode_key(row) in wanted]
        missing = wanted - {episode_key(row) for row in selected}
        if missing:
            raise ValueError(f"Requested episode IDs are absent from input: {sorted(missing)}")
    if args.max_episodes and args.max_episodes > 0 and len(selected) > args.max_episodes:
        if args.sample_mode == "random":
            rng = random.Random(args.seed)
            selected = rng.sample(selected, args.max_episodes)
            selected.sort(key=lambda row: str(row["episode_id"]))
        elif args.sample_mode == "stratified":
            selected = stratified_sample(selected, args.max_episodes, args.seed)
        else:
            selected = selected[: args.max_episodes]
    return selected


def stratified_sample(rows: Sequence[Dict[str, Any]], count: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    buckets: Dict[str, List[Dict[str, Any]]] = {"short": [], "medium": [], "long": [], "xlong": []}
    for row in rows:
        n = len(row["actions"])
        if n < 45:
            buckets["short"].append(row)
        elif n < 80:
            buckets["medium"].append(row)
        elif n < 130:
            buckets["long"].append(row)
        else:
            buckets["xlong"].append(row)
    result: List[Dict[str, Any]] = []
    while len(result) < count and any(buckets.values()):
        for key in ("short", "medium", "long", "xlong"):
            if buckets[key] and len(result) < count:
                result.append(buckets[key].pop(rng.randrange(len(buckets[key]))))
    result.sort(key=lambda row: str(row["episode_id"]))
    return result


def row_input_fingerprint(row: Mapping[str, Any], args: argparse.Namespace) -> str:
    evidence_hash = evidence_frame_fingerprint(
        row,
        image_root=args.image_root,
        max_waypoints=args.max_waypoints,
        start_window_frames=args.start_window_frames,
        endpoint_window_frames=args.endpoint_window_frames,
        route_evidence_mode=args.route_evidence_mode,
        segmented_min_actions=args.segmented_min_actions,
        segment_max_waypoints=args.segment_max_waypoints,
        mode=args.evidence_fingerprint_mode,
    )
    payload = source_row_payload(row, mode=args.mode)
    payload["evidence_frames_sha256"] = evidence_hash
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def images_for_llm(evidence: Any) -> List[Tuple[str, bytes]]:
    return [
        ("START evidence", evidence.start_sheet),
        ("ROUTE evidence", evidence.route_sheet),
        ("ENDPOINT evidence", evidence.endpoint_sheet),
        ("FINAL STOP evidence", evidence.final_sheet),
    ]


def images_for_endpoint(evidence: Any) -> List[Tuple[str, bytes]]:
    tiles = list(getattr(evidence, "final_view_tiles", []) or [])
    if tiles:
        return [("ENDPOINT approach evidence", evidence.endpoint_sheet), *tiles]
    return [("FINAL STOP evidence", evidence.final_sheet)]


def images_for_independent_review(evidence: Any) -> List[Tuple[str, bytes]]:
    """Return raw visual evidence without model-derived plans or facts."""

    images: List[Tuple[str, bytes]] = [
        ("START evidence", evidence.start_sheet),
        ("ROUTE overview evidence", evidence.route_sheet),
    ]
    images.extend(
        (f"ROUTE segment {segment.segment_id} evidence", segment.sheet)
        for segment in getattr(evidence, "segment_sheets", []) or []
    )
    images.extend(
        [("ENDPOINT approach evidence", evidence.endpoint_sheet)]
    )
    final_tiles = list(getattr(evidence, "final_view_tiles", []) or [])
    if final_tiles:
        images.extend(final_tiles)
    else:
        images.append(("FINAL STOP evidence", evidence.final_sheet))
    return images


def publishable_candidate_indices(
    judge: Mapping[str, Any], candidate_options: Sequence[Mapping[str, Any]]
) -> List[int]:
    valid = {int(option["index"]) for option in candidate_options}
    assessments = judge.get("candidate_assessments") or []
    publishable: List[int] = []
    if isinstance(assessments, Sequence) and not isinstance(assessments, (str, bytes)):
        for assessment in assessments:
            if not isinstance(assessment, Mapping):
                continue
            raw_index = assessment.get("index")
            if not isinstance(raw_index, int) or isinstance(raw_index, bool):
                continue
            index = raw_index
            if (
                index in valid
                and assessment.get("route_sequence_matches") is True
                and assessment.get("endpoint_matches") is True
                and isinstance(assessment.get("critical_problems"), list)
                and not assessment.get("critical_problems")
            ):
                publishable.append(index)
    return sorted(set(publishable))


def judge_contract_complete(
    judge: Mapping[str, Any], candidate_options: Sequence[Mapping[str, Any]]
) -> bool:
    valid = {int(option["index"]) for option in candidate_options}
    observed_route = judge.get("observed_route_sequence")
    endpoint = judge.get("observed_endpoint")
    if not isinstance(observed_route, list) or not observed_route or not all(
        isinstance(item, str) and item.strip() for item in observed_route
    ):
        return False
    if not isinstance(endpoint, Mapping):
        return False
    if not isinstance(endpoint.get("stop_location"), str) or not endpoint["stop_location"].strip():
        return False
    if not isinstance(endpoint.get("forward_anchor"), str) or not endpoint["forward_anchor"].strip():
        return False
    relation_value = endpoint.get("forward_anchor_relation")
    if not isinstance(relation_value, str):
        return False
    relation = relation_value.strip().lower()
    if relation not in {
        "reached_at_final_camera",
        "remains_ahead_after_final",
        "uncertain",
    }:
        return False
    assessments = judge.get("candidate_assessments")
    if not isinstance(assessments, list) or len(assessments) != len(valid):
        return False
    assessed: set[int] = set()
    for item in assessments:
        if not isinstance(item, Mapping):
            return False
        raw_index = item.get("index")
        if not isinstance(raw_index, int) or isinstance(raw_index, bool):
            return False
        index = raw_index
        if index not in valid or index in assessed:
            return False
        instruction_route = item.get("instruction_route_sequence")
        if not isinstance(instruction_route, list) or not instruction_route or not all(
            isinstance(route_item, str) and route_item.strip()
            for route_item in instruction_route
        ):
            return False
        if not isinstance(item.get("route_sequence_matches"), bool):
            return False
        if not isinstance(item.get("endpoint_matches"), bool):
            return False
        critical_problems = item.get("critical_problems")
        if not isinstance(critical_problems, list) or not all(
            isinstance(problem, str) for problem in critical_problems
        ):
            return False
        assessed.add(index)
    if assessed != valid or not isinstance(judge.get("needs_repair"), bool):
        return False
    repair_instruction = judge.get("repair_instruction")
    if not isinstance(repair_instruction, str):
        return False
    publishable = publishable_candidate_indices(judge, candidate_options)
    if not publishable:
        return bool(judge.get("needs_repair")) and bool(repair_instruction.strip())
    if judge.get("needs_repair") or repair_instruction.strip():
        return False
    selected = judge.get("selected_index")
    if not isinstance(selected, int) or isinstance(selected, bool):
        return False
    return selected in publishable


def selected_candidate_index(judge: Mapping[str, Any], candidate_options: Sequence[Mapping[str, Any]]) -> int:
    publishable = publishable_candidate_indices(judge, candidate_options)
    if not publishable:
        return -1
    selected = judge.get("selected_index")
    if not isinstance(selected, int) or isinstance(selected, bool):
        return -1
    return selected if selected in publishable else -1


def endpoint_relation_disagreement_diagnostic(
    review: Mapping[str, Any], endpoint_facts: Mapping[str, Any]
) -> bool:
    """Flag a high-confidence relation disagreement for later corpus review.

    This signal is deliberately diagnostic rather than a publication gate.
    Free-language anchor descriptions do not carry simulator entity IDs, so a
    lexical matcher cannot safely prove that two mentions denote the same
    physical object. The blind visual audit remains the endpoint authority.
    """

    fact_relation = str(
        (endpoint_facts or {}).get("forward_anchor_relation") or ""
    ).strip().lower()
    observed_endpoint = (review or {}).get("observed_endpoint") or {}
    if not isinstance(observed_endpoint, Mapping):
        return False
    review_relation = str(
        observed_endpoint.get("forward_anchor_relation") or ""
    ).strip().lower()
    fact_anchor = str((endpoint_facts or {}).get("forward_view_anchor") or "")
    review_anchor = str(observed_endpoint.get("forward_anchor") or "")

    def anchor_identity_terms(text: str) -> set[str]:
        tokens = re.findall(r"[a-z]+", text.lower())
        ignored = {
            "a", "an", "the", "and", "ahead", "forward", "straight",
            "visible", "view", "anchor", "area", "space", "object", "is",
            "located", "leading", "facing", "toward", "towards", "into",
            "at", "in", "on", "by", "near", "with", "of", "to",
        }
        tokens = [token for token in tokens if token not in ignored]
        synonyms = {
            "doors": "door",
            "doorways": "doorway",
            "entryway": "doorway",
            "entryways": "doorway",
            "entrance": "doorway",
            "entrances": "doorway",
            "opening": "doorway",
            "openings": "doorway",
            "threshold": "doorway",
            "thresholds": "doorway",
            "stairs": "stair",
            "staircase": "stair",
            "staircases": "stair",
            "steps": "stair",
            "windows": "window",
            "corridor": "hallway",
            "corridors": "hallway",
            "hall": "hallway",
            "halls": "hallway",
            "sofas": "sofa",
            "couch": "sofa",
            "couches": "sofa",
            "landings": "landing",
            "railings": "railing",
            "banister": "railing",
            "banisters": "railing",
        }
        return {synonyms.get(token, token) for token in tokens}

    fact_identity = anchor_identity_terms(fact_anchor)
    review_identity = anchor_identity_terms(review_anchor)
    # Exact multi-token identity is useful as a diagnostic. Generic one-word
    # anchors such as "door" or "stairs" remain intentionally uncertain.
    same_anchor = len(fact_identity) >= 2 and fact_identity == review_identity
    definite = {"reached_at_final_camera", "remains_ahead_after_final"}
    return (
        same_anchor
        and fact_relation in definite
        and review_relation in definite
        and fact_relation != review_relation
    )


def row_trajectory_metadata(row: Mapping[str, Any]) -> Dict[str, Any]:
    return enrich_trajectory_metadata(row.get("trajectory_metadata") or {})


def normalize_endpoint_facts(facts: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize structure without rewriting model-produced language.

    Endpoint facts are semantic evidence for downstream agents, not canonical
    phrases. Visual verification decides whether a detail is supported; code
    must not remove valid colors/materials or replace stair descriptions with a
    preferred wording.
    """

    normalized = dict(facts or {})
    if not normalized.get("stop_location_anchor") and normalized.get("safe_stop_phrase"):
        # Read old cached responses defensively; current prompts no longer emit
        # a publishable phrase from the endpoint stage.
        normalized["stop_location_anchor"] = normalized["safe_stop_phrase"]
    normalized.pop("safe_stop_phrase", None)
    return normalized


def endpoint_facts_complete(facts: Mapping[str, Any]) -> bool:
    """Return whether endpoint facts satisfy the minimum downstream contract."""

    location_type = str((facts or {}).get("final_location_type") or "").strip().lower()
    stop_location = str((facts or {}).get("stop_location_anchor") or "").strip()
    forward_anchor = str((facts or {}).get("forward_view_anchor") or "").strip()
    forward_relation = str((facts or {}).get("forward_anchor_relation") or "").strip().lower()
    complete = bool(
        location_type
        in {
            "inside_room",
            "threshold_or_doorway",
            "stair_landing",
            "hallway_or_corridor",
            "near_object",
            "outdoor_or_balcony",
            "uncertain",
        }
        and stop_location
        and forward_anchor
        and forward_relation
        in {"reached_at_final_camera", "remains_ahead_after_final", "uncertain"}
    )
    return complete


def normalize_segment_facts(
    facts: Mapping[str, Any],
    *,
    segment_id: int,
    segment_count: int,
    endpoint_facts: Mapping[str, Any],
) -> Dict[str, Any]:
    """Apply simulator-known segment identity instead of trusting model output."""

    normalized = dict(facts or {})
    is_final = int(segment_id) == int(segment_count)
    normalized["segment_id"] = int(segment_id)
    normalized["is_final_segment"] = is_final
    if is_final:
        if not str(normalized.get("stop_anchor_if_final") or "").strip():
            normalized["stop_anchor_if_final"] = str(
                (endpoint_facts or {}).get("stop_location_anchor") or ""
            ).strip()
    else:
        normalized["stop_anchor_if_final"] = ""
    return normalized


def use_segmented_route(evidence: Any) -> bool:
    # Evidence construction is the single authority for this decision.  This
    # prevents the renderer and runner from silently applying different route
    # complexity criteria.
    return bool(getattr(evidence, "segment_sheets", None))


def call_endpoint_facts(
    client: QwenClient,
    *,
    row: Mapping[str, Any],
    evidence: Any,
    action_summary: Mapping[str, Any],
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], str]:
    prompt = endpoint_fact_prompt(
        episode_id=row["episode_id"],
        profile=args.instruction_profile,
        action_summary=action_summary,
        trajectory_metadata=row_trajectory_metadata(row),
    )
    facts, raw = client.chat_json(
        system=SYSTEM_JSON,
        prompt=prompt,
        images=images_for_endpoint(evidence),
        temperature=args.review_temperature,
        max_tokens=args.review_max_tokens,
    )
    return normalize_endpoint_facts(facts), raw


def call_endpoint_fact_audit(
    client: QwenClient,
    *,
    row: Mapping[str, Any],
    evidence: Any,
    action_summary: Mapping[str, Any],
    endpoint_facts: Mapping[str, Any],
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    prompt = endpoint_fact_audit_prompt(
        episode_id=row["episode_id"],
        profile=args.instruction_profile,
        action_summary=action_summary,
        trajectory_metadata=row_trajectory_metadata(row),
        endpoint_facts=endpoint_facts,
    )
    audit, raw = client.chat_json(
        system=SYSTEM_JSON,
        prompt=prompt,
        images=images_for_endpoint(evidence),
        temperature=args.review_temperature,
        max_tokens=args.review_max_tokens,
    )
    audited_facts = audit.get("endpoint_facts") if isinstance(audit.get("endpoint_facts"), dict) else endpoint_facts
    return normalize_endpoint_facts(audited_facts), raw, audit


def verify_endpoint_facts(
    client: QwenClient,
    *,
    row: Mapping[str, Any],
    evidence: Any,
    action_summary: Mapping[str, Any],
    endpoint_facts: Mapping[str, Any],
    args: argparse.Namespace,
    raw_responses: Dict[str, str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], bool]:
    """Audit endpoint facts and verify a correction before propagation.

    A failed audit may propose corrected facts, but those facts are still model
    output. They receive one fresh verification pass and are never consumed by
    planning while the latest audit is failed or structurally incomplete. If
    verification still fails, derived endpoint facts are discarded and later
    agents must ground the endpoint directly from raw visual evidence.
    """

    current = normalize_endpoint_facts(endpoint_facts)
    history: List[Dict[str, Any]] = []
    for attempt in range(1, 3):
        current, raw, audit = call_endpoint_fact_audit(
            client,
            row=row,
            evidence=evidence,
            action_summary=action_summary,
            endpoint_facts=current,
            args=args,
        )
        raw_responses[f"endpoint_fact_audit_{attempt}"] = raw
        complete = endpoint_facts_complete(current)
        record = dict(audit or {})
        record["facts_complete"] = complete
        record["verification_attempt"] = attempt
        history.append(record)
        if bool(audit.get("passed")) and complete:
            return current, history, True
    return {}, history, False


def call_plan_write(
    client: QwenClient,
    *,
    row: Mapping[str, Any],
    evidence: Any,
    action_summary: Mapping[str, Any],
    endpoint_facts: Mapping[str, Any],
    args: argparse.Namespace,
    temperature: Optional[float] = None,
) -> Tuple[Dict[str, Any], str]:
    prompt = plan_write_prompt(
        episode_id=row["episode_id"],
        profile=args.instruction_profile,
        action_summary=action_summary,
        trajectory_metadata=row_trajectory_metadata(row),
        endpoint_facts=endpoint_facts,
    )
    return client.chat_json(
        system=SYSTEM_JSON,
        prompt=prompt,
        images=images_for_llm(evidence),
        temperature=args.planner_temperature if temperature is None else temperature,
        max_tokens=args.planner_max_tokens,
    )


def call_segment_facts(
    client: QwenClient,
    *,
    row: Mapping[str, Any],
    segment: Any,
    segment_count: int,
    action_summary: Mapping[str, Any],
    endpoint_facts: Mapping[str, Any],
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], str]:
    prompt = segment_fact_prompt(
        episode_id=row["episode_id"],
        profile=args.instruction_profile,
        action_summary=action_summary,
        segment_action_summary=segment.action_summary,
        trajectory_metadata=row_trajectory_metadata(row),
        endpoint_facts=endpoint_facts,
        segment_id=segment.segment_id,
        segment_count=segment_count,
        segment_frames=segment.frames,
    )
    facts, raw = client.chat_json(
        system=SYSTEM_JSON,
        prompt=prompt,
        images=[(f"SEGMENT {segment.segment_id} evidence", segment.sheet)],
        temperature=args.planner_temperature,
        max_tokens=args.segment_fact_max_tokens,
    )
    return normalize_segment_facts(
        facts,
        segment_id=segment.segment_id,
        segment_count=segment_count,
        endpoint_facts=endpoint_facts,
    ), raw


def call_segmented_merge(
    client: QwenClient,
    *,
    row: Mapping[str, Any],
    evidence: Any,
    action_summary: Mapping[str, Any],
    endpoint_facts: Mapping[str, Any],
    segment_facts: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    temperature: Optional[float] = None,
) -> Tuple[Dict[str, Any], str]:
    prompt = segmented_merge_prompt(
        episode_id=row["episode_id"],
        profile=args.instruction_profile,
        action_summary=action_summary,
        trajectory_metadata=row_trajectory_metadata(row),
        endpoint_facts=endpoint_facts,
        segment_facts=segment_facts,
    )
    return client.chat_json(
        system=SYSTEM_JSON,
        prompt=prompt,
        images=images_for_llm(evidence),
        temperature=args.planner_temperature if temperature is None else temperature,
        max_tokens=args.planner_max_tokens,
    )


def call_candidate_judge(
    client: QwenClient,
    *,
    row: Mapping[str, Any],
    evidence: Any,
    action_summary: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], str]:
    prompt = candidate_judge_prompt(
        episode_id=row["episode_id"],
        profile=args.instruction_profile,
        action_summary=action_summary,
        trajectory_metadata=row_trajectory_metadata(row),
        candidates=candidates,
    )
    return client.chat_json(
        system=SYSTEM_JSON,
        prompt=prompt,
        images=images_for_independent_review(evidence),
        temperature=args.review_temperature,
        max_tokens=args.review_max_tokens,
    )


def call_audit(
    client: QwenClient,
    *,
    row: Mapping[str, Any],
    evidence: Any,
    action_summary: Mapping[str, Any],
    instruction: str,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], str]:
    prompt = audit_prompt(
        episode_id=row["episode_id"],
        profile=args.instruction_profile,
        action_summary=action_summary,
        trajectory_metadata=row_trajectory_metadata(row),
        instruction=instruction,
    )
    return client.chat_json(
        system=SYSTEM_JSON,
        prompt=prompt,
        images=images_for_independent_review(evidence),
        temperature=args.review_temperature,
        max_tokens=args.review_max_tokens,
    )


def call_repair(
    client: QwenClient,
    *,
    row: Mapping[str, Any],
    evidence: Any,
    action_summary: Mapping[str, Any],
    endpoint_facts: Mapping[str, Any],
    route_plan: Mapping[str, Any],
    failed_instruction: str,
    issues: Any,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], str]:
    prompt = repair_prompt(
        episode_id=row["episode_id"],
        profile=args.instruction_profile,
        action_summary=action_summary,
        trajectory_metadata=row_trajectory_metadata(row),
        endpoint_facts=endpoint_facts,
        route_plan=route_plan,
        failed_instruction=failed_instruction,
        issues=issues,
    )
    return client.chat_json(
        system=SYSTEM_JSON,
        prompt=prompt,
        images=images_for_llm(evidence),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )


def generate_one(
    row: Dict[str, Any],
    *,
    args: argparse.Namespace,
    pipeline_fingerprint: str,
    input_fingerprint: str,
) -> Dict[str, Any]:
    client = QwenClient(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        timeout=args.request_timeout,
        retries=args.retries,
        disable_thinking=args.disable_thinking,
    )
    started = time.time()
    raw_responses: Dict[str, str] = {}
    try:
        evidence = build_evidence(row, args)
        actions = [int(action) for action in row["actions"]]
        action_summary = compact_action_summary(actions, len(evidence.frame_paths))
        trajectory_metadata = row_trajectory_metadata(row)
        endpoint_facts, raw_responses["endpoint_facts"] = call_endpoint_facts(
            client,
            row=row,
            evidence=evidence,
            action_summary=action_summary,
            args=args,
        )
        endpoint_facts, endpoint_fact_audit_history, endpoint_facts_verified = verify_endpoint_facts(
            client,
            row=row,
            evidence=evidence,
            action_summary=action_summary,
            endpoint_facts=endpoint_facts,
            args=args,
            raw_responses=raw_responses,
        )
        endpoint_fact_audit = endpoint_fact_audit_history[-1]
        plan: Dict[str, Any] = {}
        segment_facts: List[Dict[str, Any]] = []
        instruction = ""
        deterministic = {"passed": False, "failures": ["not_generated"], "warnings": []}
        audit: Dict[str, Any] = {}
        for stage_attempt in range(max(1, args.stage_retries + 1)):
            candidate_options: List[Dict[str, Any]] = []
            if use_segmented_route(evidence):
                segment_facts = []
                for segment in evidence.segment_sheets:
                    facts, raw = call_segment_facts(
                        client,
                        row=row,
                        segment=segment,
                        segment_count=len(evidence.segment_sheets),
                        action_summary=action_summary,
                        endpoint_facts=endpoint_facts,
                        args=args,
                    )
                    raw_responses[f"segment_{segment.segment_id:02d}_facts"] = raw
                    segment_facts.append(facts)
                for candidate_index in range(max(1, int(getattr(args, "candidate_count", 1)))):
                    candidate_temperature = (
                        args.planner_temperature
                        if candidate_index == 0
                        else float(getattr(args, "candidate_temperature", args.temperature))
                    )
                    candidate_plan, raw = call_segmented_merge(
                        client,
                        row=row,
                        evidence=evidence,
                        action_summary=action_summary,
                        endpoint_facts=endpoint_facts,
                        segment_facts=segment_facts,
                        args=args,
                        temperature=candidate_temperature,
                    )
                    raw_responses[f"segmented_merge_{candidate_index + 1}"] = raw
                    candidate_plan["segment_facts"] = segment_facts
                    candidate_plan["endpoint_facts"] = endpoint_facts
                    candidate_instruction = normalize_instruction(
                        str(candidate_plan.get("final_instruction") or candidate_plan.get("instruction") or "")
                    )
                    candidate_options.append(
                        {
                            "index": candidate_index,
                            "instruction": candidate_instruction,
                            "plan": candidate_plan,
                            "deterministic_qa": validate_instruction(
                                candidate_instruction,
                                profile=args.instruction_profile,
                                actions=actions,
                                trajectory_metadata=trajectory_metadata,
                                route_plan=candidate_plan,
                            ),
                        }
                    )
            else:
                for candidate_index in range(max(1, int(getattr(args, "candidate_count", 1)))):
                    candidate_temperature = (
                        args.planner_temperature
                        if candidate_index == 0
                        else float(getattr(args, "candidate_temperature", args.temperature))
                    )
                    candidate_plan, raw = call_plan_write(
                        client,
                        row=row,
                        evidence=evidence,
                        action_summary=action_summary,
                        endpoint_facts=endpoint_facts,
                        args=args,
                        temperature=candidate_temperature,
                    )
                    raw_responses[f"plan_write_{candidate_index + 1}"] = raw
                    candidate_plan["endpoint_facts"] = endpoint_facts
                    candidate_instruction = normalize_instruction(
                        str(candidate_plan.get("final_instruction") or candidate_plan.get("instruction") or "")
                    )
                    candidate_options.append(
                        {
                            "index": candidate_index,
                            "instruction": candidate_instruction,
                            "plan": candidate_plan,
                            "deterministic_qa": validate_instruction(
                                candidate_instruction,
                                profile=args.instruction_profile,
                                actions=actions,
                                trajectory_metadata=trajectory_metadata,
                                route_plan=candidate_plan,
                            ),
                        }
                    )
            judge: Dict[str, Any] = {}
            judge_contract_failed = False
            judge_has_publishable_candidate = True
            selected_option = candidate_options[0]
            if len(candidate_options) > 1:
                judge_candidates = [
                    {
                        "index": option["index"],
                        "instruction": option["instruction"],
                    }
                    for option in candidate_options
                ]
                random.Random(
                    f"{args.seed}:{row['episode_id']}:{stage_attempt}"
                ).shuffle(judge_candidates)
                judge, raw_responses["candidate_judge"] = call_candidate_judge(
                    client,
                    row=row,
                    evidence=evidence,
                    action_summary=action_summary,
                    candidates=judge_candidates,
                    args=args,
                )
                if not judge_contract_complete(judge, candidate_options):
                    judge, raw_responses["candidate_judge_contract_retry"] = call_candidate_judge(
                        client,
                        row=row,
                        evidence=evidence,
                        action_summary=action_summary,
                        candidates=judge_candidates,
                        args=args,
                    )
                if not judge_contract_complete(judge, candidate_options):
                    # A malformed judge cannot authorize publication or supply
                    # a trusted repair. Leave a truthy diagnostic; the empty
                    # instruction below retries the stage and eventually fails
                    # safely without entering the generic repair path.
                    judge = {
                        "contract_complete": False,
                        "contract_error": "candidate_judge_schema_incomplete_after_retry",
                    }
                    judge_contract_failed = True
                else:
                    judge["contract_complete"] = True
                judge_relation_disagreement = endpoint_relation_disagreement_diagnostic(
                    judge, endpoint_facts
                )
                judge["endpoint_relation_disagreement_diagnostic"] = (
                    judge_relation_disagreement
                )
                selected_index = selected_candidate_index(judge, candidate_options)
                judge_has_publishable_candidate = selected_index >= 0
                selected_option = next(
                    (option for option in candidate_options if int(option["index"]) == selected_index),
                    candidate_options[0],
                )
            plan = dict(selected_option["plan"])
            plan["endpoint_facts"] = endpoint_facts
            if judge:
                plan["candidate_judge"] = judge
            instruction = normalize_instruction(
                str(
                    judge.get("repair_instruction")
                    if judge.get("needs_repair") and judge.get("repair_instruction")
                    else ""
                    if judge and not judge_has_publishable_candidate
                    else selected_option["instruction"]
                )
            )
            plan["final_instruction"] = instruction
            deterministic = validate_instruction(
                instruction,
                profile=args.instruction_profile,
                actions=actions,
                trajectory_metadata=trajectory_metadata,
                route_plan=plan,
            )
            if not deterministic["passed"] and not judge_contract_failed:
                repair, raw_responses[f"repair_gate_{stage_attempt}"] = call_repair(
                    client,
                    row=row,
                    evidence=evidence,
                    action_summary=action_summary,
                    endpoint_facts=endpoint_facts,
                    route_plan=plan,
                    failed_instruction=instruction,
                    issues=deterministic,
                    args=args,
                )
                instruction = normalize_instruction(
                    str(repair.get("final_instruction") or repair.get("instruction") or "")
                )
                deterministic = validate_instruction(
                    instruction,
                    profile=args.instruction_profile,
                    actions=actions,
                    trajectory_metadata=trajectory_metadata,
                    route_plan=plan,
                )
            if deterministic["passed"]:
                break

        if deterministic["passed"] and args.blind_grounding_audit:
            audit, raw_responses["audit_1"] = call_audit(
                client,
                row=row,
                evidence=evidence,
                action_summary=action_summary,
                instruction=instruction,
                args=args,
            )
            if not audit_contract_complete(audit):
                audit, raw_responses["audit_contract_retry"] = call_audit(
                    client,
                    row=row,
                    evidence=evidence,
                    action_summary=action_summary,
                    instruction=instruction,
                    args=args,
                )
            audit_relation_disagreement = endpoint_relation_disagreement_diagnostic(
                audit, endpoint_facts
            )
            audit["endpoint_relation_disagreement_diagnostic"] = (
                audit_relation_disagreement
            )
            if not audit_passed(audit):
                corrected = normalize_instruction(str(audit.get("corrected_instruction") or ""))
                if corrected:
                    # The blind auditor saw the raw evidence without any
                    # model-derived endpoint facts or route plan.  Keep that
                    # independence by validating its correction directly
                    # instead of passing it through the correlated planner.
                    instruction = corrected
                    plan["final_instruction"] = instruction
                    deterministic = validate_instruction(
                        instruction,
                        profile=args.instruction_profile,
                        actions=actions,
                        trajectory_metadata=trajectory_metadata,
                        route_plan=plan,
                    )
                    if deterministic["passed"]:
                        audit, raw_responses["audit_2"] = call_audit(
                            client,
                            row=row,
                            evidence=evidence,
                            action_summary=action_summary,
                            instruction=instruction,
                            args=args,
                        )
                        audit["endpoint_relation_disagreement_diagnostic"] = (
                            endpoint_relation_disagreement_diagnostic(audit, endpoint_facts)
                        )

        passed = deterministic["passed"] and (
            not args.blind_grounding_audit
            or audit_passed(audit)
        )
        candidate = {
            "episode_id": row["episode_id"],
            "trajectory_id": row.get("trajectory_id"),
            "status": "success" if passed else "failed",
            "mode": args.mode,
            "source_text_blind": args.mode == "generate",
            "old_instruction": "",
            "instruction_profile": args.instruction_profile,
            "completed_at_unix": time.time(),
            "pipeline_fingerprint": pipeline_fingerprint,
            "input_fingerprint": input_fingerprint,
            "actions": actions,
            "image_key": evidence.image_key,
            "selected_frames": evidence.selected_frames,
            "start_frames": evidence.start_frames,
            "endpoint_frames": evidence.endpoint_frames,
            "segment_frames": [
                {
                    "segment_id": segment.segment_id,
                    "frames": segment.frames,
                    "action_summary": segment.action_summary,
                }
                for segment in evidence.segment_sheets
            ],
            "contact_sheets": {
                "route": evidence.route_sheet_path,
                "start": evidence.start_sheet_path,
                "endpoint": evidence.endpoint_sheet_path,
                "final": evidence.final_sheet_path,
                "segments": [
                    segment.sheet_path for segment in evidence.segment_sheets
                ],
            },
            "route_plan": plan,
            "endpoint_facts": endpoint_facts,
            "endpoint_facts_verified": endpoint_facts_verified,
            "endpoint_fact_audit": endpoint_fact_audit,
            "endpoint_fact_audit_history": endpoint_fact_audit_history,
            "segment_facts": segment_facts,
            "candidate_options": [
                {
                    "index": option["index"],
                    "instruction": option["instruction"],
                    "deterministic_qa": option["deterministic_qa"],
                    "endpoint_fact_used": (option["plan"] or {}).get("endpoint_fact_used"),
                }
                for option in candidate_options
            ],
            "instruction": instruction,
            "deterministic_qa": deterministic,
            "blind_grounding_audit": audit,
            "warnings": list(deterministic.get("warnings") or []),
            "raw_responses": raw_responses if args.keep_raw_responses else {},
            "elapsed_seconds": round(time.time() - started, 3),
        }
        if not passed:
            candidate["error"] = {
                "deterministic_qa": deterministic,
                "blind_grounding_audit": audit,
            }
        return candidate
    except Exception as error:  # noqa: BLE001 - record per-episode failure
        return {
            "episode_id": row.get("episode_id"),
            "trajectory_id": row.get("trajectory_id"),
            "status": "failed",
            "mode": args.mode,
            "source_text_blind": args.mode == "generate",
            "old_instruction": "",
            "instruction_profile": args.instruction_profile,
            "completed_at_unix": time.time(),
            "pipeline_fingerprint": pipeline_fingerprint,
            "input_fingerprint": input_fingerprint,
            "actions": row.get("actions"),
            "instruction": "",
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
            "elapsed_seconds": round(time.time() - started, 3),
        }


def write_gallery(rows: Sequence[Dict[str, Any]], candidates: Mapping[str, Dict[str, Any]], path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;line-height:1.35}"
        ".item{border:1px solid #ccc;border-radius:8px;padding:12px;margin:16px 0}"
        "img{max-width:100%;border:1px solid #ddd;margin:6px 0}"
        "pre{white-space:pre-wrap;background:#f6f8fa;padding:8px}</style>",
        "<title>PanoVLN instruction review</title></head><body>",
        "<h1>PanoVLN instruction review</h1>",
    ]
    for row in rows:
        key = episode_key(row)
        candidate = candidates.get(key) or {}
        parts.append("<div class='item'>")
        parts.append(f"<h2>Episode {html.escape(key)} — {html.escape(str(candidate.get('status')))}</h2>")
        parts.append(f"<p>{html.escape(str(candidate.get('instruction') or ''))}</p>")
        sheets = candidate.get("contact_sheets") or {}
        for label in ("start", "route", "endpoint", "final"):
            sheet_path = sheets.get(label)
            if sheet_path:
                rel = Path(sheet_path).resolve().relative_to(output.parent.resolve()) if Path(sheet_path).resolve().is_relative_to(output.parent.resolve()) else Path(sheet_path).resolve()
                parts.append(f"<h3>{label}</h3><img src='{html.escape(str(rel))}'>")
        for index, sheet_path in enumerate(sheets.get("segments") or [], start=1):
            if sheet_path:
                rel = Path(sheet_path).resolve().relative_to(output.parent.resolve()) if Path(sheet_path).resolve().is_relative_to(output.parent.resolve()) else Path(sheet_path).resolve()
                parts.append(f"<h3>segment {index}</h3><img src='{html.escape(str(rel))}'>")
        preview = {
            "route_plan": candidate.get("route_plan"),
            "endpoint_facts": candidate.get("endpoint_facts"),
            "segment_facts": candidate.get("segment_facts"),
            "deterministic_qa": candidate.get("deterministic_qa"),
            "audit": candidate.get("blind_grounding_audit"),
            "error": candidate.get("error"),
        }
        parts.append(f"<pre>{html.escape(json.dumps(preview, ensure_ascii=False, indent=2))}</pre>")
        parts.append("</div>")
    parts.append("</body></html>")
    output.write_text("\n".join(parts), encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    if args.write_gallery:
        args.save_contact_sheets = True
    if args.api_preflight:
        QwenClient(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            timeout=args.request_timeout,
            retries=1,
            disable_thinking=args.disable_thinking,
        ).preflight()
    rows = select_rows(read_jsonl(args.input_jsonl, mode=args.mode), args)
    pipeline_fingerprint = build_pipeline_fingerprint(args)
    old_candidates = existing_candidates(work_dir) if args.resume else {}
    fingerprints: Dict[str, str] = {}
    pending: List[Tuple[int, Dict[str, Any], str]] = []
    current_candidates: Dict[str, Dict[str, Any]] = {}
    for index, row in enumerate(rows):
        fingerprint = row_input_fingerprint(row, args)
        fingerprints[episode_key(row)] = fingerprint
        existing = old_candidates.get(episode_key(row))
        if candidate_is_current(
            existing,
            input_fingerprint=fingerprint,
            pipeline_fingerprint=pipeline_fingerprint,
            profile=args.instruction_profile,
        ):
            current_candidates[episode_key(row)] = existing  # type: ignore[assignment]
        else:
            pending.append((index, row, fingerprint))

    candidate_files = candidate_paths(work_dir, max(1, args.num_workers))
    failed_files = [work_dir / f"failed_rank{index}.jsonl" for index in range(max(1, args.num_workers))]
    progress = tqdm(total=len(pending), desc=f"instruction {args.instruction_profile}", unit="ep")
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as executor:
            futures = {
                executor.submit(
                    generate_one,
                    row,
                    args=args,
                    pipeline_fingerprint=pipeline_fingerprint,
                    input_fingerprint=fingerprint,
                ): (index, row, fingerprint)
                for index, row, fingerprint in pending
            }
            for future in as_completed(futures):
                index, row, _ = futures[future]
                candidate = future.result()
                rank = index % max(1, args.num_workers)
                append_jsonl(candidate_files[rank], candidate)
                if candidate.get("status") != "success":
                    append_jsonl(failed_files[rank], candidate)
                if candidate_is_current(
                    candidate,
                    input_fingerprint=fingerprints[episode_key(row)],
                    pipeline_fingerprint=pipeline_fingerprint,
                    profile=args.instruction_profile,
                ):
                    current_candidates[episode_key(row)] = candidate
                progress.update(1)
    finally:
        progress.close()

    clean_rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for row in rows:
        key = episode_key(row)
        candidate = current_candidates.get(key)
        if candidate_is_current(
            candidate,
            input_fingerprint=fingerprints[key],
            pipeline_fingerprint=pipeline_fingerprint,
            profile=args.instruction_profile,
        ):
            clean_rows.append(clean_row_from_candidate(row, candidate or {}))
        else:
            missing.append(key)

    complete = not missing
    if missing and not args.allow_incomplete:
        partial = f"{args.output_jsonl}.partial"
        if clean_rows:
            atomic_write_jsonl(partial, clean_rows)
        summary = {
            "status": "incomplete",
            "selected_rows": len(rows),
            "success_rows": len(clean_rows),
            "missing_rows": len(missing),
            "missing_preview": missing[:30],
            "output_partial": partial if clean_rows else None,
            "pipeline_fingerprint": pipeline_fingerprint,
            "source_text_blind": args.mode == "generate",
        }
        (work_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.write_gallery:
            write_gallery(rows, current_candidates, args.gallery_path or str(work_dir / "gallery.html"))
        raise RuntimeError(f"Instruction generation incomplete: {len(missing)} missing/failed rows")

    if missing and args.drop_failed:
        clean_rows = [row for row in clean_rows if episode_key(row) not in set(missing)]
    written = atomic_write_jsonl(args.output_jsonl, clean_rows)
    summary = {
        "status": "complete" if complete else "partial_allowed",
        "selected_rows": len(rows),
        "written_rows": written,
        "missing_rows": len(missing),
        "missing_preview": missing[:30],
        "output_jsonl": args.output_jsonl,
        "work_dir": str(work_dir),
        "pipeline_fingerprint": pipeline_fingerprint,
        "source_text_blind": args.mode == "generate",
        "ignored_source_text_count": sum(1 for row in rows if args.mode == "generate"),
        "source_instruction_visible_to_models": args.mode != "generate",
    }
    (work_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.write_gallery:
        write_gallery(rows, current_candidates, args.gallery_path or str(work_dir / "gallery.html"))
    return summary
