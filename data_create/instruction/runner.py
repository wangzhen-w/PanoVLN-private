"""Concurrent instruction generation runner."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import sys
import time
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
from .prompts import SYSTEM_JSON, audit_prompt, plan_write_prompt, repair_prompt
from .qa import audit_passed, validate_instruction


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
    ]


def call_plan_write(
    client: QwenClient,
    *,
    row: Mapping[str, Any],
    evidence: Any,
    action_summary: Mapping[str, Any],
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], str]:
    prompt = plan_write_prompt(
        episode_id=row["episode_id"],
        profile=args.instruction_profile,
        action_summary=action_summary,
        trajectory_metadata=row.get("trajectory_metadata") or {},
    )
    return client.chat_json(
        system=SYSTEM_JSON,
        prompt=prompt,
        images=images_for_llm(evidence),
        temperature=args.planner_temperature,
        max_tokens=args.planner_max_tokens,
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
        trajectory_metadata=row.get("trajectory_metadata") or {},
        instruction=instruction,
    )
    return client.chat_json(
        system=SYSTEM_JSON,
        prompt=prompt,
        images=images_for_llm(evidence),
        temperature=args.review_temperature,
        max_tokens=args.review_max_tokens,
    )


def call_repair(
    client: QwenClient,
    *,
    row: Mapping[str, Any],
    evidence: Any,
    action_summary: Mapping[str, Any],
    route_plan: Mapping[str, Any],
    failed_instruction: str,
    issues: Any,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], str]:
    prompt = repair_prompt(
        episode_id=row["episode_id"],
        profile=args.instruction_profile,
        action_summary=action_summary,
        trajectory_metadata=row.get("trajectory_metadata") or {},
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
        plan: Dict[str, Any] = {}
        instruction = ""
        deterministic = {"passed": False, "failures": ["not_generated"], "warnings": []}
        audit: Dict[str, Any] = {}
        for stage_attempt in range(max(1, args.stage_retries + 1)):
            plan, raw_responses["plan_write"] = call_plan_write(
                client,
                row=row,
                evidence=evidence,
                action_summary=action_summary,
                args=args,
            )
            instruction = normalize_instruction(
                str(plan.get("final_instruction") or plan.get("instruction") or "")
            )
            deterministic = validate_instruction(
                instruction,
                profile=args.instruction_profile,
                actions=actions,
                trajectory_metadata=row.get("trajectory_metadata") or {},
                route_plan=plan,
            )
            if not deterministic["passed"]:
                repair, raw_responses[f"repair_gate_{stage_attempt}"] = call_repair(
                    client,
                    row=row,
                    evidence=evidence,
                    action_summary=action_summary,
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
                    trajectory_metadata=row.get("trajectory_metadata") or {},
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
            if not audit_passed(audit):
                corrected = normalize_instruction(str(audit.get("corrected_instruction") or ""))
                if corrected:
                    repair_issues = {"audit": audit, "deterministic": deterministic}
                    repair, raw_responses["repair_audit"] = call_repair(
                        client,
                        row=row,
                        evidence=evidence,
                        action_summary=action_summary,
                        route_plan=plan,
                        failed_instruction=instruction,
                        issues=repair_issues,
                        args=args,
                    )
                    instruction = normalize_instruction(
                        str(repair.get("final_instruction") or corrected)
                    )
                deterministic = validate_instruction(
                    instruction,
                    profile=args.instruction_profile,
                    actions=actions,
                    trajectory_metadata=row.get("trajectory_metadata") or {},
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

        passed = deterministic["passed"] and (
            not args.blind_grounding_audit or audit_passed(audit)
        )
        candidate = {
            "episode_id": row["episode_id"],
            "trajectory_id": row.get("trajectory_id"),
            "status": "success" if passed else "failed",
            "mode": args.mode,
            "source_text_blind": args.mode == "generate",
            "old_instruction": "",
            "instruction_profile": args.instruction_profile,
            "pipeline_fingerprint": pipeline_fingerprint,
            "input_fingerprint": input_fingerprint,
            "actions": actions,
            "image_key": evidence.image_key,
            "selected_frames": evidence.selected_frames,
            "start_frames": evidence.start_frames,
            "endpoint_frames": evidence.endpoint_frames,
            "contact_sheets": {
                "route": evidence.route_sheet_path,
                "start": evidence.start_sheet_path,
                "endpoint": evidence.endpoint_sheet_path,
            },
            "route_plan": plan,
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
            "pipeline_fingerprint": pipeline_fingerprint,
            "input_fingerprint": input_fingerprint,
            "actions": row.get("actions"),
            "instruction": "",
            "error": {"type": type(error).__name__, "message": str(error)},
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
        for label in ("start", "route", "endpoint"):
            sheet_path = sheets.get(label)
            if sheet_path:
                rel = Path(sheet_path).resolve().relative_to(output.parent.resolve()) if Path(sheet_path).resolve().is_relative_to(output.parent.resolve()) else Path(sheet_path).resolve()
                parts.append(f"<h3>{label}</h3><img src='{html.escape(str(rel))}'>")
        preview = {
            "route_plan": candidate.get("route_plan"),
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
