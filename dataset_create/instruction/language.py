"""Segment authoring and conservative assembly with lossless clause provenance."""

from __future__ import annotations

import copy
import json
import re

from dataset_create.instruction.client import media_item, text_item, video_items
from dataset_create.instruction.prompts import AUTHOR_SYSTEM, POLISH_SYSTEM
from dataset_create.instruction.segmentation import camera_motion


MARKER_LANGUAGE = re.compile(
    r"\b(arrows?|highlight(?:ed|ing)?|compass|video|frame\s+\d+|camera|S\d+|d\d+|GT|badges?|markers?|marked|"
    r"(?:gold|yellow|red|orange|marked|drawn) (?:line|route|path)|"
    r"(?:label|candidate) [A-Z]\b)\b", re.IGNORECASE,
)


class LanguageContractError(ValueError):
    """Repeated malformed model output, not an infrastructure failure."""


def bind_clause_spans(local):
    """Normalize metadata casing/punctuation only; never alter navigation words."""
    text = local.get("text")
    if not isinstance(text, str):
        return local
    for item in [*(local.get("decisions") or []), *([local["stop"]] if isinstance(local.get("stop"), dict) else [])]:
        clause = item.get("clause")
        if not isinstance(clause, str) or not clause or clause in text:
            continue
        words = clause.strip().rstrip(".?!")
        pattern = r"\s+".join(re.escape(word) for word in words.split())
        matches = list(re.finditer(pattern, text, re.IGNORECASE)) if pattern else []
        if len(matches) == 1:
            item["clause"] = matches[0].group()
    return local


def validate_local(local, segment):
    text = local.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Missing local navigation text")
    if text != text.strip() or "\n" in text:
        raise ValueError("Navigation text must be one trimmed paragraph")
    if MARKER_LANGUAGE.search(text):
        raise ValueError("Instruction depends on visual annotations")
    if not isinstance(local.get("uncertain"), bool):
        raise ValueError("uncertain must be a boolean")
    if local.get("issue_type") not in {"none", "language", "visual", "segmentation"}:
        raise ValueError("Invalid issue_type")
    if not isinstance(local.get("issues"), list):
        raise ValueError("issues must be a list")
    decisions = local.get("decisions")
    if not isinstance(decisions, list) or [d.get("decision_id") for d in decisions] != segment.decision_ids:
        raise ValueError("Local decision IDs must exactly match assigned decisions in order")
    previous_offset = -1
    for decision in decisions:
        clause = decision.get("clause")
        if not isinstance(clause, str) or not clause or clause not in text:
            raise ValueError("Decision clause must be an exact substring of text")
        offset = text.index(clause)
        if offset < previous_offset:
            raise ValueError("Decision clauses reverse route order")
        previous_offset = offset
        evidence = decision.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError("Each choice requires pre-choice visual evidence")
        for cue in evidence:
            if (cue.get("visible_before_choice") is not True or not cue.get("claim") or
                    not isinstance(cue.get("view"), str) or not cue["view"].strip()):
                raise ValueError("Invalid pre-choice cue evidence")
    stop = local.get("stop")
    if segment.terminal:
        if (not isinstance(stop, dict) or not isinstance(stop.get("clause"), str) or
                not stop["clause"] or stop["clause"] not in text or not stop.get("evidence")):
            raise ValueError("Terminal segment must contain an exact grounded stop clause")
        if not re.search(r"\b(stop|halt|wait)\b", stop["clause"], re.IGNORECASE):
            raise ValueError("Stop clause must explicitly tell the navigator to stop")
    elif stop is not None or re.search(r"\b(stop|halt|wait)\b", text, re.IGNORECASE):
        raise ValueError("Nonterminal segment must not request stopping")


def author_content(segment, evidence, previous, feedback, media_mode):
    core = evidence["core_frame_indices"]
    info = {"segment_id": segment.segment_id, "kinds": segment.kinds,
            "assigned_decision_ids": segment.decision_ids, "terminal": segment.terminal,
            "core_video_frame_range_inclusive": [min(core), max(core)],
            "measured_camera_motion": camera_motion(evidence["cameras"]),
            "previous_segment_text_for_naming_only": previous,
            "local_repair_feedback": feedback or None}
    content = [text_item(json.dumps(info, ensure_ascii=False))]
    content += video_items(evidence["marked_video"], evidence["marked_frames"], media_mode,
                           "Current local route, in chronological order. Thin gold ground line is GT. "
                           "Describe core frames only. The last held pose is STOP only if terminal=true.")
    for decision in evidence["decisions"]:
        if decision["state_index"] in evidence["frame_state_indices"]:
            video_index = evidence["frame_state_indices"].index(decision["state_index"])
            timing = f"at video frame {video_index}, {video_index/evidence['fps']:.2f} seconds"
        else:
            timing = "from the earlier approach, before the core clip starts"
        content.extend([text_item(
                        f"Decision {decision['decision_id']}: pre-choice compass {timing}. "
                        "The ground line shows the route through the entrance. Use the video to locate the maneuver and its starting heading."),
                        media_item(decision["marked_compass"])])
    if segment.terminal:
        actual = evidence["stop"]["actual"]
        content.extend([text_item("Clean compass at the actual stop, at the end of this route:"),
                        media_item(actual["image"])])
    return content


def generate_local(client, segment, evidence, previous="", feedback=None, attempt=0):
    content = author_content(segment, evidence, previous, feedback, client.settings["media_mode"])
    keys = []
    for schema_attempt in range(3):
        local, key = client.complete("author", AUTHOR_SYSTEM, content, attempt=attempt*10+schema_attempt)
        keys.append(key)
        try:
            local = bind_clause_spans(local)
            validate_local(local, segment)
            return {**local, "request_keys": keys, "attempt": attempt}
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            client.discard_response(key)
            if schema_attempt == 2:
                raise LanguageContractError(f"Invalid author response after schema repairs: {error}") from error
            content += [text_item("Your previous JSON failed this contract check: " + str(error) +
                                  ". Produce a fresh complete JSON object obeying the schema. " +
                                  "Previous response: " + json.dumps(local, ensure_ascii=False))]
    raise AssertionError("Unreachable")


def assemble(segments, locals_by_id):
    """Lightly join already coherent local prose without rewriting route facts.

All surviving characters originate in local text. Adjacent exact duplicate
ordinary sentences can share provenance; choices and stops are never deleted.
This makes it impossible for a polishing model to change an ordinal or direction.
"""
    clauses = []
    for segment in segments:
        local = locals_by_id[segment.segment_id]
        validate_local(local, segment)
        # Keep a protected multi-sentence choice together by using segment-level
        # clauses for any decision/arrival segment.
        parts = [local["text"]] if segment.decision_ids or segment.terminal else re.split(r"(?<=[.!?])\s+", local["text"])
        for part in parts:
            clause = {"text": part, "segment_ids": [segment.segment_id],
                      "decision_ids": list(segment.decision_ids), "stop": segment.terminal}
            if (clauses and not clause["decision_ids"] and not clause["stop"] and
                    not clauses[-1]["decision_ids"] and not clauses[-1]["stop"] and
                    clauses[-1]["text"].casefold() == part.casefold()):
                if segment.segment_id not in clauses[-1]["segment_ids"]:
                    clauses[-1]["segment_ids"].append(segment.segment_id)
                continue
            clauses.append(clause)
    text = " ".join(c["text"] for c in clauses)
    offset = 0
    for i, clause in enumerate(clauses):
        clause.update({"clause_id": f"c{i:03d}", "char_start": offset, "char_end": offset + len(clause["text"])})
        offset = clause["char_end"] + 1
    protected = []
    for segment in segments:
        local = locals_by_id[segment.segment_id]
        owner = next(c for c in clauses if segment.segment_id in c["segment_ids"])
        for item in [*local["decisions"], *([local["stop"]] if segment.terminal else [])]:
            # For protected segments the whole segment is retained verbatim.
            start = owner["char_start"] + owner["text"].index(item["clause"])
            protected.append({"segment_id": segment.segment_id,
                              "decision_id": item.get("decision_id"), "stop": "decision_id" not in item,
                              "text": item["clause"], "char_start": start, "char_end": start + len(item["clause"])})
    result = {"instruction_text": text, "clauses": clauses, "protected": protected}
    validate_assembly(result, segments, locals_by_id)
    return result


def validate_assembly(final, segments, locals_by_id):
    text = final["instruction_text"]
    expected = [d for s in segments for d in s.decision_ids]
    covered = [p["decision_id"] for p in final["protected"] if p["decision_id"] is not None]
    if covered != expected or sum(bool(p["stop"]) for p in final["protected"]) != 1:
        raise ValueError("Final instruction lost/reordered a decision or STOP")
    for record in [*final["clauses"], *final["protected"]]:
        if text[record["char_start"]:record["char_end"]] != record["text"]:
            raise ValueError("Final clause provenance is inconsistent")
    covered_segments = {s for c in final["clauses"] for s in c["segment_ids"]}
    if covered_segments != {s.segment_id for s in segments}:
        raise ValueError("Final instruction lost segment coverage")
    if MARKER_LANGUAGE.search(text):
        raise ValueError("Final instruction mentions annotations")


def relevant_text(final, segment_id):
    return " ".join(c["text"] for c in final["clauses"] if segment_id in c["segment_ids"])


def validate_polish(proposal, segments, locals_by_id):
    clauses = proposal.get("clauses")
    if not isinstance(clauses, list) or [c.get("segment_id") for c in clauses] != [s.segment_id for s in segments]:
        raise ValueError("Polishing must preserve every segment's order and provenance")
    edited = copy.deepcopy(locals_by_id)
    for segment, clause in zip(segments, clauses):
        text = clause.get("text")
        if not isinstance(text, str):
            raise ValueError("Missing edited text")
        edited[segment.segment_id]["text"] = text
        # Critical choice/stop spans remain literal and traceable. Ordinary prose
        # can be smoothed freely; stage 5 reviews the resulting final movement.
        validate_local(edited[segment.segment_id], segment)
    return edited


def polish_and_assemble(client, segments, locals_by_id):
    original = [{"segment_id": s.segment_id, "text": locals_by_id[s.segment_id]["text"],
                 "protected": [d["clause"] for d in locals_by_id[s.segment_id]["decisions"]] +
                              ([locals_by_id[s.segment_id]["stop"]["clause"]] if s.terminal else [])}
                for s in segments]
    proposal, key = client.complete("polish", POLISH_SYSTEM, [text_item(json.dumps(original))])
    audit = {"applied": False, "request_keys": [key], "proposal": proposal}
    try:
        edited = validate_polish(proposal, segments, locals_by_id)
    except (ValueError, TypeError, AttributeError, KeyError) as error:
        audit["fallback_reason"] = str(error)
    else:
        if all(edited[s.segment_id]["text"] == locals_by_id[s.segment_id]["text"] for s in segments):
            audit["fallback_reason"] = "No language edit needed"
        else:
            final = assemble(segments, edited)
            final["polish"] = {**audit, "applied": True}
            return final
    final = assemble(segments, locals_by_id)
    final["polish"] = audit
    return final
