"""Deterministic quality gates for generated VLN instructions."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Sequence

from .actions import major_turns


DATA_ARTIFACT_RE = re.compile(
    r"\b(?:contact sheet|row label|route sheet|endpoint sheet|start sheet|"
    r"action sequence|action id|dataset)\b|\bframe[_\s-]?\d+\b|"
    r"\b(?:shown|visible|seen|labeled|provided)\s+(?:in|on)\s+(?:the\s+)?"
    r"(?:image|frame|sheet|panorama)\b|"
    r"\b(?:image|video|panorama|route|endpoint|start)\s+frames?\b",
    re.IGNORECASE,
)
STOP_RE = re.compile(
    r"\b(?:stop|stopping|wait|halt|stand)\b|\b(?:finish|end)\s+(?:at|by|near|inside|in|on)\b|"
    r"\b(?:destination|end\s+point|endpoint)\b",
    re.IGNORECASE,
)
DEGREE_RE = re.compile(r"\b(?:\d{2,3}\s*degrees?|turn\s+\d{2,3})\b", re.IGNORECASE)
ACTION_LIST_RE = re.compile(
    r"\b(?:move_forward|turn_left|turn_right|stop=0|action\s+\d+)\b", re.IGNORECASE
)
CONFLICTING_FINAL_FACING_RE = re.compile(
    r"\bturn(?:\s+(?:left|right|around))?\s+to\s+face\b[^.!?]{0,90}"
    r"\b(?:then|and)\s+(?:stop|wait|halt|stand)\s+facing\b",
    re.IGNORECASE,
)
FACE_BACK_TOWARD_RE = re.compile(
    r"\b(?:stop|wait|halt|stand|turn)[^.!?]{0,80}\bface\s+back\s+(?:toward|towards|to)\b",
    re.IGNORECASE,
)
FINAL_TURN_TO_FACE_RE = re.compile(
    r"\bturn(?:\s+(?:left|right|around))?\s+to\s+face\s+(?P<face>[^.!?]{3,100})",
    re.IGNORECASE,
)
TURN_FACE_THEN_STOP_NEAR_RE = re.compile(
    r"\bturn(?:\s+(?:left|right|around))?\s+to\s+face\s+(?P<face>[^.!?]{3,80}?)"
    r"\s*,?\s*(?:then|and|,)\s*(?:stop|wait|halt|stand)\s+"
    r"(?P<prep>near|at|by|beside|in front of|inside|in|on)\s+(?P<stop>[^.!?]{3,100})",
    re.IGNORECASE,
)
STOP_ANCHOR_TERMS = {
    "sink",
    "counter",
    "countertop",
    "door",
    "doorway",
    "threshold",
    "stairs",
    "staircase",
    "step",
    "landing",
    "hallway",
    "corridor",
    "bed",
    "table",
    "desk",
    "sofa",
    "couch",
    "chair",
    "fireplace",
    "cabinet",
    "closet",
    "window",
    "toilet",
    "shower",
    "tub",
    "bathroom",
    "kitchen",
    "bedroom",
    "living room",
    "television",
    "tv",
}
UNCERTAIN_ENDPOINT_MODIFIER_RE = re.compile(
    r"\b(?:white|black|gray|grey|dark|light|wooden|wood|metal|glass|tall|large|small|long)\s+"
    r"(?:shelving|shelves|shelf|display|cabinet|cabinets|unit|counter|desk)\b",
    re.IGNORECASE,
)


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text))


def sentence_count(text: str) -> int:
    parts = [part for part in re.split(r"[.!?]+", text) if part.strip()]
    return len(parts)


def anchor_count(lowered_instruction: str) -> int:
    return sum(
        1 for term in STOP_ANCHOR_TERMS if anchor_present(term, lowered_instruction)
    )


def transition_count(lowered_instruction: str) -> int:
    return len(
        re.findall(
            r"\b(?:exit|enter|through|into|past|toward|towards|across|ascend|"
            r"descend|upstairs|downstairs|stairs|staircase|hallway|corridor|"
            r"landing|doorway)\b",
            lowered_instruction,
        )
    )


def anchor_present(term: str, lowered_instruction: str) -> bool:
    if term == "living room":
        return bool(re.search(r"\bliving\s+(?:room|area|space)\b", lowered_instruction))
    if term == "hallway":
        return bool(re.search(r"\b(?:hallway|corridor|hall)\b", lowered_instruction))
    if term == "corridor":
        return bool(re.search(r"\b(?:corridor|hallway|hall)\b", lowered_instruction))
    if term == "sofa":
        return bool(re.search(r"\b(?:sofa|couch|sectional)\b", lowered_instruction))
    if term == "couch":
        return bool(re.search(r"\b(?:couch|sofa|sectional)\b", lowered_instruction))
    if term == "television":
        return bool(re.search(r"\b(?:television|tv)\b", lowered_instruction))
    if term == "tv":
        return bool(re.search(r"\b(?:tv|television)\b", lowered_instruction))
    return bool(re.search(rf"\b{re.escape(term)}\b", lowered_instruction))


def normalized_content_terms(text: str) -> set[str]:
    stopwords = {
        "the",
        "and",
        "with",
        "area",
        "room",
        "space",
        "end",
        "hall",
        "hallway",
        "corridor",
        "entrance",
        "threshold",
        "doorway",
        "door",
        "near",
        "beside",
        "inside",
        "front",
        "toward",
        "towards",
        "back",
        "white",
        "black",
        "grey",
        "gray",
        "wooden",
        "large",
        "small",
        "long",
        "open",
        "curved",
        "dark",
        "wood",
    }
    synonyms = {
        "shelving": "shelf",
        "shelves": "shelf",
        "chairs": "chair",
        "desks": "desk",
        "sofas": "sofa",
        "couches": "sofa",
        "doors": "door",
        "doorway": "door",
        "hall": "hallway",
        "staircase": "stairs",
        "stair": "stairs",
    }
    return {
        synonyms.get(token, token)
        for token in re.findall(r"[a-z]+", str(text).lower())
        if len(token) >= 4 and token not in stopwords
    }


def has_conflicting_turn_face_stop_near(text: str) -> bool:
    sentences = [part.strip() for part in re.split(r"[.!?]+", str(text or "")) if part.strip()]
    if not sentences:
        return False
    match = TURN_FACE_THEN_STOP_NEAR_RE.search(sentences[-1])
    if not match:
        return False
    face_terms = normalized_content_terms(match.group("face"))
    stop_terms = normalized_content_terms(match.group("stop"))
    if not face_terms or not stop_terms:
        return False
    return not bool(face_terms & stop_terms)


def final_stop_clause(text: str) -> str:
    sentences = [part.strip() for part in re.split(r"[.!?]+", str(text or "")) if part.strip()]
    if not sentences:
        return ""
    for sentence in reversed(sentences):
        if STOP_RE.search(sentence):
            matches = list(re.finditer(r"\b(?:stop|stopping|wait|halt|stand|finish|end)\b", sentence, re.IGNORECASE))
            if matches:
                return sentence[matches[-1].start() :]
            return sentence
    return sentences[-1]


def final_stop_sentence(text: str) -> str:
    sentences = [part.strip() for part in re.split(r"[.!?]+", str(text or "")) if part.strip()]
    if not sentences:
        return ""
    for sentence in reversed(sentences):
        if STOP_RE.search(sentence):
            return sentence
    return sentences[-1]


def endpoint_key_terms(endpoint_facts: Mapping[str, Any]) -> set[str]:
    text = " ".join(
        [str((endpoint_facts or {}).get("stop_location_anchor") or "")]
        + [str(item) for item in (endpoint_facts or {}).get("nearby_stop_anchors") or []]
    )
    return normalized_content_terms(text)


def has_final_turn_to_face_nonendpoint(text: str, endpoint_facts: Mapping[str, Any]) -> bool:
    context = final_stop_sentence(text)
    if not STOP_RE.search(context):
        return False
    matches = list(FINAL_TURN_TO_FACE_RE.finditer(context))
    if not matches:
        return False
    safe_terms = endpoint_key_terms(endpoint_facts)
    if not safe_terms:
        return False
    match = matches[-1]
    tail = context[match.end() :]
    stop_match = STOP_RE.search(tail)
    if stop_match:
        between_face_and_stop = tail[: stop_match.start()]
        if re.search(
            r"\b(?:walk|move|go|proceed|continue|pass|enter|exit|through|across|follow)\b",
            between_face_and_stop,
            re.IGNORECASE,
        ):
            return False
    face_terms = normalized_content_terms(match.group("face"))
    if not face_terms:
        return True
    return not bool(face_terms & safe_terms)


def route_event_coverage_stats(route_plan: Mapping[str, Any] | None) -> Dict[str, int]:
    """Return coarse self-reported coverage counts for long segmented routes."""

    plan = route_plan or {}
    segment_facts = plan.get("segment_facts") or []
    keep_events = 0
    decision_boundaries = 0
    if isinstance(segment_facts, Sequence) and not isinstance(segment_facts, (str, bytes)):
        for segment in segment_facts:
            if not isinstance(segment, Mapping):
                continue
            for event in segment.get("ordered_route_events") or []:
                if isinstance(event, Mapping) and bool(event.get("keep_for_instruction")):
                    keep_events += 1
            for boundary in segment.get("must_keep_decision_boundaries") or []:
                if isinstance(boundary, Mapping):
                    text = " ".join(str(boundary.get(key) or "") for key in ("boundary", "why_needed"))
                else:
                    text = str(boundary or "")
                if text.strip():
                    decision_boundaries += 1
    covered_events = len(plan.get("covered_route_events") or [])
    covered_boundaries = len(plan.get("covered_decision_boundaries") or [])
    return {
        "keep_events": keep_events,
        "covered_events": covered_events,
        "decision_boundaries": decision_boundaries,
        "covered_boundaries": covered_boundaries,
    }


def has_forbidden_terminal_stair_claim(text: str, endpoint_facts: Mapping[str, Any]) -> bool:
    """Check explicit stop wording, not earlier approach descriptions.

    A route may legitimately pass the bottom of a staircase and continue to a
    door or another room.  Only text governed by the final stop verb is treated
    as an endpoint claim here; visual auditing remains responsible for broader
    semantic grounding.
    """

    avoid_items = [str(item).lower() for item in (endpoint_facts or {}).get("avoid_endpoint_claims") or []]
    context = final_stop_clause(text).lower()
    forbidden_patterns: List[str] = []
    for item in avoid_items:
        if re.search(r"\btop\b", item):
            forbidden_patterns.append(r"\btop\b")
        if re.search(r"\bbottom\b", item):
            forbidden_patterns.append(r"\bbottom\b")
        if re.search(r"\blower\s+floor\b", item):
            forbidden_patterns.append(r"\blower\s+floor\b")
        if re.search(r"\blower\s+level\b", item):
            forbidden_patterns.append(r"\blower\s+level\b")
        if re.search(r"\bground\s+floor\b", item):
            forbidden_patterns.append(r"\bground\s+floor\b")
    return any(re.search(pattern, context) for pattern in set(forbidden_patterns))


def has_uncertain_endpoint_modifier(text: str, endpoint_facts: Mapping[str, Any]) -> bool:
    """Detect endpoint modifiers absent from the verified endpoint facts."""

    context = final_stop_sentence(text)
    if not STOP_RE.search(context):
        return False
    matches = {match.group(0).lower() for match in UNCERTAIN_ENDPOINT_MODIFIER_RE.finditer(context)}
    if not matches:
        return False
    endpoint_text = " ".join(
        [
            str((endpoint_facts or {}).get("stop_location_anchor") or ""),
            str((endpoint_facts or {}).get("forward_view_anchor") or ""),
            " ".join(str(item) for item in (endpoint_facts or {}).get("nearby_stop_anchors") or []),
        ]
    ).lower()
    for phrase in matches:
        # If the endpoint extractor itself saw this phrase in the forward or
        # nearby endpoint anchors, it is a grounding risk but not a deterministic
        # failure. Visual review remains responsible for the actual grounding.
        if phrase not in endpoint_text:
            return True
    return False


def validate_instruction(
    instruction: str,
    *,
    profile: str,
    actions: Sequence[int],
    trajectory_metadata: Mapping[str, Any],
    route_plan: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    failures: List[str] = []
    warnings: List[str] = []
    text = str(instruction or "").strip()
    lowered = text.lower()
    words = word_count(text)
    if not text or not re.search(r"[A-Za-z]", text):
        failures.append("instruction_empty_or_malformed")
    elif words < 12:
        warnings.append("instruction_may_be_too_short")
    if profile == "concise" and words > 115:
        warnings.append("concise_instruction_may_be_too_long")
    if profile == "dense" and words > 180:
        warnings.append("dense_instruction_may_be_too_long")
    if DATA_ARTIFACT_RE.search(text):
        failures.append("mentions_data_artifacts")
    if DEGREE_RE.search(text):
        warnings.append("mentions_numeric_degrees")
    if ACTION_LIST_RE.search(text):
        failures.append("mentions_action_tokens")
    if CONFLICTING_FINAL_FACING_RE.search(text):
        warnings.append("possible_conflicting_final_facing_chain")
    if FACE_BACK_TOWARD_RE.search(text):
        warnings.append("possible_face_back_toward")
    if has_conflicting_turn_face_stop_near(text):
        warnings.append("possible_conflicting_final_face_then_stop_anchor")
    if re.search(r"\b(go|walk|continue|proceed)\s+there\b", text, re.IGNORECASE):
        warnings.append("uses_there_without_anchor")

    turns = major_turns(actions)
    if len(actions) > 80 and words < 28:
        anchors = anchor_count(lowered)
        transitions = transition_count(lowered)
        warnings.append(
            "long_route_lexically_concise"
            if words < 20 or (anchors < 2 and transitions < 3)
            else "long_route_concise"
        )
    if len(turns) >= 4 and sentence_count(text) < 2 and words < 35:
        warnings.append("complex_route_may_be_overcompressed")
    if len(actions) > 80 and route_plan:
        coverage = route_event_coverage_stats(route_plan)
        if coverage["keep_events"] >= 6 and coverage["covered_events"] < min(5, max(3, coverage["keep_events"] // 2)):
            warnings.append("route_event_coverage_underpreserved")
        if (
            coverage["decision_boundaries"] >= 3
            and coverage["covered_boundaries"] < min(3, coverage["decision_boundaries"])
        ):
            warnings.append("decision_boundary_coverage_underpreserved")

    vertical = str((trajectory_metadata or {}).get("vertical_motion") or "").lower()
    down_stairs = re.search(
        r"\b(?:go|walk|head|continue|move|climb)\s+down\s+(?:the\s+)?(?:stairs|staircase|steps)\b|"
        r"\b(?:go|walk|head|continue|move|climb)\s+downstairs\b|"
        r"\bdescend(?:ing)?\b",
        lowered,
    )
    up_stairs = re.search(
        r"\b(?:go|walk|head|continue|move|climb)\s+up\s+(?:the\s+)?(?:stairs|staircase|steps)\b|"
        r"\b(?:go|walk|head|continue|move|climb)\s+upstairs\b|"
        r"\bascend(?:ing)?\b",
        lowered,
    )
    if vertical == "ascending" and down_stairs:
        failures.append("vertical_motion_conflict_ascending")
    if vertical == "descending" and up_stairs:
        failures.append("vertical_motion_conflict_descending")
    endpoint_facts = (route_plan or {}).get("endpoint_facts") or {}
    if has_forbidden_terminal_stair_claim(text, endpoint_facts):
        warnings.append("possible_endpoint_terminal_stair_claim_for_landing")
    if has_final_turn_to_face_nonendpoint(text, endpoint_facts):
        warnings.append("final_turn_to_face_nonendpoint")
    if has_uncertain_endpoint_modifier(text, endpoint_facts):
        warnings.append("endpoint_uncertain_material_or_size_claim")
    return {"passed": not failures, "failures": failures, "warnings": warnings}


def audit_contract_complete(audit: Mapping[str, Any]) -> bool:
    """Return whether a blind audit satisfies its required semantic schema."""

    if not isinstance(audit.get("route_sequence_matches"), bool):
        return False
    if not isinstance(audit.get("endpoint_matches"), bool):
        return False
    if not isinstance(audit.get("passed"), bool):
        return False
    if str(audit.get("severity") or "").strip().lower() not in {
        "none",
        "minor",
        "critical",
    }:
        return False
    if not isinstance(audit.get("problems"), list) or not all(
        isinstance(problem, str) for problem in audit["problems"]
    ):
        return False
    if not isinstance(audit.get("corrected_instruction"), str):
        return False
    endpoint = audit.get("observed_endpoint")
    if not isinstance(endpoint, Mapping):
        return False
    observed_route = audit.get("observed_route_sequence")
    instruction_route = audit.get("instruction_route_sequence")
    if not isinstance(observed_route, list) or not observed_route or not all(
        isinstance(item, str) and item.strip() for item in observed_route
    ):
        return False
    if not isinstance(instruction_route, list) or not instruction_route or not all(
        isinstance(item, str) and item.strip() for item in instruction_route
    ):
        return False
    if not isinstance(endpoint.get("stop_location"), str) or not endpoint["stop_location"].strip():
        return False
    if not isinstance(endpoint.get("forward_anchor"), str) or not endpoint["forward_anchor"].strip():
        return False
    relation_value = endpoint.get("forward_anchor_relation")
    if not isinstance(relation_value, str):
        return False
    forward_relation = relation_value.strip().lower()
    if forward_relation not in {
        "reached_at_final_camera",
        "remains_ahead_after_final",
        "uncertain",
    }:
        return False
    return True


def audit_passed(audit: Mapping[str, Any]) -> bool:
    if not audit_contract_complete(audit):
        return False
    if not bool(audit.get("passed")):
        return False
    if audit.get("route_sequence_matches") is not True:
        return False
    if audit.get("endpoint_matches") is not True:
        return False
    if any(str(item).strip() for item in (audit.get("problems") or [])):
        return False
    severity = str(audit.get("severity") or "none").lower()
    return severity == "none"
