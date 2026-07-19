"""Deterministic quality gates for generated VLN instructions."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Sequence

from .actions import major_turns


DATA_ARTIFACT_RE = re.compile(
    r"\b(?:image|images|panorama|panoramas|contact sheet|row label|"
    r"route sheet|endpoint sheet|start sheet|action sequence|action id|dataset)\b|"
    r"\bframe[_\s-]?\d+\b|\b(?:image|video|panorama|route|endpoint|start)\s+frames?\b",
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


def final_endpoint_context(text: str) -> str:
    """Return the final stop sentence plus its immediate approach sentence.

    Some grounding errors are expressed just before the stop command, e.g.
    "turn right at the bottom. Stop near the shutters."  The explicit stop
    clause alone is safe, but the preceding approach clause still changes where
    a follower would stop.  This helper keeps deterministic checks focused on
    the endpoint without scanning the whole route.
    """

    sentences = [part.strip() for part in re.split(r"[.!?]+", str(text or "")) if part.strip()]
    if not sentences:
        return ""
    stop_index = None
    for index in range(len(sentences) - 1, -1, -1):
        if STOP_RE.search(sentences[index]):
            stop_index = index
            break
    if stop_index is None:
        stop_index = len(sentences) - 1
    start = max(0, stop_index - 1)
    return ". ".join(sentences[start : stop_index + 1])


def safe_stop_key_terms(endpoint_facts: Mapping[str, Any]) -> set[str]:
    text = " ".join(
        [str((endpoint_facts or {}).get("safe_stop_phrase") or "")]
        + [str(item) for item in (endpoint_facts or {}).get("nearby_stop_anchors") or []]
    )
    return normalized_content_terms(text)


def safe_stop_phrase_terms(endpoint_facts: Mapping[str, Any]) -> set[str]:
    return normalized_content_terms(str((endpoint_facts or {}).get("safe_stop_phrase") or ""))


def has_final_turn_to_face_nonendpoint(text: str, endpoint_facts: Mapping[str, Any]) -> bool:
    context = final_stop_sentence(text)
    if not STOP_RE.search(context):
        return False
    matches = list(FINAL_TURN_TO_FACE_RE.finditer(context))
    if not matches:
        return False
    safe_terms = safe_stop_key_terms(endpoint_facts)
    if not safe_terms:
        return True
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
    """Check only the specific top/bottom/lower-floor claims facts reject."""

    avoid_items = [str(item).lower() for item in (endpoint_facts or {}).get("avoid_endpoint_claims") or []]
    context = final_endpoint_context(text).lower()
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


def safe_stop_phrase_preserved(final_clause: str, endpoint_facts: Mapping[str, Any]) -> bool:
    phrase = str((endpoint_facts or {}).get("safe_stop_phrase") or "")
    if not phrase.strip():
        return True
    final_lower = str(final_clause or "").lower()
    phrase_lower = phrase.lower()
    content_terms = normalized_content_terms(phrase)
    if content_terms and (content_terms & normalized_content_terms(final_clause)):
        return True
    anchors = [
        term
        for term in STOP_ANCHOR_TERMS
        if anchor_present(term, phrase_lower)
    ]
    if anchors and any(anchor_present(term, final_lower) for term in anchors):
        return True
    phrase_tokens = {
        token
        for token in re.findall(r"[a-z]+", phrase_lower)
        if len(token) >= 4 and token not in {"near", "beside", "inside", "front", "with"}
    }
    final_tokens = set(re.findall(r"[a-z]+", final_lower))
    return bool(phrase_tokens & final_tokens)


def has_uncertain_endpoint_modifier(text: str, endpoint_facts: Mapping[str, Any]) -> bool:
    """Detect risky final material/color/size claims that safe_stop avoided."""

    context = final_stop_sentence(text)
    if not STOP_RE.search(context):
        return False
    matches = {match.group(0).lower() for match in UNCERTAIN_ENDPOINT_MODIFIER_RE.finditer(context)}
    if not matches:
        return False
    endpoint_text = " ".join(
        [
            str((endpoint_facts or {}).get("safe_stop_phrase") or ""),
            str((endpoint_facts or {}).get("forward_view_anchor") or ""),
            " ".join(str(item) for item in (endpoint_facts or {}).get("nearby_stop_anchors") or []),
        ]
    ).lower()
    for phrase in matches:
        # If the endpoint extractor itself saw this phrase in the forward or
        # nearby endpoint anchors, it is a grounding risk but not a deterministic
        # failure.  The prompts still prefer neutral stop phrases when possible.
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
    if words < 12:
        failures.append("instruction_too_short")
    if profile == "concise" and words > 115:
        failures.append("concise_instruction_too_long")
    if profile == "dense" and words > 180:
        failures.append("dense_instruction_too_long")
    if DATA_ARTIFACT_RE.search(text):
        failures.append("mentions_data_artifacts")
    if DEGREE_RE.search(text):
        failures.append("mentions_numeric_degrees")
    if ACTION_LIST_RE.search(text):
        failures.append("mentions_action_tokens")
    if CONFLICTING_FINAL_FACING_RE.search(text):
        failures.append("conflicting_final_facing_chain")
    if FACE_BACK_TOWARD_RE.search(text):
        failures.append("unsupported_face_back_toward")
    if has_conflicting_turn_face_stop_near(text):
        failures.append("conflicting_final_face_then_stop_anchor")
    if not STOP_RE.search(text):
        failures.append("missing_explicit_stop_or_endpoint_cue")
    if re.search(r"\b(go|walk|continue|proceed)\s+there\b", text, re.IGNORECASE):
        warnings.append("uses_there_without_anchor")

    turns = major_turns(actions)
    if len(actions) > 80 and words < 28:
        anchors = anchor_count(lowered)
        transitions = transition_count(lowered)
        if words < 20 or (anchors < 2 and transitions < 3):
            failures.append("long_route_underdescribed")
        else:
            warnings.append("long_route_concise")
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
    stop_condition = str((route_plan or {}).get("stop_condition") or "").lower()
    if stop_condition:
        anchors = sorted(
            term
            for term in STOP_ANCHOR_TERMS
            if re.search(rf"\b{re.escape(term)}\b", stop_condition)
        )
        if anchors:
            missing = [
                term
                for term in anchors
                if not anchor_present(term, lowered)
            ]
            if len(anchors) <= 2 and missing:
                failures.append(f"endpoint_anchor_missing:{','.join(missing)}")
            elif len(anchors) > 2 and len(missing) > len(anchors) - 2:
                failures.append(f"endpoint_anchor_underpreserved:{','.join(missing)}")
    endpoint_facts = (route_plan or {}).get("endpoint_facts") or {}
    safe_terms = safe_stop_key_terms(endpoint_facts)
    final_clause = final_stop_clause(text)
    phrase_terms = safe_stop_phrase_terms(endpoint_facts)
    if not safe_stop_phrase_preserved(final_clause, endpoint_facts):
        if phrase_terms:
            failures.append(f"safe_stop_phrase_underpreserved:{','.join(sorted(phrase_terms))}")
        elif safe_terms:
            failures.append(f"safe_stop_phrase_underpreserved:{','.join(sorted(safe_terms))}")
        else:
            failures.append("safe_stop_phrase_underpreserved")
    if has_forbidden_terminal_stair_claim(text, endpoint_facts):
        failures.append("endpoint_terminal_stair_claim_for_landing")
    if has_final_turn_to_face_nonendpoint(text, endpoint_facts):
        failures.append("final_turn_to_face_nonendpoint")
    if has_uncertain_endpoint_modifier(text, endpoint_facts):
        failures.append("endpoint_uncertain_material_or_size_claim")
    return {"passed": not failures, "failures": failures, "warnings": warnings}


def audit_passed(audit: Mapping[str, Any]) -> bool:
    if not bool(audit.get("passed")):
        return False
    severity = str(audit.get("severity") or "none").lower()
    return severity in {"none", "minor"}
