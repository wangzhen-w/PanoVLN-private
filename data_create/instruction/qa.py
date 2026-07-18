"""Deterministic quality gates for generated VLN instructions."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Sequence

from .actions import major_turns


DATA_ARTIFACT_RE = re.compile(
    r"\b(?:image|images|frame|frames|panorama|panoramas|contact sheet|row label|"
    r"route sheet|endpoint sheet|start sheet|action sequence|action id|dataset)\b",
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

    vertical = str((trajectory_metadata or {}).get("vertical_motion") or "").lower()
    down_stairs = re.search(
        r"\b(?:go|walk|head|continue|move|climb)\s+down\s+(?:the\s+)?(?:stairs|staircase|steps)\b|"
        r"\bdownstairs\b|\bdescend(?:ing)?\b",
        lowered,
    )
    up_stairs = re.search(
        r"\b(?:go|walk|head|continue|move|climb)\s+up\s+(?:the\s+)?(?:stairs|staircase|steps)\b|"
        r"\bupstairs\b|\bascend(?:ing)?\b",
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
    return {"passed": not failures, "failures": failures, "warnings": warnings}


def audit_passed(audit: Mapping[str, Any]) -> bool:
    if not bool(audit.get("passed")):
        return False
    severity = str(audit.get("severity") or "none").lower()
    return severity in {"none", "minor"}
