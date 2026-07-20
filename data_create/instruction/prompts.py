"""Prompts for trajectory-grounded instruction generation."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping


SYSTEM_JSON = (
    "You are a careful VLN navigation-instruction annotator. Use only the "
    "trajectory evidence provided in this request. Do not copy or infer from "
    "any source instruction. Do not reveal reasoning. Output only valid JSON."
)

FINAL_INSTRUCTION_SYSTEM_JSON = (
    "You are a careful VLN navigation-instruction annotator. Use only the "
    "trajectory evidence provided in this request. In final, repaired, or corrected "
    "instructions, let the actual route determine the discourse and wording. A useful "
    "starting context, direct action, orientation, space transition, or landmark-led cue "
    "may all be natural; no opening form is required or forbidden. Do not default to one "
    "corpus-wide formula or mechanically replace it with another. Do not reveal reasoning. "
    "Output only valid JSON."
)

ALTERNATIVE_INSTRUCTION_SYSTEM_JSON = (
    "You are the language-realization writer for a grounded VLN route. The supplied route "
    "draft and semantic plan are the complete content boundary: preserve their route order, "
    "decisions, landmarks, and endpoint, without adding visual facts. Express that content "
    "as an independently organized, natural instruction rather than copying the source's "
    "first clause or sentence structure. There is no required replacement opener, verb, "
    "syntax, or length. Do not reveal reasoning. Output only valid JSON."
)


def style_requirements(profile: str) -> str:
    if profile == "dense":
        return (
            "Use enough grounded route decisions and progress cues to make a long or "
            "ambiguous route executable. Let the route determine the length, discourse, "
            "and sentence structure."
        )
    return (
        "Use the smallest set of grounded route decisions, progress cues, and endpoint "
        "facts that still makes the route executable. Let the route determine the discourse "
        "and sentence structure."
    )


def alternative_realization_prompt(
    *,
    episode_id: Any,
    profile: str,
    source_instruction: str,
    route_content: Mapping[str, Any],
    discourse_intent: Mapping[str, str],
) -> str:
    schema = {
        "preserved_route_content": [
            "brief ordered checks showing that navigation-critical source content was retained"
        ],
        "final_instruction": "independently worded instruction to publish",
    }
    return f"""
Rewrite one grounded VLN route as a natural instruction. This is language realization, not a new visual interpretation.

Episode: {episode_id}
Instruction profile: {profile}

Discourse intent for this episode:
{json.dumps(discourse_intent or {}, ensure_ascii=False, indent=2)}

The intent is a high-level organizational suggestion, not a sentence template. It never overrides grounded route content. If it does not fit the available facts, use the closest natural organization without inventing evidence.

Grounded source instruction:
{source_instruction}

Grounded semantic content:
{json.dumps(route_content or {}, ensure_ascii=False, indent=2)}

Requirements:
- Preserve the same executable route order, necessary decisions and transitions, useful landmarks, stair direction, and final stopping location.
- Do not add, remove, reverse, or relocate navigation facts merely to make the wording different. Generalize a phrase only when its extra detail has no navigation value.
- Starting context, direct action, orientation, space transition, and landmark-led phrasing are all valid when they make this particular route clear. The literal word `Start` is neither required nor forbidden.
- Reorganize the discourse independently rather than copying the source opening sentence-by-sentence or performing mechanical synonym replacement.
- Do not create apparent diversity by shifting every route to one alternative stock opener.
- Choose the structure that communicates this particular route naturally. There is no required first word, transition phrase, sentence count, or length.
- Do not mention drafts, plans, facts, images, frames, action IDs, degrees, or coordinates.

Return exactly this JSON schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}
"""


def endpoint_fact_prompt(
    *,
    episode_id: Any,
    profile: str,
    action_summary: Mapping[str, Any],
    trajectory_metadata: Mapping[str, Any],
) -> str:
    schema = {
        "final_location_type": "inside_room|threshold_or_doorway|stair_landing|hallway_or_corridor|near_object|outdoor_or_balcony|uncertain",
        "stop_location_anchor": "semantic fact identifying where the FINAL camera is standing; not drafted instruction text",
        "forward_view_anchor": "main object/space visible straight ahead at FINAL",
        "forward_anchor_relation": "reached_at_final_camera|remains_ahead_after_final|uncertain",
        "left_view_anchor": "main object/space visible left of FINAL, or empty",
        "right_view_anchor": "main object/space visible right of FINAL, or empty",
        "back_view_anchor": "main object/space behind FINAL, or empty",
        "nearby_stop_anchors": ["visible objects/spaces that help identify the stopping area"],
        "avoid_endpoint_claims": ["wrong endpoint, wrong facing, over-specific color/room-entry claims to avoid"],
        "confidence": "high|medium|low",
    }
    return f"""
You are extracting the final stopping facts for a VLN route.
You will first receive an ENDPOINT approach sheet containing the last translated positions, followed by five labeled FINAL STOP views from exactly one physical observation: FINAL FORWARD, FINAL FORWARD-DOWN, FINAL LEFT, FINAL RIGHT, FINAL BACK. The FINAL camera location itself is the true stop.

Episode: {episode_id}
Instruction profile: {profile}

Action-derived constraints for the full route:
{json.dumps(action_summary, ensure_ascii=False, indent=2)}

Trajectory metadata from geometry/GT, when available:
{json.dumps(trajectory_metadata or {}, ensure_ascii=False, indent=2)}

Rules:
- Describe only the final stopping area, not the full route.
- `stop_location_anchor` must identify where the FINAL camera is already standing. A visible object straight ahead may be distant context rather than the stopping location.
- Use the ENDPOINT approach rows and final view to decide whether the camera has reached the forward anchor or whether it remains ahead after the final observation. Record that movement relation without relying on a fixed distance threshold.
- The FINAL FORWARD view is what the agent is facing at STOP. Objects in FINAL LEFT/RIGHT/BACK may help identify `stop_location_anchor`, but they cannot replace the specific forward-facing anchor or be described as straight ahead.
- Classify a threshold, room interior, hallway, landing, or outdoor boundary from the camera's physical location, not merely from an object visible in a side view.
- Treat top and bottom as distinct from an intermediate stair landing. Claim a terminal stair position only when the evidence establishes it.
- Record only object type, color, material, and spatial relations that are visually reliable. Use additional nearby anchors when one object is ambiguous or repeated.
- Include likely false endpoint or facing claims in `avoid_endpoint_claims`, especially side-view objects that should not be described as straight ahead.
- Do not mention images, sheets, frames, rows, labels, action IDs, degrees, or coordinates.

Return exactly this JSON schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}
"""


def endpoint_fact_audit_prompt(
    *,
    episode_id: Any,
    profile: str,
    action_summary: Mapping[str, Any],
    trajectory_metadata: Mapping[str, Any],
    endpoint_facts: Mapping[str, Any],
) -> str:
    schema = {
        "passed": True,
        "problems": ["empty unless a fact uses the wrong visual column or endpoint"],
        "endpoint_facts": {
            "final_location_type": "inside_room|threshold_or_doorway|stair_landing|hallway_or_corridor|near_object|outdoor_or_balcony|uncertain",
            "stop_location_anchor": "semantic fact identifying where the FINAL camera is standing; not drafted instruction text",
            "forward_view_anchor": "main object/space visible straight ahead in the first FORWARD panel",
            "forward_anchor_relation": "reached_at_final_camera|remains_ahead_after_final|uncertain",
            "left_view_anchor": "main object/space visible in the LEFT panel, or empty",
            "right_view_anchor": "main object/space visible in the RIGHT panel, or empty",
            "back_view_anchor": "main object/space visible in the BACK panel, or empty",
            "nearby_stop_anchors": ["stable anchors near the stop when useful"],
            "avoid_endpoint_claims": ["wrong endpoint, wrong facing, or side-view destination claims to avoid"],
            "confidence": "high|medium|low",
        },
    }
    return f"""
You are auditing extracted endpoint facts for one VLN final stop.
You will receive the ENDPOINT approach sheet followed by the same five labeled FINAL STOP views, ordered as FINAL FORWARD, FINAL FORWARD-DOWN, FINAL LEFT, FINAL RIGHT, FINAL BACK.

Episode: {episode_id}
Instruction profile: {profile}

Existing endpoint facts to verify:
{json.dumps(endpoint_facts or {}, ensure_ascii=False, indent=2)}

Action-derived constraints for the full route:
{json.dumps(action_summary, ensure_ascii=False, indent=2)}

Trajectory metadata from geometry/GT, when available:
{json.dumps(trajectory_metadata or {}, ensure_ascii=False, indent=2)}

Audit task:
- Verify that `forward_view_anchor` describes FINAL FORWARD, not FINAL LEFT/RIGHT/BACK.
- Verify that `stop_location_anchor` describes the FINAL camera's current physical location, using nearby boundaries and the approach sequence. It must not simply name a distant object visible ahead.
- Verify `forward_anchor_relation` from the approach sequence and final view. Do not turn an anchor that remains ahead after the final observation into a destination requiring additional movement.
- Verify that `left_view_anchor`, `right_view_anchor`, and `back_view_anchor` describe their own panels.
- Verify that `stop_location_anchor` names the true stopping area rather than a side-view or distant forward object.
- Verify that material, color, object-category, and spatial-relation facts are visually supported; remove or generalize unsupported detail.
- If the facts are correct, return them unchanged with `passed=true`.
- If any anchor or endpoint fact is wrong, return corrected endpoint facts with `passed=false` and concise problems.
- Do not mention images, sheets, frames, rows, labels, action IDs, degrees, or coordinates.

Return exactly this JSON schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}
"""


def plan_write_prompt(
    *,
    episode_id: Any,
    profile: str,
    action_summary: Mapping[str, Any],
    trajectory_metadata: Mapping[str, Any],
    endpoint_facts: Mapping[str, Any],
) -> str:
    schema = {
        "selected_route_facts": [
            {
                "order": 1,
                "fact": "grounded movement, decision, or space transition; not drafted instruction text",
                "navigation_value": "why this fact helps execute the route",
            }
        ],
        "endpoint_fact_used": "semantic final-location fact; not a required phrase",
        "uncertain_or_avoid": ["claims that should not be made"],
        "final_instruction": "the instruction to publish",
    }
    return f"""
You will receive three visual evidence sheets, in this order:
1. EARLY ROUTE: the initial and early translated positions.
2. ROUTE: selected translated positions along the path. Each row is a physical waypoint; columns show left, forward, right, and back views.
3. ENDPOINT: the final approach and terminal view. The row labeled FINAL is the true stopping observation.

The route is panoramic VLN: write directions that a follower can execute from the initial observation to the final stop. The labels are evidence labels only; do not mention images, frames, rows, sheets, or action IDs in the final instruction.

Episode: {episode_id}
Instruction profile: {profile}
Style requirement: {style_requirements(profile)}

Action-derived constraints:
{json.dumps(action_summary, ensure_ascii=False, indent=2)}

Trajectory metadata from geometry/GT, when available:
{json.dumps(trajectory_metadata or {}, ensure_ascii=False, indent=2)}

Endpoint facts extracted from the FINAL STOP evidence:
{json.dumps(endpoint_facts or {}, ensure_ascii=False, indent=2)}

Quality rules:
- Communicate the real route in travel order, including the decisions and space transitions a follower needs and omitting low-level alignment or redundant observations.
- Turn the selected facts into coherent directions rather than one sentence per action or event. Mention a change of facing only when it is needed to choose the next path.
- Every movement, landmark, spatial relation, and room/space label that appears in the instruction must be supported by the evidence. Omit or generalize uncertain detail.
- Use left/right when it resolves a supported route choice, not merely because the camera rotated or an object appeared in a panoramic side view.
- Treat trajectory `vertical_motion` as a hard physical constraint whenever stairs are described.
- Make the final location uniquely identifiable from the endpoint facts and evidence. The FINAL camera is already at the destination; do not extend the route toward a distant or side/back context object.
- Preserve the meaning of verified endpoint facts, but choose your own wording. There is no required opening, sentence count, syntax, navigation verb, or literal stop word.
- Let this route determine whether initial-scene context or a direct navigation cue is the clearest opening; neither form is preferred globally.
- Let route complexity determine length. Missing unimportant objects or local heading adjustments is acceptable when the route remains executable.
- Do not mention images, evidence labels, frames, action IDs, degrees, or coordinates in the instruction.

Return exactly this JSON schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}
"""


def segment_fact_prompt(
    *,
    episode_id: Any,
    profile: str,
    action_summary: Mapping[str, Any],
    segment_action_summary: Mapping[str, Any],
    trajectory_metadata: Mapping[str, Any],
    endpoint_facts: Mapping[str, Any],
    segment_id: int,
    segment_count: int,
    segment_frames: Any,
) -> str:
    schema = {
        "segment_id": segment_id,
        "entry_anchor": "visible place or object at the start of this segment",
        "exit_anchor": "visible place or object at the end of this segment",
        "movement_summary": "semantic motion facts for this segment; not drafted instruction text",
        "must_keep_route_choice": "turn, doorway, stair, room transition, or empty",
        "ordered_route_events": [
            {
                "order": 1,
                "event": "grounded turn, doorway, hallway, stair, room transition, or final approach fact",
                "visual_anchor": "visible landmark or spatial boundary supporting this event",
                "keep_for_instruction": True,
            }
        ],
        "must_keep_decision_boundaries": [
            {
                "order": 1,
                "boundary": "doorway, arch, stair landing, hallway junction, island/bar bypass, room boundary, indoor/outdoor transition, or empty",
                "why_needed": "why a follower may choose the wrong path or stop early if this is omitted",
            }
        ],
        "uncertain_side_claims": [
            "left/right turn, side-of-object, or final-facing claim that should be removed or neutralized if used"
        ],
        "stable_landmarks": ["visible landmarks useful for executing this segment"],
        "stop_anchor_if_final": "semantic final-location anchor, empty for non-final segments",
        "do_not_claim": ["unsupported claims to avoid"],
        "is_final_segment": segment_id == segment_count,
        "confidence": "high|medium|low",
    }
    return f"""
You are extracting grounded route facts for one segment of a longer VLN route.
You will receive one SEGMENT evidence sheet. Each row is a physical waypoint in this segment; columns show left, forward, right, and back views. Use only this segment's visual evidence plus the action/trajectory constraints below.

Episode: {episode_id}
Instruction profile: {profile}
Segment: {segment_id} of {segment_count}
Segment frame labels: {json.dumps(segment_frames, ensure_ascii=False)}

Action-derived constraints for the full route:
{json.dumps(action_summary, ensure_ascii=False, indent=2)}

Action-derived constraints for this segment:
{json.dumps(segment_action_summary, ensure_ascii=False, indent=2)}

Trajectory metadata from geometry/GT, when available:
{json.dumps(trajectory_metadata or {}, ensure_ascii=False, indent=2)}

Endpoint facts from the FINAL STOP evidence:
{json.dumps(endpoint_facts or {}, ensure_ascii=False, indent=2)}

Rules:
- Do not write the final navigation instruction.
- Extract only route facts visible in this segment.
- Treat the segment action constraints as hard evidence about physical motion.
- Distinguish translated movement and space crossing from camera rotation or local alignment. A new view alone does not prove that the route entered or traversed a space.
- Record important route events in travel order. Mark an event `keep_for_instruction=true` only when omitting it could plausibly cause a wrong route choice, wrong space sequence, lost progress, or early stop.
- Record a decision boundary only when it is visually grounded and navigation-critical. Duplicate context, pure alignment, and incidental objects are not decision boundaries.
- Keep landmarks only when they identify a choice, confirm useful progress, or locate the final destination.
- Treat supported up/down movement and meaningful left/right branch choices as facts; put ambiguous relations in `uncertain_side_claims` rather than resolving them linguistically.
- For the final segment, use the verified `stop_location_anchor` as the destination fact. Do not substitute distant forward context or a side/back object.
- `do_not_claim` is local to this segment except for endpoint claims verified from the final stop evidence.
- Put unsupported or uncertain semantic claims in `do_not_claim` or lower their confidence.
- Do not mention images, sheets, frames, rows, labels, action IDs, degrees, or coordinates.

Return exactly this JSON schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}
"""


def segmented_merge_prompt(
    *,
    episode_id: Any,
    profile: str,
    action_summary: Mapping[str, Any],
    trajectory_metadata: Mapping[str, Any],
    endpoint_facts: Mapping[str, Any],
    segment_facts: Any,
) -> str:
    schema = {
        "covered_route_events": ["brief labels for important ordered segment events preserved in final_instruction"],
        "covered_decision_boundaries": ["brief labels for must_keep_decision_boundaries preserved in final_instruction"],
        "omitted_route_events": [
            {
                "event": "important event from segment facts that was omitted",
                "reason": "why omission does not hurt route execution",
            }
        ],
        "endpoint_fact_used": "semantic final-location fact; not a required phrase",
        "uncertain_or_avoid": ["claims that should not be made"],
        "final_instruction": "the instruction to publish",
    }
    return f"""
Write the final VLN instruction by merging ordered segment facts into one natural route instruction.
You will also receive EARLY ROUTE, ROUTE overview, and ENDPOINT evidence sheets. Use the segment facts to preserve route order and local transitions. Use ENDPOINT evidence as the authority for the final stopping condition.

Episode: {episode_id}
Instruction profile: {profile}
Style requirement: {style_requirements(profile)}

Segment facts:
{json.dumps(segment_facts, ensure_ascii=False, indent=2)}

Action-derived constraints:
{json.dumps(action_summary, ensure_ascii=False, indent=2)}

Trajectory metadata from geometry/GT, when available:
{json.dumps(trajectory_metadata or {}, ensure_ascii=False, indent=2)}

Endpoint facts extracted from the FINAL STOP evidence:
{json.dumps(endpoint_facts or {}, ensure_ascii=False, indent=2)}

Merge rules:
- Write a new natural instruction from the semantic facts; do not concatenate or copy segment summaries as drafted prose.
- Group connected events into coherent directions instead of narrating each camera alignment or semantic event separately. Retain a heading change only when it resolves a route choice.
- Preserve route order and the events or boundaries marked navigation-critical. Omit duplicate, uncertain, or incidental events when their removal does not change the executable route.
- Every published movement, landmark, space label, and spatial relation must be supported by the segment facts or raw evidence. Trajectory `vertical_motion` remains a hard physical constraint.
- Use landmarks selectively to resolve choices, confirm useful progress, or identify the destination. Do not add surface detail merely to make the instruction longer.
- Treat the FINAL camera location and verified `stop_location_anchor` as the destination. Forward distance and side/back context must not extend or redirect the route.
- Preserve endpoint meaning and necessary route decisions, but paraphrase freely. There is no required opening, sentence count, syntax, transition phrase, navigation verb, or literal stop word.
- Let this route determine whether initial-scene context or a direct navigation cue is the clearest opening; neither form is preferred globally.
- Let route complexity determine the amount of detail. Missing low-level actions, local alignments, and unimportant objects is acceptable when the route remains executable.
- Do not mention segments, evidence labels, images, frames, action IDs, degrees, or coordinates.

Return exactly this JSON schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}
"""


def candidate_judge_prompt(
    *,
    episode_id: Any,
    profile: str,
    action_summary: Mapping[str, Any],
    trajectory_metadata: Mapping[str, Any],
    candidates: Any,
) -> str:
    schema = {
        "observed_route_sequence": [
            "ordered start space, necessary intermediate space/boundary, and final space inferred from raw evidence"
        ],
        "observed_endpoint": {
            "stop_location": "where the FINAL camera is already standing",
            "forward_anchor": "one specific object or boundary straight ahead whose reached/ahead relation is judged",
            "forward_anchor_relation": "reached_at_final_camera|remains_ahead_after_final|uncertain",
        },
        "candidate_assessments": [
            {
                "index": 0,
                "instruction_route_sequence": ["space/boundary sequence actually communicated by this candidate"],
                "route_sequence_matches": True,
                "endpoint_matches": True,
                "critical_problems": [],
            }
        ],
        "selected_index": 0,
        "verdict": "brief reason for the selected candidate",
        "rejected": [
            {
                "index": 1,
                "reason": "route, grounding, endpoint, or language issue",
            }
        ],
        "needs_repair": False,
        "repair_instruction": "empty unless all candidates need a concise endpoint/route fix",
    }
    return f"""
You are choosing the best publishable VLN instruction from multiple candidates.
You will receive EARLY ROUTE, ROUTE, ENDPOINT, and FINAL STOP visual evidence. The route is panoramic navigation: choose the instruction that best helps a follower execute the true route and stop at the true final location.

Episode: {episode_id}
Instruction profile: {profile}

Candidate instructions:
{json.dumps(candidates, ensure_ascii=False, indent=2)}

Action-derived constraints:
{json.dumps(action_summary, ensure_ascii=False, indent=2)}

Trajectory metadata:
{json.dumps(trajectory_metadata or {}, ensure_ascii=False, indent=2)}

Selection criteria:
- Judge only from the raw visual evidence and action/geometry constraints in this request. The candidates may have been written from a shared, fallible route plan; do not assume their common claims are correct.
- Before comparing candidates, reconstruct the navigation-critical route sequence and final stopping state directly from the ordered evidence.
- Name one specific `forward_anchor`, then judge the relation to that same anchor. Set `reached_at_final_camera` only when the FINAL camera has physically reached it; an object or boundary farther down a hall or across visible floor is `remains_ahead_after_final`. Do not use the whole hallway or scene as the anchor when a distinct forward object remains visible.
- Fill one `candidate_assessments` entry per candidate. `instruction_route_sequence` must describe what the text actually tells a follower, not what you think the candidate intended. Set both match fields explicitly.
- A candidate is publishable only when both `route_sequence_matches` and `endpoint_matches` are true. If no candidate is publishable, return a corrected instruction with `needs_repair=true`.
- Prefer faithful route order, necessary choices and transitions, grounded landmarks, correct stair direction, and the true final camera location. Omitted detail matters only when it changes executability.
- Every communicated visual or spatial claim must be supported. Unsupported specificity, side relations, or final orientation are grounding errors, not style errors.
- Treat forward, side, and back views as context around the same final camera. A context object beyond the stop must not extend the route or replace the camera's stopping area.
- Side and back anchors may validly identify where that camera is standing even though they are not the forward destination or final facing direction.
- When candidates are equally correct and executable, prefer the one that reads naturally without redundancy. Do not reward a particular opening, sentence count, verb, phrase, or length.
- When equally grounded candidates differ only in discourse, prefer wording organized around this route's useful actions and transitions rather than reusable introductory boilerplate. A genuinely useful initial landmark is navigation content, not boilerplate.
- When the final location is semantically clear, do not add or subtract quality for the literal words "start", "stop", or "wait".
- If no single candidate is fully best, but one candidate has the better route transitions and another has the better endpoint, set `needs_repair=true` and provide one corrected full instruction grounded directly in the raw evidence.
- If all candidates have a fixable endpoint or route issue, set `needs_repair=true` and provide one corrected full instruction in `repair_instruction`.
- Do not mention images, sheets, frames, rows, action IDs, degrees, or coordinates in any repair instruction.

Return exactly this JSON schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}
"""


def audit_prompt(
    *,
    episode_id: Any,
    profile: str,
    action_summary: Mapping[str, Any],
    trajectory_metadata: Mapping[str, Any],
    instruction: str,
) -> str:
    schema = {
        "observed_route_sequence": [
            "ordered start space, necessary intermediate space/boundary, and final space inferred from raw evidence"
        ],
        "observed_endpoint": {
            "stop_location": "where the FINAL camera is already standing",
            "forward_anchor": "one specific object or boundary straight ahead whose reached/ahead relation is judged",
            "forward_anchor_relation": "reached_at_final_camera|remains_ahead_after_final|uncertain",
        },
        "instruction_route_sequence": [
            "space/boundary sequence actually communicated by the instruction"
        ],
        "route_sequence_matches": True,
        "endpoint_matches": True,
        "passed": True,
        "severity": "none|minor|critical",
        "problems": ["grounding or route problems, empty if none"],
        "corrected_instruction": "empty if no correction is needed",
    }
    return f"""
You are independently auditing one generated VLN instruction against the same EARLY ROUTE, ROUTE, and ENDPOINT evidence sheets. You did not write the instruction.

Episode: {episode_id}
Instruction profile: {profile}
Instruction to audit:
{instruction}

Action-derived constraints:
{json.dumps(action_summary, ensure_ascii=False, indent=2)}

Trajectory metadata:
{json.dumps(trajectory_metadata or {}, ensure_ascii=False, indent=2)}

Audit criteria:
- Audit only against the raw visual evidence and action/geometry constraints in this request. You are intentionally not given the writer's endpoint facts or route plan because those may be wrong.
- First reconstruct the navigation-critical route sequence and final stopping state from the ordered raw evidence. The FINAL camera is already at the destination; distinguish its stopping area from forward or side/back context beyond that location.
- Name one specific `forward_anchor`, then judge the relation to that same anchor. Set `reached_at_final_camera` only when the camera has physically reached it; an object or boundary still farther down a hall or across visible floor is `remains_ahead_after_final`. Do not classify the whole hallway or scene when a distinct forward object remains visible.
- Then fill `instruction_route_sequence` from the instruction text itself. Set `route_sequence_matches` and `endpoint_matches` explicitly; `passed` may be true only when both are true.
- Every visual or spatial claim in the instruction must be supported by the evidence.
- The instruction must preserve the route decisions and space transitions needed for execution in their real order. Missing minor landmarks, redundant boundaries, and local alignments is acceptable at any route length.
- Treat a missing event as an error only when its omission creates a plausible wrong turn, wrong space sequence, loss of progress, early stop, or overshoot.
- The endpoint description must uniquely identify the final camera location without replacing it with another room, boundary, stair position, or context object.
- Side and back anchors may identify the final camera's location; do not reject them merely because they are outside the forward panel. They become errors only when the instruction redirects movement toward them or claims they are straight ahead.
- Treat up/down direction from `vertical_motion` as a hard physical constraint. Distinguish an intermediate landing from an established top or bottom when that distinction affects where the route ends.
- Unsupported specificity and spatial relations are grounding problems only when the instruction actually asserts them. Do not penalize neutral or varied wording.
- Do not require exhaustive action detail, a particular sentence structure, fixed length, literal stop word, or any preferred navigation phrase.
- Fail critically for a wrong route choice or order, wrong endpoint, overshoot, hallucinated navigation landmark, or wrong stair direction.
- Correct only factual or executability problems. Preserve valid language when possible and do not normalize style.

Return exactly this JSON schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}
"""


def repair_prompt(
    *,
    episode_id: Any,
    profile: str,
    action_summary: Mapping[str, Any],
    trajectory_metadata: Mapping[str, Any],
    endpoint_facts: Mapping[str, Any],
    route_plan: Mapping[str, Any],
    failed_instruction: str,
    issues: Any,
) -> str:
    schema = {
        "final_instruction": "corrected instruction only",
        "changes": ["brief list of fixes"],
    }
    return f"""
Repair the generated VLN instruction using the EARLY ROUTE, ROUTE, and ENDPOINT evidence.

Episode: {episode_id}
Instruction profile: {profile}
Style requirement: {style_requirements(profile)}

Failed instruction:
{failed_instruction}

Known issues to fix:
{json.dumps(issues, ensure_ascii=False, indent=2)}

Original route plan:
{json.dumps(route_plan, ensure_ascii=False, indent=2)}

Action-derived constraints:
{json.dumps(action_summary, ensure_ascii=False, indent=2)}

Trajectory metadata:
{json.dumps(trajectory_metadata or {}, ensure_ascii=False, indent=2)}

Endpoint facts from the FINAL STOP evidence:
{json.dumps(endpoint_facts or {}, ensure_ascii=False, indent=2)}

Repair rules:
- Repair only the reported factual or executability problems and preserve all other valid content and natural wording.
- Keep the real route order, navigation-critical decisions or space transitions, geometry-supported stair direction, and final camera location.
- Restore an omitted event only when its absence could produce a wrong route choice, wrong space sequence, loss of progress, early stop, or overshoot.
- Remove or generalize unsupported movements, landmarks, spatial relations, and endpoint claims instead of inventing replacements.
- Preserve the semantic meaning of verified endpoint facts without copying their wording. Do not redirect the destination toward forward or side/back context beyond the stop.
- Let route complexity determine length and sentence structure. Do not impose a preferred opening, phrase, verb, literal stop word, or stylistic template.
- Do not mention evidence artifacts, action IDs, degrees, or coordinates.

Return exactly this JSON schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}
"""
