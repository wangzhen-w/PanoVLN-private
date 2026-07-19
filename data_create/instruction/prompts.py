"""Prompts for trajectory-grounded instruction generation."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping


SYSTEM_JSON = (
    "You are a careful VLN navigation-instruction annotator. Use only the "
    "trajectory evidence provided in this request. Do not copy or infer from "
    "any source instruction. Do not reveal reasoning. Output only valid JSON. /no_think"
)


def style_requirements(profile: str) -> str:
    if profile == "dense":
        return (
            "Write a natural instruction with enough intermediate landmarks for a "
            "long or ambiguous panoramic route. Usually 3-6 sentences. It may be "
            "conversational, but it must remain concise and executable."
        )
    return (
        "Write a compact human navigation instruction. Usually 1-3 sentences. "
        "Include only landmarks and turns needed to disambiguate the route."
    )


def endpoint_fact_prompt(
    *,
    episode_id: Any,
    profile: str,
    action_summary: Mapping[str, Any],
    trajectory_metadata: Mapping[str, Any],
) -> str:
    schema = {
        "final_location_type": "inside_room|threshold_or_doorway|stair_landing|hallway_or_corridor|near_object|outdoor_or_balcony|uncertain",
        "safe_stop_phrase": "short publishable phrase for where to stop",
        "forward_view_anchor": "main object/space visible straight ahead at FINAL",
        "left_view_anchor": "main object/space visible left of FINAL, or empty",
        "right_view_anchor": "main object/space visible right of FINAL, or empty",
        "back_view_anchor": "main object/space behind FINAL, or empty",
        "nearby_stop_anchors": ["1-3 visible objects/spaces that identify the stopping area"],
        "avoid_endpoint_claims": ["wrong endpoint, wrong facing, over-specific color/room-entry claims to avoid"],
        "confidence": "high|medium|low",
    }
    return f"""
You are extracting the final stopping facts for a VLN route.
You will receive five labeled FINAL STOP views from exactly one physical observation: the true final STOP. They are ordered as FINAL FORWARD, FINAL FORWARD-DOWN, FINAL LEFT, FINAL RIGHT, FINAL BACK. Use these FINAL views as the authority for the endpoint. Earlier route context is not shown here.

Episode: {episode_id}
Instruction profile: {profile}

Action-derived constraints for the full route:
{json.dumps(action_summary, ensure_ascii=False, indent=2)}

Trajectory metadata from geometry/GT, when available:
{json.dumps(trajectory_metadata or {{}}, ensure_ascii=False, indent=2)}

Rules:
- Describe only the final stopping area, not the full route.
- The FINAL FORWARD view is what the agent is facing at STOP. Objects in FINAL LEFT/RIGHT/BACK may be "nearby" or "visible", but they are not valid "facing" anchors and should not replace a clear forward stop anchor.
- Use "threshold" or "entrance" only when the stop is physically at a doorway, passage boundary, stair edge, or room transition. Do not call it a threshold merely because a side window, side door, wall edge, or door frame is visible.
- For stair endpoints, prefer "on the staircase landing near ..." over "at the top/bottom of the stairs" unless the FINAL observation proves that no stairs continue beyond that point. When uncertain, never use top/bottom.
- If the final stop is near stairs, railings, shutters, or a landing but not clearly the terminal top/bottom, avoid "at the bottom", "at the top", "lower floor", or "lower level" in the stop phrase.
- If the final stop is at a balcony or sliding glass door, distinguish the balcony/railing from an indoor sofa or TV seen in side/back views.
- If the final stop is in a hallway with multiple doors, avoid unsupported door colors. Prefer "at the end of the hall near..." unless the forward door color is unmistakable.
- If FORWARD shows a distinctive wall/object and LEFT/RIGHT shows a window, door, or side room, keep the forward anchor in the stop phrase; do not turn the side view into the destination.
- If the stop is already inside a room and FORWARD shows a close wall/object/furniture anchor, describe it as "near/facing/at the wall/object" rather than as a room threshold.
- Use material, color, and object-category words only when they are unmistakable in the FINAL FORWARD or FORWARD-DOWN views. If a shelf/display/cabinet/counter color or material is ambiguous, use a neutral noun such as "shelving/display unit/cabinet/counter" and pair it with a second stable nearby anchor.
- Prefer a two-anchor safe stop phrase when a single object category is ambiguous or repeated nearby, e.g. an object plus nearby furniture/wall/room area. Do not rely on an uncertain material adjective to identify the stop.
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
            "safe_stop_phrase": "specific safe phrase for the true final STOP",
            "forward_view_anchor": "main object/space visible straight ahead in the first FORWARD panel",
            "left_view_anchor": "main object/space visible in the LEFT panel, or empty",
            "right_view_anchor": "main object/space visible in the RIGHT panel, or empty",
            "back_view_anchor": "main object/space visible in the BACK panel, or empty",
            "nearby_stop_anchors": ["2-4 stable anchors near the stop"],
            "avoid_endpoint_claims": ["wrong endpoint, wrong facing, or side-view destination claims to avoid"],
            "confidence": "high|medium|low",
        },
    }
    return f"""
You are auditing extracted endpoint facts for one VLN final stop.
You will receive the same five labeled FINAL STOP views, ordered as FINAL FORWARD, FINAL FORWARD-DOWN, FINAL LEFT, FINAL RIGHT, FINAL BACK.

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
- Verify that `left_view_anchor`, `right_view_anchor`, and `back_view_anchor` describe their own panels.
- Verify that `safe_stop_phrase` names the true stopping area. It may use nearby side anchors only as context, but it must not turn a side-view window, doorway, room, or object into the destination when the FORWARD/FORWARD-DOWN panels show a clearer stop anchor.
- Verify that material/color/object-category words in `safe_stop_phrase` are unmistakable. If not, replace them with neutral wording and add a second stable nearby anchor rather than guessing a material or color.
- If the facts are correct, return them unchanged with `passed=true`.
- If any anchor or stop phrase is wrong, return corrected endpoint facts with `passed=false` and concise problems.
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
        "route_steps": [
            {
                "order": 1,
                "movement": "natural-language route segment",
                "visual_anchor": "stable visible landmark or spatial boundary",
                "confidence": "high|medium|low",
            }
        ],
        "stop_condition": "specific endpoint phrase",
        "uncertain_or_avoid": ["claims that should not be made"],
        "final_instruction": "the instruction to publish",
    }
    return f"""
You will receive three visual evidence sheets, in this order:
1. START: the start and early translated positions.
2. ROUTE: selected translated positions along the path. Each row is a physical waypoint; columns show left, forward, right, and back views.
3. ENDPOINT: the final approach and terminal view. The row labeled FINAL is the true stopping observation.

The route is panoramic VLN: write directions that a follower can execute from the start to the final stop. The labels are evidence labels only; do not mention images, frames, rows, sheets, or action IDs in the final instruction.

Episode: {episode_id}
Instruction profile: {profile}
Style requirement: {style_requirements(profile)}

Action-derived constraints:
{json.dumps(action_summary, ensure_ascii=False, indent=2)}

Trajectory metadata from geometry/GT, when available:
{json.dumps(trajectory_metadata or {{}}, ensure_ascii=False, indent=2)}

Endpoint facts extracted from the FINAL STOP evidence:
{json.dumps(endpoint_facts or {{}}, ensure_ascii=False, indent=2)}

Quality rules:
- Preserve the real route order. Do not invent a different room sequence, endpoint, or staircase direction.
- If trajectory metadata says `vertical_motion` is `ascending`, the route goes up overall; if it says `descending`, the route goes down overall. Treat that as a hard constraint when describing stairs.
- Mention major route choices, room/doorway/hallway/stair transitions, and the final stopping anchor when they are visible.
- Do not force every heading adjustment into left/right language. Around curved stairs, landings, and tight alignments, use neutral phrases like "follow the stairs/landing/hallway" unless there is a clear branch choice.
- Be conservative with stair endpoints. Say "top of the stairs" or "bottom of the stairs/landing" only when the FINAL row clearly shows that exact terminal position. If the final observation is on an intermediate stair landing, or if more stairs remain visible beyond the stop, say "descend/ascend only to the stair landing near the railing/window and stop" rather than giving an unrestricted "go down/up the stairs" command.
- Use landmarks only when they are visible and useful for navigation. Prefer generic but correct wording over unsupported specific colors or objects.
- Use left/right object relations only when they are clearly supported and useful for choosing the route. If a landmark is useful but its side is ambiguous, mention the landmark without a side.
- At the final stop, avoid generic "room" or "doorway" wording when the space type is visually clear. If the endpoint is clearly a bathroom, kitchen, bedroom, laundry room, office, balcony, or stair landing, name that space and include one visible anchor inside or beside it.
- Treat the endpoint facts as the authority for the final sentence. Preserve the `safe_stop_phrase` closely unless it conflicts with visible ENDPOINT evidence.
- Do not use any endpoint or final-facing claim listed in `avoid_endpoint_claims`.
- Do not overstate entering a final room. If the final observation is at a threshold, doorway, junction, landing, or just outside a room, write "stop at/near the entrance/threshold" rather than "enter the room".
- Avoid final-facing claims unless the ENDPOINT forward view clearly supports that exact object or wall. Prefer "stop near/by/at..." or "with ... visible" over "turn to face..." when the object may be side/back/ambiguous.
- Do not write contradictory final orientation chains such as "turn to face X, then stop facing Y"; choose the single endpoint anchor that best identifies where to stop.
- Do not describe every low-level action; do not give degrees, coordinates, or frame numbers.
- Use left/right for turns and route choices. Avoid unsupported claims like "the sofa is on your left" unless the evidence clearly supports it.
- The endpoint must be explicit enough to know where to stop.
- The final instruction must preserve the `stop_condition` exactly enough that the follower knows where to stop; include the word "stop" or "wait" in the endpoint sentence.

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
        "movement_summary": "short route facts for this segment only",
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
        "stable_landmarks": ["1-2 useful visible landmarks"],
        "stop_anchor_if_final": "specific final stop anchor, empty for non-final segments",
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
{json.dumps(trajectory_metadata or {{}}, ensure_ascii=False, indent=2)}

Endpoint facts from the FINAL STOP evidence:
{json.dumps(endpoint_facts or {{}}, ensure_ascii=False, indent=2)}

Rules:
- Do not write the final navigation instruction.
- Extract only route facts visible in this segment.
- Treat the segment action constraints as hard evidence about physical motion.
- Do not infer that the agent walked through a hallway/room just because the view rotates. When a transition is marked `mostly_orientation_or_alignment` or has only 1-2 forward actions, describe it as local turning/alignment unless a real doorway or space transition is clearly visible.
- Separate physical movement from heading adjustment: "turn to face/align with" is acceptable in facts, but do not turn that into "walk down/past/through" unless there is substantial forward movement.
- Non-final segments must not say stop, wait, finish, or destination.
- Keep landmarks sparse: only 1-2 stable anchors that help route execution.
- Preserve visible turns, doorways, hallway/room transitions, and stair movement.
- Fill `ordered_route_events` with the important route events in this segment in travel order. This list may contain 0-4 events; use multiple events when a long segment includes multiple true choices or space transitions.
- Mark `keep_for_instruction=true` for events a follower needs to choose the correct path, confirm progress through a long route, or avoid stopping/turning at the wrong place.
- Fill `must_keep_decision_boundaries` with visually grounded boundaries that should survive into the final instruction: doorways, arches, stair landings, hallway junctions, turns around kitchen islands/bar counters, room boundaries, or indoor/outdoor transitions. Leave it empty for duplicate context or pure alignment.
- Do not mark pure heading/facing adjustments as keep-for-instruction when they only point toward a side object, window, table, desk, or other non-destination anchor. Convert them into a spatial transition only if the evidence shows real movement toward the route.
- If the full action constraints show a large initial alignment before the first forward movement, include an event for it only when the segment evidence shows a real first route choice such as aligning to a hallway, doorway, stair, or room entrance.
- If action constraints show major turns between translated positions, do not automatically publish the turn degrees, but check the nearby segment views for a grounded turn/space-transition event.
- If the evidence revisits the same landmark in nearby rows, state the useful route choice once; do not describe a loop unless the segment action constraints show meaningful translation away and back.
- For the final segment, endpoint facts are the authority. Set `stop_anchor_if_final` to match `safe_stop_phrase` unless this segment's final evidence directly contradicts it.
- In the final segment, do not make `ordered_route_events` point the follower toward a side/back/non-endpoint anchor listed in `avoid_endpoint_claims` or different from `safe_stop_phrase`. If a final turn only changes facing near the stop, omit it or describe it as local alignment in `movement_summary`, not as a keep-for-instruction destination.
- For the final segment, describe whether the stop appears inside a room, at a threshold/doorway, on a landing, or near an object. Use threshold wording only for a real doorway, passage boundary, stair edge, or room transition; if endpoint facts identify a clear nearby object/wall, prefer that object/wall over generic threshold wording.
- In non-final segments, `do_not_claim` means "do not claim this happened within this segment"; it is not a global ban if a later segment or the endpoint clearly shows that the route reaches that place.
- Fill `uncertain_side_claims` when a left/right/side relation is not stable from the segment evidence, especially near final rooms, cluttered junctions, overlapping segment boundaries, panoramic wrap-around, or when the target is more safely described as "ahead", "at the end", "through the doorway", or by landmark only.
- Add wrong side doors/rooms, over-strong left/right entry claims, and unsupported final facing claims to do_not_claim.
- If a detail is uncertain, put it in do_not_claim or lower confidence.
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
        "route_steps": [
            {
                "order": 1,
                "movement": "natural-language route segment",
                "visual_anchor": "stable visible landmark or spatial boundary",
                "confidence": "high|medium|low",
            }
        ],
        "stop_condition": "specific endpoint phrase",
        "covered_route_events": ["brief labels for important ordered segment events preserved in final_instruction"],
        "covered_decision_boundaries": ["brief labels for must_keep_decision_boundaries preserved in final_instruction"],
        "omitted_route_events": [
            {
                "event": "important event from segment facts that was omitted",
                "reason": "why omission does not hurt route execution",
            }
        ],
        "neutralized_side_claims": [
            "uncertain left/right/side claims that were rewritten with neutral wording"
        ],
        "uncertain_or_avoid": ["claims that should not be made"],
        "final_instruction": "the instruction to publish",
    }
    return f"""
Write the final VLN instruction by merging ordered segment facts into one natural route instruction.
You will also receive START, ROUTE overview, and ENDPOINT evidence sheets. Use the segment facts to preserve route order and local transitions. Use ENDPOINT evidence as the authority for the final stopping condition.

Episode: {episode_id}
Instruction profile: {profile}
Style requirement: {style_requirements(profile)}

Segment facts:
{json.dumps(segment_facts, ensure_ascii=False, indent=2)}

Action-derived constraints:
{json.dumps(action_summary, ensure_ascii=False, indent=2)}

Trajectory metadata from geometry/GT, when available:
{json.dumps(trajectory_metadata or {{}}, ensure_ascii=False, indent=2)}

Endpoint facts extracted from the FINAL STOP evidence:
{json.dumps(endpoint_facts or {{}}, ensure_ascii=False, indent=2)}

Merge rules:
- Do not mechanically concatenate segment text. Write like a human navigation instruction.
- Preserve the real route order and all must-keep route choices.
- Before writing the final instruction, make a coverage pass over `ordered_route_events` from all segments. Preserve the events with `keep_for_instruction=true` unless two adjacent events are redundant or one is too uncertain to describe safely.
- Make a second coverage pass over `must_keep_decision_boundaries`. A long route must keep the decision boundaries that affect execution, such as stairs/landings, arches/doorways, hallway junctions, turns around islands/bar counters, indoor/outdoor transitions, and final-room entrances. List kept items in `covered_decision_boundaries`.
- For long routes, each physical segment should usually contribute at least one route event unless it is only local alignment or duplicate context.
- Do not compress away the first real route choice just because the route soon reaches a staircase, hallway, or room. If the first segment requires aligning toward or entering that space and the evidence supports it, mention that start transition naturally.
- Do not flatten a route with a visible turn through a hallway, doorway, stair, or room boundary into a direct move to a later landmark when that transition helps the follower choose the path.
- Do not replace a multi-space route with a direct jump to a later broad room name. If segment facts include intermediate spaces or boundaries such as a hallway, dining area, landing, arch, doorway, kitchen island/bar, office area, or lounge that help the follower localize progress, keep the most useful ones in route order.
- Coverage must not override endpoint correctness. Omit or rewrite any final-segment event that aims the follower toward a non-endpoint anchor, side/back view object, window, table, desk, or landmark listed in `avoid_endpoint_claims`.
- Avoid phrasing like "turn toward X and stop near Y" when X and Y are different endpoint anchors. Use the route event for progress, then a single endpoint phrase from `safe_stop_phrase`.
- If a kept event is omitted, list it in `omitted_route_events` with a concrete reason.
- Treat final-segment `do_not_claim` items as hard negative constraints for the endpoint. Treat non-final `do_not_claim` items as local segment constraints: they prevent saying that event happened too early, but they may be overridden by later positive segment facts or ENDPOINT evidence.
- Treat every segment's `uncertain_side_claims` as a warning: do not publish those left/right/side words unless the overview/endpoint evidence independently makes that side relation unambiguous. Prefer neutral route wording such as "enter the room at the end", "go through the doorway", "continue into the next room", "follow the hall", or landmark-only wording.
- Fill `neutralized_side_claims` with uncertain side claims you rewrote or omitted.
- Merge duplicate entry/exit anchors across segment boundaries.
- Segment sheets may overlap. If adjacent segments describe the same physical stair descent, hallway traversal, doorway crossing, or local alignment, merge it once instead of repeating it as two separate moves.
- Condense local alignment or small repositioning into a simple "turn/align toward..." phrase, or omit it if it is not needed for route execution.
- Do not preserve landmarks from segment facts merely because they were observed; keep only anchors that help choose the route, confirm progress, or identify the stop.
- Avoid unnecessary left/right side claims for non-critical landmarks in long routes. If the side is ambiguous, keep the landmark but drop the side relation.
- When merging the final segment, use its `stop_anchor_if_final` and ENDPOINT evidence to decide whether to say "inside", "at the entrance/threshold", or "on the landing"; use threshold/landing wording only for a real doorway, passage boundary, stair edge, or room transition, not for a side-view window/door near an otherwise clear in-room stop anchor.
- The endpoint facts are the strongest endpoint authority. Preserve `safe_stop_phrase` closely in the final sentence, and do not use any claim listed in `avoid_endpoint_claims`.
- If non-final segment facts or route overview conflict with endpoint facts about the final stop, trust endpoint facts.
- For a stair endpoint, bound the movement if needed: "descend only to the landing and stop" is safer than "descend the staircase" when the final stop is not the bottom.
- For stair endpoint facts with `final_location_type=stair_landing`, prefer "on the stair/staircase landing" instead of top/bottom wording.
- If endpoint facts list top/bottom/lower-floor claims in `avoid_endpoint_claims`, do not say "at the bottom", "at the top", "lower floor", or "lower level" near the final stop; use the visible landing/wall/window/railing anchor instead.
- Avoid final left/right entry claims in cluttered junctions or with multiple visible doors/rooms. Use neutral phrases like "continue to the bathroom entrance" unless a side choice is unambiguous and supported.
- Avoid final-facing claims unless ENDPOINT forward evidence uniquely supports them. Do not write "turn to face X, then stop facing Y"; use "stop near X" or "with X visible" instead.
- If the final stop phrase uses a shelf/display/cabinet/counter/desk, avoid uncertain color, material, or size adjectives unless endpoint facts explicitly make them essential. A neutral noun plus a second stable anchor is safer.
- Only the final sentence may use stop/wait/finish/destination.
- Keep concise profile to 1-3 sentences; dense profile to 3-6 sentences.
- Do not mention segments, cards, sheets, images, frames, rows, action IDs, degrees, or coordinates.
- Use landmarks only when supported by segment facts or visible START/ROUTE/ENDPOINT evidence.
- If segment facts conflict with ENDPOINT evidence about where to stop, trust ENDPOINT evidence.
- The final instruction must include the word "stop" or "wait" in the endpoint sentence.

Return exactly this JSON schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}
"""


def candidate_judge_prompt(
    *,
    episode_id: Any,
    profile: str,
    action_summary: Mapping[str, Any],
    trajectory_metadata: Mapping[str, Any],
    endpoint_facts: Mapping[str, Any],
    candidates: Any,
) -> str:
    schema = {
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
You will receive START, ROUTE, ENDPOINT, and FINAL STOP visual evidence. The route is panoramic navigation: choose the instruction that best helps a follower execute the true route and stop at the true final location.

Episode: {episode_id}
Instruction profile: {profile}

Candidate instructions:
{json.dumps(candidates, ensure_ascii=False, indent=2)}

Action-derived constraints:
{json.dumps(action_summary, ensure_ascii=False, indent=2)}

Trajectory metadata:
{json.dumps(trajectory_metadata or {{}}, ensure_ascii=False, indent=2)}

Endpoint facts from FINAL STOP evidence:
{json.dumps(endpoint_facts or {{}}, ensure_ascii=False, indent=2)}

Selection criteria:
- Prefer the candidate whose described route order, major turns, space transitions, and final stop are most faithful to the visual evidence.
- For long segmented routes, compare each candidate against its `segment_facts.ordered_route_events`, `covered_route_events`, and `omitted_route_events`. Prefer candidates that preserve more supported `keep_for_instruction=true` events without becoming a low-level action list.
- For long segmented routes, also compare `must_keep_decision_boundaries` against `covered_decision_boundaries`. Penalize candidates that omit a doorway, arch, stair landing, hallway junction, kitchen island/bar bypass, indoor/outdoor transition, or final-room entrance when it affects execution.
- Penalize candidates that omit the first real route choice, a cross-room/hallway/stair transition, or the final approach transition when that event is supported by visual evidence and affects execution.
- Do not choose a shorter candidate when it drops intermediate spaces or route events that help execute a long route. Brevity is a tie-breaker only after route coverage and endpoint accuracy are comparable.
- Penalize candidates that preserve a final-segment route event by directing the follower toward a non-endpoint object, side/back anchor, window, table, or desk instead of the `safe_stop_phrase`.
- Penalize candidates whose endpoint uses an unsupported material/color adjective for a shelf/display/cabinet/counter when a neutral noun plus a second anchor would be safer.
- Penalize candidates that publish a left/right/side relation listed in `uncertain_side_claims` or `do_not_claim`; prefer neutral wording when the route can remain executable without that side word.
- Endpoint correctness is more important than style. Penalize wrong final room, wrong threshold/landing, over-shooting into a room, stopping too early, or final facing claims not supported by the FINAL evidence.
- In the FINAL STOP evidence image, the first panel is the true forward view. Penalize candidates that treat LEFT/RIGHT/BACK side-view anchors as the final facing direction or destination when the forward stop anchor is clear.
- Prefer candidates whose final sentence preserves `safe_stop_phrase` closely. Penalize candidates that replace it with a nearby side-view door, window, room, or object.
- Penalize hallucinated or over-specific landmarks. Missing minor details is acceptable when the route remains executable.
- Do not prefer a longer instruction just because it is longer. Prefer concise, natural, executable language.
- If no single candidate is fully best, but one candidate has the better route transitions and another has the better endpoint, set `needs_repair=true` and provide one corrected full instruction that combines the correct route order with `safe_stop_phrase`.
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
    endpoint_facts: Mapping[str, Any],
    route_plan: Mapping[str, Any],
    instruction: str,
) -> str:
    schema = {
        "passed": True,
        "severity": "none|minor|critical",
        "problems": ["grounding or route problems, empty if none"],
        "corrected_instruction": "empty if no correction is needed",
    }
    return f"""
You are independently auditing one generated VLN instruction against the same START, ROUTE, and ENDPOINT evidence sheets. You did not write the instruction.

Episode: {episode_id}
Instruction profile: {profile}
Instruction to audit:
{instruction}

Action-derived constraints:
{json.dumps(action_summary, ensure_ascii=False, indent=2)}

Trajectory metadata:
{json.dumps(trajectory_metadata or {}, ensure_ascii=False, indent=2)}

Endpoint facts from the FINAL STOP evidence:
{json.dumps(endpoint_facts or {}, ensure_ascii=False, indent=2)}

Route plan and segment facts:
{json.dumps(route_plan or {}, ensure_ascii=False, indent=2)}

Audit criteria:
- Every visual or spatial claim in the instruction must be supported by the evidence.
- The route must preserve visible turns, space transitions, stairs, and final stop.
- For long segmented routes, check whether the instruction skips an intermediate space transition that is needed to execute the route. Do not fail omissions that do not affect route choice.
- For long segmented routes, use `ordered_route_events`, `covered_route_events`, and `omitted_route_events` to check whether the final instruction dropped a supported route choice that a follower would need. Missing minor or redundant events is acceptable.
- For long segmented routes, use `must_keep_decision_boundaries` and `covered_decision_boundaries` to check whether the instruction dropped a necessary doorway, arch, stair landing, hallway junction, kitchen island/bar bypass, indoor/outdoor transition, or final-room entrance.
- Flag a route issue if the instruction starts after the first real route choice, or flattens a visible turn/doorway/hallway/stair transition into a direct move to a later landmark, when segment facts show that transition is needed for execution.
- Flag a route issue if a long route instruction drops all mention of a useful intermediate space from a segment, such as a hallway, dining area, landing, office area, or lounge, and that omission makes the path less executable.
- Flag a route issue if the instruction uses a left/right entry or side-of-object claim from `uncertain_side_claims` or `do_not_claim` when neutral wording would be safer.
- Also flag over-coverage: an instruction should not add a final turn/facing/toward clause for a non-endpoint object merely to preserve a segment event. Endpoint facts remain the authority for the stop sentence.
- The endpoint must not be replaced by a different doorway, room, furniture item, landing, or stair position.
- The final stop phrase must be consistent with `safe_stop_phrase`, `forward_view_anchor`, and `nearby_stop_anchors`.
- Fail if the final sentence replaces `safe_stop_phrase` with a nearby LEFT/RIGHT/BACK side-view door, window, room, or object.
- Fail if the instruction uses an endpoint or final-facing claim listed in `avoid_endpoint_claims`.
- Check stair endpoints especially carefully: "top", "bottom", and "landing" are not interchangeable. If the final view is an intermediate landing or shows stairs continuing beyond the stop, a claim that the route stops at the top or bottom is a grounding error.
- If endpoint facts list top/bottom/lower-floor claims in `avoid_endpoint_claims`, fail any final stop sentence that says "at the bottom", "at the top", "lower floor", or "lower level".
- If the instruction says to descend/ascend a staircase and the final stop is an intermediate landing, it must explicitly say to stop on that landing; otherwise treat the instruction as potentially overshooting the destination.
- If the endpoint is at a threshold, doorway, junction, or landing, fail an instruction that says to enter fully into the room unless the final evidence clearly shows the agent inside that room.
- Fail contradictory final orientation chains such as "turn to face X, then stop facing Y". Also fail final "face/toward" claims when the named object is only side/back/ambiguous in the ENDPOINT evidence; prefer a corrected "stop near/by/at..." phrase.
- Fail uncertain color/material/size adjectives in the final stop sentence for shelf/display/cabinet/counter/desk anchors when endpoint facts provide a safer neutral stop phrase.
- Treat a generic endpoint such as "the room", "a doorway", or "the first doorway" as insufficient on a long or complex route when the endpoint clearly shows a recognizable room type or landmark.
- Do not require exhaustive low-level action detail. Missing unimportant objects is acceptable.
- Fail critically for wrong turn direction, wrong endpoint, hallucinated landmark, wrong up/down stairs, or a vague endpoint on a complex route.
- Treat `vertical_motion` as a hard physical constraint for up/down stairs; do not override it based on visual ambiguity.
- If a concise correction can fix the instruction, provide it. Otherwise leave corrected_instruction empty.

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
Repair the generated VLN instruction using the START, ROUTE, and ENDPOINT evidence.

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
- Keep only visual/spatial claims supported by evidence.
- Preserve route order and final stop.
- Preserve important intermediate space transitions when they affect route execution, while omitting minor transitions that do not help navigation.
- For long segmented routes, repair against `ordered_route_events`: restore supported events with `keep_for_instruction=true` that were dropped from the failed instruction, especially the first real route choice and the final approach transition.
- For long segmented routes, repair against `must_keep_decision_boundaries`: restore necessary doorways, arches, stair landings, hallway junctions, island/bar bypasses, indoor/outdoor transitions, and final-room entrances that were dropped.
- Keep the repaired instruction natural; do not list every event if several can be merged without changing the executable route.
- Remove or neutralize left/right/side relations listed in `uncertain_side_claims` or `do_not_claim`. Use "through the doorway", "at the end", "ahead", "into the next room", or landmark-only wording when that preserves the route without a risky side word.
- If a preserved final route event points toward a non-endpoint anchor, remove that event or rewrite it as progress context; keep only one endpoint phrase based on `safe_stop_phrase`.
- If the failed instruction over-compressed a long route, restore the most useful omitted intermediate spaces in route order before the endpoint.
- If the endpoint uses an uncertain material/color adjective, replace it with a neutral object noun and a second stable anchor from endpoint facts.
- Preserve the endpoint facts, especially `safe_stop_phrase`, unless the visual evidence directly contradicts them.
- Remove endpoint/facing claims listed in `avoid_endpoint_claims`.
- If the failed instruction replaced `safe_stop_phrase` with a nearby side-view door, window, room, or object, restore the safe stop phrase and keep the side-view anchor only as optional context.
- Preserve the up/down direction given by `vertical_motion` when stairs are described.
- Be conservative with stair endpoints: replace unsupported "top" or "bottom" claims with visible anchors such as "on the stair landing near the railing/window" when the final stop is not clearly the top or bottom.
- If endpoint facts list top/bottom/lower-floor claims in `avoid_endpoint_claims`, remove those terminal stair words and stop at the visible landing/wall/window/railing anchor.
- If the final stop is on a stair landing, explicitly bound the movement: "descend/ascend only to the landing and stop" when needed.
- If the final stop is at a doorway, threshold, junction, or just outside a room, replace "enter the room and stop" with "stop at/near the entrance/threshold".
- Remove or generalize final-facing claims unless the ENDPOINT forward view uniquely supports them. Replace "turn to face X, then stop facing Y" with a single endpoint phrase such as "stop near X" or "stop with Y visible".
- If deterministic QA reports `final_turn_to_face_nonendpoint`, remove or rewrite the final "turn to face..." clause so it points only to the endpoint described by `safe_stop_phrase`.
- If deterministic QA reports `conflicting_final_face_then_stop_anchor`, do not keep a final "turn to face X and stop near Y" chain. Rewrite it as route progress plus one stop phrase, e.g. "turn/enter into the seating area and stop near Y" or simply "stop near Y".
- Replace generic endpoint wording with the visible room type or endpoint landmark when it is clear from the final evidence.
- Do not mention images, frames, rows, sheets, action IDs, degrees, or coordinates.
- If a detail is uncertain, remove or generalize it instead of making it more specific.
- Include the word "stop" or "wait" in the endpoint sentence.

Return exactly this JSON schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}
"""
