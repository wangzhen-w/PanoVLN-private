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


def plan_write_prompt(
    *,
    episode_id: Any,
    profile: str,
    action_summary: Mapping[str, Any],
    trajectory_metadata: Mapping[str, Any],
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

Quality rules:
- Preserve the real route order. Do not invent a different room sequence, endpoint, or staircase direction.
- If trajectory metadata says `vertical_motion` is `ascending`, the route goes up overall; if it says `descending`, the route goes down overall. Treat that as a hard constraint when describing stairs.
- Mention major route choices, room/doorway/hallway/stair transitions, and the final stopping anchor when they are visible.
- Do not force every heading adjustment into left/right language. Around curved stairs, landings, and tight alignments, use neutral phrases like "follow the stairs/landing/hallway" unless there is a clear branch choice.
- Be conservative with stair endpoints. Say "top of the stairs" or "bottom of the stairs/landing" only when the FINAL row clearly shows that exact terminal position. If the final observation is on an intermediate stair landing, or if more stairs remain visible beyond the stop, describe it as a "stair landing near the railing/window" rather than top or bottom.
- Use landmarks only when they are visible and useful for navigation. Prefer generic but correct wording over unsupported specific colors or objects.
- At the final stop, avoid generic "room" or "doorway" wording when the space type is visually clear. If the endpoint is clearly a bathroom, kitchen, bedroom, laundry room, office, balcony, or stair landing, name that space and include one visible anchor inside or beside it.
- Do not describe every low-level action; do not give degrees, coordinates, or frame numbers.
- Use left/right for turns and route choices. Avoid unsupported claims like "the sofa is on your left" unless the evidence clearly supports it.
- The endpoint must be explicit enough to know where to stop.
- The final instruction must preserve the `stop_condition` exactly enough that the follower knows where to stop; include the word "stop" or "wait" in the endpoint sentence.

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

Audit criteria:
- Every visual or spatial claim in the instruction must be supported by the evidence.
- The route must preserve visible turns, space transitions, stairs, and final stop.
- The endpoint must not be replaced by a different doorway, room, furniture item, landing, or stair position.
- Check stair endpoints especially carefully: "top", "bottom", and "landing" are not interchangeable. If the final view is an intermediate landing or shows stairs continuing beyond the stop, a claim that the route stops at the top or bottom is a grounding error.
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

Repair rules:
- Keep only visual/spatial claims supported by evidence.
- Preserve route order and final stop.
- Preserve the up/down direction given by `vertical_motion` when stairs are described.
- Be conservative with stair endpoints: replace unsupported "top" or "bottom" claims with visible anchors such as "on the stair landing near the railing/window" when the final stop is not clearly the top or bottom.
- Replace generic endpoint wording with the visible room type or endpoint landmark when it is clear from the final evidence.
- Do not mention images, frames, rows, sheets, action IDs, degrees, or coordinates.
- If a detail is uncertain, remove or generalize it instead of making it more specific.
- Include the word "stop" or "wait" in the endpoint sentence.

Return exactly this JSON schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}
"""
