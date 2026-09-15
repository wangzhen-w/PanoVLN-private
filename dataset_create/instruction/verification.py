"""Stage 5: observe each clean segment once, then review its complete text."""

from __future__ import annotations

import json

from dataset_create.instruction.client import media_item, text_item, video_items
from dataset_create.instruction.language import LanguageContractError, relevant_text
from dataset_create.instruction.prompts import OBSERVE_SYSTEM, VERIFY_SYSTEM
from dataset_create.instruction.segmentation import camera_motion


def segment_info(segment, evidence):
    indices = evidence["frame_state_indices"]
    if (len(indices) < 2 or indices != sorted(set(indices)) or
            indices[0] != segment.start or indices[-1] != segment.end or
            len(evidence["clean_frames"]) != len(indices) or len(evidence["cameras"]) != len(indices)):
        raise ValueError("Incomplete segment review interval")
    return {"segment_id": segment.segment_id, "assigned_decision_ids": segment.decision_ids,
            "terminal": segment.terminal, "measured_camera_motion": camera_motion(evidence["cameras"]),
            "motion_note": "Each row describes the change between the two numbered video frames, not the bearing of an object."}


def observation_content(segment, evidence, media_mode):
    """The observer never receives the instruction, route overlay or branch answers."""
    content = [text_item(json.dumps(segment_info(segment, evidence)))]
    content += video_items(evidence["clean_video"], evidence["clean_frames"], media_mode,
                           "Complete clean core route, in chronological order:")
    for decision in evidence["decisions"]:
        index = (evidence["frame_state_indices"].index(decision["state_index"])
                 if decision["state_index"] in evidence["frame_state_indices"] else None)
        content += [text_item(f"Clean approach compass for {decision['decision_id']}, core video frame {index}; "
                              "null means before this core clip."), media_item(decision["clean_compass"])]
    if segment.terminal:
        content += [text_item("Actual stopping position:"), media_item(evidence["stop"]["actual"]["image"])]
    return content


def comparison_content(final, segment, evidence, observation):
    # Removing a choice clause allowed one turn to justify two different claims.
    ids = list(dict.fromkeys(sid for clause in final["clauses"] for sid in clause["segment_ids"]))
    index = ids.index(segment.segment_id)
    data = {**segment_info(segment, evidence), "independent_observation": observation,
            "current_local_instruction": relevant_text(final, segment.segment_id),
            "previous_local_instruction": relevant_text(final, ids[index-1]) if index else None,
            "next_local_instruction": relevant_text(final, ids[index+1]) if index+1 < len(ids) else None}
    return [text_item(json.dumps(data, ensure_ascii=False))]


def review_passed(response, minimum_confidence):
    if not isinstance(response, dict) or response.get("status") not in {"pass", "fail", "insufficient"}:
        raise ValueError("Invalid verifier status")
    confidence = response.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("Invalid verifier confidence")
    if not isinstance(response.get("reason"), str) or not response["reason"]:
        raise ValueError("Missing verifier evidence")
    if response.get("issue_type") not in {"none", "language", "visual", "segmentation"}:
        raise ValueError("Invalid verifier issue_type")
    return response["status"] == "pass" and confidence >= minimum_confidence


def decision_passed(response, minimum_confidence):
    passed = review_passed(response, minimum_confidence)
    if any(not isinstance(response.get(key), bool) for key in ("path_matches", "choice_is_clear")):
        raise ValueError("Decision review must assess path consistency and choice clarity separately")
    return passed and response["path_matches"] and response["choice_is_clear"]


def observation_reliable(response, segment):
    if not isinstance(response.get("uncertain"), bool):
        raise ValueError("Observation must report uncertainty")
    for key in ("observed_route", "reason"):
        if not isinstance(response.get(key), str) or not response[key].strip():
            raise ValueError("Missing observation evidence: " + key)
    decisions = response.get("decisions")
    if not isinstance(decisions, list) or [d.get("decision_id") for d in decisions] != segment.decision_ids:
        raise ValueError("Observation must cover every assigned decision in order")
    for decision in decisions:
        for key in ("pre_choice_cues", "alternatives"):
            if not isinstance(decision.get(key), list) or any(not isinstance(v, str) for v in decision[key]):
                raise ValueError("Invalid observation cues: " + key)
    if segment.terminal:
        if not isinstance(response.get("stop_area"), str) or not response["stop_area"].strip():
            raise ValueError("Terminal observation must describe the actual stop area")
    elif response.get("stop_area") is not None:
        raise ValueError("Nonterminal observation must not invent a stop")
    return not response["uncertain"]


def comparison_passed(response, segment, minimum):
    decisions = response.get("decisions")
    if not isinstance(decisions, list) or [d.get("decision_id") for d in decisions] != segment.decision_ids:
        raise ValueError("Review must cover every assigned decision in order")
    passed = [decision_passed(d, minimum) for d in decisions]
    passed.append(review_passed(response.get("motion"), minimum))
    if segment.terminal:
        passed.append(review_passed(response.get("stop"), minimum))
    elif response.get("stop") is not None:
        raise ValueError("Nonterminal review must not invent a stop")
    return all(passed)


def _checked_response(client, stage, system, content, check):
    for attempt in range(3):
        response, key = client.complete(stage, system, content, attempt=attempt)
        try:
            passed = check(response)
            return response, key, passed
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            client.discard_response(key)
            if attempt == 2:
                raise LanguageContractError(f"Invalid {stage} schema after retries: {error}") from error
            content = content + [text_item("Return a valid JSON object. Contract error: " + str(error))]
    raise AssertionError("Unreachable")


def verify_segment(client, final, segment, evidence, settings):
    minimum = settings["minimum_confidence"]
    observation, observation_key, reliable = _checked_response(
        client, "observe_segment", OBSERVE_SYSTEM,
        observation_content(segment, evidence, client.settings["media_mode"]),
        lambda value: observation_reliable(value, segment))
    if reliable:
        response, key, _ = _checked_response(
            client, "verify_segment", VERIFY_SYSTEM,
            comparison_content(final, segment, evidence, observation),
            lambda value: comparison_passed(value, segment, minimum))
    else:
        issue = {"status": "insufficient", "confidence": 0., "issue_type": "visual", "reason": observation["reason"]}
        response = {"decisions": [{**issue, "decision_id": d, "path_matches": False, "choice_is_clear": False}
                                  for d in segment.decision_ids],
                    "motion": issue, "stop": issue if segment.terminal else None}
        key = None
    results = [{"kind": "decision", "decision_id": decision["decision_id"],
                "segment_id": segment.segment_id, "passed": decision_passed(decision, minimum),
                "observation": observation, "observation_request_key": observation_key,
                "response": decision, "request_key": key} for decision in response["decisions"]]
    for kind in ["motion"] + (["stop"] if segment.terminal else []):
        results.append({"kind": kind, "segment_id": segment.segment_id,
                        "passed": review_passed(response[kind], minimum),
                        "observation": observation, "observation_request_key": observation_key,
                        "response": response[kind], "request_key": key})
    return results


def repair_feedback(results):
    return [{"kind": r["kind"], "decision_id": r.get("decision_id"),
             "reason": r["response"]["reason"],
             "issue_type": r["response"].get("issue_type") if r["response"].get("issue_type") != "none" else "language",
             "review_task": "Check the reported error against the complete local route, measured motion and pre-choice cues."}
            for r in results if not r["passed"]]
