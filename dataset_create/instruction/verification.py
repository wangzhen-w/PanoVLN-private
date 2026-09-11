"""Stage 5: clean decision-region, arrival, and ordinary movement reviews."""

from __future__ import annotations

import json

import numpy as np

from dataset_create.instruction.client import media_item, text_item, video_items
from dataset_create.instruction.language import relevant_text
from dataset_create.instruction.prompts import DECISION_SYSTEM, OBSERVE_DECISION_SYSTEM, STOP_SYSTEM, TRANSIT_SYSTEM
from dataset_create.instruction.segmentation import yaw_degrees


def relative_camera_poses(cameras):
    headings = np.unwrap(np.radians([yaw_degrees(c["rotation_xyzw"]) for c in cameras]))
    return [{"frame": i,
             "heading_from_start_degrees": round(float(np.degrees(headings[i]-headings[0])), 1),
             "height_from_start_m": round(c["position"][1]-cameras[0]["position"][1], 2)}
            for i, c in enumerate(cameras)]


def decision_visual_content(evidence, media_mode):
    """Observe the clean route independently: no instruction or branch answers."""
    indices = evidence["review_state_indices"]
    if (len(indices) < 2 or indices != sorted(set(indices)) or
            indices[0] != evidence["review_start_state_index"] or
            indices[-1] != evidence["review_end_state_index"] or
            len(evidence["review_frames"]) != len(indices) or len(evidence["review_cameras"]) != len(indices)):
        raise ValueError("Incomplete decision-region review interval")
    content = [text_item(json.dumps({"relative_camera_pose_by_frame": relative_camera_poses(evidence["review_cameras"])})),
               text_item("Positive heading is right, negative is left; positive height is upward.")]
    content += video_items(evidence["review_video"], evidence["review_frames"], media_mode,
                           "Clean approach, choice and entry, in chronological order:")
    content += [text_item("Clean pre-choice compass:"), media_item(evidence["clean_compass"])]
    return content


def decision_content(final, segment_id, evidence, observation):
    clause = next(p["text"] for p in final["protected"] if p["decision_id"] == evidence["decision_id"])
    return [text_item(json.dumps({"independent_observation": observation}, ensure_ascii=False)),
            text_item(json.dumps({"choice_clause_from_final_instruction": clause,
                                  "instruction_context": relevant_text(final, segment_id)}, ensure_ascii=False))]


def stop_content(final, segment_id, evidence, media_mode):
    stop = evidence["stop"]
    protected = next(p for p in final["protected"] if p["stop"])
    content = [text_item(json.dumps({"relevant_final_instruction": relevant_text(final, segment_id),
                                    "stop_clause_from_final_instruction": protected["text"],
                                    "core_video_frame_indices": evidence["core_frame_indices"]}))]
    content += video_items(evidence["clean_video"], evidence["clean_frames"], media_mode,
                           "Clean actual arrival, ending at the true stopping position:")
    content += [text_item("Clean compass at the actual stop:"), media_item(stop["actual"]["image"])]
    return content


def motion_fragments(final, segment_id):
    ordinary = relevant_text(final, segment_id)
    for clause in final["protected"]:
        if clause["segment_id"] == segment_id:
            ordinary = ordinary.replace(clause["text"], "\n")
    return [part.strip(" .,\n") for part in ordinary.split("\n") if any(c.isalpha() for c in part)]


def motion_content(final, segment_id, evidence, media_mode):
    fragments = motion_fragments(final, segment_id)
    # Neutral pose observations distinguish translation from camera rotation.
    # These describe the observed core, without the author's route markings.
    motion = relative_camera_poses(evidence.get("cameras", []))
    content = [text_item(json.dumps({"ordinary_movement_fragments": fragments,
                                    "core_video_frame_indices": evidence["core_frame_indices"],
                                    "relative_camera_pose_by_frame": motion,
                                    "choices_and_stop_reviewed_separately": True}))]
    content += video_items(evidence["clean_video"], evidence["clean_frames"], media_mode,
                           "Clean local route. Locate the portions described by the supplied movement claims.")
    return content


def _validate_review(response, statuses):
    if response.get("status") not in statuses:
        raise ValueError("Invalid verifier status")
    confidence = response.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("Invalid verifier confidence")
    if not isinstance(response.get("reason"), str) or not response["reason"]:
        raise ValueError("Missing verifier evidence")
    if response.get("issue_type") not in {"none", "language", "visual", "segmentation"}:
        raise ValueError("Invalid verifier issue_type")


def review_passed(response, minimum_confidence):
    _validate_review(response, {"pass", "fail", "insufficient"})
    return response["status"] == "pass" and response["confidence"] >= minimum_confidence


def decision_passed(response, minimum_confidence):
    passed = review_passed(response, minimum_confidence)
    if any(not isinstance(response.get(key), bool) for key in ("path_matches", "choice_is_clear")):
        raise ValueError("Decision review must assess path consistency and choice clarity separately")
    return passed and response["path_matches"] and response["choice_is_clear"]


def observation_reliable(response):
    if not isinstance(response.get("uncertain"), bool):
        raise ValueError("Observation must report uncertainty")
    for key in ("observed_route", "reason"):
        if not isinstance(response.get(key), str) or not response[key].strip():
            raise ValueError("Missing observation evidence: " + key)
    for key in ("pre_choice_cues", "alternative_entrances"):
        if not isinstance(response.get(key), list) or any(not isinstance(v, str) for v in response[key]):
            raise ValueError("Invalid observation cues: " + key)
    return not response["uncertain"]


def _checked_response(client, stage, system, content, check):
    for attempt in range(3):
        response, key = client.complete(stage, system, content, attempt=attempt)
        try:
            passed = check(response)
            return response, key, passed
        except (ValueError, TypeError) as error:
            if attempt == 2:
                raise ValueError(f"Invalid {stage} schema after retries: {error}") from error
            content = content + [text_item("Return a valid JSON object. Contract error: " + str(error))]
    raise AssertionError("Unreachable")


def verify_segment(client, final, segment, evidence, settings):
    results = []
    minimum = settings["minimum_confidence"]
    mode = client.settings["media_mode"]
    for decision in evidence["decisions"]:
        # The observation's cache key is independent of instruction text. Local
        # rewrites reuse it, and cannot bias how the reviewer sees the video.
        observation, observation_key, reliable = _checked_response(
            client, "observe_decision", OBSERVE_DECISION_SYSTEM,
            decision_visual_content(decision, mode), observation_reliable)
        if reliable:
            response, key, passed = _checked_response(
                client, "verify_decision", DECISION_SYSTEM,
                decision_content(final, segment.segment_id, decision, observation),
                lambda value: decision_passed(value, minimum))
        else:
            response = {"status": "insufficient", "confidence": 0., "path_matches": False,
                        "choice_is_clear": False, "issue_type": "visual", "reason": observation["reason"]}
            key, passed = None, False
        results.append({"kind": "decision", "decision_id": decision["decision_id"],
                        "segment_id": segment.segment_id, "passed": passed,
                        "observation": observation, "observation_request_key": observation_key,
                        "response": response, "request_key": key})
        if not passed:
            return results
    if segment.terminal:
        response, key, passed = _checked_response(
            client, "verify_stop", STOP_SYSTEM, stop_content(final, segment.segment_id, evidence, mode),
            lambda value: review_passed(value, minimum))
        results.append({"kind": "stop", "segment_id": segment.segment_id, "passed": passed,
                        "response": response, "request_key": key})
        if not passed:
            return results
    if motion_fragments(final, segment.segment_id):
        response, key, passed = _checked_response(
            client, "verify_motion", TRANSIT_SYSTEM, motion_content(final, segment.segment_id, evidence, mode),
            lambda value: review_passed(value, minimum))
    else:
        response = {"status": "pass", "confidence": 1., "issue_type": "none",
                    "reason": "No ordinary movement claims remain; choices and stopping are reviewed separately."}
        key, passed = None, True
    results.append({"kind": "motion", "segment_id": segment.segment_id, "passed": passed,
                    "response": response, "request_key": key})
    return results


def repair_feedback(results):
    return [{"kind": r["kind"], "decision_id": r.get("decision_id"),
             "reason": r["response"]["reason"],
             "issue_type": r["response"].get("issue_type") if r["response"].get("issue_type") != "none" else "language",
             "review_task": "Check this clause against the actual decision-region route and pre-choice cues."
                            if r["kind"] == "decision" else "Review the reported problem against the actual route."}
            for r in results if not r["passed"]]
