"""Regression tests for geometry, action ownership, leakage and export contracts.

Run with the standard-library unittest runner; no Habitat/GPU/API is needed.
"""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import numpy as np

from dataset_create.instruction.geometry import Camera, GroundRoute, densify_positions, look_rotation, overlay_route
from dataset_create.instruction.segmentation import (
    EvidenceError, Segment, cumulative_distance, merge_for_repair, rotation_matrix,
    sample_frames, segment_episode, validate_segments,
)
from dataset_create.instruction.language import assemble, bind_clause_spans, generate_local, validate_local, validate_polish
from dataset_create.instruction.rendering import EpisodeRenderer
from dataset_create.instruction.verification import (
    decision_content, decision_visual_content, decision_passed, motion_content,
    review_passed, stop_content, verify_segment,
)
from dataset_create.instruction.export import EMPTY_VOCAB, make_r2r_episode, validate_r2r
from dataset_create.instruction.pipeline import validate_acceptance
from dataset_create.instruction.client import digest


CONFIG = json.loads((Path(__file__).parents[1] / "config/default.json").read_text())


def synthetic_route():
    states = [{"position": [0., 0., 0.], "rotation_xyzw": [0., 0., 0., 1.]}]
    actions = []
    for i in range(1, 9):
        states.append({"position": [0., 0., -float(i)], "rotation_xyzw": [0., 0., 0., 1.]})
        actions.append(1)
    for angle in range(15, 91, 15):
        states.append({"position": [0., 0., -8.], "rotation_xyzw": look_rotation([0., 0., 0., 1.], angle)})
        actions.append(3)
    for i in range(1, 9):
        states.append({"position": [float(i), 0., -8.], "rotation_xyzw": look_rotation([0., 0., 0., 1.], 90)})
        actions.append(1)
    states.append(copy.deepcopy(states[-1]))
    actions.append(0)
    event = {"action_index": 8, "selected": {"connection_id": "selected", "region_id": 1, "anchor_position": [2., 0., -8.]},
             "alternatives": [{"connection_id": "alternative", "region_id": 2, "anchor_position": [0., 0., -12.]}]}
    episode = {"trajectory_id": "synthetic", "scene_id": "hm3d/train/scene/scene.basis.glb",
               "start_position": states[0]["position"], "start_rotation_xyzw": states[0]["rotation_xyzw"],
               "final_position": states[-1]["position"], "goal_position": [8.2, 0., -8.],
               "action_ids": actions, "decision_events": [event]}
    return episode, states


class ProjectionTests(unittest.TestCase):
    def test_unsupported_ground_samples_are_never_painted(self):
        camera = Camera(80, 60, 90., np.array([0., 1.25, 0.]), [0., 0., 0., 1.])
        points, tangent, arc = densify_positions([[0, 0, -3], [0, 0, -6]], .04)
        route = GroundRoute(points, tangent, arc, np.zeros(len(arc), dtype=bool))
        rgb = np.zeros((60, 80, 3), dtype=np.uint8)
        painted, stats = overlay_route(rgb, np.full((60, 80), 3.), camera, route, CONFIG['render'])
        self.assertEqual(stats['route_pixels'], 0)
        np.testing.assert_array_equal(painted, rgb)

    def test_unproject_reproject_with_yaw_and_pitch(self):
        # Nonzero x quaternion rotation represents an actual tilted sensor.
        base = [-.13, 0., 0., .991514]
        camera = Camera(80, 60, 90., np.array([2., 1.25, -3.]), look_rotation(base, 73.))
        depth = np.full((60, 80), 3.)
        world = camera.unproject(depth)
        uv, z = camera.project(world.reshape(-1, 3))
        rows, cols = np.indices(depth.shape)
        np.testing.assert_allclose(uv, np.c_[cols.ravel(), rows.ravel()], atol=1e-9)
        np.testing.assert_allclose(z, 3., atol=1e-9)

    def test_compass_left_right_is_relative_to_agent(self):
        for base in ([0., 0., 0., 1.], look_rotation([0., 0., 0., 1.], 117)):
            inverse = rotation_matrix(base).T
            for bearing, expected in [(90, [1., 0., 0.]), (-90, [-1., 0., 0.]), (180, [0., 0., 1.])]:
                actual = inverse @ rotation_matrix(look_rotation(base, bearing)) @ np.array([0., 0., -1.])
                np.testing.assert_allclose(actual, expected, atol=1e-7)

    def test_surface_decal_respects_walls_and_wrong_floors(self):
        cfg = CONFIG["render"]
        camera = Camera(160, 120, 90., np.array([0., 1.25, 0.]), [0., 0., 0., 1.])
        rows, _ = np.indices((120, 160))
        down = (rows + .5 - 60) / 80
        depth = np.where(down > 0, 1.25 / np.maximum(down, 1e-6), cfg["far_m"])
        positions, tangent, arc = densify_positions([[0, 0, -3], [0, 0, -6]], .04)
        route = GroundRoute(positions, tangent, arc, np.ones(len(arc), dtype=bool))
        rgb = np.zeros((120, 160, 3), dtype=np.uint8)
        painted, stats = overlay_route(rgb, depth, camera, route, cfg)
        self.assertGreater(stats["route_pixels"], 20)
        wall_depth = np.minimum(depth, 2.)
        occluded, stats = overlay_route(rgb, wall_depth, camera, route, cfg)
        self.assertEqual(stats["route_pixels"], 0)
        np.testing.assert_array_equal(occluded, rgb)
        route.points[:, 1] += 1.0
        _, stats = overlay_route(rgb, depth, camera, route, cfg)
        self.assertEqual(stats["route_pixels"], 0)


class SegmentationTests(unittest.TestCase):
    def test_completed_choices_separate_despite_shared_approach(self):
        states = [{"position": [0., 0., -float(i)], "rotation_xyzw": [0., 0., 0., 1.]}
                  for i in range(14)]
        states.append(copy.deepcopy(states[-1]))
        episode = {"action_ids": [1]*13 + [0], "decision_events": [
            {"action_index": i, "selected": {"anchor_position": states[i+1]["position"]}}
            for i in (4, 7)]}
        segments = segment_episode(episode, states, CONFIG["segmentation"])
        first, second = [next(s for s in segments if d in s.decision_ids) for d in ("d0", "d1")]
        self.assertNotEqual(first.segment_id, second.segment_id)
        self.assertEqual(first.end, second.start)
        self.assertLess(second.context_start, second.start)
        self.assertLessEqual(first.end, 7)
        self.assertGreater(first.end, 5)
        validate_segments(segments, episode, states, CONFIG["segmentation"])

    def test_context_motion_is_excluded_from_both_primary_media_modes(self):
        renderer = object.__new__(EpisodeRenderer)
        renderer.settings = CONFIG["render"]
        rgb, depth = np.zeros((8, 8, 3), dtype=np.uint8), np.ones((8, 8))
        camera = Camera(8, 8, 90., np.array([0., 1.25, 0.]), [0., 0., 0., 1.])
        renderer.observe = Mock(return_value=(rgb, depth, camera))
        states = [{"position": [0., 0., -float(i)], "rotation_xyzw": [0., 0., 0., 1.]}
                  for i in range(5)]
        segment = Segment("s", 1, 3, 0, 4, [], ["transit"])
        route = Mock()
        route.subset.return_value = route
        module = "dataset_create.instruction.rendering."
        with tempfile.TemporaryDirectory() as tmp, \
                patch(module + "sample_frames", return_value=list(range(5))), \
                patch(module + "overlay_route", return_value=(rgb, {"route_pixels": 50})), \
                patch(module + "video_from_frames", side_effect=lambda paths, dest, fps, hold: str(dest)) as video:
            evidence = renderer.render_segment({}, states, segment, route, tmp, CONFIG["segmentation"])
        self.assertEqual(evidence["frame_state_indices"], [1, 2, 3])
        self.assertEqual(evidence["context"]["frame_state_indices"], list(range(5)))
        for paths in (evidence["clean_frames"], evidence["marked_frames"],
                      video.call_args_list[0].args[0], video.call_args_list[1].args[0]):
            self.assertEqual([Path(p).name.split("_")[1] for p in paths], ["001", "002", "003"])

    def test_complete_decision_and_real_stop_ownership(self):
        episode, states = synthetic_route()
        segments = segment_episode(episode, states, CONFIG["segmentation"])
        owner = next(s for s in segments if "d0" in s.decision_ids)
        self.assertLess(owner.start, 8)
        self.assertGreater(owner.end, 16)
        self.assertFalse(any(8 < s.end < 14 for s in segments))
        self.assertEqual([d for s in segments for d in s.decision_ids], ["d0"])
        self.assertEqual(segments[-1].end, len(episode["action_ids"]))
        self.assertTrue(segments[-1].terminal)
        self.assertGreater(cumulative_distance(states)[-1] - cumulative_distance(states)[segments[-1].start], 1.)
        validate_segments(segments, episode, states, CONFIG["segmentation"])

    def test_decision_review_starts_at_approach_and_keeps_natural_segment_end(self):
        renderer = object.__new__(EpisodeRenderer)
        renderer.settings = CONFIG["render"]
        rgb, depth = np.zeros((8, 8, 3), dtype=np.uint8), np.ones((8, 8))
        camera = Camera(8, 8, 90., np.array([0., 1.25, 0.]), [0., 0., 0., 1.])
        renderer.observe = Mock(return_value=(rgb, depth, camera))
        renderer.decision_compass = Mock(return_value={"decision_id": "d0", "state_index": 3,
                                                       "review_start_state_index": 2})
        states = [{"position": [0., 0., -float(i)], "rotation_xyzw": [0., 0., 0., 1.]} for i in range(8)]
        segment = Segment("s", 1, 6, 0, 7, ["d0"], ["decision"])
        route = Mock()
        route.subset.return_value = route
        module = "dataset_create.instruction.rendering."
        with tempfile.TemporaryDirectory() as tmp, \
                patch(module + "sample_frames", return_value=list(range(8))), \
                patch(module + "overlay_route", return_value=(rgb, {"route_pixels": 50})), \
                patch(module + "video_from_frames", side_effect=lambda paths, dest, fps, hold=0: str(dest)):
            evidence = renderer.render_segment({}, states, segment, route, tmp, CONFIG["segmentation"])
        decision = evidence["decisions"][0]
        self.assertEqual(decision["review_state_indices"], [2, 3, 4, 5, 6])
        self.assertEqual([c["state_index"] for c in decision["review_cameras"]], [2, 3, 4, 5, 6])
        self.assertTrue(all("_clean.jpg" in p for p in decision["review_frames"]))
        self.assertEqual(decision["review_end_state_index"], segment.end)

    def test_upstream_merge_preserves_coverage(self):
        episode, states = synthetic_route()
        segments = segment_episode(episode, states, CONFIG["segmentation"])
        repaired, combined = merge_for_repair(segments, segments[1].segment_id)
        self.assertEqual(len(repaired), len(segments)-1)
        validate_segments(repaired, episode, states, CONFIG["segmentation"])

    def test_video_budget_never_silently_drops_decision(self):
        episode, states = synthetic_route()
        segment = Segment("s", 0, len(states)-1, 0, len(states)-1, ["d0"], ["decision"], True)
        cfg = dict(CONFIG["render"], max_video_frames=2)
        with self.assertRaises(EvidenceError):
            sample_frames(segment, states, cfg, required=[8, 14])


class LeakageTests(unittest.TestCase):
    def test_motion_reviews_stairs_without_retrying_the_entrance_choice(self):
        final = {"clauses": [{"text": "Enter the right doorway. Descend the stairs. Stop beside the bench.",
                              "segment_ids": ["s"]}], "protected": [
            {"segment_id": "s", "text": "Enter the right doorway.", "decision_id": "d0", "stop": False},
            {"segment_id": "s", "text": "Stop beside the bench.", "decision_id": None, "stop": True}]}
        evidence = {"core_frame_indices": [0], "clean_video": "/clean/clip.mp4", "clean_frames": []}
        payload = json.loads(motion_content(final, "s", evidence, "video")[0]["text"])
        self.assertEqual(payload["ordinary_movement_fragments"], ["Descend the stairs"])
        self.assertNotIn("right doorway", json.dumps(payload))
        self.assertNotIn("bench", json.dumps(payload))

    def test_motion_pose_trace_distinguishes_turns_from_translation(self):
        evidence = {"core_frame_indices": [0, 1, 2], "clean_video": "/clean/clip.mp4",
                    "clean_frames": [], "cameras": [
                        {"rotation_xyzw": look_rotation([0., 0., 0., 1.], yaw),
                         "position": [0., height, -float(i)]}
                        for i, (yaw, height) in enumerate([(170., 2.), (170., 2.), (-160., 1.)])]}
        final = {"clauses": [{"text": "Walk straight, then right and down.", "segment_ids": ["s"]}],
                 "protected": []}
        payload = json.loads(motion_content(final, "s", evidence, "video")[0]["text"])
        samples = payload["relative_camera_pose_by_frame"]
        self.assertEqual([s["heading_from_start_degrees"] for s in samples], [0., 0., 30.])
        self.assertEqual([s["height_from_start_m"] for s in samples], [0., 0., -1.])

    def test_decision_review_has_complete_clean_interval_and_no_answer_annotations(self):
        final = {"clauses": [{"text": "Enter the right doorway.", "segment_ids": ["s0"]}],
                 "protected": [{"decision_id": "d0", "text": "Enter the right doorway.", "stop": False}]}
        evidence = {"decision_id": "d0", "state_index": 8, "review_state_indices": [4, 8, 12],
                    "review_start_state_index": 4, "review_end_state_index": 12,
                    "review_video": "/clean/choice.mp4", "review_frames": ["/clean/a.jpg"]*3,
                    "review_cameras": [{"rotation_xyzw": [0, 0, 0, 1], "position": [0, 1.25, 0]}]*3,
                    "clean_compass": "/clean/compass.jpg",
                    "ground_truth_label": "SECRET_GT", "marked_compass": "/SECRET_MARKED.jpg",
                    "route_exit_bearing_degrees": "SECRET_GT_DIRECTION",
                    "label_to_branch": {"SECRET_SELECTED": "SECRET_ANCHOR"}}
        content = decision_visual_content(evidence, "video")
        self.assertNotIn("SECRET", json.dumps(content))
        self.assertNotIn("Enter the right doorway.", json.dumps(content))
        self.assertIn("/clean/choice.mp4", json.dumps(content))
        observation = {"observed_route": "Turn right into a doorway.", "uncertain": False}
        comparison = decision_content(final, "s0", evidence, observation)
        self.assertIn("choice_clause_from_final_instruction", json.dumps(comparison))
        self.assertIn("Enter the right doorway.", json.dumps(comparison))
        self.assertNotIn("SECRET", json.dumps(comparison))
        self.assertTrue(all(item["type"] == "text" for item in comparison))
        evidence["review_state_indices"][-1] = 13
        with self.assertRaises(ValueError):
            decision_visual_content(evidence, "frames")

    def test_stop_and_motion_have_separate_responsibilities_and_clean_evidence(self):
        final = {"clauses": [{"text": "Stop at the carpet edge.", "segment_ids": ["s0"]}],
                 "protected": [{"segment_id": "s0", "decision_id": None, "text": "Stop at the carpet edge.", "stop": True}]}
        evidence = {"clean_video": "/clean/arrival.mp4", "clean_frames": ["/clean/arrival.jpg"],
                    "core_frame_indices": [0], "marked_video": "/SECRET_MARKED.mp4",
                    "stop": {"actual": {"image": "/clean/actual.jpg", "state": {"SECRET": 1}},
                             "comparisons": [{"kind": "earlier", "image": "/clean/earlier.jpg"}]}}
        content = stop_content(final, "s0", evidence, "video")
        encoded = json.dumps(content)
        self.assertNotIn("SECRET", encoded)
        self.assertIn("/clean/arrival.mp4", encoded)
        self.assertIn("/clean/actual.jpg", encoded)
        self.assertNotIn("/clean/earlier.jpg", encoded)
        ordinary = motion_content(final, "s0", evidence, "video")
        self.assertNotIn("/clean/actual.jpg", json.dumps(ordinary))
        self.assertIn("choices_and_stop_reviewed_separately", ordinary[0]["text"])
        self.assertNotIn("Stop at the carpet edge", json.dumps(ordinary))
        for status in ("fail", "insufficient"):
            self.assertFalse(review_passed({"status": status, "confidence": .95,
                                           "reason": "Wrong stopping area", "issue_type": "language"}, .7))

    def test_ambiguous_or_incorrect_review_never_passes(self):
        result = {"status": "insufficient", "confidence": .95,
                  "reason": "Both visible entrances fit", "issue_type": "language"}
        self.assertFalse(review_passed(result, .7))
        result.update(status="fail", reason="Wrong direction")
        self.assertFalse(review_passed(result, .7))
        result.update(status="pass", confidence=.5)
        self.assertFalse(review_passed(result, .7))
        result.update(confidence=.95, path_matches=True, choice_is_clear=False)
        self.assertFalse(decision_passed(result, .7))
        result.update(path_matches=False, choice_is_clear=True)
        self.assertFalse(decision_passed(result, .7))
        result.update(path_matches=True)
        self.assertTrue(decision_passed(result, .7))

    def test_failed_choice_defers_later_requests_and_empty_motion_needs_no_call(self):
        final = {"clauses": [{"text": "Enter the doorway.", "segment_ids": ["s"]}],
                 "protected": [{"segment_id": "s", "decision_id": "d0", "text": "Enter the doorway."}]}
        segment = Segment("s", 0, 3, 0, 3, ["d0"], ["decision"])
        client = SimpleNamespace(settings={"media_mode": "video"})
        evidence = {"decisions": [{"decision_id": "d0"}]}
        module = "dataset_create.instruction.verification."
        with patch(module + "decision_visual_content", return_value=[]), patch(module + "_checked_response") as check:
            check.side_effect = [({"uncertain": False}, "observed", True), ({"status": "fail"}, "failed", False)]
            results = verify_segment(client, final, segment, evidence, CONFIG["validation"])
            self.assertEqual([r["kind"] for r in results], ["decision"])
            self.assertEqual(check.call_count, 2)
            check.reset_mock()
            check.side_effect = [({"uncertain": False}, "observed", True), ({"status": "pass"}, "passed", True)]
            results = verify_segment(client, final, segment, evidence, CONFIG["validation"])
            self.assertEqual([r["kind"] for r in results], ["decision", "motion"])
            self.assertTrue(all(r["passed"] for r in results))
            self.assertEqual(check.call_count, 2)
            self.assertIsNone(results[-1]["request_key"])


class LanguageExportTests(unittest.TestCase):
    def test_architectural_frames_are_not_model_annotations(self):
        segment = Segment("s", 0, 3, 0, 3, [], ["transit"])
        local = {"text": "Walk through the door frame and past the window frames.",
                 "decisions": [], "stop": None, "uncertain": False, "issue_type": "none", "issues": []}
        validate_local(local, segment)
        for text in ("Follow the gold line.", "Turn left as shown in frame 3."):
            with self.assertRaises(ValueError):
                validate_local(dict(local, text=text), segment)

    def local(self):
        return {"text": "Enter the second doorway on the left, then stop just beyond the threshold.",
                "decisions": [{"decision_id": "d0", "clause": "Enter the second doorway on the left",
                               "evidence": [{"claim": "Both doorways visible", "view": "left", "visible_before_choice": True}]}],
                "stop": {"clause": "stop just beyond the threshold", "evidence": "Threshold behind"},
                "uncertain": False, "issue_type": "none", "issues": []}

    def test_protected_language_and_final_provenance(self):
        segment = Segment("s0", 0, 20, 0, 20, ["d0"], ["decision", "arrival"], True)
        local = self.local()
        result = assemble([segment], {"s0": local})
        self.assertEqual(result["instruction_text"], local["text"])
        for item in result["protected"]:
            self.assertEqual(result["instruction_text"][item["char_start"]:item["char_end"]], item["text"])
        local["text"] = "Follow the arrows. " + local["text"]
        with self.assertRaises(ValueError):
            validate_local(local, segment)

    def test_clause_metadata_casing_is_bound_without_changing_words(self):
        local = self.local()
        local["decisions"][0]["clause"] = "enter the second doorway on the left."
        bound = bind_clause_spans(local)
        self.assertEqual(bound["decisions"][0]["clause"], "Enter the second doorway on the left")
        self.assertEqual(bound["text"], self.local()["text"])

    def test_cue_view_spelling_does_not_reject_navigable_text(self):
        segment = Segment("s0", 0, 20, 0, 20, ["d0"], ["decision", "arrival"], True)
        local = self.local()
        local['decisions'][0]['evidence'][0]['view'] = 'front / front-left'
        validate_local(local, segment)
        local['decisions'][0]['evidence'][0]['visible_before_choice'] = False
        with self.assertRaises(ValueError):
            validate_local(local, segment)

    def test_malformed_metadata_is_retried_without_crashing_the_pipeline(self):
        segment = Segment("s0", 0, 20, 0, 20, ["d0"], ["decision", "arrival"], True)
        malformed = dict(self.local(), decisions=['d0'])
        client = SimpleNamespace(settings={'media_mode': 'video'},
                                 complete=Mock(side_effect=[(malformed, 'bad'), (self.local(), 'good')]))
        with patch('dataset_create.instruction.language.author_content', return_value=[]):
            local = generate_local(client, segment, {})
        self.assertEqual(local['request_keys'], ['bad', 'good'])
        self.assertEqual(local['text'], self.local()['text'])

    def test_polish_preserves_choice_and_stop_without_a_word_whitelist(self):
        segment = Segment("s0", 0, 20, 0, 20, ["d0"], ["decision", "arrival"], True)
        source = self.local()
        for text in [source["text"].replace("left", "right"), source["text"].replace("beyond", "before")]:
            with self.assertRaises(ValueError):
                validate_polish({"clauses": [{"segment_id": "s0", "text": text}]}, [segment], {"s0": source})
        text = source["text"].replace(", then ", "; once through, ")
        edited = validate_polish({"clauses": [{"segment_id": "s0", "text": text}]}, [segment], {"s0": source})
        self.assertEqual(edited['s0']['text'], text)

    def test_r2r_uses_actual_stop_and_resolvable_portable_scene(self):
        episode, states = synthetic_route()
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp) / "train/scene/scene.basis.glb"
            scene.parent.mkdir(parents=True)
            scene.touch()
            result = make_r2r_episode(episode, 11, states, self.local()["text"], 16., scene, tmp)
            self.assertEqual(result["goals"][0]["position"], states[-1]["position"])
            self.assertNotEqual(result["goals"][0]["position"], episode["goal_position"])
            self.assertEqual(result["scene_id"], "train/scene/scene.basis.glb")
            self.assertEqual(result["start_rotation"], episode["start_rotation_xyzw"])
            validate_r2r({"episodes": [result], "instruction_vocab": EMPTY_VOCAB}, tmp)
            result["audit"] = {"ground_truth": "leak"}
            with self.assertRaises(ValueError):
                validate_r2r({"episodes": [result], "instruction_vocab": EMPTY_VOCAB}, tmp)

    def test_export_rejects_missing_checks_or_changed_final_text(self):
        text = "Enter the right doorway and stop beyond the threshold."
        record = {"source_trajectory": {"decision_events": [{}]}, "segments": [{"segment_id": "s0"}],
                  "verification": [
                      {"kind": "decision", "decision_id": "d0", "segment_id": "s0", "passed": True,
                       "observation": {"uncertain": False},
                       "response": {"status": "pass", "path_matches": True, "choice_is_clear": True}},
                      {"kind": "stop", "segment_id": "s0", "passed": True,
                       "response": {"status": "pass"}},
                      {"kind": "motion", "segment_id": "s0", "passed": True, "response": {"status": "pass"}}],
                  "final": {"instruction_text": text}, "acceptance": {"final_instruction_hash": digest(text)},
                  "r2r_episode": {"instruction": {"instruction_text": text}}}
        validate_acceptance(record)
        missing = copy.deepcopy(record)
        missing["verification"].pop(0)
        with self.assertRaises(ValueError):
            validate_acceptance(missing)
        changed = copy.deepcopy(record)
        changed["final"]["instruction_text"] = text.replace("right", "left")
        with self.assertRaises(ValueError):
            validate_acceptance(changed)


if __name__ == "__main__":
    unittest.main()
