import os
import tempfile
import unittest
from collections import Counter

from PIL import Image

from src.data.prepare_training_data import (
    build_dagger_oracle_action_chunks,
    compute_ebs_candidate_counts,
    frame_index_from_filename,
    process_dataset,
)
from src.data.generate_dagger_data import (
    action_sequence_requests_stop,
    is_raw_episode_complete,
    is_successful_dagger_termination,
)


def _dagger_annotation():
    return {
        "episode_id": 7,
        "trajectory_id": "r2r_7",
        "instruction": "Walk to the goal.",
        "actions": [3, 1, 1, 1, 0],
        "oracle_chunks": [
            {
                "step_index": 0,
                "oracle_actions": [1, 1, 2, 1],
            },
            {
                "step_index": 4,
                "oracle_actions": [0, 0, 0, 0],
            },
        ],
    }


class DaggerChunkRelabelTest(unittest.TestCase):
    def test_stop_detection_covers_every_chunk_position(self):
        self.assertTrue(action_sequence_requests_stop([0, 0, 0, 0]))
        self.assertTrue(action_sequence_requests_stop([1, 0, 0, 0]))
        self.assertTrue(action_sequence_requests_stop([1, 1, 0, 0]))
        self.assertTrue(action_sequence_requests_stop([1, 1, 1, 0]))
        self.assertFalse(action_sequence_requests_stop([1, 1, 2, 1]))

    def test_only_real_stop_inside_single_goal_radius_is_successful(self):
        self.assertTrue(
            is_successful_dagger_termination("terminal", [1, 0], 0.3, 0.3)
        )
        self.assertFalse(
            is_successful_dagger_termination("terminal", [1, 0], 0.301, 0.3)
        )
        self.assertFalse(
            is_successful_dagger_termination("max_steps", [1, 1], 0.1, 0.3)
        )
        self.assertFalse(
            is_successful_dagger_termination("env_episode_over", [1, 1], 0.1, 0.3)
        )

    def test_uses_previewed_oracle_chunk_instead_of_executed_chunk(self):
        chunks = build_dagger_oracle_action_chunks(_dagger_annotation())

        self.assertEqual(chunks[0]["action_ids"], [1, 1, 2, 1])
        self.assertNotEqual(chunks[0]["action_ids"], [3, 1, 1, 1])
        self.assertEqual(chunks[1]["action_ids"], [0, 0, 0, 0])
        self.assertEqual(chunks[1]["real_action_count"], 1)

    def test_rejects_missing_dagger_decision_chunks(self):
        annotation = _dagger_annotation()
        annotation["oracle_chunks"] = annotation["oracle_chunks"][:1]
        with self.assertRaisesRegex(ValueError, "final DAgger oracle chunk"):
            build_dagger_oracle_action_chunks(annotation)

    def test_rejects_misaligned_final_oracle_stop(self):
        annotation = _dagger_annotation()
        annotation["oracle_chunks"][-1]["oracle_actions"] = [1, 0, 0, 0]
        with self.assertRaisesRegex(ValueError, "not aligned"):
            build_dagger_oracle_action_chunks(annotation)

    def test_rejects_stop_in_non_final_oracle_chunk(self):
        annotation = _dagger_annotation()
        annotation["oracle_chunks"][0]["oracle_actions"] = [1, 0, 0, 0]
        with self.assertRaisesRegex(ValueError, "Only the final"):
            build_dagger_oracle_action_chunks(annotation)

        with tempfile.TemporaryDirectory() as output_root:
            image_dir = os.path.join(output_root, "images", "dagger", "r2r_7")
            os.makedirs(image_dir)
            for frame_index in range(len(annotation["actions"])):
                Image.new("RGB", (1280, 640)).save(
                    os.path.join(image_dir, f"frame_{frame_index}.jpg")
                )
            self.assertFalse(
                is_raw_episode_complete(output_root, "dagger", annotation, "jpeg")
            )

    def test_rejects_non_four_step_decision_gap(self):
        annotation = _dagger_annotation()
        annotation["actions"] = [3, 1, 1, 1, 1, 0]
        annotation["oracle_chunks"][-1]["step_index"] = 5
        with self.assertRaisesRegex(ValueError, "step_index"):
            build_dagger_oracle_action_chunks(annotation)

    def test_accepts_early_terminal_replan_after_model_reaches_goal(self):
        annotation = _dagger_annotation()
        annotation["actions"] = [3, 1, 0]
        annotation["oracle_chunks"][-1]["step_index"] = 2
        chunks = build_dagger_oracle_action_chunks(annotation)
        self.assertEqual([chunk["start_step"] for chunk in chunks], [0, 2])
        self.assertEqual(chunks[-1]["action_ids"], [0, 0, 0, 0])

    def test_ebs_statistics_exclude_chunk_relabelled_dagger(self):
        counts = compute_ebs_candidate_counts({"dagger": [_dagger_annotation()]})
        self.assertEqual(counts, Counter())

    def test_resume_validation_rejects_missing_decision_chunks(self):
        annotation = _dagger_annotation()
        with tempfile.TemporaryDirectory() as output_root:
            image_dir = os.path.join(output_root, "images", "dagger", "r2r_7")
            os.makedirs(image_dir)
            for frame_index in range(len(annotation["actions"])):
                Image.new("RGB", (1280, 640)).save(
                    os.path.join(image_dir, f"frame_{frame_index}.jpg")
                )

            self.assertTrue(
                is_raw_episode_complete(output_root, "dagger", annotation, "jpeg")
            )
            annotation["oracle_chunks"] = annotation["oracle_chunks"][:1]
            self.assertFalse(
                is_raw_episode_complete(output_root, "dagger", annotation, "jpeg")
            )

    def test_prepare_keeps_actual_history_and_oracle_label(self):
        annotation = _dagger_annotation()
        with tempfile.TemporaryDirectory() as input_root:
            image_dir = os.path.join(input_root, "images", "dagger", "r2r_7")
            os.makedirs(image_dir)
            for frame_index in range(len(annotation["actions"])):
                open(os.path.join(image_dir, f"frame_{frame_index}.jpg"), "wb").close()

            samples = process_dataset(
                selected_subset_list=["dagger"],
                dataset_config={
                    "dagger": {
                        "image_path": os.path.join(input_root, "images", "dagger"),
                        "annotation_path": "unused",
                        "dataset_label": "dagger",
                        "frame_index_fn": frame_index_from_filename,
                    }
                },
                annotations_by_subset={"dagger": [annotation]},
                input_root=input_root,
                # DAgger decisions must not be discarded or reconstructed by
                # primitive-trajectory EBS probabilities.
                event_keep_prob=0.0,
                background_keep_prob=0.0,
                tail_dense_keep_prob=0.0,
            )

        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0]["action_sequence"], ["forward", "forward", "left", "forward"])
        self.assertEqual(samples[0]["history_actions"], [])
        self.assertEqual(samples[1]["action_sequence"], ["stop", "stop", "stop", "stop"])
        self.assertEqual(samples[1]["history_actions"], ["right", "forward", "forward", "forward"])
        self.assertEqual(len(samples[1]["images"]), 5)


if __name__ == "__main__":
    unittest.main()
