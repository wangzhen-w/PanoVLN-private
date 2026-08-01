import unittest

from src.train.data.data import (
    apply_vln_memory_policy,
    build_vln_image_selection,
    build_vln_interframe_action_texts,
    build_vln_user_content,
    format_vln_interframe_actions,
)
from src.data.prepare_training_data import resolve_executed_action_history
from realworld.go2_client import selected_history_with_action_spans
from realworld.inference import _select_interframe_action_texts


class InterframeActionTextTest(unittest.TestCase):
    def test_interleaved_actions_collapse_to_net_motion_summary(self):
        self.assertEqual(
            format_vln_interframe_actions(
                ["right"] * 6 + ["forward"] * 8 + ["left"] * 2
            ),
            "right 60 degrees; forward 2 meters",
        )

        self.assertEqual(
            format_vln_interframe_actions(
                ["left", "forward", "left", "forward", "forward", "right"]
            ),
            "left 15 degrees; forward 0.75 meters",
        )

    def test_cancelled_and_full_circle_turns_describe_endpoint_heading(self):
        self.assertEqual(
            format_vln_interframe_actions(["left", "forward", "right"]),
            "forward 0.25 meters",
        )
        self.assertEqual(
            format_vln_interframe_actions(["left", "right"]),
            "no net change",
        )
        self.assertEqual(
            format_vln_interframe_actions(["left"] * 24),
            "no net change",
        )
        self.assertEqual(
            format_vln_interframe_actions(["left"] * 23),
            "right 15 degrees",
        )
        self.assertEqual(
            format_vln_interframe_actions(["left"] * 25),
            "left 15 degrees",
        )
        self.assertEqual(
            format_vln_interframe_actions(["left"] * 12),
            "left 180 degrees",
        )
        self.assertEqual(
            format_vln_interframe_actions(["right"] * 12),
            "right 180 degrees",
        )

    def test_single_component_summaries_and_units_are_explicit(self):
        self.assertEqual(
            format_vln_interframe_actions(["left", "left"]),
            "left 30 degrees",
        )
        self.assertEqual(
            format_vln_interframe_actions(["forward"] * 4),
            "forward 1 meter",
        )

    def test_short_histories_keep_every_edge_and_summary(self):
        for num_frames in (1, 2, 9, 10, 11):
            with self.subTest(num_frames=num_frames):
                selected_indices = build_vln_image_selection(
                    current_step=num_frames - 1,
                    last_frame_index=num_frames - 1,
                )
                action_texts = build_vln_interframe_action_texts(
                    history_actions=["forward"] * (num_frames - 1),
                    selected_frame_indices=selected_indices,
                )
                self.assertEqual(selected_indices, list(range(num_frames)))
                self.assertEqual(len(action_texts), num_frames - 1)
                self.assertTrue(
                    all(
                        text == "forward 0.25 meters"
                        for text in action_texts
                    )
                )

    def test_selected_edges_use_half_open_absolute_slices(self):
        history_actions = [
            "right",
            "right",
            "forward",
            "left",
            "forward",
            "forward",
        ]
        self.assertEqual(
            build_vln_interframe_action_texts(
                history_actions=history_actions,
                selected_frame_indices=[1, 4, 6],
            ),
            [
                "forward 0.25 meters",
                "forward 0.5 meters",
            ],
        )

    def test_recent_window_does_not_prepend_actions_before_first_selected_frame(self):
        selected_indices = build_vln_image_selection(
            current_step=120,
            last_frame_index=120,
        )
        self.assertEqual(selected_indices, [21, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])

        history_actions = ["left"] * 21 + ["right"] * 9 + ["forward"] * 90
        action_texts = build_vln_interframe_action_texts(
            history_actions=history_actions,
            selected_frame_indices=selected_indices,
        )
        self.assertEqual(action_texts[0], "right 135 degrees")
        self.assertNotIn("left", action_texts[0])

    def test_prompt_interleaves_each_transition_between_its_images(self):
        content = build_vln_user_content(
            instruction="Walk into the kitchen.",
            num_images=3,
            interframe_action_texts=[
                "right 30 degrees",
                "forward 1 meter",
            ],
        )

        image_positions = [
            index for index, item in enumerate(content) if item["type"] == "image"
        ]
        self.assertEqual(image_positions, [2, 4, 7])
        self.assertIn("right 30 degrees", content[3]["text"])
        self.assertIn("forward 1 meter", content[5]["text"])
        self.assertIn("Motion between consecutive shown observations", content[1]["text"])
        self.assertIn("Current observation", content[6]["text"])
        self.assertIn("Devise the next action sequence", content[8]["text"])

    def test_training_policy_preserves_uniform_indices_for_action_alignment(self):
        images = [f"frame_{index}.jpg" for index in range(15)]
        history_actions = ["right"] * 6 + ["forward"] * 8
        normalized = apply_vln_memory_policy(
            {
                "instruction": "Go to the doorway.",
                "action_sequence": ["left", "forward", "forward", "forward"],
                "images": images,
                "history_actions": history_actions,
                "step_index": 14,
            },
            interframe_action_text_enabled=True,
        )
        selected_indices = build_vln_image_selection(
            current_step=14,
            last_frame_index=14,
        )
        self.assertEqual(
            normalized["images"],
            [images[index] for index in selected_indices],
        )
        action_annotations = [
            item["text"]
            for item in normalized["messages"][1]["content"]
            if item["type"] == "text"
            and item["text"].startswith("\nMotion:")
        ]
        self.assertEqual(len(action_annotations), len(selected_indices) - 1)
        self.assertIn("right", action_annotations[0])
        self.assertIn("forward 0.5 meters", action_annotations[-1])

    def test_current_only_selection_has_no_transition(self):
        self.assertEqual(
            build_vln_image_selection(
                current_step=20,
                last_frame_index=20,
                max_memory_images=0,
            ),
            [20],
        )
        content = build_vln_user_content(
            instruction="Stop here.",
            num_images=1,
            interframe_action_texts=[],
        )
        self.assertEqual(
            sum(item["type"] == "image" for item in content),
            1,
        )

    def test_invalid_history_length_and_stop_fail_fast(self):
        with self.assertRaisesRegex(ValueError, "one historical action"):
            apply_vln_memory_policy(
                {
                    "instruction": "Go forward.",
                    "action_sequence": ["forward"] * 4,
                    "images": ["frame_0.jpg", "frame_1.jpg"],
                    "history_actions": [],
                    "step_index": 1,
                },
                interframe_action_text_enabled=True,
            )
        with self.assertRaisesRegex(ValueError, "left/right/forward"):
            format_vln_interframe_actions(["stop"])
        with self.assertRaisesRegex(ValueError, "requires an action span"):
            format_vln_interframe_actions([])

    def test_step_index_and_current_endpoint_are_strict(self):
        with self.assertRaisesRegex(ValueError, "step_index"):
            apply_vln_memory_policy(
                {
                    "instruction": "Go forward.",
                    "action_sequence": ["forward"] * 4,
                    "images": ["frame_0.jpg", "frame_1.jpg"],
                    "history_actions": ["forward"],
                    "step_index": 0,
                },
                interframe_action_text_enabled=True,
            )
        with self.assertRaisesRegex(ValueError, "final selected"):
            build_vln_interframe_action_texts(
                history_actions=["forward", "right"],
                selected_frame_indices=[0, 1],
            )

    def test_dagger_history_uses_executed_actions_not_expert_targets(self):
        expert_actions = [1, 2, 0]
        self.assertEqual(
            resolve_executed_action_history(
                {
                    "episode_id": 7,
                    "actions": expert_actions,
                    "executed_actions": [3, 1, 0],
                },
                expert_actions=expert_actions,
            ),
            [3, 1, 0],
        )

    def test_action_text_feature_is_explicitly_switchable(self):
        normalized = apply_vln_memory_policy(
            {
                "instruction": "Go forward.",
                "action_sequence": ["forward"] * 4,
                "images": ["frame_0.jpg", "frame_1.jpg"],
            },
            interframe_action_text_enabled=False,
        )
        user_content = normalized["messages"][1]["content"]
        self.assertFalse(
            any(
                item["type"] == "text"
                and item["text"].startswith("\nMotion:")
                for item in user_content
            )
        )

    def test_realworld_selection_keeps_action_spans_aligned(self):
        selected_images, action_spans = selected_history_with_action_spans(
            history=[bytes([index]) for index in range(7)],
            action_history=["right"] * 2 + ["forward"] * 4,
            max_memory_images=1,
            memory_pool_window_frames=100,
        )
        self.assertEqual(selected_images, [b"\x00", b"\x06"])
        self.assertEqual(
            action_spans,
            [["right", "right", "forward", "forward", "forward", "forward"]],
        )
        self.assertEqual(
            _select_interframe_action_texts(
                interframe_actions=action_spans,
                selected_indices=[0, 1],
                num_input_images=2,
            ),
            ["right 30 degrees; forward 1 meter"],
        )

    def test_training_and_habitat_eval_build_identical_user_content(self):
        from src.eval.eval import build_eval_messages

        history_actions = ["right", "forward"]
        training_sample = apply_vln_memory_policy(
            {
                "instruction": "Go to the doorway.",
                "action_sequence": ["left"] * 4,
                "images": ["frame_0.jpg", "frame_1.jpg", "frame_2.jpg"],
                "history_actions": history_actions,
                "step_index": 2,
            },
            interframe_action_text_enabled=True,
        )
        action_texts = build_vln_interframe_action_texts(
            history_actions=history_actions,
            selected_frame_indices=[0, 1, 2],
        )
        eval_messages = build_eval_messages(
            instruction="Go to the doorway.",
            images=[object(), object(), object()],
            interframe_action_texts=action_texts,
        )
        self.assertEqual(
            training_sample["messages"][1]["content"],
            eval_messages[1]["content"],
        )

    def test_episode_final_action_is_not_added_without_a_following_frame(self):
        from src.eval.eval import PanoVLN_Agent

        agent = object.__new__(PanoVLN_Agent)
        agent.interframe_action_history = []
        agent.last_returned_action = 1
        agent._record_previous_action()
        self.assertEqual(agent.interframe_action_history, [1])

        agent.last_returned_action = 0
        agent.finalize_episode()
        self.assertEqual(agent.interframe_action_history, [1])
        self.assertIsNone(agent.last_returned_action)


if __name__ == "__main__":
    unittest.main()
