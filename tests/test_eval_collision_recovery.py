"""Exercise the real agent queue and RGB feedback without loading model weights."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from PIL import Image
from transformers import BatchEncoding

from src.eval.collision_recovery import CollisionRecovery, FORWARD, LEFT, RIGHT, STOP
from src.eval.eval import PanoVLN_Agent, extract_first_turn_logits


class AgentCollisionRecoveryTests(unittest.TestCase):
    def agent(self, predictions, steps=2, stop_cap=12):
        agent = PanoVLN_Agent.__new__(PanoVLN_Agent)
        agent.save_topdown = False
        agent.actions_per_replan = 4
        agent.stop_commit_max_actions = stop_cap
        agent.max_memory_images = 10
        agent.memory_pool_window_frames = 100
        agent.collision_recovery = CollisionRecovery(steps)
        agent.reset()
        plans = iter(predictions)

        def predict(instruction, selected_images):
            agent.prediction_turn_logits = [1.0, 0.0]
            return 'test prediction', next(plans)

        agent._predict_action_sequence_from_images = Mock(side_effect=predict)
        agent._prepare_selected_images = lambda indices: [agent.rgb_history[i] for i in indices]
        return agent

    def act(self, agent, collided=False, value=0):
        return agent.act(
            {'rgb': np.full((64, 128, 3), value, dtype=np.uint8), 'instruction': {'text': 'Walk to the door.'}},
            {'collisions': {'is_collision': collided}}, 'episode',
        )['action']

    def test_feedback_replans_after_second_collision_and_uses_old_queue(self):
        agent = self.agent([[FORWARD, FORWARD, RIGHT, RIGHT], [FORWARD, LEFT] * 9, [STOP]])
        self.assertEqual(self.act(agent), FORWARD)
        self.assertEqual(self.act(agent, collided=True), FORWARD)
        self.assertEqual(self.act(agent, collided=True), RIGHT)
        self.assertEqual(agent._predict_action_sequence_from_images.call_count, 2)
        self.assertEqual(self.act(agent), FORWARD)
        self.assertEqual(self.act(agent, value=100), FORWARD)
        self.assertEqual(self.act(agent, value=200), STOP)
        self.assertEqual(agent._predict_action_sequence_from_images.call_count, 3)
        self.assertEqual(len(agent.rgb_history), 6)

    def test_rgb_motion_prevents_recovery_even_when_collision_flag_is_set(self):
        agent = self.agent([[FORWARD] * 4, [STOP]])
        for i in range(4):
            self.assertEqual(self.act(agent, collided=i > 0, value=i * 10), FORWARD)
        self.assertEqual(agent._predict_action_sequence_from_images.call_count, 1)
        self.assertEqual(self.act(agent, collided=True, value=40), STOP)

    def test_already_committed_stop_is_never_interrupted(self):
        agent = self.agent([[FORWARD] * 11 + [STOP]])
        self.assertEqual(self.act(agent), FORWARD)
        for _ in range(10):
            self.assertEqual(self.act(agent, collided=True), FORWARD)
        self.assertEqual(self.act(agent, collided=True), STOP)
        self.assertEqual(agent._predict_action_sequence_from_images.call_count, 1)

    def test_stop_window_remains_twelve(self):
        prediction = [FORWARD] * 12 + [STOP]
        for cap, expected in [(12, LEFT), (13, FORWARD)]:
            with self.subTest(cap=cap):
                agent = self.agent([[FORWARD] * 18, prediction], stop_cap=cap)
                self.act(agent)
                self.act(agent, collided=True)
                self.assertEqual(self.act(agent, collided=True), expected)
                self.assertEqual(agent.collision_recovery.terminal_queue, cap == 13)

    def test_disable_restores_original_fixed_queue(self):
        agent = self.agent([[FORWARD] * 4, [STOP]], steps=0)
        for i in range(4):
            self.assertEqual(self.act(agent, collided=i > 0), FORWARD)
        self.assertEqual(agent._predict_action_sequence_from_images.call_count, 1)
        self.assertEqual(self.act(agent, collided=True), STOP)

    def test_reset_forgets_previous_action_and_collision_streak(self):
        agent = self.agent([[FORWARD] * 4, [FORWARD] * 4])
        self.act(agent)
        self.act(agent, collided=True)
        agent.reset()
        self.assertEqual(self.act(agent, collided=True), FORWARD)
        self.assertIsNone(agent.collision_recovery.heading)
        self.assertEqual(agent.collision_recovery.count, 0)
        self.assertEqual(len(agent.rgb_history), 1)

    def test_history_window_keeps_original_action_timestamps(self):
        agent = self.agent([[FORWARD] * 4] * 26, steps=0)
        for i in range(102):
            self.act(agent, value=i)
        self.assertEqual(len(agent.rgb_history), 102)
        self.assertEqual(agent._select_image_indices(), [2, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101])

    def test_fallback_scores_use_first_content_token_and_correct_prefix(self):
        lookup = {1: (FORWARD, (0, 1, 2, 3)), 5: (FORWARD, (4, 5, 6, 7))}
        first = torch.tensor([[0., 0., 10., 20., 0., 0., 3., 7.]])
        later = torch.tensor([[0., 0., 50., 1., 0., 0., 50., 1.]])
        self.assertEqual(extract_first_turn_logits([99, 5, 1], [later, first, later], lookup, {99}), [3., 7.])
        self.assertEqual(extract_first_turn_logits([1], [first], lookup, set()), [10., 20.])
        self.assertIsNone(extract_first_turn_logits([99], [first], lookup, {99}))

    def test_generation_collects_recovery_logits_in_fixed_and_uncertainty_modes(self):
        for mode in (4, 'uncertainty'):
            for steps in (0, 2):
                with self.subTest(mode=mode, steps=steps):
                    agent = self.agent([], steps=steps)
                    agent.actions_per_replan = mode
                    agent.action_sequence_length = 18
                    agent.replan_action_range = (4, 8)
                    agent.device = 'cpu'
                    agent.view_mode = 'panorama'
                    agent.erp_top_crop_degrees = agent.erp_bottom_crop_degrees = 0.0
                    agent.current_images = agent.rgb_history = [Image.new('RGB', (2, 2))]
                    agent.eos_token_id = agent.pad_token_id = None
                    agent.tokenizer = SimpleNamespace(all_special_ids=[])
                    agent.action_token_lookup = {i: (i, (0, 1, 2, 3)) for i in range(4)}
                    inputs = BatchEncoding({'input_ids': torch.tensor([[8, 9]])})
                    agent.processor = Mock(return_value=inputs)
                    agent.processor.batch_decode.return_value = ['forward forward']
                    sequence = torch.tensor([[8, 9, FORWARD, FORWARD]])
                    logits = (torch.tensor([[0., 10., 7., 8.]]), torch.tensor([[0., 10., 9., 1.]]))
                    need_logits = mode == 'uncertainty' or steps > 0
                    output = SimpleNamespace(sequences=sequence, logits=logits) if need_logits else sequence
                    agent.model = Mock(config=SimpleNamespace(panovggt_enabled=False))
                    agent.model.generate.return_value = output
                    with patch('src.eval.eval.build_eval_generation_prompt', return_value='prompt'), \
                         patch('src.eval.eval.build_vln_image_geometry_batch', return_value=None):
                        self.assertEqual(agent.predict_inference(), 'forward forward')
                    self.assertEqual(agent.prediction_turn_logits, [7., 8.] if steps else None)
                    kwargs = agent.model.generate.call_args.kwargs
                    self.assertEqual(kwargs.get('output_logits', False), need_logits)
                    self.assertFalse(kwargs['do_sample'])


if __name__ == '__main__':
    unittest.main()
