import unittest

from src.eval.collision_recovery import CollisionRecovery, FORWARD, LEFT, RIGHT, STOP


class CollisionRecoveryTests(unittest.TestCase):
    def request_recovery(self, recovery):
        for _ in range(recovery.steps):
            recovery.observe(FORWARD, True, 0.0)
        self.assertTrue(recovery.requested)

    def start_recovery(self, recovery):
        self.request_recovery(recovery)
        return recovery.choose_queue([FORWARD, LEFT], [FORWARD], [], [1.0, 0.0])

    def test_disabled_preserves_normal_queue(self):
        recovery = CollisionRecovery(0)
        for _ in range(10):
            self.assertIsNone(recovery.observe(FORWARD, True, 0.0))
        self.assertFalse(recovery.requested)
        self.assertEqual(recovery.choose_queue([LEFT], [FORWARD, RIGHT], [], None), [FORWARD, RIGHT])

    def test_threshold_and_static_rgb_are_both_required(self):
        recovery = CollisionRecovery(2)
        recovery.observe(FORWARD, True, 0.5)
        self.assertFalse(recovery.requested)
        recovery.observe(FORWARD, True, 0.51)
        self.assertEqual(recovery.count, 0)
        recovery.observe(FORWARD, True, 0.5)
        recovery.observe(FORWARD, True, 0.5)
        self.assertTrue(recovery.requested)

    def test_turn_or_collision_free_forward_breaks_streak(self):
        for action, collided in [(LEFT, False), (RIGHT, True), (FORWARD, False)]:
            with self.subTest(action=action, collided=collided):
                recovery = CollisionRecovery(2)
                recovery.observe(FORWARD, True, 0.0)
                recovery.observe(action, collided, 0.0)
                recovery.observe(FORWARD, True, 0.0)
                self.assertFalse(recovery.requested)

    def test_streak_survives_a_normal_replan(self):
        recovery = CollisionRecovery(2)
        recovery.observe(FORWARD, True, 0.0)
        recovery.choose_queue([FORWARD] * 18, [FORWARD] * 4, [], None)
        recovery.observe(FORWARD, True, 0.0)
        self.assertTrue(recovery.requested)

    def test_three_steps_remains_configurable(self):
        recovery = CollisionRecovery(3)
        recovery.observe(FORWARD, True, 0.0)
        recovery.observe(FORWARD, True, 0.0)
        self.assertFalse(recovery.requested)
        recovery.observe(FORWARD, True, 0.0)
        self.assertTrue(recovery.requested)

    def test_pending_turn_wins_over_fresh_direction(self):
        recovery = CollisionRecovery()
        self.request_recovery(recovery)
        queue = recovery.choose_queue([FORWARD, LEFT], [FORWARD, LEFT], [FORWARD, RIGHT, RIGHT], [9.0, 1.0])
        self.assertEqual(queue, [RIGHT, FORWARD])

    def test_use_next_predicted_turn_before_logits(self):
        recovery = CollisionRecovery()
        self.request_recovery(recovery)
        self.assertEqual(recovery.choose_queue([FORWARD, RIGHT], [FORWARD], [], [9.0, 1.0]), [RIGHT, FORWARD])

    def test_fallback_ignores_turns_after_stop(self):
        recovery = CollisionRecovery()
        self.request_recovery(recovery)
        self.assertEqual(recovery.choose_queue([FORWARD, STOP, RIGHT], [FORWARD], [], [9.0, 1.0]), [LEFT, FORWARD])

    def test_new_stop_plan_overrides_recovery_and_pending_turn(self):
        recovery = CollisionRecovery()
        self.request_recovery(recovery)
        terminal = [FORWARD] * 11 + [STOP]
        self.assertEqual(recovery.choose_queue(terminal, terminal, [RIGHT], None), terminal)
        for _ in range(11):
            self.assertIsNone(recovery.observe(FORWARD, True, 0.0))
            self.assertFalse(recovery.requested)

    def test_native_stop_queue_is_also_protected(self):
        recovery = CollisionRecovery()
        terminal = [FORWARD, FORWARD, FORWARD, STOP]
        recovery.choose_queue(terminal, terminal, [], None)
        for _ in range(3):
            recovery.observe(FORWARD, True, 0.0)
        self.assertFalse(recovery.requested)

    def test_two_free_forward_steps_return_control_to_model(self):
        recovery = CollisionRecovery()
        self.assertEqual(self.start_recovery(recovery), [LEFT, FORWARD])
        self.assertIsNone(recovery.observe(LEFT, False))
        self.assertEqual(recovery.observe(FORWARD, True), [LEFT, FORWARD])
        recovery.observe(LEFT, False)
        self.assertEqual(recovery.observe(FORWARD, False), [FORWARD])
        self.assertEqual(recovery.observe(FORWARD, False), [])
        self.assertIsNone(recovery.heading)
        self.assertFalse(recovery.requested)

    def test_second_forward_collision_resumes_same_turn_direction(self):
        recovery = CollisionRecovery()
        self.start_recovery(recovery)
        recovery.observe(LEFT, False)
        recovery.observe(FORWARD, False)
        self.assertEqual(recovery.observe(FORWARD, True), [LEFT, FORWARD])
        self.assertEqual(recovery.observe(FORWARD, False), [FORWARD])

    def test_search_exhaustion_and_ten_action_cooldown(self):
        recovery = CollisionRecovery()
        self.start_recovery(recovery)
        for attempt in range(24):
            recovery.observe(LEFT, False)
            queue = recovery.observe(FORWARD, True)
            self.assertEqual(queue, [LEFT, FORWARD] if attempt < 23 else [])
        self.assertIsNone(recovery.heading)
        recovery.choose_queue([FORWARD] * 18, [FORWARD] * 4, [], None)
        for _ in range(10):
            recovery.observe(FORWARD, True, 0.0)
            self.assertEqual(recovery.count, 0)
            self.assertFalse(recovery.requested)
        recovery.observe(FORWARD, True, 0.0)
        self.assertFalse(recovery.requested)
        recovery.observe(FORWARD, True, 0.0)
        self.assertTrue(recovery.requested)

    def test_episode_reset_clears_recovery_and_terminal_state(self):
        recovery = CollisionRecovery()
        self.start_recovery(recovery)
        recovery.reset()
        self.assertIsNone(recovery.heading)
        self.assertFalse(recovery.requested)
        recovery.choose_queue([STOP], [STOP], [], None)
        recovery.reset()
        self.request_recovery(recovery)

    def test_invalid_thresholds_are_rejected(self):
        for value in (-1, True, 1.5, '2'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                CollisionRecovery(value)


if __name__ == '__main__':
    unittest.main()
