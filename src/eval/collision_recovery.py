"""Bounded recovery using executed actions, collision feedback and RGB change."""

STOP, FORWARD, LEFT, RIGHT = range(4)
DEFAULT_COLLISION_RECOVERY_STEPS = 2


def validate_collision_recovery_steps(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("collision-recovery-steps must be a nonnegative integer (0 disables)")
    return value


class CollisionRecovery:
    """Keep a chosen turn direction until two forward actions are collision-free.

    observe() consumes feedback for the last executed action, before the next
    action is selected. None preserves the pending queue; [] requests a fresh
    model plan. A requested recovery preserves the old queue until choose_queue()
    can use its next turn, after giving the new model's STOP plan priority.
    """

    MAX_RGB_MAE = 0.5  # 64 x 32 RGB thumbnails, with pixel values in [0, 255].
    MAX_HEADING_ATTEMPTS = 24  # One full rotation with the existing 15-degree turns.
    FREE_FORWARD_STEPS = 2
    COOLDOWN_STEPS = 10

    def __init__(self, steps=DEFAULT_COLLISION_RECOVERY_STEPS):
        self.steps = validate_collision_recovery_steps(steps)
        self.reset()

    @property
    def enabled(self):
        return self.steps > 0

    def reset(self):
        self.count = 0
        self.requested = False
        self.heading = None
        self.free_forward_count = 0
        self.attempts = 0
        self.cooldown = 0
        self.terminal_queue = False

    def observe(self, action, collided, rgb_mae=None):
        if not self.enabled or action is None:
            return None
        if self.heading is not None:
            if action == FORWARD:
                if collided:
                    self.free_forward_count = 0
                    self.attempts += 1
                    if self.attempts > self.MAX_HEADING_ATTEMPTS:
                        self.heading = None
                        self.cooldown = self.COOLDOWN_STEPS
                        self.count = 0
                        return []
                    return [self.heading, FORWARD]
                self.free_forward_count += 1
                if self.free_forward_count >= self.FREE_FORWARD_STEPS:
                    self.heading = None
                    self.count = 0
                    return []
                return [FORWARD]
        elif self.terminal_queue:
            self.count = 0
        elif self.cooldown:
            self.cooldown -= 1
            self.count = 0
        elif action == FORWARD and collided and rgb_mae is not None and rgb_mae <= self.MAX_RGB_MAE:
            self.count += 1
            self.requested = self.count >= self.steps
        else:
            self.count = 0
        return None

    def choose_queue(self, prediction, default_queue, pending_queue, first_turn_logits):
        if not self.enabled:
            return list(default_queue)
        self.terminal_queue = STOP in default_queue
        queue = list(default_queue)
        if self.terminal_queue:
            self.count = 0
            self.heading = None
        elif self.requested:
            turn = next((a for a in pending_queue if a in (LEFT, RIGHT)), None)
            if turn is None:
                before_stop = prediction[:prediction.index(STOP)] if STOP in prediction else prediction
                turn = next((a for a in before_stop if a in (LEFT, RIGHT)), None)
            if turn is None:
                if first_turn_logits is None:
                    raise ValueError("Collision recovery requires first-action LEFT/RIGHT logits")
                turn = LEFT if first_turn_logits[0] >= first_turn_logits[1] else RIGHT
            self.heading = turn
            self.free_forward_count = 0
            self.attempts = 1
            self.count = 0
            queue = [turn, FORWARD]
        self.requested = False
        return queue
