"""Shared simulation/real-world action uncertainty policy.

No Habitat, model, or torch imports: the robot client only needs scalar budgets;
logit extraction uses tensors supplied by the inference server.
"""

import math
from typing import Optional, Sequence

ATOMIC_ACTION_NAMES = ("stop", "forward", "left", "right")
STOP_ACTION_ID = 0
DEFAULT_REPLAN_ACTION_RANGE = (4, 8)
DEFAULT_STOP_COMMIT_MAX_ACTIONS = 12
DEFAULT_UNCERTAINTY_BUDGET = 1.2


def _validate_action_count(value):
    if isinstance(value, bool):
        raise ValueError("action count must be a positive integer")
    value = int(value)
    if value <= 0:
        raise ValueError("action count must be a positive integer")
    return value


def validate_uncertainty_budget(value):
    budget = float(value)
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("uncertainty-budget must be finite and positive")
    return budget


def validate_replan_action_range(action_range):
    if not isinstance(action_range, (tuple, list)) or len(action_range) != 2:
        raise ValueError("replan-action-range requires two positive integers: MIN MAX")
    minimum, maximum = map(_validate_action_count, action_range)
    if minimum > maximum:
        raise ValueError("replan-action-range requires MIN <= MAX")
    return minimum, maximum


def validate_stop_commit_max_actions(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("stop-commit-max-actions must be a nonnegative integer (0 disables)")
    return value


def select_stop_commit_horizon(
    action_ids: Sequence[int],
    stop_commit_max_actions: int = DEFAULT_STOP_COMMIT_MAX_ACTIONS,
) -> Optional[int]:
    """Commit through the first STOP in the inclusive window, counting STOP itself."""
    limit = validate_stop_commit_max_actions(stop_commit_max_actions)
    for position, action_id in enumerate(action_ids[:limit], start=1):
        if action_id == STOP_ACTION_ID:
            return position
    return None


def select_uncertainty_horizon(
    action_uncertainties, budget, action_range=DEFAULT_REPLAN_ACTION_RANGE,
):
    """Longest prefix within action_range with sum(-log p(action)) <= budget.

    The lower bound is the minimum nominal K, even if it exceeds the budget.
    STOP and short predictions are subsequently handled by the action queue.
    This is model uncertainty, not a calibrated probability of execution error.
    """
    budget = validate_uncertainty_budget(budget)
    minimum, maximum = validate_replan_action_range(action_range)
    horizon, cumulative = minimum, 0.0
    for position, uncertainty in enumerate(action_uncertainties[:maximum], start=1):
        if not math.isfinite(uncertainty) or uncertainty < 0:
            raise ValueError("Action uncertainty must be finite and nonnegative")
        cumulative += uncertainty
        if cumulative > budget:
            break
        horizon = max(minimum, position)
    return horizon


def build_action_token_lookup(tokenizer):
    """Map both first-word and space-prefixed action tokens to four-way logits."""
    lookup = {}
    for prefix in ("", " "):
        encoded = [tokenizer.encode(prefix + name, add_special_tokens=False)
                   for name in ATOMIC_ACTION_NAMES]
        if any(len(ids) != 1 for ids in encoded):
            raise ValueError("Uncertainty mode requires one-token canonical actions")
        token_ids = tuple(ids[0] for ids in encoded)
        if len(set(token_ids)) != len(ATOMIC_ACTION_NAMES):
            raise ValueError("Action token IDs must be distinct")
        for action_id, token_id in enumerate(token_ids):
            lookup[token_id] = (action_id, token_ids)
    return lookup


def extract_action_uncertainties(
    generated_ids, logits, token_lookup, special_ids,
    max_actions=DEFAULT_REPLAN_ACTION_RANGE[1],
):
    """Read raw, temperature-1 four-action log probabilities from this generation."""
    max_actions = _validate_action_count(max_actions)
    if logits is None or len(generated_ids) != len(logits):
        raise ValueError("Generated tokens and raw logits are not aligned")
    actions, uncertainties = [], []
    for token_id, raw_logits in zip(generated_ids, logits):
        if token_id in special_ids:
            continue
        if token_id not in token_lookup:
            raise ValueError(f"Uncertainty mode received a noncanonical action token: {token_id}")
        action_id, token_ids = token_lookup[token_id]
        action_logits = raw_logits[0, list(token_ids)].float()
        uncertainty = -action_logits.log_softmax(dim=-1)[action_id].item()
        if not math.isfinite(uncertainty):
            raise ValueError("Non-finite action logits in uncertainty mode")
        actions.append(action_id)
        uncertainties.append(uncertainty)
        if action_id == STOP_ACTION_ID or len(actions) == max_actions:
            break
    return actions, uncertainties
