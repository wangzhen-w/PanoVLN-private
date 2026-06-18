import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Sequence, Tuple

from torch.utils.data import Sampler


@dataclass(frozen=True)
class TraceSample:
    episode_key: str
    step_index: int
    end_step_index: int
    action_sequence: Tuple[str, ...]


class TraceSampler(Sampler[int]):
    """TRACE: Temporal Route-Action Conflict Exclusion sampler."""

    def __init__(
        self,
        episode_keys: Sequence[str],
        step_indices: Sequence[int],
        end_step_indices: Sequence[int],
        action_sequences: Sequence[Sequence[str]],
        window_size: int = 1,
        conflict_step_window: int = 4,
        seed: int = 0,
        shuffle: bool = True,
    ):
        lengths = {
            len(episode_keys),
            len(step_indices),
            len(end_step_indices),
            len(action_sequences),
        }
        if len(lengths) != 1:
            raise ValueError("TRACE metadata fields must have identical lengths")

        self.samples = [
            TraceSample(
                episode_key=str(episode_key),
                step_index=int(step_index),
                end_step_index=int(end_step_index),
                action_sequence=tuple(str(action) for action in action_sequence),
            )
            for episode_key, step_index, end_step_index, action_sequence in zip(
                episode_keys,
                step_indices,
                end_step_indices,
                action_sequences,
            )
        ]
        self.window_size = max(1, int(window_size))
        self.conflict_step_window = max(0, int(conflict_step_window))
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        self.num_episode_groups = len({sample.episode_key for sample in self.samples})

    def __len__(self) -> int:
        return len(self.samples)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        return iter(self._build_epoch_order())

    def _build_epoch_order(self) -> List[int]:
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        base_order = list(range(len(self.samples)))
        if self.shuffle:
            rng.shuffle(base_order)
        else:
            base_order.sort(
                key=lambda index: (
                    self.samples[index].episode_key,
                    self.samples[index].step_index,
                )
            )

        pending: Deque[int] = deque(base_order)
        carry: Deque[int] = deque()
        order: List[int] = []

        while pending or carry:
            window: List[int] = []
            blocked: Deque[int] = deque()
            window_by_episode: Dict[str, List[int]] = defaultdict(list)
            scan_budget = len(carry) + len(pending)

            for _ in range(scan_budget):
                if len(window) >= self.window_size:
                    break

                if carry:
                    sample_index = carry.popleft()
                else:
                    sample_index = pending.popleft()

                if self._conflicts_with_window(sample_index, window_by_episode):
                    blocked.append(sample_index)
                    continue

                self._append_to_window(sample_index, window, window_by_episode)

            # If every remaining candidate conflicts with this partially built
            # window, keep epoch progress by accepting the conflict instead of
            # dropping or oversampling.
            while len(window) < self.window_size and blocked:
                sample_index = blocked.popleft()
                self._append_to_window(sample_index, window, window_by_episode)

            if not window:
                break

            order.extend(window)

            if blocked:
                carry = deque(list(blocked) + list(carry))

        return order

    def _append_to_window(
        self,
        sample_index: int,
        window: List[int],
        window_by_episode: Dict[str, List[int]],
    ) -> None:
        window.append(sample_index)
        sample = self.samples[sample_index]
        window_by_episode[sample.episode_key].append(sample_index)

    def _conflicts_with_window(
        self,
        sample_index: int,
        window_by_episode: Dict[str, List[int]],
    ) -> bool:
        sample = self.samples[sample_index]
        for existing_index in window_by_episode.get(sample.episode_key, []):
            if self._samples_conflict(sample, self.samples[existing_index]):
                return True
        return False

    def _samples_conflict(self, left: TraceSample, right: TraceSample) -> bool:
        if left.action_sequence == right.action_sequence:
            return False

        step_close = abs(left.step_index - right.step_index) <= self.conflict_step_window
        interval_overlap = (
            left.step_index <= right.end_step_index
            and right.step_index <= left.end_step_index
        )
        return step_close or interval_overlap
