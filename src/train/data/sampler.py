import random
from collections import defaultdict
from typing import List, Optional

from torch.utils.data import Sampler


class TaskTypeBlockSampler(Sampler[int]):
    def __init__(
        self,
        task_types: List[Optional[str]],
        block_size: int,
        seed: int = 42,
    ):
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

        self.task_types = list(task_types)
        self.block_size = int(block_size)
        self.seed = int(seed)
        self.epoch = 0

        grouped_indices = defaultdict(list)
        for index, task_type in enumerate(self.task_types):
            grouped_indices[task_type].append(index)
        self.grouped_indices = dict(grouped_indices)

        self.total_size = 0
        for indices in self.grouped_indices.values():
            num_blocks = (len(indices) + self.block_size - 1) // self.block_size
            self.total_size += num_blocks * self.block_size

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return self.total_size

    def _build_full_block(self, shuffled_indices: List[int], start: int) -> List[int]:
        block = shuffled_indices[start:start + self.block_size]
        if len(block) == self.block_size:
            return block

        if not shuffled_indices:
            raise ValueError("Cannot build a task-type block from an empty index list")

        padded_block = list(block)
        pad_offset = 0
        while len(padded_block) < self.block_size:
            padded_block.append(shuffled_indices[pad_offset % len(shuffled_indices)])
            pad_offset += 1
        return padded_block

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)

        blocks = []
        for indices in self.grouped_indices.values():
            shuffled_indices = list(indices)
            rng.shuffle(shuffled_indices)
            for start in range(0, len(shuffled_indices), self.block_size):
                blocks.append(self._build_full_block(shuffled_indices, start))

        rng.shuffle(blocks)

        for block in blocks:
            for index in block:
                yield index
