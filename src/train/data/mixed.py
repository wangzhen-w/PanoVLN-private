import random
from collections import Counter
from typing import Dict, List, Tuple

from torch.utils.data import Dataset, Sampler


ManifestEntry = Tuple[str, int]
MIXING_STRATEGIES = {"sample", "task"}


class MixedSupervisedDataset(Dataset):
    def __init__(
        self,
        *,
        vln_dataset: Dataset,
        panoworld_dataset: Dataset,
        panoworld_keep_ratio: float,
        seed: int,
        shuffle: bool = True,
        mixing_strategy: str = "sample",
    ):
        keep_ratio = float(panoworld_keep_ratio)
        if keep_ratio < 0.0 or keep_ratio > 1.0:
            raise ValueError(
                "data.panoworld.keep_ratio must be in [0, 1], "
                f"got {panoworld_keep_ratio}"
            )
        if mixing_strategy not in MIXING_STRATEGIES:
            raise ValueError(
                "data.panoworld.mixing_strategy must be one of "
                f"{sorted(MIXING_STRATEGIES)}, got {mixing_strategy}"
            )

        self.datasets: Dict[str, Dataset] = {
            "vln": vln_dataset,
            "panoworld": panoworld_dataset,
        }
        self.shuffle = bool(shuffle)
        self.mixing_strategy = mixing_strategy
        rng = random.Random(int(seed))

        panoworld_count = int(round(len(panoworld_dataset) * keep_ratio))
        panoworld_count = max(0, min(len(panoworld_dataset), panoworld_count))
        panoworld_indices = list(range(len(panoworld_dataset)))
        rng.shuffle(panoworld_indices)
        panoworld_indices = panoworld_indices[:panoworld_count]

        manifest: List[ManifestEntry] = [("vln", index) for index in range(len(vln_dataset))]
        manifest.extend(("panoworld", index) for index in panoworld_indices)

        if shuffle:
            rng.shuffle(manifest)

        self.manifest = manifest
        self.source_counts = dict(Counter(source for source, _ in manifest))
        self.source_order = [
            source for source in ("vln", "panoworld")
            if self.source_counts.get(source, 0) > 0
        ]
        self.source_to_indices: Dict[str, List[int]] = {
            source: [] for source in self.source_order
        }
        for mixed_index, (source, _) in enumerate(manifest):
            self.source_to_indices[source].append(mixed_index)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int):
        source, sample_index = self.manifest[index]
        return self.datasets[source][sample_index]


class SourceGroupedSampler(Sampler[int]):
    def __init__(
        self,
        dataset: MixedSupervisedDataset,
        *,
        batch_size: int,
        seed: int,
        shuffle: bool = True,
        world_size: int = 1,
        gradient_accumulation_steps: int = 1,
        drop_last: bool = False,
    ):
        if not hasattr(dataset, "source_to_indices"):
            raise TypeError("SourceGroupedSampler requires a MixedSupervisedDataset")
        self.source_to_indices = {
            source: list(indices)
            for source, indices in dataset.source_to_indices.items()
            if indices
        }
        self.source_order = [
            source for source in dataset.source_order
            if source in self.source_to_indices
        ]
        self.batch_size = max(1, int(batch_size))
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.world_size = max(1, int(world_size))
        self.gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
        self.drop_last = bool(drop_last)
        self.block_batch_count = self.world_size * self.gradient_accumulation_steps
        self._iteration = 0

    def _make_source_batches(
        self,
        indices: List[int],
        rng: random.Random,
    ) -> List[List[int]]:
        indices = list(indices)
        if self.shuffle:
            rng.shuffle(indices)
        batches = []
        for start in range(0, len(indices), self.batch_size):
            batch = indices[start:start + self.batch_size]
            if len(batch) < self.batch_size:
                if self.drop_last:
                    continue
                while len(batch) < self.batch_size:
                    batch.append(rng.choice(indices))
            batches.append(batch)
        return batches

    def _make_source_blocks(
        self,
        batches: List[List[int]],
        rng: random.Random,
    ) -> List[List[List[int]]]:
        blocks = []
        for start in range(0, len(batches), self.block_batch_count):
            block = batches[start:start + self.block_batch_count]
            if len(block) < self.block_batch_count:
                if self.drop_last:
                    continue
                while len(block) < self.block_batch_count:
                    block.append(list(rng.choice(batches)))
            blocks.append(block)
        return blocks

    def __iter__(self):
        rng = random.Random(self.seed + self._iteration)
        self._iteration += 1
        blocks: List[List[List[int]]] = []
        for source in self.source_order:
            source_batches = self._make_source_batches(
                self.source_to_indices[source],
                rng,
            )
            if not source_batches:
                continue
            blocks.extend(self._make_source_blocks(source_batches, rng))
        if self.shuffle:
            rng.shuffle(blocks)
        for block in blocks:
            for batch in block:
                yield from batch

    def __len__(self) -> int:
        total_batches = 0
        for indices in self.source_to_indices.values():
            source_batches = len(indices) // self.batch_size
            if len(indices) % self.batch_size and not self.drop_last:
                source_batches += 1
            if source_batches == 0:
                continue
            source_blocks = source_batches // self.block_batch_count
            if source_batches % self.block_batch_count and not self.drop_last:
                source_blocks += 1
            total_batches += source_blocks * self.block_batch_count
        return total_batches * self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self._iteration = int(epoch)
