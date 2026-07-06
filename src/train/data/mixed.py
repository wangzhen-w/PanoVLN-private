import random
from collections import Counter
from typing import Dict, List, Tuple

from torch.utils.data import Dataset


ManifestEntry = Tuple[str, int]


class MixedSupervisedDataset(Dataset):
    def __init__(
        self,
        *,
        vln_dataset: Dataset,
        panoworld_dataset: Dataset,
        panoworld_keep_ratio: float,
        seed: int,
        shuffle: bool = True,
    ):
        keep_ratio = float(panoworld_keep_ratio)
        if keep_ratio < 0.0 or keep_ratio > 1.0:
            raise ValueError(
                "data.panoworld.keep_ratio must be in [0, 1], "
                f"got {panoworld_keep_ratio}"
            )

        self.datasets: Dict[str, Dataset] = {
            "vln": vln_dataset,
            "panoworld": panoworld_dataset,
        }
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

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int):
        source, sample_index = self.manifest[index]
        return self.datasets[source][sample_index]
