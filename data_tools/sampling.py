# Utils
import numpy as np
from typing import Optional
from enum import Enum

# Sklearn
from sklearn.utils import class_weight

# Torch
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from pathlib import Path


# Enum for different sampling strategies
class Sampling(Enum):
    NUMPY = 1
    SKLEARN = 2
    NONE = 3


class SamplerFactory:
    @staticmethod
    def create_sampler(
        sampling: Sampling,
        train_dataset: Dataset,
        train_labels: Optional[list],
    ) -> Optional[WeightedRandomSampler]:
        """Create a sampler based on the specified sampling strategy.

        Args:
        ----
            sampling (Sampling): Enum value indicating the sampling strategy.
            train_dataset (Dataset): Training dataset.
            train_labels (optional): Train labels.

        Returns:
        -------
            Sampler or None: Sampler instance based on the specified strategy,
            or None if no sampler is needed.

        """
        if sampling == Sampling.NONE:
            return None

        if sampling == Sampling.NUMPY:
            class_counts = np.array(
                [np.sum(train_labels == c) for c in np.unique(train_labels)],
            )
            class_weights = 1 / class_counts

            return WeightedRandomSampler(class_weights, len(train_dataset))
        # else
        class_weights = class_weight.compute_class_weight(
            class_weight="balanced",
            classes=np.unique(train_labels),
            y=train_labels,
        )
        return WeightedRandomSampler(class_weights, len(train_dataset))


def sample_patient_images(
    directory: str,
    image_num: int,
    patient_dir: str,
    label: str,
    strategy: str = "FIFO",
) -> list[int]:
    if strategy == "FIFO":
        return torch.arange(1, image_num + 1, dtype=torch.int64).tolist()
    patient_dir = Path(f"{directory}/{label}s/{patient_dir}")
    images = [
        patient_image.name
        for patient_image in patient_dir.glob("*.jpg")
        if patient_image.is_file()
    ]
    max_images = len(images)
    rng = torch.Generator()
    end_idx = torch.randint(17, max_images - 1, (1,), generator=rng).item()
    start_idx = max_images - end_idx
    indices = torch.linspace(start_idx, end_idx, steps=image_num)
    indices = torch.clamp(indices, start_idx, end_idx - 1).to(torch.int64)

    return indices.tolist()
