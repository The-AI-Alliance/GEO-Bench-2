# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""GeoBench BigEarthNetV2 DataModule."""

from collections.abc import Callable
from typing import Any, Sequence
import kornia.augmentation as K
import torch


from geobench_v2.datasets import GeoBenchEverWatch
from torch.utils.data import random_split

from .base import GeoBenchObjectDetectionDataModule
import torch.nn as nn

# TODO everwatch collate_fn check the different image sizes


def everwatch_collate_fn(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collate function for EverWatch dataset.

    Args:
        batch: A list of dictionaries containing the data for each sample

    Returns:
        A dictionary containing the collated data
    """
    # collate images
    images = [sample["image"] for sample in batch]
    images = torch.stack(images, dim=0)

    # collate boxes into list of boxes
    boxes = [sample["bbox_xyxy"] for sample in batch]
    label = [sample["label"] for sample in batch]

    return {"image": images, "bbox_xyxy": boxes, "label": label}


class GeoBenchEverWatchDataModule(GeoBenchObjectDetectionDataModule):
    """GeoBench EverWatch Data Module."""

    # norm stats
    band_means = {"red": 0.0, "green": 0.0, "blue": 0.0}
    band_stds = {"red": 1.0, "green": 1.0, "blue": 1.0}

    def __init__(
        self,
        img_size: int,
        band_order: Sequence[float | str] = GeoBenchEverWatch.band_default_order,
        batch_size: int = 32,
        eval_batch_size: int = 64,
        num_workers: int = 0,
        collate_fn: Callable | None = everwatch_collate_fn,
        train_augmentations: nn.Module | None = None,
        eval_augmentations: nn.Module | None = None,
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize GeoBench CaFFe dataset module.

        Args:
            img_size: Image size
            batch_size: Batch size during
            eval_batch_size: Evaluation batch size
            num_workers: Number of workers
            collate_fn: Collate function
            train_augmentations: Transforms/Augmentations to apply during training, they will be applied
                at the sample level and should include normalization. See :method:`define_augmentations`
                for the default transformation.
            eval_augmentations: Transforms/Augmentations to apply during evaluation, they will be applied
                at the sample level and should include normalization. See :method:`define_augmentations`
                for the default transformation.
            pin_memory: Pin memory
            **kwargs: Additional keyword arguments for the dataset class
        """
        super().__init__(
            dataset_class=GeoBenchEverWatch,
            img_size=img_size,
            band_order=band_order,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            train_augmentations=train_augmentations,
            eval_augmentations=eval_augmentations,
            pin_memory=pin_memory,
            **kwargs,
        )

    def setup(self, stage: str) -> None:
        """Setup the dataset for training or evaluation."""
        train_dataset = self.dataset_class(
            split="train", band_order=self.band_order, **self.kwargs
        )
        # split into train and validation
        generator = torch.Generator().manual_seed(0)
        # random 80-20 split
        self.train_dataset, self.val_dataset = random_split(
            train_dataset, [1 - 0.2, 0.2], generator
        )

        self.test_dataset = self.dataset_class(
            split="test", band_order=self.band_order, **self.kwargs
        )

    def visualize_geolocation_distribution(self) -> None:
        """Visualize geolocation distribution."""
        raise AttributeError("EverWAtch does not have geolocation information.")
