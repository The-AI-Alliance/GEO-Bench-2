# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""GeoBench BigEarthNetV2 DataModule."""

from collections.abc import Callable
from typing import Any

from geobench_v2.datasets import GeoBenchBENV2
import kornia.augmentation as K

from .base import GeoBenchClassificationDataModule


class GeoBenchBENV2DataModule(GeoBenchClassificationDataModule):
    """GeoBench BigEarthNetV2 Data Module."""

    # Normalization stats

    def __init__(
        self,
        img_size: int,
        batch_size: int = 32,
        eval_batch_size: int = 64,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        train_augmentations: nn.Module | None = None,
        eval_augmentations: nn.Module | None = None,
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize GeoBench BigEarthNetV2 dataset module.

        Args:
            img_size: Image size
            batch_size: Batch size
            eval_batch_size: Evaluation batch size
            num_workers: Number of workers
            collate_fn: Collate function
            pin_memory: Pin memory
            **kwargs: Additional keyword arguments
        """
        super().__init__(
            dataset_class=GeoBenchBENV2,
            img_size=img_size,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            train_augmentations=train_augmentations,
            eval_augmentations=eval_augmentations,
            pin_memory=pin_memory,
            **kwargs,
        )

    def setup(self, stage: str | None = None) -> None:
        """Setup the dataset.

        Args:
            stage: Stage
        """
        norm_transform = K.AugmentationSequential(
            K.Normalize(self.mean, self.std, keepdim=True), data_keys=["image", "mask"]
        )
        self.train_dataset = self.dataset_class(split="train", **self.kwargs)
        self.val_dataset = self.dataset_class(split="val", **self.kwargs)
        self.test_dataset = self.dataset_class(split="test", **self.kwargs)

    def visualize_geolocation_distribution(self) -> None:
        """Visualize the geolocation distribution."""
        raise NotImplementedError
