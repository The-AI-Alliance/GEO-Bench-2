# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Flair 2 Aerial DataModule."""

from collections.abc import Callable
from typing import Any, Sequence

import torch

from geobench_v2.datasets import GeoBenchFLAIR2

from .base import GeoBenchSegmentationDataModule
import torch.nn as nn
from torch.utils.data import random_split


class GeoBenchFLAIR2DataModule(GeoBenchSegmentationDataModule):
    """GeoBench FLAIR2 Data Module."""

    def __init__(
        self,
        img_size: int,
        band_order: Sequence[float | str] = GeoBenchFLAIR2.band_default_order,
        batch_size: int = 32,
        eval_batch_size: int = 64,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        train_augmentations: nn.Module | None = None,
        eval_augmentations: nn.Module | None = None,
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize GeoBench FLAIR2 DataModule.

        Args:
            img_size: Image size
            batch_size: Batch size during training
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
            **kwargs: Additional keyword arguments
                :class:`~geobench_v2.datasets.flair2.GeoBenchFLAIR2`.
        """
        super().__init__(
            dataset_class=GeoBenchFLAIR2,
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

    def collect_metadata(self) -> None:
        """Collect metadata for the dataset."""
        pass

    def visualize_geolocation_distribution(self) -> None:
        """Visualize the geolocation distribution of the dataset."""
        pass
