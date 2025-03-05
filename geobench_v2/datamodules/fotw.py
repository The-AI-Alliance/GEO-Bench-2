# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Fields of the World DataModule."""

from collections.abc import Callable
from typing import Any

import torch

from geobench_v2.datasets import GeoBenchFieldsOfTheWorld

from .base import GeoBenchSegmentationDataModule


class GeoBenchFieldsOfTheWorldDataModule(GeoBenchSegmentationDataModule):
    """GeoBench Fields of the World Data Module."""

    # https://github.com/microsoft/torchgeo/blob/592f8926c1601bc94d0936f91196425b590b369d/torchgeo/datamodules/ftw.py#L21C5-L22C31
    mean = torch.tensor([0])
    std = torch.tensor([3000])

    # TODO also compute other band statistics

    band_means = {"red": 0.0, "green": 0.0, "blue": 0.0, "nir": 0.0}

    band_stds = {"red": 1.0, "green": 1.0, "blue": 1.0, "nir": 1.0}

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
        """Initialize GeoBench Fields of the World DataModule.

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
                :class:`~geobench_v2.datasets.fotw.GeoBenchFieldsOfTheWorld`.
        """
        super().__init__(
            dataset_class=GeoBenchFieldsOfTheWorld,
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
        """Setup data for train, val, test.

        Args:
            stage: One of 'fit', 'validate', 'test', or 'predict'.
        """
        self.train_dataset = self.dataset_class(split="train", **self.kwargs)
        self.val_dataset = self.dataset_class(split="val", **self.kwargs)
        self.test_dataset = self.dataset_class(split="test", **self.kwargs)

    def collect_metadata(self) -> None:
        """Collect metadata for the dataset."""
        pass

    def visualize_geolocation_distribution(self) -> None:
        """Visualize the geolocation distribution of the dataset."""
        pass
