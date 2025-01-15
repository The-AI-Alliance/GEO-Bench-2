# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Fields of the World DataModule."""

from collections.abc import Callable
from typing import Any

import torch

from ..datasets.fotw import GeoBenchFieldsOfTheWorld
from .base import GeoBenchSegmentationDataModule


class GeoBenchFieldsOfTheWorldDataModule(GeoBenchSegmentationDataModule):
    """GeoBench Fields of the World Data Module."""

    # https://github.com/microsoft/torchgeo/blob/592f8926c1601bc94d0936f91196425b590b369d/torchgeo/datamodules/ftw.py#L21C5-L22C31
    mean = torch.tensor([0])
    std = torch.tensor([3000])

    # TODO also compute other band statistics

    def __init__(
        self,
        img_size: int,
        batch_size: int = 32,
        eval_batch_size: int = 64,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize Fields of the World dataset module.

        Args:
            img_size: Image size
            batch_size: Batch size during training
            eval_batch_size: Evaluation batch size
            num_workers: Number of workers
            collate_fn: Collate function
            pin_memory: Pin memory
            **kwargs: Additional keyword arguments
        """
        super().__init__(
            dataset_class=GeoBenchFieldsOfTheWorld,
            img_size=img_size,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            **kwargs,
        )

    def setup(self, stage: str | None = None) -> None:
        """Setup data for train, val, test.

        Args:
            stage: One of 'fit', 'validate', 'test', or 'predict'.
        """
        self.dataset_train = self.dataset_class(split="train", **self.kwargs)
        self.dataset_val = self.dataset_class(split="val", **self.kwargs)
        self.dataset_test = self.dataset_class(split="test", **self.kwargs)

    def visualize_geolocation_distribution(self) -> None:
        """Visualize the geolocation distribution of the dataset."""
        pass
