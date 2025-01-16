# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""SpaceNet6 DataModule."""

from collections.abc import Callable
from typing import Any

from geobench_v2.datasets import GeoBenchSpaceNet6

from .base import GeoBenchSegmentationDataModule


class GeoBenchSpaceNet6DataModule(GeoBenchSegmentationDataModule):
    """GeoBench SpaceNet6 Data Module."""

    #

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
        """Initialize SpaceNet6 dataset module.

        Args:
            img_size: Image size
            batch_size: Batch size during training
            eval_batch_size: Evaluation batch size
            num_workers: Number of workers
            collate_fn: Collate function
            pin_memory: Pin memory
            **kwargs: Additional keyword arguments for the dataset class
        """
        super().__init__(
            dataset_class=GeoBenchSpaceNet6,
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
        self.train_dataset = self.dataset_class(split="train", **self.kwargs)
        self.val_dataset = self.dataset_class(split="val", **self.kwargs)

    def visualize_geolocation_distribution(self) -> None:
        """Visualize the geolocation distribution of the dataset."""
        pass
