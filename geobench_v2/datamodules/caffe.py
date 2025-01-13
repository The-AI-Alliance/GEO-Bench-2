# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""CaFFe Dataset."""

from collections.abc import Callable
from typing import Any

from .base import GeoBenchSegmentationDataModule


class GeoBenchCaFFeDataModule(GeoBenchSegmentationDataModule):
    """GeoBench CaFFe Data Module."""

    band_means = {"gray": 0.5517}
    band_stds = {"gray": 11.8478}

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
        """Initialize CaFFe dataset module.

        Args:
            img_size: Image size
            batch_size: Batch size
            eval_batch_size: Evaluation batch size
            num_workers: Number of workers
            collate_fn: Collate function
            pin_memory: Pin memory
            **kwargs: Additional keyword arguments for the dataset class
        """

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
