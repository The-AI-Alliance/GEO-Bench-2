# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Resisc45 dataset module."""

from collections.abc import Callable
from typing import Any

from ..datasets.resisc45 import GeoBenchRESISC45
from .base import GeoBenchClassificationDataModule


class GeoBenchRESISC45DataModule(GeoBenchClassificationDataModule):
    """GeoBench RESISC45 Data Module."""

    # https://github.com/microsoft/torchgeo/blob/68e0cfebcd18edb6605008eeeaba96388e63eca7/torchgeo/datamodules/resisc45.py#L21
    band_means = {"r": 93.89391792, "g": 97.11226906, "b": 87.56775284}

    band_stds = {"r": 51.84919672, "g": 47.2365918, "b": 47.06308786}

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
        """Initialize Resisc45 dataset module.

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
            dataset_class=GeoBenchRESISC45,
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

    def visualize_geolocation_distribution(self) -> None:
        """Visualize geolocation distribution."""
        raise AttributeError("RESISC45 does not have geolocation information.")
