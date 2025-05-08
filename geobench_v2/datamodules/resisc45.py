# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Resisc45 DataModule."""

import os
from collections.abc import Callable, Sequence
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
from torch import Tensor

from geobench_v2.datasets import GeoBenchRESISC45

from .base import GeoBenchClassificationDataModule


class GeoBenchRESISC45DataModule(GeoBenchClassificationDataModule):
    """GeoBench RESISC45 Data Module."""

    # https://github.com/microsoft/torchgeo/blob/68e0cfebcd18edb6605008eeeaba96388e63eca7/torchgeo/datamodules/resisc45.py#L21
    band_means = {"r": 93.89391792, "g": 97.11226906, "b": 87.56775284}

    band_stds = {"r": 51.84919672, "g": 47.2365918, "b": 47.06308786}

    def __init__(
        self,
        img_size: int = 256,
        band_order: Sequence[float | str] = GeoBenchRESISC45.band_default_order,
        batch_size: int = 32,
        eval_batch_size: int = 64,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        train_augmentations: nn.Module | None = None,
        eval_augmentations: nn.Module | None = None,
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize GeoBench Resisc45 dataset module.

        Args:
            img_size: Image size
            batch_size: Batch size
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
        """
        super().__init__(
            dataset_class=GeoBenchRESISC45,
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

    def load_metadata(self) -> pd.DataFrame:
        """Load metadata file.

        Returns:
            pandas DataFrame with metadata.
        """
        return pd.read_parquet(
            os.path.join(self.kwargs["root"], "geobench_resisc45.parquet")
        )

    def visualize_batch(
        self, split: str = "train"
    ) -> tuple[plt.Figure, dict[str, Tensor]]:
        """Visualize a batch of data.

        Args:
            split: One of 'train', 'val', 'test'

        Returns:
            The matplotlib figure and the batch of data
        """
        pass

    def visualize_geolocation_distribution(self) -> None:
        """Visualize the geolocation distribution of the dataset."""
        pass
