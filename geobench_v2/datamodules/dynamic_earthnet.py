# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""DynamicEarthNet DataModule."""

from collections.abc import Callable
from typing import Any, Sequence

from geobench_v2.datasets import GeoBenchDynamicEarthNet

import pandas as pd
from torch import Tensor
import os
import matplotlib.pyplot as plt

import torch
from .base import GeoBenchSegmentationDataModule
import torch.nn as nn
from torch import Tensor




# TODO add timeseries argument
class GeoBenchDynamicEarthNetDataModule(GeoBenchSegmentationDataModule):
    """GeoBench DynamicEarthNet Data Module."""

    def __init__(
        self,
        img_size: int,
        band_order: Sequence[float | str] = GeoBenchDynamicEarthNet.band_default_order,
        batch_size: int = 32,
        eval_batch_size: int = 64,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        train_augmentations: nn.Module | None = None,
        eval_augmentations: nn.Module | None = None,
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize GeoBench DynamicEarthNet DataModule.

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
            **kwargs: Additional keyword arguments to
                :class:`~geobench_v2.datasets.DynamicEarthNet.GeoBenchDynamicEarthNet`.
        """
        super().__init__(
            dataset_class=GeoBenchDynamicEarthNet,
            band_order=band_order,
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

    def load_metadata(self) -> pd.DataFrame:
        """Load metadata file.

        Returns:
            pandas DataFrame with metadata.
        """
        return pd.read_parquet(
            os.path.join(self.kwargs["root"], "geobench_dynamic_earthnet.parquet")
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
