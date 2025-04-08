# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""GeoBench WindTurbine DataModule."""

from collections.abc import Callable
from typing import Any, Sequence
import kornia.augmentation as K
import torch

import pandas as pd
from torch import Tensor
import os
import matplotlib.pyplot as plt


from geobench_v2.datasets import GeoBenchWindTurbine
from torch.utils.data import random_split

from .base import GeoBenchObjectDetectionDataModule
import torch.nn as nn

# TODO WindTurbine collate_fn check the different image sizes


def wind_turbine_collate_fn(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collate function for WindTurbine dataset.

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


class GeoBenchWindTurbineDataModule(GeoBenchObjectDetectionDataModule):
    """GeoBench WindTurbine Data Module."""

    def __init__(
        self,
        img_size: int = 512,
        band_order: Sequence[float | str] = GeoBenchWindTurbine.band_default_order,
        batch_size: int = 32,
        eval_batch_size: int = 64,
        num_workers: int = 0,
        collate_fn: Callable | None = wind_turbine_collate_fn,
        train_augmentations: nn.Module | None = None,
        eval_augmentations: nn.Module | None = None,
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize GeoBench DOTAV2 dataset module.

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
            dataset_class=GeoBenchWindTurbine,
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
            os.path.join(self.kwargs["root"], "geobench_windturbine.parquet")
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
