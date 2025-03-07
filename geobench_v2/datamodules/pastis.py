# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""PASTIS DataModule."""

from collections.abc import Callable
from typing import Any, Sequence

from geobench_v2.datasets import GeoBenchPASTIS

import torch
from .base import GeoBenchSegmentationDataModule
import torch.nn as nn
from torch import Tensor


# def pastis_collate_fn(batch: Sequence[dict[str, Any]]) -> dict[str, Tensor]:
#     """Collate function for PASTIS dataset to deal with timeseries

#     Args:
#         batch: A list of samples from PASTIS dataset

#     Returns:
#         A dictionary containing the collated batch
#     """
#     collated_batch = {}
#     # deal with various timeseries, collate to min-number of time steps
#     min_time_steps = min([sample["image"].shape[0] for sample in batch])
#     images = [sample["image"][:min_time_steps] for sample in batch]
#     images = torch.stack(images, dim=0)
#     collated_batch["image"] = images

#     collate_batch["mask"] = torch.stack([sample["mask"] for sample in batch], dim=0)

#     return collated_batch


# TODO add timeseries argument
class GeoBenchPASTISDataModule(GeoBenchSegmentationDataModule):
    """GeoBench PASIS Data Module."""

    #

    def __init__(
        self,
        img_size: int,
        band_order: Sequence[float | str] = GeoBenchPASTIS.band_default_order,
        batch_size: int = 32,
        eval_batch_size: int = 64,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        train_augmentations: nn.Module | None = None,
        eval_augmentations: nn.Module | None = None,
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize GeoBench PASIS DataModule.

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
                :class:`~geobench_v2.datasets.pastis.GeoBenchPASTIS`.
        """
        super().__init__(
            dataset_class=GeoBenchPASTIS,
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

    def visualize_geolocation_distribution(self) -> None:
        """Visualize geolocation distribution."""
        pass
