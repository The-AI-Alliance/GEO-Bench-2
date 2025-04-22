# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""SpaceNet7 DataModule."""

from collections.abc import Callable
from typing import Any, Sequence
import matplotlib.pyplot as plt
import pandas as pd
import os
from torch import Tensor
import torch
import numpy as np
from torchgeo.datasets.utils import percentile_normalization
from einops import rearrange


from geobench_v2.datasets import GeoBenchSpaceNet7

from .base import GeoBenchSegmentationDataModule
import torch.nn as nn


class GeoBenchSpaceNet7DataModule(GeoBenchSegmentationDataModule):
    """GeoBench SpaceNet7 Data Module."""

    #

    def __init__(
        self,
        img_size: int = 512,
        band_order: Sequence[float | str] = GeoBenchSpaceNet7.band_default_order,
        batch_size: int = 32,
        eval_batch_size: int = 64,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        train_augmentations: nn.Module | None = None,
        eval_augmentations: nn.Module | None = None,
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize GeoBench SpaceNet7 dataset module.

        Args:
            img_size: Image size, created patches are of size 512
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
            **kwargs: Additional keyword arguments for the dataset class
        """
        super().__init__(
            dataset_class=GeoBenchSpaceNet7,
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

    def visualize_batch(
        self, split: str = "train"
    ) -> tuple[plt.Figure, dict[str, Tensor]]:
        """Visualize a batch of data.

        Args:
            split: One of 'train', 'val', 'test'

        Returns:
            The matplotlib figure and the batch of data

        Raises:
            AssertionError: If bands needed for plotting are missing
        """
        if split == "train":
            batch = next(iter(self.train_dataloader()))
        elif split == "validation":
            batch = next(iter(self.val_dataloader()))
        else:
            batch = next(iter(self.test_dataloader()))

        batch = self.data_normalizer.unnormalize(batch)

        images = batch["image"]
        masks = batch["mask"]

        n_samples = min(8, images.shape[0])
        indices = torch.randperm(images.shape[0])[:n_samples]

        images = images[indices]
        masks = masks[indices]

        plot_bands = self.dataset_band_config.plot_bands
        rgb_indices = [
            self.band_order.index(band)
            for band in plot_bands
            if band in self.band_order
        ]
        images = images[:, rgb_indices, :, :]
        # Create figure with 3 columns: image, mask, and legend
        fig, axes = plt.subplots(
            n_samples,
            3,
            figsize=(12, 3 * n_samples),
            gridspec_kw={"width_ratios": [1, 1, 0.5]},
        )

        if n_samples == 1:
            axes = axes.reshape(1, -1)

        unique_classes = torch.unique(masks).cpu().numpy()
        unique_classes = [
            int(cls) for cls in unique_classes if cls < len(self.class_names)
        ]

        fig, axes = plt.subplots(n_samples, 2, figsize=(9, 3 * n_samples))

        if n_samples == 1:
            axes = axes.reshape(1, -1)

        masks = batch["mask"][indices]
        unique_classes = torch.unique(masks).cpu().numpy()
        unique_classes = [
            int(cls) for cls in unique_classes if cls < len(self.class_names)
        ]

        colors = {0: "black", 1: "gray", 2: "orange"}

        from matplotlib.colors import ListedColormap

        class_colors = [colors[i] for i in range(len(colors))]
        build_cmap = ListedColormap(class_colors)

        for i in range(n_samples):
            ax = axes[i, 0]
            img = rearrange(images[i], "c h w -> h w c").cpu().numpy()
            img = percentile_normalization(img, lower=2, upper=98)
            ax.imshow(img)
            ax.set_title("Aerial Image" if i == 0 else "")
            ax.axis("off")

            ax = axes[i, 1]
            mask_img = masks[i].cpu().numpy()
            im = ax.imshow(mask_img, cmap=build_cmap, vmin=0, vmax=2)
            ax.set_title("Building Mask" if i == 0 else "")
            ax.axis("off")

            if i == 0:
                legend_elements = []
                for cls in unique_classes:
                    if cls < len(self.class_names) and cls in colors:
                        legend_elements.append(
                            plt.Rectangle(
                                (0, 0),
                                1,
                                1,
                                color=colors[cls],
                                label=f"{self.class_names[cls]}",
                            )
                        )

        plt.tight_layout()

        fig.legend(
            handles=legend_elements,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=len(legend_elements),
            frameon=True,
            fontsize=20,
        )

        plt.subplots_adjust(bottom=0.1)

        return fig, batch

    def visualize_geolocation_distribution(self) -> None:
        """Visualize the geolocation distribution of the dataset."""
        pass

    def load_metadata(self) -> pd.DataFrame:
        """Load metadata file.

        Returns:
            pandas DataFrame with metadata.
        """
        return pd.read_parquet(
            os.path.join(self.kwargs["root"], "geobench_spacenet7.parquet")
        )
