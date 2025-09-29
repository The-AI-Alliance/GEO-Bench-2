# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""SpaceNet8 DataModule."""

import os
from collections.abc import Callable, Sequence
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import tacoreader
import torch
import torch.nn as nn
from einops import rearrange
from matplotlib.colors import ListedColormap
from torchgeo.datasets.utils import percentile_normalization

from geobench_v2.datasets import GeoBenchSpaceNet8

from .base import GeoBenchSegmentationDataModule


class GeoBenchSpaceNet8DataModule(GeoBenchSegmentationDataModule):
    """GeoBench SpaceNet8 Data Module."""

    def __init__(
        self,
        img_size: int = 512,
        band_order: Sequence[float | str] = GeoBenchSpaceNet8.band_default_order,
        batch_size: int = 32,
        eval_batch_size: int = 64,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        train_augmentations: nn.Module | None = None,
        eval_augmentations: nn.Module | None = None,
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize GeoBench SpaceNet8 dataset module.

        Args:
            img_size: Image size, created patches are of size 512
            band_order: The order of bands to return in the sample
            batch_size: Batch size during training
            eval_batch_size: Evaluation batch size
            num_workers: Number of workers
            collate_fn: Collate function
            train_augmentations: Transforms/Augmentations to apply during training, they will be applied
                at the sample level and should include normalization. See :meth:`define_augmentations`
                for the default transformation.
            eval_augmentations: Transforms/Augmentations to apply during evaluation, they will be applied
                at the sample level and should include normalization. See :meth:`define_augmentations`
                for the default transformation.
            pin_memory: Pin memory
            **kwargs: Additional keyword arguments for the dataset class
        """
        super().__init__(
            dataset_class=GeoBenchSpaceNet8,
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
        self.data_df = tacoreader.load(
            [os.path.join(self.kwargs["root"], f) for f in GeoBenchSpaceNet8.paths]
        )
        return self.data_df

    def visualize_batch(
        self, batch: dict[str, Any] | None = None, split: str = "train"
    ) -> tuple[Any, dict[str, Any]]:
        """Visualize a batch of data.

        Args:
            batch: Optional batch of data. If not provided, a batch will be fetched from the dataloader.
            split: One of 'train', 'validation', 'test'

        Returns:
            The matplotlib figure and the batch of data
        """
        if batch is None:
            if split == "train":
                batch = next(iter(self.train_dataloader()))
            elif split == "validation":
                batch = next(iter(self.val_dataloader()))
            else:
                batch = next(iter(self.test_dataloader()))

        if hasattr(self.data_normalizer, "unnormalize"):
            batch = self.data_normalizer.unnormalize(batch)

        pre_images = batch["image_pre"]
        post_images = batch["image_post"]
        masks = batch["mask"]

        batch_size = pre_images.shape[0]
        n_samples = min(8, batch_size)
        indices = torch.randperm(batch_size)[:n_samples]

        pre_images = pre_images[indices]
        post_images = post_images[indices]
        masks = masks[indices]

        plot_bands = self.dataset_band_config.plot_bands
        rgb_indices = [
            self.band_order.index(band)
            for band in plot_bands
            if band in self.band_order
        ]
        pre_images = pre_images[:, rgb_indices, :, :]
        post_images = post_images[:, rgb_indices, :, :]

        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3 * n_samples))

        if n_samples == 1:
            axes = axes.reshape(1, -1)

        unique_classes = torch.unique(masks).cpu().numpy()
        unique_classes = [
            int(cls) for cls in unique_classes if cls < len(self.class_names)
        ]

        colors = {
            0: "#000000",  # Black for background
            1: "#4daf4a",  # Green for road (not flooded)
            2: "#377eb8",  # Blue for road (flooded)
            3: "#e41a1c",  # Red for building (not flooded)
            4: "#984ea3",  # Purple for building (flooded)
        }

        class_colors = [colors[i] for i in range(len(colors))]
        flood_cmap = ListedColormap(class_colors)

        for i in range(n_samples):
            # pre-event image
            ax = axes[i, 0]
            img = rearrange(pre_images[i], "c h w -> h w c").cpu().numpy()
            img = percentile_normalization(img, lower=2, upper=98)
            ax.imshow(img)
            ax.set_title("Aerial Pre Image" if i == 0 else "")
            ax.axis("off")

            # post-event image
            ax = axes[i, 1]
            img = rearrange(post_images[i], "c h w -> h w c").cpu().numpy()
            img = percentile_normalization(img, lower=2, upper=98)
            ax.imshow(img)
            ax.set_title("Aerial Post Image" if i == 0 else "")
            ax.axis("off")

            ax = axes[i, 2]
            mask_img = masks[i].cpu().numpy()

            ax.imshow(mask_img, cmap=flood_cmap, vmin=0, vmax=4)
            ax.set_title("Flood Mask" if i == 0 else "")
            ax.axis("off")

        legend_elements = []
        for cls in unique_classes:
            if cls in colors:
                legend_elements.append(
                    plt.Rectangle(
                        (0, 0),
                        1,
                        1,
                        color=colors[cls],
                        label=f"{cls}: {self.class_names[cls]}"
                        if cls < len(self.class_names)
                        else f"Class {cls}",
                    )
                )

        plt.tight_layout()

        # Only add legend if we have legend elements
        if legend_elements:
            fig.legend(
                handles=legend_elements,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.01),
                ncol=len(legend_elements),
                frameon=True,
                fontsize=12,
            )
            plt.subplots_adjust(bottom=0.1)

        return fig, batch

    def visualize_geolocation_distribution(self) -> None:
        """Visualize the geolocation distribution of the dataset."""
        pass
