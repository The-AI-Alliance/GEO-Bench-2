# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Fields of the World DataModule."""

import os
from collections.abc import Callable, Sequence
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import tacoreader
import torch
import torch.nn as nn
from matplotlib.colors import ListedColormap
from torchgeo.datasets.utils import percentile_normalization

from geobench_v2.datasets import GeoBenchFieldsOfTheWorld

from .base import GeoBenchSegmentationDataModule


class GeoBenchFieldsOfTheWorldDataModule(GeoBenchSegmentationDataModule):
    """GeoBench Fields of the World Data Module."""

    def __init__(
        self,
        img_size: int = 256,
        band_order: Sequence[float | str] = GeoBenchFieldsOfTheWorld.band_default_order,
        batch_size: int = 32,
        eval_batch_size: int = 64,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        train_augmentations: nn.Module | None = None,
        eval_augmentations: nn.Module | None = None,
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize GeoBench Fields of the World DataModule.

        Args:
            img_size: Image size
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
            **kwargs: Additional keyword arguments for :class:`~geobench_v2.datasets.fotw.GeoBenchFieldsOfTheWorld`.
        """
        super().__init__(
            dataset_class=GeoBenchFieldsOfTheWorld,
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
            [
                os.path.join(self.kwargs["root"], f)
                for f in GeoBenchFieldsOfTheWorld.paths
            ]
        )
        return self.data_df

    def visualize_batch(
        self, batch: dict[str, Any] | None = None, split: str = "train"
    ) -> tuple[Any, dict[str, Any]]:
        """Visualize a batch of data from the Fields of the World dataset.

        Args:
            batch: Optional batch of data. If not provided, a batch will be fetched from the dataloader.
            split: One of 'train', 'validation', 'test'

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

        if hasattr(self.data_normalizer, "unnormalize"):
            batch = self.data_normalizer.unnormalize(batch)

        batch_size = batch["mask"].shape[0]
        n_samples = min(8, batch_size)
        indices = torch.randperm(batch_size)[:n_samples]

        # Determine available image types and setup columns
        image_types = [key for key in ["image_a", "image_b", "image"] if key in batch]
        num_cols = len(image_types) + 1  # +1 for mask

        fig, axes = plt.subplots(
            n_samples,
            num_cols,
            figsize=(num_cols * 4, 3 * n_samples),
            gridspec_kw={"width_ratios": num_cols * [1]},
        )

        if n_samples == 1:
            axes = axes.reshape(1, -1)

        # Get RGB indices once
        rgb_indices = []
        for band in ["red", "green", "blue"]:
            if band in self.band_order:
                rgb_indices.append(self.band_order.index(band))
        has_rgb = len(rgb_indices) == 3

        # Setup mask visualization
        masks = batch["mask"][indices]
        unique_classes = torch.unique(masks).cpu().numpy()
        unique_classes = [
            int(cls) for cls in unique_classes if cls < len(self.class_names)
        ]

        # Define colors for the classes
        colors = {0: "black", 1: "green", 2: "yellow"}
        class_colors = [colors[i] for i in range(len(colors))]
        field_cmap = ListedColormap(class_colors)

        # Create legend elements
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

        # Visualization function for any image type
        def visualize_image(img_tensor, ax, title=""):
            if has_rgb:
                # Use RGB bands if available
                display_img = img_tensor[rgb_indices].permute(1, 2, 0).cpu().numpy()
            else:
                # Otherwise use first three bands
                display_img = img_tensor[:3].permute(1, 2, 0).cpu().numpy()

            display_img = percentile_normalization(display_img, lower=2, upper=98)
            ax.imshow(display_img)
            ax.set_title(title, fontsize=20)
            ax.axis("off")

        # Plot each sample
        for i in range(n_samples):
            # Plot each image type
            for j, img_type in enumerate(image_types):
                img = batch[img_type][indices[i]]
                ax = axes[i, j]
                title = f"{img_type.replace('_', ' ').title()}" if i == 0 else ""
                visualize_image(img, ax, title)

            # Plot mask
            ax = axes[i, len(image_types)]
            mask_img = masks[i].cpu().numpy()
            ax.imshow(mask_img, cmap=field_cmap, vmin=0, vmax=2)
            ax.set_title("Field Mask" if i == 0 else "", fontsize=20)
            ax.axis("off")

        plt.tight_layout()

        if legend_elements:
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
