# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Burn Scars DataMdule."""

import os
from collections.abc import Callable, Sequence
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import tacoreader
import torch
import torch.nn as nn
from torch import Tensor
from torchgeo.datasets.utils import percentile_normalization

from geobench_v2.datasets.burn_scars import GeoBenchBurnScars

from .base import GeoBenchSegmentationDataModule


class GeoBenchBurnScarsDataModule(GeoBenchSegmentationDataModule):
    """GeoBench Burn Scars Data Module."""

    def __init__(
        self,
        img_size: int = 512,
        band_order: Sequence[float | str] = GeoBenchBurnScars.band_default_order,
        batch_size: int = 32,
        eval_batch_size: int = 64,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        train_augmentations: nn.Module | None = None,
        eval_augmentations: nn.Module | None = None,
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize GeoBench Burn Scars dataset module.

        Args:
            img_size: Image size
            band_order: The order of bands to return in the sample
            batch_size: Batch size during
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
            **kwargs: Additional keyword arguments for :class:`geobench_v2.datasets.burn_scars.GeoBenchBurnScars`
        """
        super().__init__(
            dataset_class=GeoBenchBurnScars,
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
            [os.path.join(self.kwargs["root"], f) for f in GeoBenchBurnScars.paths]
        )
        return self.data_df

    def visualize_batch(
        self, batch: dict[str, Tensor] | None = None, split: str = "train"
    ) -> tuple[plt.Figure, dict[str, Tensor]]:
        """Visualize a batch of data.

        Args:
            batch: Batch of data to visualize
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

        images = batch["image"]
        masks = batch["mask"]

        n_samples = min(8, images.shape[0])
        indices = torch.randperm(images.shape[0])[:n_samples]

        images = images[indices]
        masks = masks[indices]

        plot_bands = self.dataset_band_config.plot_bands
        plot_indices = [self.band_order.index(band) for band in plot_bands]
        images = images[:, plot_indices, :, :]

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

        cmap = plt.cm.tab20

        for i in range(n_samples):
            ax = axes[i, 0]
            img = images[i].cpu().numpy()
            img = percentile_normalization(img, lower=2, upper=98).transpose((1, 2, 0))
            ax.imshow(img, cmap="gray")
            ax.set_title("HLS Image" if i == 0 else "")
            ax.axis("off")

            ax = axes[i, 1]
            mask_img = masks[i].cpu().numpy()
            ax.imshow(mask_img, cmap="tab20", vmin=0, vmax=19)
            ax.set_title("Mask" if i == 0 else "")
            ax.axis("off")

            ax = axes[i, 2]
            ax.axis("off")

            if i == 0:
                legend_elements = []
                for cls in unique_classes:
                    if cls < len(self.class_names):
                        color = cmap(cls / 20.0 if cls < 20 else 0)
                        legend_elements.append(
                            plt.Rectangle(
                                (0, 0),
                                1,
                                1,
                                color=color,
                                label=f"{cls}: {self.class_names[cls]}",
                            )
                        )

                ax.legend(
                    handles=legend_elements,
                    loc="center",
                    frameon=True,
                    fontsize="small",
                    title="Classes",
                )

        plt.tight_layout()
        return fig, batch

    def visualize_geolocation_distribution(self) -> None:
        """Visualize the geolocation distribution of the dataset."""
        pass
