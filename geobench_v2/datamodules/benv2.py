# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""GeoBench BigEarthNetV2 DataModule."""

import os
from collections.abc import Callable, Sequence
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tacoreader
import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torchgeo.datasets.utils import percentile_normalization

from geobench_v2.datasets import GeoBenchBENV2

from .base import GeoBenchClassificationDataModule


class GeoBenchBENV2DataModule(GeoBenchClassificationDataModule):
    """GeoBench BigEarthNetV2 Data Module."""

    has_extra_test_samples = True

    def __init__(
        self,
        img_size: int = 120,
        band_order: Sequence[float | str] = GeoBenchBENV2.band_default_order,
        batch_size: int = 32,
        eval_batch_size: int = 64,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        train_augmentations: nn.Module | None = None,
        eval_augmentations: nn.Module | None = None,
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize GeoBench BigEarthNetV2 dataset module.

        Args:
            img_size: Image size
            band_order: The order of bands to return in the sample
            batch_size: Batch size
            eval_batch_size: Evaluation batch size
            num_workers: Number of workers
            collate_fn: Collate function
            train_augmentations: Augmentations to apply during training
            eval_augmentations: Augmentations to apply during evaluation
            pin_memory: Pin memory
            **kwargs: Additional keyword arguments
        """
        super().__init__(
            dataset_class=GeoBenchBENV2,
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
            [os.path.join(self.kwargs["root"], f) for f in GeoBenchBENV2.paths]
        )
        return self.data_df

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

        if hasattr(self.data_normalizer, "unnormalize"):
            batch = self.data_normalizer.unnormalize(batch)

        batch_size = batch["label"].shape[0]
        n_samples = min(8, batch_size)
        indices = torch.randperm(batch_size)[:n_samples]

        modalities = {}

        for mod in self.band_order.keys():
            mod_plot_bands = self.dataset_band_config.modalities[mod].plot_bands
            missing_bands = [
                band for band in mod_plot_bands if band not in self.band_order[mod]
            ]
            if missing_bands:
                raise AssertionError(
                    f"Plotting bands {missing_bands} for modality '{mod}' not found in band_order {self.band_order[mod]}"
                )

            # Get plot indices for bands that exist
            mod_plot_indices = [
                self.band_order[mod].index(band) for band in mod_plot_bands
            ]
            mod_images = batch[f"image_{mod}"][:, mod_plot_indices, :, :][indices]
            mod_images = rearrange(mod_images, "b c h w -> b h w c").cpu().numpy()
            modalities[mod] = mod_images

        num_modalities = len(modalities)
        fig = plt.figure(figsize=(num_modalities * 4 + 2, 3 * n_samples))
        gs = fig.add_gridspec(
            n_samples, num_modalities + 1, width_ratios=[*[1] * num_modalities, 0.4]
        )

        labels = batch["label"][indices]
        sample_labels = []
        for i in range(n_samples):
            present_labels = torch.where(labels[i] == 1)[0].cpu().tolist()
            sample_labels.append(present_labels)

        for i in range(n_samples):
            for j, (mod, modality_img) in enumerate(modalities.items()):
                ax = fig.add_subplot(gs[i, j])
                plot_img = modality_img[i]

                if mod == "s1":
                    vv = plot_img[..., 0]
                    vh = plot_img[..., 1]

                    vv = percentile_normalization(vv, lower=2, upper=98)
                    vh = percentile_normalization(vh, lower=2, upper=98)

                    ratio = np.divide(vv, vh, out=np.zeros_like(vv), where=vh != 0)

                    vv = np.clip(vv / 0.3, a_min=0, a_max=1)
                    vh = np.clip(vh / 0.05, a_min=0, a_max=1)
                    ratio = np.clip(ratio / 25, a_min=0, a_max=1)
                    img = np.stack((vv, vh, ratio), axis=2)
                else:
                    img = percentile_normalization(plot_img, lower=2, upper=98)

                ax.imshow(img)
                ax.set_title(f"{mod} image" if i == 0 else "", fontsize=20)
                ax.axis("off")

            label_ax = fig.add_subplot(gs[i, -1])
            label_ax.axis("off")

            label_names = [f"- {self.class_names[label]}" for label in sample_labels[i]]

            label_text = "\n".join(label_names)

            label_ax.text(
                0.05,
                0.5,
                label_text,
                ha="left",
                va="center",
                fontsize=9,
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor="lightyellow",
                    alpha=0.8,
                    edgecolor="lightgray",
                ),
                transform=label_ax.transAxes,
                wrap=True,
            )

            if i == 0:
                label_ax.set_title("Labels", fontsize=15)

        plt.tight_layout()

        plt.subplots_adjust(bottom=0.1)

        return fig, batch

    def visualize_geospatial_distribution(
        self,
        output_path: str = None,
        split_column="tortilla:data_split",
        buffer_degrees: float = 5.0,
        sample_fraction: float = 1.0,
        scale: Literal["10m", "50m", "110m"] = "50m",
        alpha: float = 0.5,
        s: float = 0.5,
    ) -> None:
        """Visualize the geospatial distribution of the dataset.

        Args:
            output_path: Path to save the visualization
            split_column: Column name for the split
            buffer_degrees: Buffer size in degrees
            sample_fraction: Fraction of samples to visualize
            scale: Scale of the map
            alpha: Alpha value for the points
            s: Size of the points
        """
        return super().visualize_geospatial_distribution(split_column=split_column)
