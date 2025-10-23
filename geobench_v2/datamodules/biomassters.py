# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Biomassters DataModule."""

import os
from collections.abc import Callable, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tacoreader
import torch
import torch.nn as nn
from einops import rearrange
from torchgeo.datasets.utils import percentile_normalization

from geobench_v2.datasets import GeoBenchBioMassters

from .base import GeoBenchSegmentationDataModule


class GeoBenchBioMasstersDataModule(GeoBenchSegmentationDataModule):
    """GeoBench BioMassters Data Module."""

    def __init__(
        self,
        img_size: int = 256,
        band_order: Sequence[float | str]
        | dict[str, Sequence[float | str]] = GeoBenchBioMassters.band_default_order,
        batch_size: int = 32,
        eval_batch_size: int = 64,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        train_augmentations: nn.Module | None = None,
        eval_augmentations: nn.Module | None = None,
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize GeoBench BioMassters DataModule.

        Args:
            img_size: Image size, original size is 256
            band_order: The order of bands to return in the sample
            batch_size: Batch size
            eval_batch_size: Evaluation batch size
            num_workers: Number of workers
            collate_fn: Collate function
            train_augmentations: Augmentations to apply during training
            eval_augmentations: Augmentations to apply during evaluation
            pin_memory: Pin memory
            **kwargs: Additional keyword arguments for :class:`geobench_v2.datasets.biomassters.GeoBenchBioMassters`
        """
        super().__init__(
            dataset_class=GeoBenchBioMassters,
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
        return tacoreader.load(
            [os.path.join(self.kwargs["root"], f) for f in GeoBenchBioMassters.paths]
        )

    def visualize_batch(
        self, batch: dict[str, Any] | None = None, split: str = "train"
    ) -> tuple[Any, dict[str, Any]]:
        """Visualize a batch of data.

        Args:
            batch: Batch of data
            split: One of 'train', 'validation', 'test'

        Returns:
            The matplotlib figure and the batch of data
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
        n_samples = min(4, batch_size)
        indices = torch.randperm(batch_size)[:n_samples]

        # Collect modality images and timesteps per modality
        modalities: dict[str, np.ndarray] = {}
        timesteps_per_mod: dict[str, int] = {}

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

            tensor = batch[f"image_{mod}"]
            if tensor.ndim == 5:
                # time series data [B, T, C, H, W] -> [b, t, h, w, c]
                mod_images = tensor[indices][:, :, mod_plot_indices, :, :]
                mod_images = (
                    rearrange(mod_images, "b t c h w -> b t h w c").cpu().numpy()
                )
                timesteps_per_mod[mod] = mod_images.shape[1]
            else:
                # single image data [B, C, H, W] -> [b, 1, h, w, c]
                mod_images = tensor[indices][:, mod_plot_indices, :, :]
                mod_images = rearrange(mod_images, "b c h w -> b 1 h w c").cpu().numpy()
                timesteps_per_mod[mod] = 1

            modalities[mod] = mod_images

        # Layout: for each sample, stack timesteps vertically
        t_max = max(timesteps_per_mod.values()) if timesteps_per_mod else 1
        num_columns = len(modalities) + 1  # +1 for mask

        fig, axes = plt.subplots(
            n_samples * t_max,
            num_columns,
            figsize=(num_columns * 4.2, 3.0 * n_samples * t_max),
            gridspec_kw={"width_ratios": num_columns * [1]},
        )
        if axes.ndim == 1:
            axes = axes.reshape(1, -1)

        # Add timestep row labels (t=0, t=1, ...)
        for i in range(n_samples):
            for t in range(t_max):
                row_idx = i * t_max + t
                ax_label = axes[row_idx, 0]
                ax_label.text(
                    -0.06,
                    0.5,
                    f"t={t}",
                    transform=ax_label.transAxes,
                    va="center",
                    ha="right",
                    fontsize=10,
                )

        masks = batch["mask"][indices]
        # Squeeze channel if present (B, 1, H, W) -> (B, H, W)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)
        masks_np = masks.cpu().numpy()
        vmin = float(np.nanmin(masks_np))
        vmax = float(np.nanmax(masks_np))

        # Plot modalities and regression mask with per-sample colorbar
        for i in range(n_samples):
            for j, mod in enumerate(modalities.keys()):
                mod_images = modalities[mod]  # [b, t, h, w, c]
                t_len = timesteps_per_mod[mod]
                for t in range(t_max):
                    row_idx = i * t_max + t
                    ax = axes[row_idx, j]

                    if t < t_len:
                        if mod == "s1":
                            vv = mod_images[i, t, :, :, 0]
                            vh = mod_images[i, t, :, :, 1]

                            vv = percentile_normalization(vv, lower=2, upper=98)
                            vh = percentile_normalization(vh, lower=2, upper=98)

                            ratio = np.divide(
                                vv, vh, out=np.zeros_like(vv), where=vh != 0
                            )

                            vv = np.clip(vv / 0.3, a_min=0, a_max=1)
                            vh = np.clip(vh / 0.05, a_min=0, a_max=1)
                            ratio = np.clip(ratio / 25, a_min=0, a_max=1)
                            img = np.stack((vv, vh, ratio), axis=2)
                        else:
                            img = percentile_normalization(
                                mod_images[i, t], lower=2, upper=98
                            )

                        ax.imshow(img)

                        if i == 0 and t == 0:
                            ax.set_title(f"{mod.upper()}", fontsize=14)
                    else:
                        ax.axis("off")
                    ax.axis("off")

            # Mask column at the end (only first t shown)
            for t in range(t_max):
                row_idx = i * t_max + t
                ax = axes[row_idx, -1]
                if t == 0:
                    mask_img = masks_np[i]
                    im = ax.imshow(mask_img, cmap="viridis", vmin=vmin, vmax=vmax)
                    ax.set_title("Target", fontsize=14)
                    # Add per-sample colorbar
                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=8)
                else:
                    ax.axis("off")
                ax.axis("off")

        plt.tight_layout()
        return fig, batch

    def visualize_geolocation_distribution(self) -> None:
        """Visualize the geolocation distribution of the dataset."""
        pass
