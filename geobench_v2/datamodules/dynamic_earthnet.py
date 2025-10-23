# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""DynamicEarthNet DataModule."""

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
from torchgeo.datasets.utils import percentile_normalization

from geobench_v2.datasets import GeoBenchDynamicEarthNet

from .base import GeoBenchSegmentationDataModule
from .utils import TimeSeriesResize


# TODO add timeseries argument
class GeoBenchDynamicEarthNetDataModule(GeoBenchSegmentationDataModule):
    """GeoBench DynamicEarthNet Data Module."""

    # TODO img_size will change to 512
    def __init__(
        self,
        img_size: int = 512,
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
            **kwargs: Additional keyword arguments for :class:`geobench_v2.datasets.dynamic_earthnet.GeoBenchDynamicEarthNet`
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
        return tacoreader.load(
            [
                os.path.join(self.kwargs["root"], f)
                for f in GeoBenchDynamicEarthNet.paths
            ]
        )

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

        for k, v in batch.items():
            orig_dim = v.dim()
            if orig_dim == 5:  # BxCxTxHxW -> BxTxCxHxW
                batch[k] = v.permute(0, 2, 1, 3, 4)

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
            figsize=(num_columns * 4.0, 2.6 * n_samples * t_max),  # slightly tighter
            gridspec_kw={
                "width_ratios": num_columns * [1],
                "wspace": 0.02,
                "hspace": 0.02,
            },
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

        # Prepare mask and legend
        masks = batch["mask"][indices]
        unique_classes = torch.unique(masks).cpu().numpy()
        unique_classes = sorted({int(c) for c in unique_classes if int(c) >= 0})

        class_names = getattr(self, "class_names", None)
        if not class_names or max(unique_classes, default=0) >= len(class_names):
            # Fallback numeric names
            class_names = [
                f"class {i}" for i in range(max(unique_classes, default=-1) + 1)
            ]

        cmap = plt.cm.tab20
        colors = {i: cmap(i % 20) for i in unique_classes}
        class_cmap = plt.cm.colors.ListedColormap(
            [colors[i] for i in unique_classes] or [(0, 0, 0, 1)]
        )

        legend_elements = []
        for cls_id in unique_classes:
            legend_elements.append(
                plt.Rectangle(
                    (0, 0),
                    1,
                    1,
                    color=colors[cls_id],
                    label=class_names[cls_id]
                    if cls_id < len(class_names)
                    else f"class {cls_id}",
                )
            )

        # Plot modalities
        for i in range(n_samples):
            for j, mod in enumerate(modalities.keys()):
                mod_images = modalities[mod]  # [b, t, h, w, c]
                t_len = timesteps_per_mod[mod]
                for t in range(t_max):
                    row_idx = i * t_max + t
                    ax = axes[row_idx, j]

                    if t < t_len:
                        img = mod_images[i, t]  # h, w, c
                        ax.imshow(percentile_normalization(img, lower=2, upper=98))

                        if i == 0 and t == 0:
                            ax.set_title(f"{mod.upper()}", fontsize=13)
                    else:
                        ax.axis("off")
                    ax.axis("off")

            # Mask column at the end (only first t shown)
            for t in range(t_max):
                row_idx = i * t_max + t
                ax = axes[row_idx, -1]
                if t == 0:
                    mask_img = masks[i].squeeze(0).cpu().numpy()
                    vmax = max(unique_classes) if unique_classes else 1
                    ax.imshow(
                        mask_img,
                        cmap=class_cmap,
                        vmin=min(unique_classes) if unique_classes else 0,
                        vmax=vmax,
                    )
                    ax.set_title("Label", fontsize=13)
                else:
                    ax.axis("off")
                ax.axis("off")

        # Compute legend layout
        n_classes = len(legend_elements)
        ncols = min(6, max(1, n_classes))

        legend = fig.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=ncols,
            fontsize=12.0,
            title="Classes",
            title_fontsize=14,
            frameon=False,
            # tighter legend paddings reduce vertical space
            borderaxespad=0.0,
            handlelength=0.9,
            handletextpad=0.3,
            columnspacing=0.8,
            labelspacing=0.2,
        )
        # Render once to get accurate legend bbox
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox = legend.get_window_extent(renderer=renderer)
        fig_w, fig_h = fig.get_size_inches()

        legend_h_frac = (bbox.height / (fig_h * fig.dpi)) + 0.006
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=1.0 - legend_h_frac)

        return fig, batch

    def visualize_geospatial_distribution(
        self,
        split_column: str = "tortilla:data_split",
        buffer_degrees: float = 5.0,
        sample_fraction: float | None = None,
        scale: Literal["10m", "50m", "110m"] = "50m",
        alpha: float = 0.8,
        s: float = 10,
    ) -> plt.Figure:
        """Visualize the geospatial distribution of dataset samples on a map.

        Creates a plot showing the geographic locations of samples, colored by dataset split
        (train, validation, test, extra_test). This helps to understand the spatial distribution
        and potential geographic biases in the dataset.

        Args:
            split_column: Column name in the metadata DataFrame that indicates the dataset split.
            buffer_degrees: Buffer around the data extent in degrees.
            sample_fraction: Optional fraction of samples to plot (0.0-1.0) for performance with large datasets.
            scale: Scale of cartopy features (e.g., '10m', '50m', '110m').
            alpha: Transparency of plotted points
            s: Size of plotted points.

        Returns:
            A matplotlib Figure object with the geospatial distribution plot.
        """
        return super().visualize_geospatial_distribution(
            split_column=split_column,
            buffer_degrees=buffer_degrees,
            sample_fraction=sample_fraction,
            scale=scale,
            alpha=alpha,
            s=s,
        )

    def setup_image_size_transforms(self) -> tuple[nn.Module, nn.Module, nn.Module]:
        """Setup image resizing transforms for train, val, test.

        Image resizing and normalization happens on dataset level on individual data samples.
        """
        return (
            TimeSeriesResize(self.img_size),
            TimeSeriesResize(self.img_size),
            TimeSeriesResize(self.img_size),
        )
