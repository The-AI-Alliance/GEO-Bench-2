# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""PASTIS DataModule."""

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

from geobench_v2.datasets import GeoBenchPASTIS

from .base import GeoBenchSegmentationDataModule


# TODO add timeseries argument
class GeoBenchPASTISDataModule(GeoBenchSegmentationDataModule):
    """GeoBench PASIS Data Module."""

    def __init__(
        self,
        img_size: int = 128,
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
            **kwargs: Additional keyword arguments for:class:`~geobench_v2.datasets.pastis.GeoBenchPASTIS`.
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

    def load_metadata(self) -> pd.DataFrame:
        """Load metadata file.

        Returns:
            pandas DataFrame with metadata.
        """
        return tacoreader.load(
            [os.path.join(self.kwargs["root"], f) for f in GeoBenchPASTIS.paths]
        )

    def visualize_batch(
        self, batch: dict[str, Any] | None = None, split: str = "train"
    ) -> tuple[Any, dict[str, Any]]:
        """Visualize a batch of data.

        Args:
            batch: Optional batch of data. If not provided, a batch will be fetched from the dataloader
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

        for k, v in batch.items():
            orig_dim = v.dim()
            if orig_dim == 4:  # CxTxHxW -> TxCxHxW
                batch[k] = batch[k].permute(1, 0, 2, 3)
            elif orig_dim == 5:  # BxCxTxHxW -> BxTxCxHxW
                batch[k] = batch[k].permute(0, 2, 1, 3, 4)

        if hasattr(self.data_normalizer, "unnormalize"):
            batch = self.data_normalizer.unnormalize(batch)

        batch_size = batch["mask"].shape[0]
        n_samples = min(8, batch_size)
        indices = torch.randperm(batch_size)[:n_samples]

        # Collect modality images and determine timesteps per modality
        modalities = {}
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
                # time series data [B, T, C H, W] -> [b, t, h, w, c]
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

        # Global layout: for each sample, stack timesteps vertically
        t_max = max(timesteps_per_mod.values()) if timesteps_per_mod else 1
        num_modalities = len(modalities) + 1  # +1 for mask column

        fig, axes = plt.subplots(
            n_samples * t_max,
            num_modalities,
            figsize=(num_modalities * 4, 3 * n_samples * t_max),
            gridspec_kw={"width_ratios": num_modalities * [1]},
        )

        if axes.ndim == 1:
            axes = axes.reshape(1, -1)

        # Add row labels for timesteps (t=0, t=1, ...)
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
        unique_classes = torch.unique(masks).cpu().numpy()
        unique_classes = [
            int(cls) for cls in unique_classes if cls < len(self.class_names)
        ]

        # use tab20 colormap to color the unique classes found
        cmap = plt.cm.tab20
        colors = {
            i: cmap(i) for i in range(len(self.class_names)) if i in unique_classes
        }
        class_cmap = plt.cm.colors.ListedColormap(colors.values())

        # Build legend handles once
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

        # Plot
        modality_list = list(modalities.keys())
        for i in range(n_samples):
            for j, mod in enumerate(modality_list):
                mod_images = modalities[mod]  # [b, t, h, w, c]
                t_len = timesteps_per_mod[mod]
                for t in range(t_max):
                    row_idx = i * t_max + t
                    ax = axes[row_idx, j]

                    if t < t_len:
                        plot_img = mod_images[i, t]
                        # Special handling for SAR style if applicable
                        if mod in ["s1_asc", "s1_desc"] and plot_img.shape[-1] >= 2:
                            vv = plot_img[..., 0]
                            vh = plot_img[..., 1]

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
                            img = percentile_normalization(plot_img, lower=2, upper=98)

                        ax.imshow(img)
                        if i == 0 and t == 0:
                            ax.set_title(f"{mod} image", fontsize=16)
                    else:
                        ax.axis("off")

                    ax.axis("off")

            # Mask column (last)
            for t in range(t_max):
                row_idx = i * t_max + t
                ax = axes[row_idx, -1]
                if t == 0:
                    mask_img = masks[i].cpu().numpy()
                    ax.imshow(
                        mask_img,
                        cmap=class_cmap,
                        vmin=0,
                        vmax=max(unique_classes) if unique_classes else 1,
                    )
                    ax.set_title("Mask", fontsize=16)
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

    def visualize_geolocation_distribution(self) -> None:
        """Visualize the geolocation distribution of the dataset."""
        pass
