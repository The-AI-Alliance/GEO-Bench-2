# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""PASTIS DataModule."""

import os
from collections.abc import Callable, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torchgeo.datasets.utils import percentile_normalization

from geobench_v2.datasets import GeoBenchPASTIS

from .base import GeoBenchSegmentationDataModule

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

    def load_metadata(self) -> pd.DataFrame:
        """Load metadata file.

        Returns:
            pandas DataFrame with metadata.
        """
        return pd.read_parquet(
            os.path.join(self.kwargs["root"], "geobench_pastis.parquet")
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

        batch_size = batch["mask"].shape[0]
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

        num_modalities = len(modalities) + 1
        fig, axes = plt.subplots(
            n_samples,
            num_modalities,
            figsize=(num_modalities * 4, 3 * n_samples),
            gridspec_kw={"width_ratios": num_modalities * [1]},
        )

        if n_samples == 1:
            axes = axes.reshape(1, -1)

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

        for i in range(n_samples):
            for j, (mod, modality_img) in enumerate(modalities.items()):
                plot_img = modality_img[i]

                if mod in ["s1_asc", "s1_desc"]:
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

                ax = axes[i, j]

                ax.imshow(img)
                ax.set_title(f"{mod} image" if i == 0 else "", fontsize=20)
                ax.axis("off")

            ax = axes[i, -1]
            mask_img = masks[i].cpu().numpy()
            im = ax.imshow(mask_img, cmap=class_cmap, vmin=0, vmax=2)
            ax.set_title("Building Mask" if i == 0 else "", fontsize=20)
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
            ncol=3,
            fontsize=15,
            title="Classes",
            title_fontsize=15,
            columnspacing=100,
            mode="expand",
            borderaxespad=0.0,
        )

        plt.subplots_adjust(bottom=0.1)

        return fig, batch

    def visualize_geolocation_distribution(self) -> None:
        """Visualize the geolocation distribution of the dataset."""
        pass
