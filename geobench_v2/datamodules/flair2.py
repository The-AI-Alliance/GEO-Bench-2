# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Flair 2 Aerial DataModule."""

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

from geobench_v2.datasets import GeoBenchFLAIR2

from .base import GeoBenchSegmentationDataModule


class GeoBenchFLAIR2DataModule(GeoBenchSegmentationDataModule):
    """GeoBench FLAIR2 Data Module."""

    def __init__(
        self,
        img_size: int = 512,
        band_order: Sequence[float | str] = GeoBenchFLAIR2.band_default_order,
        batch_size: int = 32,
        eval_batch_size: int = 64,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        train_augmentations: nn.Module | None = None,
        eval_augmentations: nn.Module | None = None,
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize GeoBench FLAIR2 DataModule.

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
            **kwargs: Additional keyword arguments
                :class:`~geobench_v2.datasets.flair2.GeoBenchFLAIR2`.
        """
        super().__init__(
            dataset_class=GeoBenchFLAIR2,
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
            [os.path.join(self.kwargs["root"], f) for f in GeoBenchFLAIR2.paths]
        )
        return self.data_df

    def visualize_batch(
        self, batch: dict[str, Any] | None = None, split: str = "train"
    ) -> tuple[Any, dict[str, Any]]:
        """Visualize a batch of data.

        Args:
            batch: A batch of data
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
        fig, axes = plt.subplots(
            n_samples,
            num_modalities,
            figsize=(num_modalities * 4, 3 * n_samples),
            gridspec_kw={"width_ratios": num_modalities * [1]},
        )

        if n_samples == 1 and num_modalities == 1:
            axes = np.array([[axes]])
        elif n_samples == 1:
            axes = axes.reshape(1, -1)
        elif num_modalities == 1:
            axes = axes.reshape(-1, 1)

        masks = batch["mask"][indices]

        unique_classes = torch.unique(masks).cpu().numpy()
        unique_classes = [
            int(cls) for cls in unique_classes if cls < len(self.class_names)
        ]

        cmap = plt.cm.tab20

        for i in range(n_samples):
            for j, (mod, modality_img) in enumerate(modalities.items()):
                plot_img = modality_img[i]
                img = percentile_normalization(plot_img, lower=2, upper=98)
                ax = axes[i, j]
                ax.imshow(img)
                ax.set_title(f"{mod} image" if i == 0 else "", fontsize=20)
                ax.axis("off")

            ax = axes[i, -1]
            mask_img = masks[i].cpu().numpy()
            ax.imshow(mask_img, cmap=cmap, vmin=0, vmax=19)
            ax.set_title("Mask" if i == 0 else "")
            ax.axis("off")

            if i == 0:
                legend_elements = []
                for class_val in unique_classes:
                    if class_val < len(self.class_names):
                        color = cmap(class_val / 20.0 if class_val < 20 else 0)
                        legend_elements.append(
                            plt.Rectangle(
                                (0, 0),
                                1,
                                1,
                                color=color,
                                label=f"{class_val}: {self.class_names[class_val]}",
                            )
                        )

                ax.legend(
                    handles=legend_elements,
                    loc="center left",
                    bbox_to_anchor=(1.05, 0.5),
                    frameon=True,
                    fontsize="small",
                    title="Classes",
                )

        plt.tight_layout()
        return fig, batch

    def visualize_geolocation_distribution(self) -> None:
        """Visualize the geolocation distribution of the dataset."""
        pass
