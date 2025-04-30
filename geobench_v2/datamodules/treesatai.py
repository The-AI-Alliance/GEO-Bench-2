# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""TreeSatAI dataset."""

from collections.abc import Callable
from typing import Any, Sequence

import pandas as pd
from torch import Tensor
import torch
import numpy as np
from einops import rearrange
from torchgeo.datasets.utils import percentile_normalization
import os
import matplotlib.pyplot as plt

from geobench_v2.datasets import GeoBenchTreeSatAI

from .base import GeoBenchClassificationDataModule
import torch.nn as nn


class GeoBenchTreeSatAIDataModule(GeoBenchClassificationDataModule):
    """GeoBench TreeSatAI Data Module."""

    def __init__(
        self,
        img_size: int = 304,
        band_order: Sequence[float | str] = GeoBenchTreeSatAI.band_default_order,
        batch_size: int = 32,
        eval_batch_size: int = 64,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        train_augmentations: nn.Module | None = None,
        eval_augmentations: nn.Module | None = None,
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize GeoBench TreeSatAI dataset module.

        Args:
            img_size: Image size originally 304
            batch_size: Batch size
            eval_batch_size: Evaluation batch size
            num_workers: Number of workers
            collate_fn: Collate function
            pin_memory: Pin memory
            **kwargs: Additional keyword arguments
        """
        super().__init__(
            dataset_class=GeoBenchTreeSatAI,
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
        return pd.read_parquet(
            os.path.join(self.kwargs["root"], "geobench_treesatai.parquet")
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

        labels = batch["label"][indices]
        sample_labels = []
        for i in range(n_samples):
            present_labels = torch.where(labels[i] == 1)[0].cpu().tolist()
            sample_labels.append(present_labels)

        for i in range(n_samples):
            for j, (mod, modality_img) in enumerate(modalities.items()):
                plot_img = modality_img[i]

                img = percentile_normalization(plot_img, lower=2, upper=98)

                ax = axes[i, j]
                ax.imshow(img)
                ax.set_title(f"{mod} image" if i == 0 else "", fontsize=20)
                ax.axis("off")

            label_names = [self.class_names[label] for label in sample_labels[i]]
            separator = ", \n"
            suptitle = f"Labels: {separator.join(label_names)}"
            ax = axes[i, -1]
            ax.set_title(suptitle, fontsize=8)

        plt.tight_layout()

        plt.subplots_adjust(bottom=0.1)

        return fig, batch

    def visualize_geolocation_distribution(self) -> None:
        """Visualize the geolocation distribution of the dataset."""
        pass
