# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""GeoBench So2Sat DataModule."""

import os
from collections.abc import Callable, Sequence
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import tacoreader
import torch
import torch.nn as nn
from torchgeo.datasets.utils import percentile_normalization

from geobench_v2.datasets import GeoBenchForestnet

from .base import GeoBenchClassificationDataModule


class GeoBenchForestnetDataModule(GeoBenchClassificationDataModule):
    """GeoBench Forestnet Data Module."""

    def __init__(
        self,
        img_size: int = 332,
        band_order: Sequence[float | str]
        | dict[str, Sequence[float | str]] = GeoBenchForestnet.band_default_order,
        batch_size: int = 32,
        eval_batch_size: int = 64,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        train_augmentations: nn.Module | None = None,
        eval_augmentations: nn.Module | None = None,
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize GeoBench Forestnet dataset module.

        Args:
            img_size: Image size
            band_order: band order
            batch_size: Batch size
            eval_batch_size: Evaluation batch size
            num_workers: Number of workers
            collate_fn: Collate function
            eval_augmentations: augmentations for validation and test splits
            train_augmentations: augmentations for train split
            pin_memory: Pin memory
            **kwargs: Additional keyword arguments for :class:geobench_v2.datasets.forestnet.GeoBenchForestnet
        """
        super().__init__(
            dataset_class=GeoBenchForestnet,
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
            [os.path.join(self.kwargs["root"], f) for f in GeoBenchForestnet.paths]
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

        num_modalities = 1

        fig = plt.figure(figsize=(num_modalities * 4 + 2, 2 * n_samples))
        gs = fig.add_gridspec(
            n_samples, num_modalities + 1, width_ratios=[*[1] * num_modalities, 0.4]
        )

        labels = batch["label"][indices]
        sample_labels = []
        for i in range(n_samples):
            # present_labels = torch.where(labels[i] == 1)[0].cpu().tolist()
            sample_labels.append([labels[i]])

        images = batch["image"][indices]

        plot_bands = [
            self.band_order.index(x) for x in self.dataset_band_config.plot_bands
        ]

        for i in range(n_samples):
            ax = fig.add_subplot(gs[i, 0])
            plot_img = images[i][plot_bands]

            img = percentile_normalization(plot_img, lower=2, upper=98)

            ax.imshow(img.permute(1, 2, 0))
            ax.set_title("Landsat 8" if i == 0 else "", fontsize=20)
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
                label_ax.set_title("Label", fontsize=15)

        plt.tight_layout()

        plt.subplots_adjust(bottom=0.1)

        return fig, batch
