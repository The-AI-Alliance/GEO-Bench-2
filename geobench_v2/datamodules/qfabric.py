# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.


"""QFabric Datamodule."""

import os
from collections.abc import Callable, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torchgeo.datasets.utils import percentile_normalization

from geobench_v2.datasets import GeoBenchQFabric

from .base import GeoBenchSegmentationDataModule


class GeoBenchQFabricDataModule(GeoBenchSegmentationDataModule):
    """GeoBench QFabric Data Module."""

    def __init__(
        self,
        img_size: int = 2048,
        band_order: Sequence[float | str] = GeoBenchQFabric.band_default_order,
        batch_size: int = 4,
        eval_batch_size: int = 8,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        train_augmentations: nn.Module | None = None,
        eval_augmentations: nn.Module | None = None,
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize GeoBench QFabric dataset module.

        Args:
            img_size: Image size, in geobench version patches of 512
            band_order: The order of bands to return in the sample
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
            **kwargs: Additional keyword arguments for the dataset class
        """
        super().__init__(
            dataset_class=GeoBenchQFabric,
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
            os.path.join(self.kwargs["root"], "geobench_qfabric.parquet")
        )

    def visualize_batch(
        self, split: str = "train"
    ) -> tuple[plt.Figure, dict[str, Tensor]]:
        """Visualize a batch of QFabric data with temporal dimension.

        Args:
            split: One of 'train', 'val', 'test'

        Returns:
            The matplotlib figure and the batch of data
        """
        if split == "train":
            batch = next(iter(self.train_dataloader()))
        elif split == "validation":
            batch = next(iter(self.val_dataloader()))
        else:
            batch = next(iter(self.test_dataloader()))

        batch = self.data_normalizer.unnormalize(batch)

        images = batch["image"]  # Shape: [B, T, C, H, W]
        change_status_masks = batch["mask_status"]  # Shape: [B, T, H, W]
        change_type_mask = batch["mask_change"]  # Shape: [B, H, W]

        B, T, C, H, W = images.shape

        n_samples = min(4, B)
        indices = torch.randperm(B)[:n_samples]

        images = images[indices]
        change_status_masks = change_status_masks[indices]
        change_type_mask = change_type_mask[indices]

        plot_bands = self.dataset_band_config.plot_bands
        rgb_indices = [
            self.band_order.index(band)
            for band in plot_bands
            if band in self.band_order
        ]

        images = images[:, :, rgb_indices, :, :]  # Shape: [n_samples, T, 3, H, W]

        fig, axes = plt.subplots(
            n_samples * 2,
            T + 1,
            figsize=(3 * (T + 1), 3 * n_samples * 2),
            gridspec_kw={"width_ratios": [1] * T + [1.2]},
        )

        status_class_names = self.dataset_class.status_classes
        change_type_names = self.dataset_class.classes

        n_status_classes = len(status_class_names)
        n_change_classes = len(change_type_names)

        cmap_status = plt.cm.get_cmap("viridis", n_status_classes)
        cmap_change = plt.cm.get_cmap("tab10", n_change_classes)

        status_names = status_class_names

        all_change_types = torch.unique(change_type_mask).cpu().numpy()
        unique_change_types = sorted(
            [int(cls) for cls in all_change_types if cls < len(change_type_names)]
        )

        for i in range(n_samples):
            for t in range(T):
                ax = axes[i * 2, t]
                img = images[i, t].permute(1, 2, 0).cpu().numpy()
                img = percentile_normalization(img, lower=2, upper=98)
                ax.imshow(img)

                if "image_dates" in batch:
                    date_str = batch["image_dates"][indices[i]][t]
                    ax.set_title(f"T{t}: {date_str}" if i == 0 else f"T{t}")
                else:
                    ax.set_title(f"Time {t}" if i == 0 else "")

                ax.axis("off")

            axes[i * 2, T].axis("off")

            for t in range(T):
                ax = axes[i * 2 + 1, t]
                status_mask = change_status_masks[i, t].cpu().numpy()
                ax.imshow(
                    status_mask, cmap=cmap_status, vmin=0, vmax=n_status_classes - 1
                )

                if i == 0:
                    ax.set_title(f"Status T{t}")
                ax.axis("off")

            ax = axes[i * 2 + 1, T]
            type_mask = change_type_mask[i].cpu().numpy()
            ax.imshow(
                type_mask, cmap=cmap_change, vmin=0, vmax=n_change_classes - 1
            )

            if i == 0:
                ax.set_title("Change Type")
            ax.axis("off")

            if i == 0:
                status_cax = fig.add_axes([0.92, 0.6, 0.01, 0.3])
                status_cb = plt.colorbar(
                    plt.cm.ScalarMappable(
                        cmap=cmap_status, norm=plt.Normalize(0, n_status_classes - 1)
                    ),
                    cax=status_cax,
                )

                tick_positions = np.linspace(0, n_status_classes - 1, n_status_classes)
                status_cb.set_ticks(tick_positions)

                truncated_names = [
                    name[:10] + ("..." if len(name) > 10 else "")
                    for name in status_names
                ]
                status_cb.set_ticklabels(truncated_names)
                status_cb.set_label("Change Status")

                type_cax = fig.add_axes([0.92, 0.1, 0.01, 0.3])
                type_cb = plt.colorbar(
                    plt.cm.ScalarMappable(
                        cmap=cmap_change, norm=plt.Normalize(0, n_change_classes - 1)
                    ),
                    cax=type_cax,
                )

                if unique_change_types:
                    type_cb.set_ticks(unique_change_types)
                    type_labels = [
                        change_type_names[t]
                        if t < len(change_type_names)
                        else f"Unknown {t}"
                        for t in unique_change_types
                    ]
                    type_cb.set_ticklabels(type_labels)
                type_cb.set_label("Change Type")

        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.subplots_adjust(hspace=0.3, wspace=0.1)

        return fig, batch

    def visualize_geolocation_distribution(self) -> None:
        """Visualize the geolocation distribution of the dataset."""
        pass
