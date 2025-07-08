# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""GeoBench Substation DataModule."""

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

from geobench_v2.datasets import GeoBenchSubstation

from .base import GeoBenchObjectDetectionDataModule
import skimage
from matplotlib import patches


def substation_collate_fn(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collate function for Substation dataset.

    Args:
        batch: A list of dictionaries containing the data for each sample

    Returns:
        A dictionary containing the collated data
    """
    # collate images
    images = [sample["image"] for sample in batch]
    images = torch.stack(images, dim=0)
    # collate boxes into list of boxes
    boxes = [sample["bbox_xyxy"] for sample in batch]
    label = [sample["label"] for sample in batch]
    masks = [sample["mask"] for sample in batch]

    return {"image": images, "bbox_xyxy": boxes, "label": label, "mask": masks}


class GeoBenchSubstationDataModule(GeoBenchObjectDetectionDataModule):
    """GeoBench Substation Data Module."""

    def __init__(
        self,
        img_size: int = 228,
        band_order: Sequence[float | str] = GeoBenchSubstation.band_default_order,
        batch_size: int = 32,
        eval_batch_size: int = 64,
        num_workers: int = 0,
        collate_fn: Callable | None = substation_collate_fn,
        train_augmentations: nn.Module | None = None,
        eval_augmentations: nn.Module | None = None,
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        
        """Initialize GeoBench Substation dataset module.

        Args:
            img_size: Image size
            band_order: The order of bands to return, defaults to ['red', 'green', 'blue'], if one would
                specify ['red', 'green', 'blue', 'blue'], the dataset would return images with 4 channels
                in that order. This is useful for models that expect a certain band order, or
                test the impact of band order on model performance.
            batch_size: Batch size during
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
            dataset_class=GeoBenchSubstation,
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

    def visualize_batch(
        self, split: str = "train"
    ) -> tuple[plt.Figure, dict[str, Tensor]]:
        """Visualize a batch of data.

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

        images = batch["image"]
        boxes_batch = batch["bbox_xyxy"]
        labels_batch = batch["label"]
        masks_batch = batch["mask"]

        batch_size = images.shape[0]
        n_samples = min(8, batch_size)
        indices = torch.randperm(batch_size)[:n_samples]

        images = images[indices]
        boxes_batch = [boxes_batch[i] for i in indices]
        labels_batch = [labels_batch[i] for i in indices]
        masks_batch =  [masks_batch[i] for i in indices]

        plot_bands = self.dataset_band_config.plot_bands
        rgb_indices = [
            self.band_order.index(band)
            for band in plot_bands
            if band in self.band_order
        ]
        images = images[:, rgb_indices, :, :]

        fig, axes = plt.subplots(
            n_samples,
            2,
            figsize=(14, 5 * n_samples),
            gridspec_kw={"width_ratios": [3, 1]},
        )

        if n_samples == 1:
            axes = np.array([axes])

        num_classes = len(self.class_names)
        colors = plt.cm.tab20(np.linspace(0, 1, num_classes))

        legend_elements = []
        for i, name in enumerate(self.class_names):
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="w",
                    markerfacecolor=colors[i],
                    markersize=10,
                    label=name,
                )
            )

        for i in range(n_samples):
            ax_img = axes[i, 0]
            img = rearrange(images[i], "c h w -> h w c").cpu().numpy()
            img = percentile_normalization(img, lower=2, upper=98)
            ax_img.imshow(img)

            boxes = boxes_batch[i]
            labels = labels_batch[i]
            masks = masks_batch[i]

            class_counts = {}
            for label in labels:
                if isinstance(label, torch.Tensor):
                    label = label.item()
                class_name = self.class_names[int(label)]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

            for box, label, mask in zip(boxes, labels, masks):
                if isinstance(box, torch.Tensor):
                    box = box.cpu().numpy()
                if isinstance(label, torch.Tensor):
                    label = label.item()
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()

                x1, y1, x2, y2 = box
                color = colors[int(label)]

                rect = plt.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none",
                )
                ax_img.add_patch(rect)
                contours = skimage.measure.find_contours(mask, 0.5)
                for verts in contours:
                    verts = np.fliplr(verts)
                    p = patches.Polygon(
                        verts, facecolor=color, alpha=0.4, edgecolor='white'
                    )
                    ax_img.add_patch(p)

            ax_img.set_title(f"Sample {i + 1}" if i == 0 else "")
            ax_img.set_xticks([])
            ax_img.set_yticks([])

            ax_stats = axes[i, 1]

            ax_stats.axis("off")
            if class_counts:
                sorted_items = sorted(
                    class_counts.items(), key=lambda x: x[1], reverse=True
                )

                start_y_pos = 0.9
                y_pos = start_y_pos

                total = sum(class_counts.values())
                ax_stats.text(
                    0.1,
                    y_pos,
                    f"Total: {total}",
                    va="top",
                    fontsize=15,
                    fontweight="bold",
                )
                y_pos -= 0.05

                for name, count in sorted_items:
                    y_pos -= 0.04
                    class_idx = self.class_names.index(name)
                    color = colors[class_idx]

                    square = plt.Rectangle(
                        (0.05, y_pos), 0.03, 0.03, facecolor=color, edgecolor="black"
                    )
                    ax_stats.add_patch(square)

                    ax_stats.text(
                        0.1, y_pos, f" {name}: {count}", va="center", fontsize=15
                    )

                counts_box = plt.Rectangle(
                    (0.01, y_pos - 0.02),
                    0.9,
                    (start_y_pos + 0.02) - (y_pos - 0.02),
                    fill=False,
                    edgecolor="gray",
                    linestyle="--",
                    transform=ax_stats.transAxes,
                )
                ax_stats.add_patch(counts_box)
            else:
                ax_stats.text(0.1, 0.5, "No objects detected", va="center")

        plt.tight_layout()

        return fig, batch

    def visualize_geolocation_distribution(self) -> None:
        """Visualize the geolocation distribution of the dataset."""
        pass
