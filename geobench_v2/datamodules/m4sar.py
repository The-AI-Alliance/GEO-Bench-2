# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""GeoBench M4SAR DataModule."""

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
from torch import Tensor
from torchgeo.datasets.utils import percentile_normalization

from geobench_v2.datasets import GeoBenchM4SAR

from .base import GeoBenchObjectDetectionDataModule


def m4sar_collate_fn(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collate function for M4SAR dataset.

    Args:
        batch: A list of dictionaries containing the data for each sample

    Returns:
        A dictionary containing the collated data
    """
    collated_batch = {}
    # collate images
    keys = batch[0].keys()
    for key in keys:
        if key.startswith("image_"):
            images = [sample[key] for sample in batch]
            images = torch.stack(images, dim=0)
            collated_batch[key] = images
        elif key == "image":
            images = [sample[key] for sample in batch]
            images = torch.stack(images, dim=0)
            collated_batch[key] = images
        else:
            collated_batch[key] = [sample[key] for sample in batch]

    return collated_batch


class GeoBenchM4SARDataModule(GeoBenchObjectDetectionDataModule):
    """GeoBench M4SAR Data Module."""

    def __init__(
        self,
        img_size: int = 512,
        band_order: Sequence[float | str] = GeoBenchM4SAR.band_default_order,
        batch_size: int = 32,
        eval_batch_size: int = 64,
        num_workers: int = 0,
        collate_fn: Callable | None = m4sar_collate_fn,
        train_augmentations: nn.Module | None = None,
        eval_augmentations: nn.Module | None = None,
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize GeoBench DOTAV2 dataset module.

        Args:
            img_size: Image size
            band_order: The order of bands to return, defaults to ['red', 'green', 'blue'], if one would
                specify ['red', 'green', 'blue', 'blue'], the dataset would return images with 4 channels
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
            dataset_class=GeoBenchM4SAR,
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
            [os.path.join(self.kwargs["root"], f) for f in GeoBenchM4SAR.paths]
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
        """
        if split == "train":
            batch = next(iter(self.train_dataloader()))
        elif split == "validation":
            batch = next(iter(self.val_dataloader()))
        else:
            batch = next(iter(self.test_dataloader()))

        if hasattr(self.data_normalizer, "unnormalize"):
            batch = self.data_normalizer.unnormalize(batch)

        batch_size = len(batch["label"])
        boxes_batch = batch["bbox_xyxy"]
        labels_batch = batch["label"]

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

        if n_samples == 1 and num_modalities == 1:
            axes = np.array([[axes]])
        elif n_samples == 1:
            axes = axes.reshape(1, -1)
        elif num_modalities == 1:
            axes = axes.reshape(-1, 1)

        boxes_batch = [boxes_batch[i] for i in indices]
        labels_batch = [labels_batch[i] for i in indices]

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
            boxes = boxes_batch[i]
            labels = labels_batch[i]
            for j, (mod, modality_img) in enumerate(modalities.items()):
                plot_img = modality_img[i]

                # if mod == "sar":
                #     vv = plot_img[..., 0]
                #     vh = plot_img[..., 1]

                #     vv = percentile_normalization(vv, lower=2, upper=98)
                #     vh = percentile_normalization(vh, lower=2, upper=98)

                #     ratio = np.divide(vv, vh, out=np.zeros_like(vv), where=vh != 0)

                #     vv = np.clip(vv / 0.3, a_min=0, a_max=1)
                #     vh = np.clip(vh / 0.05, a_min=0, a_max=1)
                #     ratio = np.clip(ratio / 25, a_min=0, a_max=1)
                #     img = np.stack((vv, vh, ratio), axis=2)
                # else:
                img = percentile_normalization(plot_img, lower=2, upper=98)
                ax = axes[i, j]
                ax.imshow(img)
                ax.set_title(f"{mod} image" if i == 0 else "", fontsize=20)
                ax.axis("off")

                for box, label in zip(boxes, labels):
                    if isinstance(box, torch.Tensor):
                        box = box.cpu().numpy()
                    if isinstance(label, torch.Tensor):
                        label = label.item()

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
                    ax.add_patch(rect)

            class_counts = {}
            for label in labels:
                if isinstance(label, torch.Tensor):
                    label = label.item()
                class_name = self.class_names[int(label)]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

            ax = axes[i, -1]
            ax.set_xticks([])
            ax.set_yticks([])

            ax.axis("off")
            if class_counts:
                sorted_items = sorted(
                    class_counts.items(), key=lambda x: x[1], reverse=True
                )

                start_y_pos = 0.9
                y_pos = start_y_pos

                total = sum(class_counts.values())
                ax.text(
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
                    ax.add_patch(square)

                    ax.text(0.1, y_pos, f" {name}: {count}", va="center", fontsize=15)

                counts_box = plt.Rectangle(
                    (0.01, y_pos - 0.02),
                    0.9,
                    (start_y_pos + 0.02) - (y_pos - 0.02),
                    fill=False,
                    edgecolor="gray",
                    linestyle="--",
                    transform=ax.transAxes,
                )
                ax.add_patch(counts_box)
            else:
                ax.text(0.1, 0.5, "No objects detected", va="center")

        plt.tight_layout()

        return fig, batch

    def visualize_geolocation_distribution(self) -> None:
        """Visualize the geolocation distribution of the dataset."""
        pass
