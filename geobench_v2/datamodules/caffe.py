# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""CaFFe DataMdule."""

import os
from collections.abc import Callable, Sequence
from typing import Any, Literal

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import pandas as pd
import tacoreader
import torch
import torch.nn as nn
from matplotlib.lines import Line2D
from torch import Tensor
from torchgeo.datasets.utils import percentile_normalization

from geobench_v2.datasets.caffe import GeoBenchCaFFe

from .base import GeoBenchSegmentationDataModule


class GeoBenchCaFFeDataModule(GeoBenchSegmentationDataModule):
    """GeoBench CaFFe Data Module."""

    def __init__(
        self,
        img_size: int = 512,
        band_order: Sequence[float | str] = GeoBenchCaFFe.band_default_order,
        batch_size: int = 32,
        eval_batch_size: int = 64,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        train_augmentations: nn.Module | None = None,
        eval_augmentations: nn.Module | None = None,
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize GeoBench CaFFe dataset module.

        Args:
            img_size: Image size
            band_order: The order of bands to return in the sample
            batch_size: Batch size during
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
            **kwargs: Additional keyword arguments for :class:`geobench_v2.datasets.caffe.GeoBenchCaFFe`
        """
        super().__init__(
            dataset_class=GeoBenchCaFFe,
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
            [os.path.join(self.kwargs["root"], f) for f in GeoBenchCaFFe.paths]
        )
        return self.data_df

    def visualize_geospatial_distribution(
        self,
        split_column: str = "tortilla:data_split",
        buffer_degrees: float = 5.0,
        sample_fraction: float | None = None,
        scale: Literal["10m", "50m", "110m"] = "50m",
        alpha: float = 0.7,
        s: float = 5,
    ) -> plt.Figure:
        """Visualize the geospatial distribution of CaFFe samples on polar projections.

        CaFFe dataset contains glacier data from high northern and southern latitudes,
        focusing on Greenland/Canada in the north and West Antarctica in the south.
        This creates a split view with zoomed polar projections for each region.

        Args:
            split_column: Column name in the metadata DataFrame that indicates the dataset split.
            buffer_degrees: Buffer around the data extent in degrees.
            sample_fraction: Optional fraction of samples to plot (0.0-1.0) for performance.
            scale: Scale of cartopy features (e.g., '10m', '50m', '110m').
            alpha: Transparency of plotted points.
            s: Size of plotted points.

        Returns:
            A matplotlib Figure object with the geospatial distribution plot.
        """
        data_df = self.load_metadata()

        # Standardize coordinate columns
        if "lat" not in data_df.columns or "lon" not in data_df.columns:
            if "latitude" in data_df.columns and "longitude" in data_df.columns:
                data_df = data_df.rename(
                    columns={"latitude": "lat", "longitude": "lon"}
                )
            else:
                raise ValueError(
                    "Metadata is missing required latitude and longitude information"
                )

        # Optional sub-sampling for performance
        if sample_fraction is not None and 0.0 < sample_fraction < 1.0:
            data_df = data_df.sample(frac=sample_fraction, random_state=0)

        # Normalize split names on the main dataframe first
        plot_col = "plot_split"
        data_df[plot_col] = (
            data_df[split_column].astype(str).replace({"val": "validation"})
        )

        # Split data into northern and southern hemisphere (already has plot_col)
        northern_data = data_df[data_df["lat"] > 0].copy()
        southern_data = data_df[data_df["lat"] < 0].copy()

        # Determine splits present
        desired_order = ["train", "validation", "test", "extra_test"]
        present = [sp for sp in desired_order if sp in set(data_df[plot_col].unique())]
        others = [sp for sp in data_df[plot_col].unique() if sp not in present]
        splits = present + others

        split_colors = {"train": "blue", "validation": "green", "test": "red"}

        # Create figure with two subplots (north and south)
        fig = plt.figure(figsize=(20, 10))

        # Northern hemisphere subplot - focused on Greenland/Canada region
        ax_north = fig.add_subplot(1, 2, 1, projection=ccrs.NorthPolarStereo())

        # Compute extent for northern hemisphere with tighter bounds
        if len(northern_data) > 0:
            min_lon = northern_data["lon"].min() - buffer_degrees
            max_lon = northern_data["lon"].max() + buffer_degrees
            min_lat = max(60, northern_data["lat"].min() - buffer_degrees)
            max_lat = min(90, northern_data["lat"].max() + buffer_degrees)
            # Focus on Greenland/Canada region (roughly -100 to -10 longitude)
            ax_north.set_extent(
                [min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree()
            )
        else:
            # Default to Greenland/Canada region
            ax_north.set_extent([-100, -10, 60, 85], crs=ccrs.PlateCarree())

        # Add features to northern plot
        ax_north.add_feature(cfeature.LAND.with_scale(scale), facecolor="lightgray")
        ax_north.add_feature(cfeature.OCEAN.with_scale(scale), facecolor="lightblue")
        ax_north.add_feature(cfeature.COASTLINE.with_scale(scale), linewidth=0.8)
        ax_north.gridlines(linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
        ax_north.set_title(
            "Northern Hemisphere (Greenland/Canada Glaciers)", fontsize=12, pad=20
        )

        # Plot northern hemisphere data
        legend_elements: list[Line2D] = []
        for split in splits:
            split_data = northern_data[northern_data[plot_col] == split]
            if len(split_data) == 0:
                continue
            color = split_colors.get(split, "gray")
            ax_north.scatter(
                split_data["lon"],
                split_data["lat"],
                transform=ccrs.PlateCarree(),
                c=color,
                s=s,
                alpha=alpha,
                label=split,
            )
            # Only create legend elements once
            if not legend_elements or split not in [
                le.get_label() for le in legend_elements
            ]:
                total_count = len(data_df[data_df[plot_col] == split])
                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=color,
                        markersize=8,
                        label=f"{split} (n={total_count})",
                    )
                )

        # Southern hemisphere subplot - focused on West Antarctica
        ax_south = fig.add_subplot(1, 2, 2, projection=ccrs.SouthPolarStereo())

        # Compute extent for southern hemisphere with tighter bounds
        if len(southern_data) > 0:
            min_lon = southern_data["lon"].min() - buffer_degrees
            max_lon = southern_data["lon"].max() + buffer_degrees
            min_lat = max(-90, southern_data["lat"].min() - buffer_degrees)
            max_lat = min(-60, southern_data["lat"].max() + buffer_degrees)
            # Focus on West Antarctica region
            ax_south.set_extent(
                [min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree()
            )
        else:
            # Default to West Antarctica region
            ax_south.set_extent([-180, 0, -90, -65], crs=ccrs.PlateCarree())

        # Add features to southern plot
        ax_south.add_feature(cfeature.LAND.with_scale(scale), facecolor="lightgray")
        ax_south.add_feature(cfeature.OCEAN.with_scale(scale), facecolor="lightblue")
        ax_south.add_feature(cfeature.COASTLINE.with_scale(scale), linewidth=0.8)
        ax_south.gridlines(linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
        ax_south.set_title(
            "Southern Hemisphere (West Antarctica Glaciers)", fontsize=12, pad=20
        )

        # Plot southern hemisphere data
        for split in splits:
            split_data = southern_data[southern_data[plot_col] == split]
            if len(split_data) == 0:
                continue
            color = split_colors.get(split, "gray")
            ax_south.scatter(
                split_data["lon"],
                split_data["lat"],
                transform=ccrs.PlateCarree(),
                c=color,
                s=s,
                alpha=alpha,
                label=split,
            )

        # Add shared legend below both plots
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            ncol=len(legend_elements),
            frameon=True,
            fontsize=10,
            title="Dataset Splits",
            bbox_to_anchor=(0.5, -0.05),
        )

        plt.suptitle(
            "Geographic Distribution of CaFFe Glacier Samples by Split",
            fontsize=14,
            y=0.98,
        )

        plt.tight_layout()
        return fig

    def visualize_batch(
        self, batch: dict[str, Tensor] | None = None, split: str = "train"
    ) -> tuple[plt.Figure, dict[str, Tensor]]:
        """Visualize a batch of data.

        Args:
            batch: Batch of data to visualize
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

        if hasattr(self.data_normalizer, "unnormalize"):
            batch = self.data_normalizer.unnormalize(batch)

        images = batch["image"]
        masks = batch["mask"]

        n_samples = min(8, images.shape[0])
        indices = torch.randperm(images.shape[0])[:n_samples]

        images = images[indices]
        masks = masks[indices]

        plot_bands = self.dataset_band_config.plot_bands
        plot_index = self.band_order.index(plot_bands[0])
        images = images[:, plot_index, :, :]

        # Create figure with 3 columns: image, mask, and legend
        fig, axes = plt.subplots(
            n_samples,
            3,
            figsize=(12, 3 * n_samples),
            gridspec_kw={"width_ratios": [1, 1, 0.5]},
        )

        if n_samples == 1:
            axes = axes.reshape(1, -1)

        unique_classes = torch.unique(masks).cpu().numpy()
        unique_classes = [
            int(cls) for cls in unique_classes if cls < len(self.class_names)
        ]

        cmap = plt.cm.tab20

        for i in range(n_samples):
            ax = axes[i, 0]
            img = images[i].cpu().numpy()
            img = percentile_normalization(img, lower=2, upper=98)
            ax.imshow(img, cmap="gray")
            ax.set_title("SAR Image" if i == 0 else "")
            ax.axis("off")

            ax = axes[i, 1]
            mask_img = masks[i].cpu().numpy()
            ax.imshow(mask_img, cmap="tab20", vmin=0, vmax=19)
            ax.set_title("Mask" if i == 0 else "")
            ax.axis("off")

            ax = axes[i, 2]
            ax.axis("off")

            if i == 0:
                legend_elements = []
                for cls in unique_classes:
                    if cls < len(self.class_names):
                        color = cmap(cls / 20.0 if cls < 20 else 0)
                        legend_elements.append(
                            plt.Rectangle(
                                (0, 0),
                                1,
                                1,
                                color=color,
                                label=f"{cls}: {self.class_names[cls]}",
                            )
                        )

                ax.legend(
                    handles=legend_elements,
                    loc="center",
                    frameon=True,
                    fontsize="small",
                    title="Classes",
                )

        plt.tight_layout()
        return fig, batch

    def visualize_geolocation_distribution(self) -> None:
        """Visualize the geolocation distribution of the dataset."""
        pass
