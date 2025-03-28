# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Utility functions for benchmark generation."""

from matplotlib.lines import Line2D
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def plot_sample_locations(
    metadata_df: pd.DataFrame,
    output_path: str = None,
    buffer_degrees: float = 5.0,
    split_column: str = "split",
    sample_fraction: float = 1.0,  # Reduced default to 10%
    alpha: float = 0.5,
    s: float = 0.5,
    dataset_name: str = "BigEarthNetV2",
) -> None:
    """Plot the geolocation of samples on a map, differentiating by dataset splits.

    Args:
        metadata_df: DataFrame with metadata including lat and lon columns
        output_path: Path to save the figure. If None, the figure is displayed but not saved.
        buffer_degrees: Buffer around the data extent in degrees
        split_column: Column name that indicates the dataset split
        sample_fraction: Fraction of samples to plot for better performance (0.0-1.0)
        alpha: Transparency of plotted points
        s: Size of plotted points
    """
    if sample_fraction < 1.0:
        sample_size = int(len(metadata_df) * sample_fraction)
        metadata_df = metadata_df.sample(sample_size, random_state=42)
        print(f"Sampled {sample_size} points for plotting")

    if "latitude" in metadata_df.columns:
        metadata_df.rename(
            columns={"latitude": "lat", "longitude": "lon"}, inplace=True
        )

    min_lon = metadata_df["lon"].min() - buffer_degrees
    max_lon = metadata_df["lon"].max() + buffer_degrees
    min_lat = metadata_df["lat"].min() - buffer_degrees
    max_lat = metadata_df["lat"].max() + buffer_degrees

    min_lon = max(-180, min_lon)
    max_lon = min(180, max_lon)
    min_lat = max(-90, min_lat)
    max_lat = min(90, max_lat)

    print(
        f"Map extent: Longitude [{min_lon:.2f}° to {max_lon:.2f}°], "
        f"Latitude [{min_lat:.2f}° to {max_lat:.2f}°]"
    )

    plt.figure(figsize=(12, 10))

    lon_extent = max_lon - min_lon
    lat_extent = max_lat - min_lat

    if lon_extent > 180:
        # Global extent, Robinson is a good choice
        projection = ccrs.Robinson()
    else:
        central_lon = (min_lon + max_lon) / 2
        central_lat = (min_lat + max_lat) / 2

        if lat_extent > 60:
            projection = ccrs.AlbersEqualArea(
                central_longitude=central_lon, central_latitude=central_lat
            )
        else:
            projection = ccrs.LambertConformal(
                central_longitude=central_lon, central_latitude=central_lat
            )

    ax = plt.axes(projection=projection)

    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    scale = "110m"
    ax.add_feature(cfeature.LAND.with_scale(scale), facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN.with_scale(scale), facecolor="lightblue")
    ax.add_feature(cfeature.COASTLINE.with_scale(scale), linewidth=0.5)
    ax.add_feature(cfeature.BORDERS.with_scale(scale), linewidth=0.3, linestyle=":")

    if max_lon - min_lon < 90:
        ax.add_feature(cfeature.RIVERS, linewidth=0.2, alpha=0.5)
        ax.add_feature(cfeature.LAKES, facecolor="lightblue", alpha=0.5)

    splits = metadata_df[split_column].unique()
    print(f"Found {len(splits)} dataset splits: {', '.join(map(str, splits))}")

    split_colors = {
        "train": "blue",
        "val": "green",
        "validation": "green",
        "test": "red",
    }

    legend_elements = []

    # scatter plot for each split
    for split in splits:
        split_data = metadata_df[metadata_df[split_column] == split]
        if len(split_data) > 0:
            color = split_colors[split]
            ax.scatter(
                split_data["lon"],
                split_data["lat"],
                transform=ccrs.PlateCarree(),
                c=color,
                s=s,
                alpha=alpha,
                label=split,
            )
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markersize=8,
                    label=f"{split} (n={len(split_data)})",
                )
            )

    ax.legend(handles=legend_elements, loc="lower right", title="Dataset Splits")

    title = f"Geographic Distribution of {dataset_name} Samples by Split"

    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False

    plt.title(title, fontsize=14)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Map saved to {output_path}")


def plot_enhanced_hemisphere_locations(
    metadata_df: pd.DataFrame,
    output_path: str = None,
    buffer_degrees: float = 5.0,
    split_column: str = "split",
    alpha: float = 0.5,
    s: float = 0.5,
    dataset_name: str = "CaFFe",
    west_east_split: float = -80.0,
) -> None:
    """Plot the geolocation of samples on three maps - two for northern hemisphere regions and one for southern hemisphere.

    Args:
        metadata_df: DataFrame with metadata including lat and lon columns
        output_path: Path to save the figure
        buffer_degrees: Buffer around the data extent in degrees
        split_column: Column name that indicates the dataset split
        alpha: Transparency of plotted points
        s: Size of plotted points
        dataset_name: Name of the dataset for the title
        west_east_split: Longitude value to split western/eastern northern hemisphere
    """
    if "latitude" in metadata_df.columns:
        metadata_df = metadata_df.copy()
        metadata_df.rename(
            columns={"latitude": "lat", "longitude": "lon"}, inplace=True
        )
    north_df = metadata_df[metadata_df["lat"] >= 0].copy()
    north_west_df = north_df[north_df["lon"] <= west_east_split].copy()
    north_east_df = north_df[north_df["lon"] > west_east_split].copy()
    south_df = metadata_df[metadata_df["lat"] < 0].copy()

    print(f"Northern hemisphere: {len(north_df)} samples")
    print(f"  - Western region: {len(north_west_df)} samples")
    print(f"  - Eastern region: {len(north_east_df)} samples")
    print(f"Southern hemisphere: {len(south_df)} samples")

    fig = plt.figure(figsize=(20, 16))

    gs = fig.add_gridspec(2, 2)
    ax_north_west = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax_north_east = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    ax_south = fig.add_subplot(gs[1, :], projection=ccrs.PlateCarree())

    split_colors = {"train": "blue", "val": "green", "test": "red"}
    if len(north_west_df) > 0:
        _plot_region(
            ax_north_west,
            north_west_df,
            split_column,
            split_colors,
            buffer_degrees,
            s,
            alpha,
            f"Northern Hemisphere (Western) - {len(north_west_df)} samples",
        )

    if len(north_east_df) > 0:
        _plot_region(
            ax_north_east,
            north_east_df,
            split_column,
            split_colors,
            buffer_degrees,
            s,
            alpha,
            f"Northern Hemisphere (Eastern) - {len(north_east_df)} samples",
        )

    if len(south_df) > 0:
        _plot_region(
            ax_south,
            south_df,
            split_column,
            split_colors,
            buffer_degrees,
            s,
            alpha,
            f"Southern Hemisphere - {len(south_df)} samples",
        )

    fig.suptitle(
        f"Geographic Distribution of {dataset_name} Samples by Split", fontsize=16
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Map saved to {output_path}")
    else:
        plt.show()


def _plot_region(ax, df, split_column, split_colors, buffer_degrees, s, alpha, title):
    """Helper function to plot a specific region on the given axis."""

    min_lon = df["lon"].min() - buffer_degrees
    max_lon = df["lon"].max() + buffer_degrees
    min_lat = df["lat"].min() - buffer_degrees
    max_lat = df["lat"].max() + buffer_degrees

    min_lon = max(-180, min_lon)
    max_lon = min(180, max_lon)
    min_lat = max(-90, min_lat)
    max_lat = min(90, max_lat)

    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    scale = "50m"
    ax.add_feature(cfeature.LAND.with_scale(scale), facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN.with_scale(scale), facecolor="lightblue")
    ax.add_feature(cfeature.COASTLINE.with_scale(scale), linewidth=0.5)
    ax.add_feature(cfeature.BORDERS.with_scale(scale), linewidth=0.3, linestyle=":")

    if max_lon - min_lon < 90:
        ax.add_feature(cfeature.RIVERS, linewidth=0.2, alpha=0.5)
        ax.add_feature(cfeature.LAKES, facecolor="lightblue", alpha=0.5)

    # Plot each split
    legend_elements = []
    for split in df[split_column].unique():
        split_data = df[df[split_column] == split]
        if len(split_data) > 0:
            color = split_colors.get(split, "purple")

            ax.scatter(
                split_data["lon"],
                split_data["lat"],
                transform=ccrs.PlateCarree(),
                c=color,
                s=s,
                alpha=alpha,
                label=split,
            )

            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markersize=8,
                    label=f"{split} (n={len(split_data)})",
                )
            )

    ax.legend(handles=legend_elements, loc="lower right", title="Dataset Splits")

    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False
    ax.set_title(title, fontsize=12)


import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from torchgeo.datasets.utils import percentile_normalization
from matplotlib.colors import ListedColormap


def visualize_dynamic_earthnet_sample(
    root_dir, df, sample_idx, output_path=None, figsize=(12, 16), resize_factor=None
):
    """
    Visualize a sample from the DynamicEarthNet dataset with a 4x2 grid layout:
    - Rows 1-2: 4 random Planet timesteps
    - Row 3: 1 Planet timestep and Label
    - Row 4: Sentinel-1 and Sentinel-2

    Args:
        root_dir: Root directory of the dataset
        df: DataFrame containing metadata
        sample_idx: Sample index to visualize
        output_path: Path to save the visualization
        figsize: Figure size as (width, height)
        resize_factor: Factor to resize images for faster plotting
    """
    import numpy as np
    import rasterio
    from matplotlib.colors import ListedColormap
    from skimage.transform import resize as sk_resize

    sample_df = df[df["sample_idx"] == sample_idx].reset_index(drop=True)

    if len(sample_df) >= 5:
        planet_samples = sample_df.sample(n=5)
    else:
        planet_samples = sample_df

    fig, axes = plt.subplots(4, 2, figsize=figsize)

    planet_iter = iter(planet_samples.iterrows())

    # First two rows: 4 Planet samples
    for row in range(2):
        for col in range(2):
            try:
                _, row_data = next(planet_iter)

                planet_path = os.path.join(root_dir, row_data["planet_path"])
                with rasterio.open(planet_path) as src:
                    planet_data = src.read()

                    if resize_factor is not None:
                        new_h = int(planet_data.shape[1] * resize_factor)
                        new_w = int(planet_data.shape[2] * resize_factor)
                        resized_data = np.zeros((planet_data.shape[0], new_h, new_w))
                        for b in range(planet_data.shape[0]):
                            resized_data[b] = sk_resize(
                                planet_data[b],
                                (new_h, new_w),
                                preserve_range=True,
                                anti_aliasing=True,
                            )
                        planet_data = resized_data

                    rgb = planet_data[[2, 1, 0]].transpose(1, 2, 0)
                    rgb = percentile_normalization(rgb)

                    axes[row, col].imshow(rgb)
                    axes[row, col].set_title(f"Planet {row_data['planet_date']}")
                    axes[row, col].axis("off")
            except StopIteration:
                axes[row, col].axis("off")

    # Row 3: Last Planet sample and Label
    try:
        _, row_data = next(planet_iter)

        planet_path = os.path.join(root_dir, row_data["planet_path"])
        with rasterio.open(planet_path) as src:
            planet_data = src.read()

            if resize_factor is not None:
                new_h = int(planet_data.shape[1] * resize_factor)
                new_w = int(planet_data.shape[2] * resize_factor)
                resized_data = np.zeros((planet_data.shape[0], new_h, new_w))
                for b in range(planet_data.shape[0]):
                    resized_data[b] = sk_resize(
                        planet_data[b],
                        (new_h, new_w),
                        preserve_range=True,
                        anti_aliasing=True,
                    )
                planet_data = resized_data

            rgb = rgb = planet_data[[2, 1, 0]].transpose(1, 2, 0)
            rgb = percentile_normalization(rgb)

            print("SIZE", rgb.shape)

            axes[2, 0].imshow(rgb)
            axes[2, 0].set_title(f"Planet {row_data['planet_date']}")
            axes[2, 0].axis("off")
    except StopIteration:
        axes[2, 0].axis("off")

    # Label data
    reference_row = sample_df.iloc[0]
    label_path = os.path.join(root_dir, reference_row["label_path"])
    with rasterio.open(label_path) as src:
        label_data = src.read()

        if resize_factor is not None:
            new_h = int(label_data.shape[1] * resize_factor)
            new_w = int(label_data.shape[2] * resize_factor)
            resized_data = np.zeros((label_data.shape[0], new_h, new_w))
            for b in range(label_data.shape[0]):
                resized_data[b] = sk_resize(
                    label_data[b],
                    (new_h, new_w),
                    preserve_range=True,
                    anti_aliasing=True,
                )
            label_data = resized_data

        land_cover = np.argmax(label_data, axis=0)

        colors = [
            "lightgreen",
            "darkgreen",
            "blue",
            "gray",
            "yellow",
            "white",
            "silver",
        ]
        cmap = ListedColormap(colors[: label_data.shape[0]])

        axes[2, 1].imshow(land_cover, cmap=cmap, vmin=0, vmax=label_data.shape[0] - 1)
        axes[2, 1].set_title("Land Cover")
        axes[2, 1].axis("off")

    # Row 4: S1 and S2
    s1_path = os.path.join(root_dir, reference_row["s1_path"])
    with rasterio.open(s1_path) as src:
        s1_data = src.read()

        if resize_factor is not None:
            new_h = int(s1_data.shape[1] * resize_factor)
            new_w = int(s1_data.shape[2] * resize_factor)
            resized_data = np.zeros((s1_data.shape[0], new_h, new_w))
            for b in range(s1_data.shape[0]):
                resized_data[b] = sk_resize(
                    s1_data[b], (new_h, new_w), preserve_range=True, anti_aliasing=True
                )
            s1_data = resized_data

        vv = s1_data[0]
        vh = s1_data[4] if s1_data.shape[0] > 4 else s1_data[1]

        s1_rgb = np.stack([vv, vh, vv - vh], axis=2)
        s1_rgb = percentile_normalization(s1_rgb)

        axes[3, 0].imshow(s1_rgb)
        axes[3, 0].set_title("Sentinel-1")
        axes[3, 0].axis("off")

    s2_path = os.path.join(root_dir, reference_row["s2_path"])
    with rasterio.open(s2_path) as src:
        s2_data = src.read()

        if resize_factor is not None:
            new_h = int(s2_data.shape[1] * resize_factor)
            new_w = int(s2_data.shape[2] * resize_factor)
            resized_data = np.zeros((s2_data.shape[0], new_h, new_w))
            for b in range(s2_data.shape[0]):
                resized_data[b] = sk_resize(
                    s2_data[b], (new_h, new_w), preserve_range=True, anti_aliasing=True
                )
            s2_data = resized_data

        s2_rgb = np.stack([s2_data[3], s2_data[2], s2_data[1]], axis=2)
        s2_rgb = percentile_normalization(s2_rgb)

        axes[3, 1].imshow(s2_rgb)
        axes[3, 1].set_title("Sentinel-2")
        axes[3, 1].axis("off")

    plt.tight_layout()
    fig.suptitle(
        f"DynamicEarthNet Sample: {sample_df.iloc[0]['new_id']}", fontsize=14, y=0.98
    )

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to {output_path}")

    plt.show()

    return fig
