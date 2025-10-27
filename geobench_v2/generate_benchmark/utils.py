# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Utility functions for benchmark generation."""

import os
from glob import glob

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import pandas as pd
import tacoreader
from matplotlib.lines import Line2D


def plot_sample_locations(
    metadata_df: pd.DataFrame,
    output_path: str = None,
    buffer_degrees: float = 5.0,
    split_column: str = "split",
    sample_fraction: float = 1.0,
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
        dataset_name: Name of the dataset for the title
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

    scale = "50m"
    ax.add_feature(cfeature.LAND.with_scale(scale), facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN.with_scale(scale), facecolor="lightblue")
    ax.add_feature(cfeature.COASTLINE.with_scale(scale), linewidth=0.8)
    ax.add_feature(cfeature.BORDERS.with_scale(scale), linewidth=0.8, linestyle=":")

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


def create_subset_from_df(
    metadata_df: pd.DataFrame,
    n_train_samples: int,
    n_val_samples: int,
    n_test_samples: int,
    n_additional_test_samples: int = 0,
    random_state: int = 42,
    split_column: str = "split",
) -> pd.DataFrame:
    """Create a subset of a DataFrame based on the specified number of samples for each split.

    Args:
        metadata_df: DataFrame containing metadata with a 'split' column
        n_train_samples: Number of training samples to include in the subset
        n_val_samples: Number of validation samples to include in the subset
        n_test_samples: Number of test samples to include in the subset
        n_additional_test_samples: Number of additional test samples to create from remaining train samples
        random_state: Random seed for reproducibility
        split_column: Column name that indicates the dataset split

    Returns:
        A DataFrame containing the selected subset of samples with an additional 'test_from_train' split
    """
    train_count = len(metadata_df[metadata_df[split_column] == "train"])
    val_count = len(metadata_df[metadata_df[split_column] == "validation"])
    test_count = len(metadata_df[metadata_df[split_column] == "test"])

    total_train_needed = n_train_samples + n_additional_test_samples
    if (
        n_train_samples != -1
        and n_additional_test_samples > 0
        and total_train_needed > train_count
    ):
        raise ValueError(
            f"Not enough training samples available. Need {total_train_needed} "
            f"({n_train_samples} train + {n_additional_test_samples} additional test) "
            f"but only {train_count} available."
        )

    n_train_samples = (
        train_count if n_train_samples == -1 else min(n_train_samples, train_count)
    )
    n_val_samples = val_count if n_val_samples == -1 else min(n_val_samples, val_count)
    n_test_samples = (
        test_count if n_test_samples == -1 else min(n_test_samples, test_count)
    )

    print(
        f"Selecting {n_train_samples} train, {n_val_samples} validation, "
        f"{n_test_samples} test, and {n_additional_test_samples} additional test samples"
    )

    train_data = metadata_df[metadata_df[split_column] == "train"]

    if n_additional_test_samples > 0:
        total_train_sample = train_data.sample(
            n_train_samples + n_additional_test_samples, random_state=random_state
        )

        train_samples = total_train_sample.iloc[:n_train_samples].copy()
        additional_test_samples = total_train_sample.iloc[n_train_samples:].copy()
    else:
        train_samples = train_data.sample(n_train_samples, random_state=random_state)
        additional_test_samples = pd.DataFrame()

    val_samples = metadata_df[metadata_df[split_column] == "validation"].sample(
        n_val_samples, random_state=random_state
    )
    test_samples = metadata_df[metadata_df[split_column] == "test"].sample(
        n_test_samples, random_state=random_state
    )

    train_samples["add_test_split"] = False
    val_samples["add_test_split"] = False
    test_samples["add_test_split"] = False
    additional_test_samples["add_test_split"] = True

    subset_dfs = [train_samples, val_samples, test_samples]
    if len(additional_test_samples) > 0:
        subset_dfs.append(additional_test_samples)

    subset_df = pd.concat(subset_dfs, ignore_index=True)

    split_counts = subset_df[split_column].value_counts()
    print("Final subset composition:")
    for split, count in split_counts.items():
        print(f"  {split}: {count} samples")

    return subset_df


def create_unittest_subset(
    data_dir: str,
    tortilla_pattern: str,
    test_dir_name: str,
    n_train_samples: int,
    n_val_samples: int,
    n_test_samples: int,
    n_additional_test_samples: int = 0,
    random_state: int = 42,
) -> None:
    """Create a unittest version tortilla.

    Args:
        data_dir: Directory containing the tortilla files
        tortilla_pattern: Pattern to match tortilla files
        test_dir_name: Name of the directory to save the unittest subset
        n_train_samples: Number of training samples to include in the subset
        n_val_samples: Number of validation samples to include in the subset
        n_test_samples: Number of test samples to include in the subset
        n_additional_test_samples: Number of additional test samples from train split
        random_state: Random seed for reproducibility
    """
    taco_glob = sorted(glob(os.path.join(data_dir, tortilla_pattern)))

    taco_subset = tacoreader.load(taco_glob)

    unit_test_taco = create_subset_from_df(
        taco_subset,
        n_train_samples=n_train_samples,
        n_val_samples=n_val_samples,
        n_test_samples=n_test_samples,
        n_additional_test_samples=n_additional_test_samples,
        random_state=random_state,
        split_column="tortilla:data_split",
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(script_dir))
    test_data_dir = os.path.join(repo_root, "tests", "data", test_dir_name)
    os.makedirs(test_data_dir, exist_ok=True)
    tortilla_path = os.path.join(test_data_dir, f"{test_dir_name}.tortilla")
    tacoreader.compile(dataframe=unit_test_taco, output=tortilla_path)

    print(f"Unit test subset saved to {tortilla_path}")
    print(f"Filesize: {os.path.getsize(tortilla_path) / (1024 * 1024):.2f} MB")
