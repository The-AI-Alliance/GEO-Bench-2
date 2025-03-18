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
    sample_fraction: float = 0.8,
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

    # Sample data if fraction is less than 1.0
    if sample_fraction < 1.0:
        sample_size = int(len(metadata_df) * sample_fraction)
        metadata_df = metadata_df.sample(sample_size, random_state=42)
        print(f"Sampled {sample_size} points for plotting")

    if "latitude" in metadata_df.columns:
        metadata_df.rename(
            columns={"latitude": "lat", "longitude": "lon"}, inplace=True
        )

    # Determine the geographic extent of the data with buffer
    min_lon = metadata_df["lon"].min() - buffer_degrees
    max_lon = metadata_df["lon"].max() + buffer_degrees
    min_lat = metadata_df["lat"].min() - buffer_degrees
    max_lat = metadata_df["lat"].max() + buffer_degrees

    # Ensure the extent is valid
    min_lon = max(-180, min_lon)
    max_lon = min(180, max_lon)
    min_lat = max(-90, min_lat)
    max_lat = min(90, max_lat)

    print(
        f"Map extent: Longitude [{min_lon:.2f}° to {max_lon:.2f}°], "
        f"Latitude [{min_lat:.2f}° to {max_lat:.2f}°]"
    )

    # Create figure with a suitable projection for this extent
    plt.figure(figsize=(12, 10))

    # Choose an appropriate projection depending on the extent
    lon_extent = max_lon - min_lon
    lat_extent = max_lat - min_lat

    if lon_extent > 180:
        # Global extent, Robinson is a good choice
        projection = ccrs.Robinson()
    else:
        # Regional extent, use a projection centered on the data
        central_lon = (min_lon + max_lon) / 2
        central_lat = (min_lat + max_lat) / 2

        if lat_extent > 60:  # Large latitude range
            projection = ccrs.AlbersEqualArea(
                central_longitude=central_lon, central_latitude=central_lat
            )
        else:  # Smaller extent
            projection = ccrs.LambertConformal(
                central_longitude=central_lon, central_latitude=central_lat
            )

    ax = plt.axes(projection=projection)

    # Set the map extent
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    # Add map features
    scale = "50m"
    ax.add_feature(cfeature.LAND.with_scale(scale), facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN.with_scale(scale), facecolor="lightblue")
    ax.add_feature(cfeature.COASTLINE.with_scale(scale), linewidth=0.5)
    ax.add_feature(cfeature.BORDERS.with_scale(scale), linewidth=0.3, linestyle=":")

    # Add more detailed features based on the extent
    if max_lon - min_lon < 90:
        ax.add_feature(cfeature.RIVERS, linewidth=0.2, alpha=0.5)
        ax.add_feature(cfeature.LAKES, facecolor="lightblue", alpha=0.5)

    # Get unique splits
    splits = metadata_df[split_column].unique()
    print(f"Found {len(splits)} dataset splits: {', '.join(map(str, splits))}")

    # Define colors for different splits (with defaults for train/val/test)
    split_colors = {
        "train": "blue",
        "val": "green",
        "test": "red",
        "validation": "green",
        "testing": "red",
    }

    # Create a legend handle list
    legend_elements = []

    # Create a scatter plot for each split
    for split in splits:
        split_data = metadata_df[metadata_df[split_column] == split]
        if len(split_data) > 0:
            # Get color (default to a predictable color if not in split_colors)
            color = split_colors[split]

            # Plot the points
            ax.scatter(
                split_data["lon"],
                split_data["lat"],
                transform=ccrs.PlateCarree(),
                c=color,
                s=s,
                alpha=alpha,
                label=split,
            )

            # Add to legend
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

    # Add grid lines
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False

    # Set title
    plt.title(title, fontsize=14)

    # Save the figure if output_path is provided
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
    west_east_split: float = -80.0,  # Longitude to split western/eastern hemispheres
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
    # Ensure we have lat/lon columns with consistent naming
    if "latitude" in metadata_df.columns:
        metadata_df = metadata_df.copy()
        metadata_df.rename(
            columns={"latitude": "lat", "longitude": "lon"}, inplace=True
        )

    # Split data into hemispheres and regions
    north_df = metadata_df[metadata_df["lat"] >= 0].copy()
    north_west_df = north_df[north_df["lon"] <= west_east_split].copy()
    north_east_df = north_df[north_df["lon"] > west_east_split].copy()
    south_df = metadata_df[metadata_df["lat"] < 0].copy()

    print(f"Northern hemisphere: {len(north_df)} samples")
    print(f"  - Western region: {len(north_west_df)} samples")
    print(f"  - Eastern region: {len(north_east_df)} samples")
    print(f"Southern hemisphere: {len(south_df)} samples")

    # Create figure with three subplots in a 2x2 grid (with one empty spot)
    fig = plt.figure(figsize=(20, 16))

    # Setup the grid: 2 rows, 2 columns
    gs = fig.add_gridspec(2, 2)

    # Create our three subplots
    ax_north_west = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax_north_east = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    ax_south = fig.add_subplot(gs[1, :], projection=ccrs.PlateCarree())

    # Define colors for different splits
    split_colors = {
        "train": "blue",
        "val": "green",
        "test": "red",
        "validation": "green",
        "testing": "red",
    }

    # Plot the northwestern region
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

    # Plot the northeastern region
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

    # Plot the southern hemisphere
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

    # Add overall title
    fig.suptitle(
        f"Geographic Distribution of {dataset_name} Samples by Split", fontsize=16
    )

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the figure if output_path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Map saved to {output_path}")
    else:
        plt.show()


def _plot_region(ax, df, split_column, split_colors, buffer_degrees, s, alpha, title):
    """Helper function to plot a specific region on the given axis."""
    if len(df) == 0:
        return

    # Determine the geographic extent with buffer
    min_lon = df["lon"].min() - buffer_degrees
    max_lon = df["lon"].max() + buffer_degrees
    min_lat = df["lat"].min() - buffer_degrees
    max_lat = df["lat"].max() + buffer_degrees

    # Ensure valid extent
    min_lon = max(-180, min_lon)
    max_lon = min(180, max_lon)
    min_lat = max(-90, min_lat)
    max_lat = min(90, max_lat)

    # Set up the plot extent
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    # Add map features
    scale = "50m"
    ax.add_feature(cfeature.LAND.with_scale(scale), facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN.with_scale(scale), facecolor="lightblue")
    ax.add_feature(cfeature.COASTLINE.with_scale(scale), linewidth=0.5)
    ax.add_feature(cfeature.BORDERS.with_scale(scale), linewidth=0.3, linestyle=":")

    # Add detailed features if zoomed in enough
    if max_lon - min_lon < 90:
        ax.add_feature(cfeature.RIVERS, linewidth=0.2, alpha=0.5)
        ax.add_feature(cfeature.LAKES, facecolor="lightblue", alpha=0.5)

    # Plot each split
    legend_elements = []
    for split in df[split_column].unique():
        split_data = df[df[split_column] == split]
        if len(split_data) > 0:
            color = split_colors.get(
                split, "purple"
            )  # Default to purple if split not found

            # Plot points
            ax.scatter(
                split_data["lon"],
                split_data["lat"],
                transform=ccrs.PlateCarree(),
                c=color,
                s=s,
                alpha=alpha,
                label=split,
            )

            # Add to legend
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

    # Add gridlines
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False

    # Set title
    ax.set_title(title, fontsize=12)
