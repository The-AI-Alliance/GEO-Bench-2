# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Utility functions for benchmark generation."""

from matplotlib.lines import Line2D
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def validate_metadata_with_geo(metadata_df: pd.DataFrame) -> None:
    """Validate the metadata DataFrame for benchmark generation.

    Args:
        metadata_df: DataFrame with metadata columns

    Raises:
        AssertionError: If metadata is missing required columns
    """
    assert "split" in metadata_df.columns, "Metadata must contain 'split' column"
    assert "lat" in metadata_df.columns, "Metadata must contain 'lat' column"
    assert "lon" in metadata_df.columns, "Metadata must contain 'lon' column"
    assert "crs" in metadata_df.columns, "Metadata must contain 'crs' column"
    assert "sample_id" in metadata_df.columns, (
        "Metadata must contain 'sample_id' column"
    )


def validate_metadata(metadata_df: pd.DataFrame) -> None:
    """Validate the metadata DataFrame for benchmark generation, for datasets without geospatial information.

    Args:
        metadata_df: DataFrame with metadata columns

    Raises:
        AssertionError: If metadata is missing required columns
    """
    assert "split" in metadata_df.columns, "Metadata must contain 'split' column"
    assert "sample_id" in metadata_df.columns, (
        "Metadata must contain 'sample_id' column"
    )


def plot_sample_locations(
    metadata_df: pd.DataFrame,
    output_path: str = None,
    buffer_degrees: float = 5.0,
    split_column: str = "split",
    sample_fraction: float = 0.8,
    alpha: float = 0.5,
    s: float = 0.5,
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
            columns={"latitude": "lat", "longitude": "long"}, inplace=True
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
    scale = "110m"
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

    title = "Geographic Distribution of CloudSen12 Samples by Split"

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
