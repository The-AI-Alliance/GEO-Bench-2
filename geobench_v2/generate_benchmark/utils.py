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
        # Use stratified sampling to maintain split proportions
        if split_column in metadata_df.columns:
            sampled_dfs = []
            for split in metadata_df[split_column].unique():
                split_df = metadata_df[metadata_df[split_column] == split]
                split_sample_size = max(
                    1, int(sample_size * len(split_df) / len(metadata_df))
                )
                sampled_dfs.append(
                    split_df.sample(
                        n=min(len(split_df), split_sample_size), random_state=42
                    )
                )
            metadata_df = pd.concat(sampled_dfs)
        else:
            metadata_df = metadata_df.sample(sample_size, random_state=42)
        print(f"Sampled {len(metadata_df)} points for plotting")

    # Ensure column names are standardized
    if "latitude" in metadata_df.columns:
        metadata_df = metadata_df.rename(
            columns={"latitude": "lat", "longitude": "lon"}
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
    plt.figure(figsize=(10, 8))  # Smaller figure for faster rendering

    # For France-specific data, use an optimized projection
    if 41 < min_lat < 51.5 and -5 < min_lon < 10:
        # France-specific projection
        projection = ccrs.LambertConformal(
            central_longitude=3.0, central_latitude=46.5, standard_parallels=(44, 49)
        )
    else:
        # Choose appropriate projection for other regions
        central_lon = (min_lon + max_lon) / 2
        central_lat = (min_lat + max_lat) / 2
        projection = ccrs.LambertConformal(
            central_longitude=central_lon, central_latitude=central_lat
        )

    ax = plt.axes(projection=projection)

    # Set the map extent
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    # Use lower resolution features for speed
    scale = "50m"  # Very low resolution for speed

    # Add ONLY essential map features - removing detailed features that slow down rendering
    ax.add_feature(cfeature.LAND.with_scale(scale), facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN.with_scale(scale), facecolor="lightblue")
    ax.add_feature(cfeature.COASTLINE.with_scale(scale), linewidth=0.5)
    ax.add_feature(cfeature.BORDERS.with_scale(scale), linewidth=0.3)

    # REMOVED: Rivers and Lakes features which cause the slowdown
    # Also removed the conditional for detailed features

    # Get unique splits
    if split_column in metadata_df.columns:
        splits = metadata_df[split_column].unique()
    else:
        splits = ["train", "val", "test"]  # Default splits if column doesn't exist

    print(f"Found {len(splits)} dataset splits: {', '.join(map(str, splits))}")

    # Define colors for different splits
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
        if split_column in metadata_df.columns:
            split_data = metadata_df[metadata_df[split_column] == split]
        else:
            split_data = metadata_df  # Use all data if split column doesn't exist

        if len(split_data) > 0:
            # Get color (default to blue if not in split_colors)
            color = split_colors.get(split, "blue")

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

    # Set appropriate title based on output path
    if "pastis" in output_path.lower():
        title = "Geographic Distribution of PASTIS Samples"
    elif "flair" in output_path.lower():
        title = "Geographic Distribution of FLAIR2 Samples"
    else:
        title = "Geographic Distribution of Dataset Samples"

    # Use simplified gridlines (no labels)
    gl = ax.gridlines(
        draw_labels=False, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
    )

    # Set title
    plt.title(title, fontsize=14)

    # Save the figure if output_path is provided
    if output_path:
        try:
            plt.savefig(
                output_path, dpi=150, bbox_inches="tight"
            )  # Reduced DPI for faster saving
            print(f"Map saved to {output_path}")
        except Exception as e:
            print(f"Error saving figure: {e}")
    else:
        plt.show()

    # Explicitly close the figure to free memory
    plt.close()
