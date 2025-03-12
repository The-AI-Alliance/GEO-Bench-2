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
    dpi: int = 300,
    optimize_for_speed: bool = True,
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
        dpi: Resolution of the output image
        optimize_for_speed: Whether to apply optimizations for large datasets
    """
    import time

    start_time = time.time()

    # Handle column naming
    if "latitude" in metadata_df.columns:
        metadata_df = metadata_df.rename(
            columns={"latitude": "lat", "longitude": "lon"}
        )

    # For extremely large datasets, apply more aggressive downsampling
    if optimize_for_speed and len(metadata_df) > 10000:
        # Determine optimal sample size based on dataset size
        if len(metadata_df) > 100000:
            sample_fraction = min(sample_fraction, 0.05)  # Very large dataset
        elif len(metadata_df) > 50000:
            sample_fraction = min(sample_fraction, 0.1)  # Large dataset

        # Use stratified sampling to maintain representative distribution
        sample_size = int(len(metadata_df) * sample_fraction)
        sampled_df = metadata_df.groupby(split_column, group_keys=False).apply(
            lambda x: x.sample(
                min(len(x), max(1, int(sample_size * len(x) / len(metadata_df)))),
                random_state=42,
            )
        )
        print(
            f"Optimizing for speed: Sampled {len(sampled_df)} points from {len(metadata_df)} "
            f"({100 * len(sampled_df) / len(metadata_df):.1f}%)"
        )
    else:
        # Standard sampling
        if sample_fraction < 1.0:
            sample_size = int(len(metadata_df) * sample_fraction)
            sampled_df = metadata_df.sample(sample_size, random_state=42)
            print(f"Sampled {sample_size} points for plotting")
        else:
            sampled_df = metadata_df

    # Determine the geographic extent of the data with buffer
    min_lon = sampled_df["lon"].min() - buffer_degrees
    max_lon = sampled_df["lon"].max() + buffer_degrees
    min_lat = sampled_df["lat"].min() - buffer_degrees
    max_lat = sampled_df["lat"].max() + buffer_degrees

    # Ensure the extent is valid
    min_lon = max(-180, min_lon)
    max_lon = min(180, max_lon)
    min_lat = max(-90, min_lat)
    max_lat = min(90, max_lat)

    print(
        f"Map extent: Longitude [{min_lon:.2f}° to {max_lon:.2f}°], "
        f"Latitude [{min_lat:.2f}° to {max_lat:.2f}°]"
    )

    # Use a smaller figure size and simplified features for speed
    if optimize_for_speed:
        figsize = (10, 8)
        use_detailed_features = False
    else:
        figsize = (12, 10)
        use_detailed_features = max_lon - min_lon < 90

    # Create figure with a suitable projection for this extent
    plt.figure(figsize=figsize)

    # Choose an appropriate projection depending on the extent
    lon_extent = max_lon - min_lon
    lat_extent = max_lat - min_lat

    # For France-specific data (like FLAIR2), a Lambert Conformal Conic centered on France is good
    if (
        41 < min_lat < 52
        and -5 < min_lon < 10
        and 41 < max_lat < 52
        and -5 < max_lon < 10
    ):
        # This is approximately France
        projection = ccrs.LambertConformal(
            central_longitude=2.0, central_latitude=46.5, standard_parallels=(44, 49)
        )
    elif lon_extent > 180:
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

    # Add map features with varying detail based on speed optimization
    if optimize_for_speed:
        # Simpler features for speed
        ax.add_feature(cfeature.LAND, facecolor="lightgray")
        ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    else:
        # More detailed features
        ax.add_feature(cfeature.LAND, facecolor="lightgray")
        ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")

        # Add more detailed features based on the extent
        if use_detailed_features:
            ax.add_feature(cfeature.RIVERS, linewidth=0.2, alpha=0.5)
            ax.add_feature(cfeature.LAKES, facecolor="lightblue", alpha=0.5)

    # Get unique splits
    splits = ["train", "val", "test"]
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

    # For very large datasets, use hexbin instead of scatter for performance
    if optimize_for_speed and len(sampled_df) > 5000:
        for split in splits:
            split_data = sampled_df[sampled_df[split_column] == split]
            if len(split_data) > 0:
                color = split_colors[split]

                # Use hexbin for better performance with many points
                hexbin = ax.hexbin(
                    split_data["lon"],
                    split_data["lat"],
                    transform=ccrs.PlateCarree(),
                    gridsize=50,  # Adjust this value based on density and desired appearance
                    cmap=plt.cm.get_cmap(
                        f"Blues"
                        if split == "train"
                        else "Greens"
                        if split == "val"
                        else "Reds"
                    ),
                    mincnt=1,
                    alpha=0.7,
                )

                # Add to legend
                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        marker="h",
                        color="w",
                        markerfacecolor=color,
                        markersize=8,
                        label=f"{split} (n={len(split_data)})",
                    )
                )
    else:
        # Use regular scatter plot for smaller datasets
        for split in splits:
            split_data = sampled_df[sampled_df[split_column] == split]
            if len(split_data) > 0:
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

    # Add legend in a good position
    ax.legend(handles=legend_elements, loc="lower right", title="Dataset Splits")

    # Set dataset-specific title or use default
    if "flair" in output_path.lower():
        title = "Geographic Distribution of FLAIR2 Samples by Split"
    else:
        title = "Geographic Distribution of Dataset Samples by Split"

    # Add grid lines (simplified for speed if needed)
    if optimize_for_speed:
        gl = ax.gridlines(linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
    else:
        gl = ax.gridlines(
            draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
        )
        gl.top_labels = False
        gl.right_labels = False

    # Set title
    plt.title(title, fontsize=14)

    # Save the figure if output_path is provided
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=dpi)
        elapsed_time = time.time() - start_time
        print(f"Map saved to {output_path} (took {elapsed_time:.1f} seconds)")
    else:
        plt.tight_layout()
        plt.show()

    plt.close()  # Close the figure to free memory
