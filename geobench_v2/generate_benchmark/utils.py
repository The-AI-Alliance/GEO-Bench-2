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
    sample_fraction: float = 1.0,
    alpha: float = 0.5,
    s: float = 0.5,
) -> None:
    """Plot the geolocation of samples on a map, differentiating by dataset splits.
    Creates a subplot for each region defined in the 'region' column.

    Args:
        metadata_df: DataFrame with metadata including lat/lon columns and region column
        output_path: Path to save the figure
        buffer_degrees: Buffer around the data extent in degrees
        split_column: Column name that indicates the dataset split
        sample_fraction: Fraction of samples to plot
        alpha: Transparency of plotted points
        s: Size of plotted points
    """
    # Ensure column names are standardized
    if "latitude" in metadata_df.columns:
        metadata_df = metadata_df.rename(columns={"latitude": "lat", "longitude": "lon"})
    
    # Sample data if fraction is less than 1.0
    if sample_fraction < 1.0:
        sample_size = int(len(metadata_df) * sample_fraction)
        metadata_df = metadata_df.sample(sample_size, random_state=42)
        print(f"Sampled {len(metadata_df)} points for plotting")
    
    # Check if region column exists
    if "region" not in metadata_df.columns:
        # Fall back to single plot
        print("No 'region' column found. Creating a single plot for all data.")
        # Single plot code here
        return
    
    # Get unique regions
    regions = metadata_df["region"].unique()
    n_regions = len(regions)
    print(f"Found {n_regions} regions: {', '.join(regions)}")
    
    # Create figure first
    if n_regions == 1:
        fig = plt.figure(figsize=(10, 8))
        n_rows, n_cols = 1, 1
    else:
        # For many regions, determine appropriate layout
        n_cols = min(n_regions, 2)  # Maximum 2 columns
        n_rows = (n_regions + n_cols - 1) // n_cols
        fig = plt.figure(figsize=(n_cols*8, n_rows*6))
    
    # For dataset name in title
    if output_path:
        if "spacenet" in output_path.lower():
            dataset_name = "SpaceNet8"
        elif "flair" in output_path.lower():
            dataset_name = "FLAIR2"
        elif "pastis" in output_path.lower():
            dataset_name = "PASTIS"
        else:
            dataset_name = "Dataset"
        
        fig.suptitle(f"Geographic Distribution of {dataset_name} Samples", fontsize=14, y=0.98)
    
    # Plot each region
    for i, region_name in enumerate(regions):
        # Get data for this region
        region_df = metadata_df[metadata_df["region"] == region_name]
        
        # Skip empty regions
        if len(region_df) == 0:
            continue
        
        # Calculate bounds for this region
        min_lon = region_df["lon"].min() - buffer_degrees
        max_lon = region_df["lon"].max() + buffer_degrees
        min_lat = region_df["lat"].min() - buffer_degrees
        max_lat = region_df["lat"].max() + buffer_degrees
        
        # Ensure valid bounds
        min_lon, max_lon = max(-180, min_lon), min(180, max_lon)
        min_lat, max_lat = max(-90, min_lat), min(90, max_lat)
        
        # Create subplot with projection from the start
        # IMPORTANT: Use fig.add_subplot() with projection parameter
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection=ccrs.PlateCarree())
        
        # Add title
        ax.set_title(f"{region_name} (n={len(region_df)})", fontsize=12)
        
        # Add map features
        scale = "50m"  # Medium resolution
        ax.add_feature(cfeature.LAND.with_scale(scale), facecolor="lightgray")
        ax.add_feature(cfeature.OCEAN.with_scale(scale), facecolor="lightblue")
        ax.add_feature(cfeature.COASTLINE.with_scale(scale), linewidth=0.5)
        ax.add_feature(cfeature.BORDERS.with_scale(scale), linewidth=0.3)
        
        # Set extent
        ax.set_extent([min_lon, max_lon, min_lat, max_lat])
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.3, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
        
        # Legend elements
        legend_elements = []
        
        # Plot each split
        for split in sorted(region_df[split_column].unique()):
            split_data = region_df[region_df[split_column] == split]
            
            # Skip empty splits
            if len(split_data) == 0:
                continue
            
            # Get color
            color = {"train": "blue", "val": "green", "validation": "green", "test": "red"}.get(split, "blue")
            
            # Plot points
            ax.scatter(
                split_data["lon"],
                split_data["lat"],
                c=color,
                s=s,
                alpha=alpha,
                label=split,
            )
            
            # Add to legend
            legend_elements.append(
                Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                    markersize=6, label=f"{split} (n={len(split_data)})")
            )
        
        # Add legend
        if legend_elements:
            ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    
    plt.tight_layout()
    
    # Save the figure if output_path is provided
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Map saved to {output_path}")
    else:
        plt.show()
    
    plt.close(fig)