# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of BenV2 dataset."""

from torchgeo.datasets import BigEarthNetV2
import argparse
import rasterio
import os

from typing import Any
import pandas as pd
import glob
import concurrent.futures
from tqdm import tqdm

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


def create_subset(
    ds: BigEarthNetV2, metadata_df: pd.DataFrame, save_dir: str
) -> None:
    """Create a subset of BigEarthNet dataset.

    Args:
        ds: BigEarthNet dataset.
        metadata_df: Metadata DataFrame.
        save_dir: Directory to save the subset.
    """
    # based on the metadata_df create a subset of the dataset and copy it
    # with the same structure to the save_dir
    pass

def plot_sample_locations(
    metadata_df: pd.DataFrame, 
    output_path: str = None, 
    buffer_degrees: float = 5.0,
    split_column: str = "split",
    sample_fraction: float = 0.8,
    alpha: float = 0.5,
    s: float = 0.5
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
    # Filter out rows with missing coordinates
    valid_data = metadata_df.dropna(subset=['lon', 'lat'])
    
    print(f"Plotting {len(valid_data)} samples with valid coordinates "
          f"({len(metadata_df) - len(valid_data)} samples had missing coordinates)")
    
    if len(valid_data) == 0:
        print("No valid coordinates found, cannot create plot")
        return
    
    # Sample data if fraction is less than 1.0
    if sample_fraction < 1.0:
        sample_size = int(len(valid_data) * sample_fraction)
        valid_data = valid_data.sample(sample_size, random_state=42)
        print(f"Sampled {sample_size} points for plotting")
    
    # Determine the geographic extent of the data with buffer
    min_lon = valid_data['lon'].min() - buffer_degrees
    max_lon = valid_data['lon'].max() + buffer_degrees
    min_lat = valid_data['lat'].min() - buffer_degrees
    max_lat = valid_data['lat'].max() + buffer_degrees
    
    # Ensure the extent is valid
    min_lon = max(-180, min_lon)
    max_lon = min(180, max_lon)
    min_lat = max(-90, min_lat)
    max_lat = min(90, max_lat)
    
    print(f"Map extent: Longitude [{min_lon:.2f}° to {max_lon:.2f}°], "
          f"Latitude [{min_lat:.2f}° to {max_lat:.2f}°]")
    
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
            projection = ccrs.AlbersEqualArea(central_longitude=central_lon, 
                                             central_latitude=central_lat)
        else:  # Smaller extent
            projection = ccrs.LambertConformal(central_longitude=central_lon, 
                                              central_latitude=central_lat)
    
    ax = plt.axes(projection=projection)
    
    # Set the map extent
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')
    
    # Add more detailed features based on the extent
    if max_lon - min_lon < 90:
        ax.add_feature(cfeature.RIVERS, linewidth=0.2, alpha=0.5)
        ax.add_feature(cfeature.LAKES, facecolor='lightblue', alpha=0.5)
    
    # Check if split column exists
    if split_column in valid_data.columns:
        # Get unique splits
        splits = valid_data[split_column].unique()
        print(f"Found {len(splits)} dataset splits: {', '.join(map(str, splits))}")
        
        # Define colors for different splits (with defaults for train/val/test)
        split_colors = {
            'train': 'blue',
            'val': 'green',
            'test': 'red',
            'validation': 'green',
            'testing': 'red',
        }
        
        # Create a legend handle list
        from matplotlib.lines import Line2D
        legend_elements = []
        
        # Create a scatter plot for each split
        for split in splits:
            split_data = valid_data[valid_data[split_column] == split]
            if len(split_data) > 0:
                # Get color (default to a predictable color if not in split_colors)
                if split in split_colors:
                    color = split_colors[split]
                else:
                    # Use a hash of the split name to get a consistent color
                    import hashlib
                    color_hash = int(hashlib.md5(str(split).encode()).hexdigest(), 16)
                    color = plt.cm.tab10(color_hash % 10)
                
                # Plot the points
                ax.scatter(
                    split_data['lon'], 
                    split_data['lat'],
                    transform=ccrs.PlateCarree(),
                    c=color,
                    s=s,
                    alpha=alpha,
                    label=split
                )
                
                # Add to legend
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                           markersize=8, label=f"{split} (n={len(split_data)})")
                )
                
        # Add legend
        ax.legend(handles=legend_elements, loc='lower right', title="Dataset Splits")
        
        title = 'Geographic Distribution of BigEarthNetV2 Samples by Split'
    else:
        # If no split column, plot all points in a single color
        ax.scatter(
            valid_data['lon'], 
            valid_data['lat'],
            transform=ccrs.PlateCarree(),
            c='blue',
            s=s,
            alpha=alpha
        )
        title = 'Geographic Distribution of BigEarthNetV2 Samples'
        
    # Add grid lines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Set title
    plt.title(title, fontsize=14)
    
    # Save the figure if output_path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Map saved to {output_path}")


def process_row(args: tuple) -> dict[str, Any]:
    """Process a single row from the metadata DataFrame.
    
    Args:
        args: Tuple containing (row, root, dir_file_names)
        
    Returns:
        Dictionary with patch_id, lon, and lat
    """
    row, root, dir_file_names = args
    patch_id = row["patch_id"]
    patch_dir = '_'.join(patch_id.split('_')[0:-2])
    
    # Find the first TIF file in the patch directory
    path_pattern = os.path.join(
        root, dir_file_names['s2'], patch_dir, patch_id, '*.tif'
    )
    paths = glob.glob(path_pattern)
    
    if not paths:
        return {"patch_id": patch_id, "lon": None, "lat": None, "error": "No TIF files found"}
    
    try:
        with rasterio.open(paths[0]) as src:
            lon, lat = src.lnglat()
            return {
                "patch_id": patch_id,
                "lon": lon,
                "lat": lat,
            }
    except Exception as e:
        return {
            "patch_id": patch_id,
            "lon": None,
            "lat": None,
            "error": str(e)
        }


def generate_metadata_df(ds: BigEarthNetV2, num_workers: int = 8) -> pd.DataFrame:
    """Generate metadata DataFrame for BigEarthNet dataset with parallel processing.
    
    Args:
        ds: BigEarthNet dataset
        num_workers: Number of parallel workers to use
        
    Returns:
        DataFrame with metadata including geolocation for each patch
    """
    full_metadata_df = pd.read_parquet(os.path.join(ds.root, 'metadata.parquet'))
    print(f"Generating metadata for {len(full_metadata_df)} patches using {num_workers} workers...")

    
    # Prepare arguments for parallel processing
    args_list = [(row, ds.root, ds.dir_file_names) for _, row in full_metadata_df.iterrows()]
    
    # Process in parallel with progress bar
    metadata = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Use tqdm to show progress
        for result in tqdm(
            executor.map(process_row, args_list),
            total=len(args_list),
            desc="Processing patches"
        ):
            metadata.append(result)
    
    # Create DataFrame from results
    metadata_df = pd.DataFrame(metadata)

    metadata_df = pd.merge(full_metadata_df, metadata_df, how="left", on="patch_id")

    return metadata_df


def create_unit_test_subset() -> None:
    """Create a subset of BigEarthNet dataset for GeoBench unit tests."""

    # create random images etc that respect the structure of the dataset in minimal format
    pass


def main():
    """Generate BigEarthNet Benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for BigEarthNet dataset"
    )
    parser.add_argument(
        "--output-dir", default="geobenchBenV2", help="Output directory for the benchmark"
    )
    args = parser.parse_args()

    new_metadata_path = os.path.join(args.output_dir, "geobench_metadata.parquet")

    orig_dataset = BigEarthNetV2(root=args.root, download=False)

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(new_metadata_path):
        metadata_df = generate_metadata_df(orig_dataset)
        metadata_df.to_parquet(new_metadata_path)
    else:
        metadata_df = pd.read_parquet(new_metadata_path)

    plot_sample_locations(metadata_df, output_path=os.path.join(args.output_dir, "sample_locations.png"), sample_fraction=0.20)



if __name__ == "__main__":
    main()
