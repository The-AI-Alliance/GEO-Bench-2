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
from .utils import plot_sample_locations


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
