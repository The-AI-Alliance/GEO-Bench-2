# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of Sen4Agrinet dataset."""

import geopandas as gpd
import pandas as pd
import os
import argparse
import rasterio
from tqdm import tqdm
import re
from geobench_v2.generate_benchmark.utils import plot_sample_locations
import tacotoolbox
import tacoreader
import glob
import numpy as np
import datetime


from geobench_v2.generate_benchmark.geospatial_split_utils import (
    show_samples_per_valid_ratio,
    split_geospatial_tiles_into_patches,
    visualize_checkerboard_pattern,
    visualize_geospatial_split,
    checkerboard_split,
    geographic_buffer_split,
    geographic_distance_split,
    visualize_distance_clusters,
)
import numpy as np
from itertools import combinations
from sklearn.model_selection import train_test_split

from typing import List, Tuple, Dict, Any, Optional, Union
import os
import re
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from pathlib import Path
from tqdm import tqdm

import h5py
import glob
from pathlib import Path


def create_geographic_splits(df, val_size=0.1, test_size=0.2, seed=42):
    """Create geographically distinct train/val/test splits based on tiles.

    Args:
        df: Metadata DataFrame with patch_tile information
        val_size: Target validation size (proportion)
        test_size: Target test size (proportion)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with 'split' column added
    """
    unique_tiles = sorted(df["patch_tile"].unique())
    print(f"Dataset contains {len(unique_tiles)} unique tiles: {unique_tiles}")

    n_tiles = len(unique_tiles)
    n_test_tiles = max(1, int(np.ceil(n_tiles * test_size)))
    n_val_tiles = max(1, int(np.ceil(n_tiles * val_size)))
    n_train_tiles = n_tiles - n_test_tiles - n_val_tiles

    print(
        f"Target split: {n_train_tiles} train, {n_val_tiles} val, {n_test_tiles} test tiles"
    )

    np.random.seed(seed)

    tile_counts = df["patch_tile"].value_counts().to_dict()
    total_samples = len(df)

    unique_tiles = sorted(unique_tiles)

    def evaluate_split(train_tiles, val_tiles, test_tiles):
        train_count = sum(tile_counts[t] for t in train_tiles)
        val_count = sum(tile_counts[t] for t in val_tiles)
        test_count = sum(tile_counts[t] for t in test_tiles)

        train_ratio = train_count / total_samples
        val_ratio = val_count / total_samples
        test_ratio = test_count / total_samples

        train_diff = abs(train_ratio - (1 - val_size - test_size))
        val_diff = abs(val_ratio - val_size)
        test_diff = abs(test_ratio - test_size)

        return train_diff + val_diff + test_diff

    best_score = float("inf")
    best_split = None

    if n_tiles <= 10:
        for test_tiles in combinations(unique_tiles, n_test_tiles):
            remaining_tiles = [t for t in unique_tiles if t not in test_tiles]
            for val_tiles in combinations(remaining_tiles, n_val_tiles):
                train_tiles = [t for t in remaining_tiles if t not in val_tiles]

                score = evaluate_split(train_tiles, val_tiles, test_tiles)
                if score < best_score:
                    best_score = score
                    best_split = (train_tiles, val_tiles, test_tiles)
    else:
        for _ in range(1000):
            np.random.shuffle(unique_tiles)

            test_tiles = unique_tiles[:n_test_tiles]
            val_tiles = unique_tiles[n_test_tiles : n_test_tiles + n_val_tiles]
            train_tiles = unique_tiles[n_test_tiles + n_val_tiles :]

            score = evaluate_split(train_tiles, val_tiles, test_tiles)
            if score < best_score:
                best_score = score
                best_split = (train_tiles, val_tiles, test_tiles)

    train_tiles, val_tiles, test_tiles = best_split

    train_count = sum(tile_counts[t] for t in train_tiles)
    val_count = sum(tile_counts[t] for t in val_tiles)
    test_count = sum(tile_counts[t] for t in test_tiles)

    train_ratio = train_count / total_samples
    val_ratio = val_count / total_samples
    test_ratio = test_count / total_samples

    print(f"Final split distribution:")
    print(
        f"Train: {train_count} samples ({train_ratio:.2%}) from tiles: {sorted(train_tiles)}"
    )
    print(f"Val: {val_count} samples ({val_ratio:.2%}) from tiles: {sorted(val_tiles)}")
    print(
        f"Test: {test_count} samples ({test_ratio:.2%}) from tiles: {sorted(test_tiles)}"
    )

    # Update the DataFrame
    df["split"] = "train"
    df.loc[df["patch_tile"].isin(val_tiles), "split"] = "validation"
    df.loc[df["patch_tile"].isin(test_tiles), "split"] = "test"

    return df


def generate_metadata_df(root: str) -> pd.DataFrame:
    """Generate metadata DataFrame for Sen4AgriNet dataset.

    Args:
        root: Root directory for Sen4AgriNet dataset

    Returns:
        DataFrame with metadata and file paths
    """
    metadata_records = []

    CAT_TILES = ["31TBF", "31TCF", "31TCG", "31TDF", "31TDG"]
    FR_TILES = ["31TCJ", "31TDK", "31TCL", "31TDM", "31UCP", "31UDR"]
    ALL_TILES = CAT_TILES + FR_TILES
    YEARS = ["2019", "2020"]

    nc_files = glob.glob(os.path.join(root, "**", "**", "*.nc"), recursive=True)
    print(f"Found {len(nc_files)} NetCDF files")

    for file_path in tqdm(nc_files, desc="Processing NetCDF files"):
        # replace with h5py
        netcdf = netCDF4.Dataset(file_path)

        record = {
            "path": os.path.relpath(file_path, root),
            "patch_full_name": netcdf.patch_full_name,
            "patch_year": netcdf.patch_year,
            "patch_name": netcdf.patch_name,
            "patch_country_code": netcdf.patch_country_code,
            "patch_tile": netcdf.patch_tile,
        }

        v = xr.open_dataset(xr.backends.NetCDF4DataStore(netcdf["B02"]))

        unix_timestamps = (v.time.values.astype(np.int64) // 10**9).tolist()
        record["timestamps"] = unix_timestamps

        dates = [
            datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            for ts in unix_timestamps
        ]
        record["dates"] = dates

        if dates:
            record["first_date"] = dates[0]
            record["last_date"] = dates[-1]
            record["num_timestamps"] = len(dates)

        match = re.search(r"patch_(\d+)_(\d+)", record["patch_name"])
        if match:
            record["patch_x"] = int(match.group(1))
            record["patch_y"] = int(match.group(2))

        b2_data = getattr(v, "B02")
        if b2_data is not None:
            record["height"] = b2_data.shape[1]
            record["width"] = b2_data.shape[2]

        labels_data = xr.open_dataset(xr.backends.NetCDF4DataStore(netcdf["labels"]))
        labels_array = getattr(labels_data, "labels").values
        unique_labels = np.unique(labels_array)
        record["unique_labels"] = unique_labels.tolist()
        record["num_classes"] = len(unique_labels) - 1

        label_counts = {}
        for label in unique_labels:
            count = np.sum(labels_array == label)
            label_counts[int(label)] = int(count)
        record["label_counts"] = label_counts

        total_pixels = labels_array.size
        valid_pixels = total_pixels - int(label_counts.get(0, 0))
        record["valid_ratio"] = valid_pixels / total_pixels if total_pixels > 0 else 0

        metadata_records.append(record)

    df = pd.DataFrame(metadata_records)

    print(f"Created metadata for {len(df)} samples")
    print(f"Year distribution: {df['patch_year'].value_counts().to_dict()}")
    print(f"Tile distribution: {df['patch_tile'].value_counts().to_dict()}")

    df.drop(columns=["label_counts"], inplace=True)

    df["split"] = "train"
    df.loc[df["patch_tile"].isin(FR_TILES), "split"] = "validation"
    df.loc[df["patch_tile"].isin(CAT_TILES), "split"] = "test"

    df = create_geographic_splits(df, val_size=0.1, test_size=0.2, seed=42)

    return df


def main():
    """Generate Sen4Agrinet Benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for Sen4Agrinet dataset"
    )
    parser.add_argument(
        "--save_dir",
        default="geobenchV2/Sen4Agrinet",
        help="Directory to save the subset",
    )
    args = parser.parse_args()

    metadata_path = os.path.join(args.save_dir, "geobench_sen4agrinet.parquet")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if os.path.exists(metadata_path):
        metadata_df = pd.read_parquet(metadata_path)
    else:
        metadata_df = generate_metadata_df(args.root)
        metadata_df.to_parquet(metadata_path)


if __name__ == "__main__":
    main()
