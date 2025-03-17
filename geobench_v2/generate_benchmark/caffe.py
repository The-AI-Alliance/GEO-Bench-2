# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of Caffe dataset."""

from torchgeo.datasets import CaFFe
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

from geobench_v2.generate_benchmark.utils import plot_sample_locations, plot_enhanced_hemisphere_locations


import os
import re
import numpy as np
import pandas as pd
import json
from pathlib import Path
from argparse import ArgumentParser
import pickle
import cv2
from sklearn.model_selection import train_test_split


def load_metadata(metadata_path):
    try:
        metadata_df = pd.read_csv(metadata_path, delimiter=";", encoding="latin-1")
        metadata_df.columns = metadata_df.columns.str.strip()
        metadata_df["date"] = pd.to_datetime(
            metadata_df["date"], format="%d.%m.%Y", errors="coerce"
        )

        def parse_bbox(bbox_str):
            pattern = r"BoundingBox\(left=([-\d.]+),\s*bottom=([-\d.]+),\s*right=([-\d.]+),\s*top=([-\d.]+)\)"
            match = re.search(pattern, bbox_str)
            if match:
                return (
                    float(match.group(1)),
                    float(match.group(2)),
                    float(match.group(3)),
                    float(match.group(4)),
                )
            return None, None, None, None

        metadata_df[["left", "bottom", "right", "top"]] = metadata_df[
            "Bounding box coordinates"
        ].apply(lambda x: pd.Series(parse_bbox(x)))
        return metadata_df
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return pd.DataFrame()


def calculate_patch_coordinates(
    img_width,
    img_height,
    patch_x,
    patch_y,
    patch_size,
    bbox_left,
    bbox_bottom,
    bbox_right,
    bbox_top,
    coord_system,
):
    patch_center_x = patch_x + patch_size / 2
    patch_center_y = patch_y + patch_size / 2

    coord_x = bbox_left + (bbox_right - bbox_left) * (patch_center_x / img_width)
    coord_y = bbox_bottom + (bbox_top - bbox_bottom) * (
        1 - (patch_center_y / img_height)
    )

    lat, lon = None, None
    if coord_system.startswith("EPSG:"):
        try:
            import pyproj

            if coord_system != "EPSG:4326":
                projection = pyproj.Transformer.from_crs(
                    coord_system, "EPSG:4326", always_xy=True
                )
                lon, lat = projection.transform(coord_x, coord_y)
            else:
                lon, lat = coord_x, coord_y
        except ImportError:
            pass

    return coord_x, coord_y, lat, lon


def process_files_for_coordinates(
    files,
    modality_dir,
    data_split_dir,
    patch_size,
    overlap,
    metadata_df,
    patch_metadata,
):
    parent_dir = os.path.dirname(os.getcwd())

    for file in files:
        file_basename = os.path.basename(file)
        img_name = os.path.splitext(file_basename)[0]

        original_name = img_name.split("__")[0] if "__" in img_name else img_name
        tif_match = metadata_df[metadata_df["image_name"].str.startswith(original_name)]

        if len(tif_match) == 0:
            print(f"WARNING: No metadata match found for {original_name}")
            continue

        img_metadata = tif_match.iloc[0]
        image_date = img_metadata["date"]
        bbox_left = img_metadata["left"]
        bbox_bottom = img_metadata["bottom"]
        bbox_right = img_metadata["right"]
        bbox_top = img_metadata["top"]
        coord_system = img_metadata["Coordinate system"]

        try:
            image = cv2.imread(file.__str__(), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                orig_height, orig_width = image.shape
            else:
                resolution_m = float(img_metadata["resolution (m)"])
                orig_width = int((bbox_right - bbox_left) / resolution_m)
                orig_height = int((bbox_top - bbox_bottom) / resolution_m)
        except:
            resolution_m = float(img_metadata["resolution (m)"])
            orig_width = int((bbox_right - bbox_left) / resolution_m)
            orig_height = int((bbox_top - bbox_bottom) / resolution_m)

        bottom = patch_size - (orig_height % patch_size)
        bottom = bottom % patch_size
        right = patch_size - (orig_width % patch_size)
        right = right % patch_size

        if overlap > 0:
            bottom = (patch_size - overlap) - (
                (orig_height - patch_size) % (patch_size - overlap)
            )
            right = (patch_size - overlap) - (
                (orig_width - patch_size) % (patch_size - overlap)
            )

        stride = (patch_size - overlap, patch_size - overlap)
        padded_height = orig_height + bottom
        padded_width = orig_width + right

        x_tmp = np.arange(0, padded_height - patch_size + 1, stride[0])
        y_tmp = np.arange(0, padded_width - patch_size + 1, stride[1])

        x_coord, y_coord = np.meshgrid(x_tmp, y_tmp)
        x_coord = x_coord.ravel()
        y_coord = y_coord.ravel()

        for j in range(len(x_coord)):
            patch_x = x_coord[j]
            patch_y = y_coord[j]

            center_x, center_y, lat, lon = calculate_patch_coordinates(
                orig_width,
                orig_height,
                patch_x,
                patch_y,
                patch_size,
                bbox_left,
                bbox_bottom,
                bbox_right,
                bbox_top,
                coord_system,
            )

            add_to_name = f"__{bottom}_{right}_{j}_{patch_x}_{patch_y}.png"
            patch_filename = img_name + add_to_name

            patch_metadata[patch_filename] = {
                "timestamp": image_date.strftime("%Y-%m-%d")
                if not pd.isna(image_date)
                else None,
                "center_x": float(center_x),
                "center_y": float(center_y),
                "latitude": float(lat) if lat is not None else None,
                "longitude": float(lon) if lon is not None else None,
                "coordinate_system": coord_system,
                "original_image": img_metadata["image_name"],
                "glacier_name": img_metadata["glacier_name"].strip()
                if not pd.isna(img_metadata["glacier_name"])
                else None,
                "sensor": img_metadata["sensor"].strip()
                if not pd.isna(img_metadata["sensor"])
                else None,
                "resolution_m": float(img_metadata["resolution (m)"])
                if not pd.isna(img_metadata["resolution (m)"])
                else None,
                "polarization": img_metadata["polarization"].strip()
                if not pd.isna(img_metadata["polarization"])
                else None,
                "data_split": data_split_dir,
                "patch_x": int(patch_x),
                "patch_y": int(patch_y),
                "patch_idx": int(j),
            }


def save_patch_coordinates_only(
    raw_data_dir, patch_size, overlap, overlap_test, overlap_val, metadata_csv
):
    parent_dir = os.path.dirname(os.getcwd())
    patch_metadata = {}

    metadata_df = load_metadata(metadata_csv)
    if metadata_df.empty:
        print(f"ERROR: Failed to load metadata from {metadata_csv}")
        return

    for modality_dir in ["sar_images"]:
        for data_split_dir in ["test", "train"]:
            raw_dir_path = os.path.join(
                parent_dir, raw_data_dir, modality_dir, data_split_dir
            )
            if not os.path.exists(raw_dir_path):
                print(f"Directory not found: {raw_dir_path}")
                continue

            folder = sorted(Path(raw_dir_path).rglob("*.png"))
            files = [x for x in folder]

            if data_split_dir == "train":
                if not os.path.exists("data_splits"):
                    os.makedirs("data_splits")

                data_idx = np.arange(len(files))
                train_idx, val_idx = train_test_split(
                    data_idx, test_size=0.1, random_state=1
                )

                with open(os.path.join("data_splits", "train_idx.txt"), "wb") as fp:
                    pickle.dump(train_idx, fp)

                with open(os.path.join("data_splits", "val_idx.txt"), "wb") as fp:
                    pickle.dump(val_idx, fp)

                process_files_for_coordinates(
                    [files[i] for i in train_idx],
                    modality_dir,
                    data_split_dir,
                    patch_size,
                    overlap,
                    metadata_df,
                    patch_metadata,
                )

                process_files_for_coordinates(
                    [files[i] for i in val_idx],
                    modality_dir,
                    "val",
                    patch_size,
                    overlap_val,
                    metadata_df,
                    patch_metadata,
                )
            else:
                process_files_for_coordinates(
                    files,
                    modality_dir,
                    data_split_dir,
                    patch_size,
                    overlap_test,
                    metadata_df,
                    patch_metadata,
                )

    patches_df = pd.DataFrame.from_dict(patch_metadata, orient="index").reset_index()
    patches_df.rename(columns={"index": "filename"}, inplace=True)
    return patches_df


def create_subset(ds: CaFFe, metadata_df: pd.DataFrame, save_dir: str) -> None:
    """Create a subset of CaFFe dataset.

    Args:
        ds: CaFFe dataset.
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
    patch_dir = "_".join(patch_id.split("_")[0:-2])

    # Find the first TIF file in the patch directory
    path_pattern = os.path.join(
        root, dir_file_names["s2"], patch_dir, patch_id, "*.tif"
    )
    paths = glob.glob(path_pattern)

    if not paths:
        return {
            "patch_id": patch_id,
            "lon": None,
            "lat": None,
            "error": "No TIF files found",
        }

    try:
        with rasterio.open(paths[0]) as src:
            lon, lat = src.lnglat()
            return {"patch_id": patch_id, "lon": lon, "lat": lat}
    except Exception as e:
        return {"patch_id": patch_id, "lon": None, "lat": None, "error": str(e)}


def generate_metadata_df() -> pd.DataFrame:
    """Generate metadata DataFrame for CaFFe dataset with parallel processing.

    Args:
        ds: CaFFe dataset
        num_workers: Number of parallel workers to use

    Returns:
        DataFrame with metadata including geolocation for each patch
    """
    # TODO add download and unzip of raw data to the root directory
    # and the metadata CSV file from huggingface, both available on torchgeo huggingface
    df = save_patch_coordinates_only(
        raw_data_dir="data",
        patch_size=512,
        overlap=0,
        overlap_test=128,
        overlap_val=128,
        metadata_csv=os.path.join("data", "meta_data.csv"),
    )

    return df


def create_unit_test_subset() -> None:
    """Create a subset of CaFFe dataset for GeoBench unit tests."""

    # create random images etc that respect the structure of the dataset in minimal format
    pass

def plot_samples_by_glacier(metadata_df: pd.DataFrame, output_path: str, dataset_name: str) -> None:
    """Create plots of sample locations grouped by glacier.
    
    Args:
        metadata_df: DataFrame containing metadata with glacier_name, latitude, longitude, and split
        output_path: Path to save the output plot
        dataset_name: Name of the dataset for the title
    """
    # Ensure we have required columns
    required_cols = ['glacier_name', 'latitude', 'longitude', 'split']
    if not all(col in metadata_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in metadata_df.columns]
        raise ValueError(f"Missing required columns: {missing}")
    
    # Drop rows with missing lat/lon
    valid_df = metadata_df.dropna(subset=['latitude', 'longitude', 'glacier_name'])
    if len(valid_df) == 0:
        print("No valid samples with location data to plot")
        return
    
    # Get unique glaciers
    glaciers = valid_df['glacier_name'].unique()
    num_glaciers = len(glaciers)
    
    # Calculate grid dimensions for subplots
    n_cols = min(3, num_glaciers)  # Max 3 columns
    n_rows = (num_glaciers + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure
    fig = plt.figure(figsize=(n_cols * 6, n_rows * 5))
    fig.suptitle(f"{dataset_name} Samples by Glacier (Total: {len(valid_df)} samples)", fontsize=16, y=0.98)
    
    # Create a shared colormap for all glaciers
    split_cmap = {
        'train': '#1f77b4',  # Blue
        'val': '#ff7f0e',    # Orange
        'test': '#2ca02c'    # Green
    }
    
    # Get global min/max longitude for all glaciers
    global_stats = {}
    for glacier in glaciers:
        glacier_df = valid_df[valid_df['glacier_name'] == glacier]
        global_stats[glacier] = {
            'min_lon': glacier_df['longitude'].min(),
            'max_lon': glacier_df['longitude'].max(),
            'min_lat': glacier_df['latitude'].min(),
            'max_lat': glacier_df['latitude'].max(),
            'lon_range': glacier_df['longitude'].max() - glacier_df['longitude'].min(),
            'lat_range': glacier_df['latitude'].max() - glacier_df['latitude'].min()
        }
    
    # Create subplots
    for i, glacier in enumerate(glaciers):
        glacier_df = valid_df[valid_df['glacier_name'] == glacier]
        
        split_counts = glacier_df['split'].value_counts().to_dict()
        for split in ['train', 'val', 'test']:
            if split not in split_counts:
                split_counts[split] = 0
        
        # Calculate subplot position
        ax = fig.add_subplot(n_rows, n_cols, i+1, projection=ccrs.PlateCarree())

        min_lon = global_stats[glacier]['min_lon']
        max_lon = global_stats[glacier]['max_lon']
        min_lat = global_stats[glacier]['min_lat']
        max_lat = global_stats[glacier]['max_lat']
        
        lon_buffer = global_stats[glacier]['lon_range'] * 0.1
        lat_buffer = global_stats[glacier]['lat_range'] * 0.1
        lon_buffer = max(lon_buffer, 0.01)
        lat_buffer = max(lat_buffer, 0.01)
        
        ax.set_extent([min_lon - lon_buffer, max_lon + lon_buffer, 
                      min_lat - lat_buffer, max_lat + lat_buffer])
        
        # Add coastlines and borders
        scale='110m'
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.STATES, linewidth=0.3)
        ax.gridlines(draw_labels=True, linewidth=0.2, color='gray', alpha=0.5)
        
        # Plot each split with different colors
        for split in ['train', 'val', 'test']:
            split_data = glacier_df[glacier_df['split'] == split]
            if len(split_data) > 0:
                ax.scatter(
                    split_data['longitude'], split_data['latitude'],
                    c=split_cmap[split], label=f"{split} ({len(split_data)})",
                    alpha=0.7, s=20, transform=ccrs.PlateCarree()
                )
        
        total_count = len(glacier_df)
        title = f"{glacier} Glacier ({total_count} samples)\n"
        title += f"Train: {split_counts['train']}, Val: {split_counts['val']}, Test: {split_counts['test']}"
        ax.set_title(title, fontsize=12)
        
        # Add legend
        ax.legend(loc='upper right')
    
    total_counts = valid_df['split'].value_counts().to_dict()
    for split in ['train', 'val', 'test']:
        if split not in total_counts:
            total_counts[split] = 0
    
    # Add an overall summary in the figure footer
    summary = (f"Summary - Total: {len(valid_df)} samples across {num_glaciers} glaciers | " 
               f"Train: {total_counts.get('train', 0)} | "
               f"Val: {total_counts.get('val', 0)} | "
               f"Test: {total_counts.get('test', 0)}")
    
    fig.text(0.5, 0.01, summary, ha='center', fontsize=14)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Plot saved to {output_path}")
    plt.close(fig)

    # Create and save a summary DataFrame
    summary_data = []
    for glacier in glaciers:
        glacier_df = valid_df[valid_df['glacier_name'] == glacier]
        split_counts = glacier_df['split'].value_counts().to_dict()
        summary_data.append({
            'glacier_name': glacier,
            'train_samples': split_counts.get('train', 0),
            'val_samples': split_counts.get('val', 0),
            'test_samples': split_counts.get('test', 0),
            'total_samples': len(glacier_df)
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('total_samples', ascending=False)
    
    # Add total row
    total_row = {
        'glacier_name': 'TOTAL',
        'train_samples': total_counts.get('train', 0),
        'val_samples': total_counts.get('val', 0),
        'test_samples': total_counts.get('test', 0),
        'total_samples': len(valid_df)
    }
    summary_df = pd.concat([summary_df, pd.DataFrame([total_row])], ignore_index=True)
    
    # Save summary to CSV
    summary_path = output_path.replace('.png', '_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")


def main():
    """Generate CaFFe Benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for CaFFe dataset"
    )
    parser.add_argument(
        "--output-dir",
        default="geobenchBenV2/caffe",
        help="Output directory for the benchmark",
    )
    args = parser.parse_args()

    new_metadata_path = os.path.join(args.output_dir, "geobench_metadata.parquet")

    orig_dataset = CaFFe(root=args.root, download=False)

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(new_metadata_path):
        metadata_df = generate_metadata_df(orig_dataset)
        metadata_df.to_parquet(new_metadata_path)
    else:
        metadata_df = pd.read_parquet(new_metadata_path)

    metadata_df.rename(columns={"data_split": "split"}, inplace=True)

    # The quality factor (with 1 being the best and 6 the worst)
    #

    def extract_quality_factor(filename):
        try:
            parts = os.path.basename(filename).split("__")[0].split("_")
            # The quality factor should be the 5th element (index 4)
            if len(parts) >= 5:
                return str(parts[4])
            return None
        except (IndexError, ValueError):
            return None

    # Extract quality factor from the original image name
    metadata_df["quality_factor"] = metadata_df["filename"].apply(
        extract_quality_factor
    )

    plot_enhanced_hemisphere_locations(
        metadata_df,
        output_path=os.path.join(args.output_dir, "caffe_hemispheres.png"),
        dataset_name="CaFFe",
        buffer_degrees=1.0,
        s=5,  # Increase point size slightly for better visibility
        alpha=0.7
    )


if __name__ == "__main__":
    main()
