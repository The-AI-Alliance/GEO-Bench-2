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
import shutil
import concurrent.futures
from tqdm import tqdm

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from geobench_v2.generate_benchmark.utils import (
    plot_sample_locations,
    create_subset_from_df,
    create_unittest_subset,
)

import tacotoolbox
import tacoreader

import os
import re
import numpy as np
import pandas as pd
import json
from pathlib import Path
from argparse import ArgumentParser
import pickle
from PIL import Image
import rasterio
import pyproj
from tqdm import tqdm
import random


def create_geospatial_train_val_split(df):
    """Split training data into train/validation sets based on regions."""
    test_df = df[df["split"] == "test"].copy()
    train_df = df[df["split"] == "train"].copy()

    region_counts = train_df["region"].value_counts()
    total_train = len(train_df)
    target_val = total_train * 0.2

    sorted_regions = region_counts.sort_values(ascending=True).index.tolist()
    val_regions = []
    val_count = 0

    for region in sorted_regions:
        region_size = region_counts[region]

        if val_count + region_size <= target_val * 1.1:
            val_regions.append(region)
            val_count += region_size
        if val_count >= target_val * 0.9:
            break

    if val_count < target_val * 0.8:
        val_regions = []
        val_count = 0

        size_diff = [
            (r, abs(region_counts[r] - target_val)) for r in region_counts.index
        ]
        size_diff.sort(key=lambda x: x[1])

        best_region = size_diff[0][0]
        val_regions = [best_region]
        val_count = region_counts[best_region]

        if val_count < target_val * 0.8:
            for region in sorted_regions:
                if (
                    region != best_region
                    and val_count + region_counts[region] <= target_val * 1.1
                ):
                    val_regions.append(region)
                    val_count += region_counts[region]
                    if val_count >= target_val * 0.9:
                        break

    train_df.loc[train_df["region"].isin(val_regions), "split"] = "validation"

    result_df = pd.concat([train_df, test_df])

    train_count = (result_df["split"] == "train").sum()
    val_count = (result_df["split"] == "validation").sum()
    test_count = (result_df["split"] == "test").sum()

    print(f"Split statistics:")
    print(f"  Train: {train_count} samples ({train_count / len(result_df):.1%})")
    print(f"  Validation: {val_count} samples ({val_count / len(result_df):.1%})")
    print(f"  Test: {test_count} samples ({test_count / len(result_df):.1%})")
    print(f"Validation regions: {val_regions}")

    return result_df


def generate_metadata_df(root: str) -> pd.DataFrame:
    """Generate metadata for CaFFe dataset by matching masks with images.

    Args:
        root: Root directory for CaFFe dataset.

    Returns:
        DataFrame containing matched metadata for image and mask pairs.
    """
    mask_paths = glob.glob(
        os.path.join(root, "data_raw", "zones", "**", "*.png"), recursive=True
    )
    df = pd.DataFrame(mask_paths, columns=["mask_path"])

    df["mask_basename"] = df["mask_path"].apply(os.path.basename)
    df["region"] = df["mask_basename"].str.split("_").str[0]  # thinks like COL, DBE
    df["date"] = df["mask_basename"].str.split("_").str[1]  # liek 2016-08-14
    df["sensor"] = df["mask_basename"].str.split("_").str[2]  # like S1, TDX

    df["split"] = df["mask_path"].apply(
        lambda x: "train" if "train" in x else "test" if "test" in x else "unknown"
    )

    geo_tiffs_path = glob.glob(
        os.path.join(root, "geotiffs", "**", "**", "*.tif"), recursive=True
    )
    geo_df = pd.DataFrame(geo_tiffs_path, columns=["image_path"])

    geo_df["image_basename"] = geo_df["image_path"].apply(os.path.basename)
    geo_df["region"] = geo_df["image_basename"].str.split("_").str[0]
    geo_df["date"] = geo_df["image_basename"].str.split("_").str[1]
    geo_df["sensor"] = geo_df["image_basename"].str.split("_").str[2]

    merged_df = pd.merge(
        df,
        geo_df[["image_path", "region", "date", "sensor"]],
        on=["region", "date", "sensor"],
        how="inner",
    )

    print(f"Found {len(merged_df)} matching pairs out of {len(df)} masks")

    result_df = merged_df[
        ["mask_path", "image_path", "region", "date", "sensor", "split"]
    ]

    # Check for duplicates (same mask matched to multiple images)
    mask_counts = merged_df["mask_path"].value_counts()

    def extract_quality_factor(filename):
        try:
            parts = os.path.basename(filename).split("__")[0].split("_")
            if len(parts) >= 5:
                return str(parts[4])
            return None
        except (IndexError, ValueError):
            return None

    # The quality factor (with 1 being the best and 6 the worst)
    result_df["quality_factor"] = result_df["mask_path"].apply(extract_quality_factor)
    result_df = result_df[result_df["quality_factor"] == "1"]

    def extract_mask_px_dims(path):
        mask = Image.open(path)
        width, height = mask.size
        return width, height

    result_df["mask_width"], result_df["mask_height"] = zip(
        *result_df["mask_path"].apply(extract_mask_px_dims)
    )

    def extract_img_px_dims(path):
        with rasterio.open(path) as img:
            width = img.width
            height = img.height
            lng, lat = img.lnglat()
        return width, height, lng, lat

    (
        result_df["img_width"],
        result_df["img_height"],
        result_df["lon"],
        result_df["lat"],
    ) = zip(*result_df["image_path"].apply(extract_img_px_dims))

    assert result_df["mask_width"].equals(result_df["img_width"]), (
        "Image and mask widths do not match"
    )
    assert result_df["mask_height"].equals(result_df["img_height"]), (
        "Image and mask heights do not match"
    )

    result_df["image_path"] = result_df["image_path"].str.replace(args.save_dir, "")
    result_df["mask_path"] = result_df["mask_path"].str.replace(args.save_dir, "")

    result_df = create_geospatial_train_val_split(result_df)

    return result_df


def process_caffe_sample(args):
    """Process a single CaFFe sample by extracting 512x512 patches from the center."""
    idx, row, root_dir, save_dir = args

    try:
        mask_path = os.path.join(root_dir, row["mask_path"])
        image_path = os.path.join(root_dir, row["image_path"])

        basename = f"{row['region']}_{row['date']}_{row['sensor']}"

        image_dir = os.path.join(save_dir, "images")
        mask_dir = os.path.join(save_dir, "masks")
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        patch_size = 512
        patch_metadata = []

        with rasterio.open(image_path) as src:
            image_data = src.read()
            image_transform = src.transform
            image_crs = src.crs
            height, width = src.height, src.width

            # Create transformer for coordinate conversion if necessary
            transformer = None
            if image_crs and image_crs.is_valid:
                try:
                    if image_crs.to_epsg() != 4326:  # If not already in WGS84
                        transformer = pyproj.Transformer.from_crs(
                            image_crs, 4326, always_xy=True
                        )
                except (ValueError, pyproj.exceptions.CRSError):
                    try:
                        # Try using WKT representation instead of EPSG
                        transformer = pyproj.Transformer.from_crs(
                            image_crs.wkt, 4326, always_xy=True
                        )
                    except Exception as e:
                        print(
                            f"Warning: Could not create transformer for {basename}: {str(e)}"
                        )

            mask = np.array(Image.open(mask_path))
            if len(mask.shape) == 2:
                mask = mask[np.newaxis, :, :]
            else:
                mask = np.transpose(mask, (2, 0, 1))
            rows = max(1, height // patch_size)
            cols = max(1, width // patch_size)

            border_y = (height - rows * patch_size) // 2
            border_x = (width - cols * patch_size) // 2

            border_y = max(0, border_y)
            border_x = max(0, border_x)

            for r in range(rows):
                for c in range(cols):
                    row_start = border_y + r * patch_size
                    col_start = border_x + c * patch_size

                    if row_start + patch_size > height:
                        row_start = height - patch_size
                    if col_start + patch_size > width:
                        col_start = width - patch_size

                    mask_patch = mask[
                        :,
                        row_start : row_start + patch_size,
                        col_start : col_start + patch_size,
                    ]
                    img_patch = image_data[
                        :,
                        row_start : row_start + patch_size,
                        col_start : col_start + patch_size,
                    ]

                    patch_transform = rasterio.transform.from_origin(
                        image_transform.c + col_start * image_transform.a,
                        image_transform.f + row_start * image_transform.e,
                        image_transform.a,
                        image_transform.e,
                    )

                    patch_id = f"{basename}_r{r}_c{c}"
                    img_filename = f"{patch_id}.tif"
                    mask_filename = f"{patch_id}_mask.tif"

                    mask_profile = {
                        "driver": "GTiff",
                        "height": patch_size,
                        "width": patch_size,
                        "count": mask_patch.shape[0],
                        "dtype": mask_patch.dtype,
                        "tiled": True,
                        "blockxsize": patch_size,
                        "blockysize": patch_size,
                        "interleave": "pixel",
                        "compress": "zstd",
                        "zstd_level": 13,
                        "predictor": 2,
                        "crs": image_crs,
                        "transform": patch_transform,
                    }

                    mask_out_path = os.path.join(mask_dir, mask_filename)
                    with rasterio.open(mask_out_path, "w", **mask_profile) as dst:
                        dst.write(mask_patch)

                    img_profile = {
                        "driver": "GTiff",
                        "height": patch_size,
                        "width": patch_size,
                        "count": img_patch.shape[0],
                        "dtype": img_patch.dtype,
                        "tiled": True,
                        "blockxsize": patch_size,
                        "blockysize": patch_size,
                        "interleave": "pixel",
                        "compress": "zstd",
                        "zstd_level": 13,
                        "predictor": 2,
                        "crs": image_crs,
                        "transform": patch_transform,
                    }

                    img_out_path = os.path.join(image_dir, img_filename)
                    with rasterio.open(img_out_path, "w", **img_profile) as dst:
                        dst.write(img_patch)

                    # Calculate patch center coordinates in the original CRS
                    bounds = rasterio.transform.array_bounds(
                        patch_size, patch_size, patch_transform
                    )
                    west, south, east, north = bounds
                    center_x_projected = (west + east) / 2
                    center_y_projected = (north + south) / 2

                    # Transform to geographic coordinates (WGS84 lat/lon)
                    if transformer:
                        center_lon, center_lat = transformer.transform(
                            center_x_projected, center_y_projected
                        )
                    else:
                        center_lon, center_lat = center_x_projected, center_y_projected

                    patch_metadata.append(
                        {
                            "original_image": image_path,
                            "original_mask": mask_path,
                            "image_path": os.path.relpath(img_out_path, save_dir),
                            "mask_path": os.path.relpath(mask_out_path, save_dir),
                            "patch_id": patch_id,
                            "region": row["region"],
                            "date": row["date"],
                            "sensor": row["sensor"],
                            "split": row["split"],
                            "row_idx": r,
                            "col_idx": c,
                            "row_px": int(row_start),
                            "col_px": int(col_start),
                            "lon": center_lon,
                            "lat": center_lat,
                            "projected_x": center_x_projected,
                            "projected_y": center_y_projected,
                            "crs": str(image_crs),
                        }
                    )

        return patch_metadata

    except Exception as e:
        print(f"Error processing sample {idx} (mask: {row['mask_path']}): {str(e)}")
        import traceback

        traceback.print_exc()
        return []


def create_caffe_patches(metadata_df, root_dir, save_dir, num_workers=None):
    """Create 512x512 patches from CaFFe images and masks.

    Args:
        metadata_df: DataFrame with metadata including mask and image paths
        root_dir: Root directory containing dataset
        save_dir: Directory to save patches
        num_workers: Ignored parameter (kept for API compatibility)

    Returns:
        DataFrame with metadata for all created patches
    """
    os.makedirs(save_dir, exist_ok=True)

    all_patch_metadata = []

    print(f"Processing {len(metadata_df)} image-mask pairs")

    for idx, row in tqdm(
        metadata_df.iterrows(), total=len(metadata_df), desc="Creating patches"
    ):
        result = process_caffe_sample((idx, row, root_dir, save_dir))
        if result:
            all_patch_metadata.extend(result)

    patches_df = pd.DataFrame(all_patch_metadata)

    print(f"Created {len(patches_df)} patches from {len(metadata_df)} image-mask pairs")

    patches_df["image_path"] = patches_df["image_path"].str.replace(save_dir, "")
    patches_df["mask_path"] = patches_df["mask_path"].str.replace(save_dir, "")
    patches_df["original_image"] = patches_df["original_image"].str.replace(
        root_dir, ""
    )
    patches_df["original_mask"] = patches_df["original_mask"].str.replace(root_dir, "")

    return patches_df


def create_tortilla(root_dir, df, save_dir, tortilla_name):
    """Create a tortilla version of the dataset."""

    tortilla_dir = os.path.join(save_dir, "tortilla")
    os.makedirs(tortilla_dir, exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating tortilla"):
        modalities = ["image", "mask"]
        modality_samples = []

        for modality in modalities:
            path = os.path.join(root_dir, row[modality + "_path"])
            with rasterio.open(path) as src:
                profile = src.profile

            if 'PROJCS["unknown"' in str(profile["crs"]):
                # Define standard Antarctic Polar Stereographic projection
                antarctic_wkt = """PROJCS["Antarctic_Polar_Stereographic",
                    GEOGCS["WGS 84",
                        DATUM["WGS_1984",
                            SPHEROID["WGS 84",6378137,298.257223563]],
                        PRIMEM["Greenwich",0],
                        UNIT["degree",0.0174532925199433]],
                    PROJECTION["Polar_Stereographic"],
                    PARAMETER["latitude_of_origin",-71],
                    PARAMETER["central_meridian",0],
                    PARAMETER["false_easting",0],
                    PARAMETER["false_northing",0],
                    UNIT["metre",1]]"""

                stac_data = {
                    "crs": antarctic_wkt,
                    "geotransform": profile["transform"].to_gdal(),
                    "raster_shape": (profile["height"], profile["width"]),
                    "time_start": row["date"],
                }
            else:
                crs_str = "EPSG:" + str(profile["crs"].to_epsg())

                stac_data = {
                    "crs": crs_str,
                    "geotransform": profile["transform"].to_gdal(),
                    "raster_shape": (profile["height"], profile["width"]),
                    "time_start": row["date"],
                }

            sample = tacotoolbox.tortilla.datamodel.Sample(
                id=modality,
                path=path,
                file_format="GTiff",
                data_split=row["split"],
                stac_data=stac_data,
                source_mask_file=row["original_mask"],
                source_img_file=row["original_image"],
                region=row["region"],
                sensor=row["sensor"],
            )

            modality_samples.append(sample)

        taco_samples = tacotoolbox.tortilla.datamodel.Samples(samples=modality_samples)
        samples_path = os.path.join(tortilla_dir, f"sample_{idx}.tortilla")
        tacotoolbox.tortilla.create(taco_samples, samples_path, quiet=True)

    # merge tortillas into a single dataset
    all_tortilla_files = sorted(glob.glob(os.path.join(tortilla_dir, "*.tortilla")))

    samples = []

    for idx, tortilla_file in tqdm(
        enumerate(all_tortilla_files),
        total=len(all_tortilla_files),
        desc="Building taco",
    ):
        sample_data = tacoreader.load(tortilla_file).iloc[0]

        sample_tortilla = tacotoolbox.tortilla.datamodel.Sample(
            id=os.path.basename(tortilla_file).split(".")[0],
            path=tortilla_file,
            file_format="TORTILLA",
            stac_data={
                "crs": sample_data["stac:crs"],
                "geotransform": sample_data["stac:geotransform"],
                "raster_shape": sample_data["stac:raster_shape"],
                "time_start": sample_data["stac:time_start"],
            },
            data_split=sample_data["tortilla:data_split"],
            source_mask_file=sample_data["source_mask_file"],
            source_img_file=sample_data["source_img_file"],
            region=sample_data["region"],
            sensor=sample_data["sensor"],
        )
        samples.append(sample_tortilla)

    # create final taco file
    final_samples = tacotoolbox.tortilla.datamodel.Samples(samples=samples)
    tacotoolbox.tortilla.create(
        final_samples, os.path.join(save_dir, tortilla_name), quiet=True
    )


def create_geobench_version(
    metadata_df: pd.DataFrame,
    n_train_samples: int,
    n_val_samples: int,
    n_test_samples: int,
) -> None:
    """Create a GeoBench version of the dataset.
    Args:
        metadata_df: DataFrame with metadata including geolocation for each patch
        n_train_samples: Number of final training samples, -1 means all
        n_val_samples: Number of final validation samples, -1 means all
        n_test_samples: Number of final test samples, -1 means all
    """
    random_state = 24

    subset_df = create_subset_from_df(
        metadata_df,
        n_train_samples=n_train_samples,
        n_val_samples=n_val_samples,
        n_test_samples=n_test_samples,
        random_state=random_state,
    )

    return subset_df


def main():
    """Generate CaFFe Benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for CaFFe dataset"
    )
    parser.add_argument(
        "--save_dir",
        default="geobenchBenV2/caffe",
        help="Output directory for the benchmark",
    )
    args = parser.parse_args()

    new_metadata_path = os.path.join(args.save_dir, "geobench_metadata.parquet")

    os.makedirs(args.save_dir, exist_ok=True)

    if os.path.exists(new_metadata_path):
        metadata_df = pd.read_parquet(new_metadata_path)
    else:
        metadata_df = generate_metadata_df(args.root)
        metadata_df.to_parquet(new_metadata_path)

    patches_path = os.path.join(args.save_dir, "caffe_patches.parquet")

    if os.path.exists(patches_path):
        patches_df = pd.read_parquet(patches_path)
    else:
        patches_df = create_caffe_patches(
            metadata_df, args.root, args.save_dir, num_workers=8
        )
        patches_df.to_parquet(patches_path)

    result_df_path = os.path.join(args.save_dir, "geobench_caffe.parquet")
    if os.path.exists(result_df_path):
        result_df = pd.read_parquet(result_df_path)
    else:
        result_df = create_geobench_version(
            patches_df, n_train_samples=4000, n_val_samples=1000, n_test_samples=2000
        )
        result_df.to_parquet(result_df_path)

    # Create a tortilla version of the dataset
    tortilla_name = "geobench_caffe.tortilla"
    create_tortilla(args.save_dir, result_df, args.save_dir, tortilla_name)

    create_unittest_subset(
        data_dir=args.save_dir,
        tortilla_pattern=tortilla_name,
        test_dir_name="caffe",
        n_train_samples=4,
        n_val_samples=2,
        n_test_samples=2,
    )


if __name__ == "__main__":
    # full pipeline todo
    # RAW DATA download automation to merge the metadata inof
    # Torchgeo patch data generation dataset version
    # copy files from those
    main()
