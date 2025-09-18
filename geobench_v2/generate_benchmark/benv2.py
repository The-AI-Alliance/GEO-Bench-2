# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of BenV2 dataset."""

import argparse
import concurrent.futures
import glob
import os
import random
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import tacoreader
import tacotoolbox
from rasterio.enums import Resampling
from tqdm import tqdm

from geobench_v2.generate_benchmark.utils import (
    create_subset_from_df,
    create_unittest_subset,
)


def extract_date_from_patch_id(patch_id: str) -> str:
    """Extract the date from a BigEarthNet patch ID.

    Args:
        patch_id: BigEarthNet patch ID string

    Returns:
        Date string in ISO format (YYYY-MM-DD)
    """
    # The date is in the third segment of the patch ID with format YYYYMMDD
    segments = patch_id.split("_")
    if len(segments) < 3:
        return None

    # Extract the date portion from the timestamp (first 8 characters)
    timestamp = segments[2]
    if len(timestamp) < 8 or not timestamp[:8].isdigit():
        return None

    year = timestamp[:4]
    month = timestamp[4:6]
    day = timestamp[6:8]

    return f"{year}-{month}-{day}"


def process_row(args: tuple) -> dict[str, Any]:
    """Process a single row from the metadata DataFrame.

    Args:
        args: Tuple containing (row, root, dir_file_names)

    Returns:
        Dictionary with patch_id, lon, and lat
    """
    row, root, dir_name = args
    patch_id = row["patch_id"]
    date = extract_date_from_patch_id(patch_id)
    patch_dir = "_".join(patch_id.split("_")[0:-2])

    path_pattern = os.path.join(root, dir_name, patch_dir, patch_id, "*.tif")
    paths = glob.glob(path_pattern)

    if not paths:
        return {
            "patch_id": patch_id,
            "lon": None,
            "lat": None,
            "error": "No TIF files found",
            "date": date,
        }

    try:
        with rasterio.open(paths[0]) as src:
            lon, lat = src.lnglat()
            return {"patch_id": patch_id, "lon": lon, "lat": lat, "date": date}
    except Exception as e:
        return {
            "patch_id": patch_id,
            "lon": None,
            "lat": None,
            "error": str(e),
            "date": date,
        }


def generate_metadata_df(root_dir: str, num_workers: int = 8) -> pd.DataFrame:
    """Generate metadata DataFrame for BigEarthNet dataset with parallel processing.

    Args:
        root_dir: root directory of the BigEarthNet dataset
        num_workers: Number of parallel workers to use

    Returns:
        DataFrame with metadata including geolocation for each patch
    """
    full_metadata_df = pd.read_parquet(os.path.join(root_dir, "metadata.parquet"))
    print(
        f"Generating metadata for {len(full_metadata_df)} patches using {num_workers} workers..."
    )

    args_list = [
        (row, root_dir, "BigEarthNet-S2") for _, row in full_metadata_df.iterrows()
    ]

    metadata = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for result in tqdm(
            executor.map(process_row, args_list),
            total=len(args_list),
            desc="Processing patches",
        ):
            metadata.append(result)

    metadata_df = pd.DataFrame(metadata)

    metadata_df = pd.merge(full_metadata_df, metadata_df, how="left", on="patch_id")

    return metadata_df


def create_tortilla(root_dir, metadata_df, save_dir, tortilla_name):
    """Create a tortilla version of the dataset."""
    tortilla_dir = os.path.join(save_dir, "tortilla")
    os.makedirs(tortilla_dir, exist_ok=True)

    for idx, row in tqdm(
        metadata_df.iterrows(), total=len(metadata_df), desc="Creating tortilla"
    ):
        modalities = ["s1", "s2"]

        modality_samples = []

        for modality in modalities:
            path = os.path.join(root_dir, row[modality + "_path"])

            with rasterio.open(path) as src:
                profile = src.profile

            sample = tacotoolbox.tortilla.datamodel.Sample(
                id=modality,
                path=path,
                file_format="GTiff",
                data_split=row["split"],
                stac_data={
                    "crs": "EPSG:" + str(profile["crs"].to_epsg()),
                    "geotransform": profile["transform"].to_gdal(),
                    "raster_shape": (profile["height"], profile["width"]),
                    "time_start": row["date"],
                    "time_end": row["date"],
                },
                add_test_split=row["is_additional_test"],
                lon=row["lon"],
                lat=row["lat"],
                country=row["country"],
                contains_seasonal_snow=row["contains_seasonal_snow"],
                contains_cloud_or_shadow=row["contains_cloud_or_shadow"],
                labels=row["labels"],
                patch_id=row["patch_id"],
            )

            modality_samples.append(sample)

        taco_samples = tacotoolbox.tortilla.datamodel.Samples(samples=modality_samples)
        samples_path = os.path.join(tortilla_dir, f"sample_{idx}.tortilla")
        tacotoolbox.tortilla.create(taco_samples, samples_path, quiet=True)

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
                "time_end": sample_data["stac:time_end"],
            },
            data_split=sample_data["tortilla:data_split"],
            add_test_split=sample_data["add_test_split"],
            lon=sample_data["lon"],
            lat=sample_data["lat"],
            country=sample_data["country"],
            contains_seasonal_snow=sample_data["contains_seasonal_snow"],
            contains_cloud_or_shadow=sample_data["contains_cloud_or_shadow"],
            labels=sample_data["labels"],
            patch_id=sample_data["patch_id"],
        )
        samples.append(sample_tortilla)

    final_samples = tacotoolbox.tortilla.datamodel.Samples(samples=samples)
    tacotoolbox.tortilla.create(
        final_samples, os.path.join(save_dir, tortilla_name), quiet=True
    )


def process_sample(args):
    """Process a single BigEarthNet sample to create optimized GeoTIFF files.

    Args:
        args: Dictionary containing sample information and paths

    Returns:
        Dictionary with updated paths and processing status
    """
    try:
        idx, row, root_dir, save_dir = (
            args["idx"],
            args["row"],
            args["root_dir"],
            args["save_dir"],
        )
        result = {"idx": idx, "s1_path": None, "s2_path": None, "status": "success"}

        s1_bands = ["VH", "VV"]
        s1_patch_id = row["s1_name"]
        s1_patch_dir = "_".join(s1_patch_id.split("_")[0:-3])
        s1_dir = os.path.join(root_dir, "BigEarthNet-S1", s1_patch_dir, s1_patch_id)

        s1_band_paths = [
            os.path.join(s1_dir, f"{s1_patch_id}_{band}.tif") for band in s1_bands
        ]

        s1_output_path = os.path.join(save_dir, "S1", f"{s1_patch_id}.tif")
        result["s1_path"] = os.path.join("S1", f"{s1_patch_id}.tif")

        with rasterio.open(s1_band_paths[0]) as src:
            height = src.height
            width = src.width
            crs = src.crs
            transform = src.transform

            s1_profile = {
                "driver": "GTiff",
                "height": height,
                "width": width,
                "count": 2,  # VH, VV
                "dtype": src.dtypes[0],
                "tiled": True,
                "blockxsize": 64,
                "blockysize": 64,
                "interleave": "pixel",
                "compress": "zstd",
                "zstd_level": 13,
                "predictor": 2,
                "crs": crs,
                "transform": transform,
            }

        s1_data = []
        for path in s1_band_paths:
            with rasterio.open(path) as src:
                s1_data.append(src.read(1))

        s1_data_array = np.stack(s1_data)

        with rasterio.open(s1_output_path, "w", **s1_profile) as dst:
            dst.write(s1_data_array)

        s2_bands = [
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B09",
            "B11",
            "B12",
        ]
        s2_patch_id = row["patch_id"]
        s2_patch_dir = "_".join(s2_patch_id.split("_")[0:-2])
        s2_dir = os.path.join(root_dir, "BigEarthNet-S2", s2_patch_dir, s2_patch_id)

        s2_band_paths = [
            os.path.join(s2_dir, f"{s2_patch_id}_{band}.tif") for band in s2_bands
        ]

        s2_output_path = os.path.join(save_dir, "S2", f"{s2_patch_id}.tif")
        result["s2_path"] = os.path.join("S2", f"{s2_patch_id}.tif")

        with rasterio.open(s2_band_paths[0]) as src:
            crs = src.crs
            transform = src.transform

            s2_profile = {
                "driver": "GTiff",
                "height": 120,
                "width": 120,
                "count": 12,  # 12 S2 bands
                "dtype": src.dtypes[0],
                "tiled": True,
                "blockxsize": 128,
                "blockysize": 128,
                "interleave": "pixel",
                "compress": "zstd",
                "zstd_level": 13,
                "predictor": 2,
                "crs": crs,
                "transform": transform,
            }

        s2_data = []
        for path in s2_band_paths:
            with rasterio.open(path) as src:
                array = src.read(
                    indexes=1,
                    out_shape=(120, 120),
                    out_dtype="int32",
                    resampling=Resampling.bilinear,
                )
                s2_data.append(array)

        s2_data_array = np.stack(s2_data)

        with rasterio.open(s2_output_path, "w", **s2_profile) as dst:
            dst.write(s2_data_array)

        return result

    except Exception as e:
        return {
            "idx": idx,
            "s1_path": None,
            "s2_path": None,
            "status": "error",
            "error_message": str(e),
        }


def create_optimized_geotiffs(
    metadata_df: pd.DataFrame, root_dir: str, save_dir: str, num_workers: int = 8
) -> pd.DataFrame:
    """Create optimized GeoTIFF files for BigEarthNet dataset using parallel processing.

    Args:
        metadata_df: DataFrame with metadata including geolocation for each patch
        root_dir: Root directory for the dataset
        save_dir: Directory to save the optimized files
        num_workers: Number of parallel workers to use

    Returns:
        DataFrame with updated paths to optimized files
    """
    os.makedirs(os.path.join(save_dir, "S1"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "S2"), exist_ok=True)

    result_df = metadata_df.copy()
    result_df["s1_path"] = None
    result_df["s2_path"] = None

    task_args = []
    for idx, row in result_df.iterrows():
        task_args.append(
            {"idx": idx, "row": row, "root_dir": root_dir, "save_dir": save_dir}
        )

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(
                executor.map(process_sample, task_args),
                total=len(task_args),
                desc="Creating optimized GeoTIFFs",
            )
        )

    for result in results:
        idx = result["idx"]
        if result["status"] == "success":
            result_df.at[idx, "s1_path"] = result["s1_path"]
            result_df.at[idx, "s2_path"] = result["s2_path"]
        else:
            print(
                f"Error processing sample {idx}: {result.get('error_message', 'Unknown error')}"
            )

    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = len(results) - success_count
    print(
        f"Processed {len(results)} samples: {success_count} successful, {error_count} failed"
    )

    return result_df


def create_geobench_version(
    metadata_df: pd.DataFrame,
    n_train_samples: int,
    n_val_samples: int,
    n_test_samples: int,
    n_additional_test_samples: int = 0,
) -> None:
    """Create a GeoBench version of the dataset.

    Args:
        metadata_df: DataFrame with metadata including geolocation for each patch
        n_train_samples: Number of final training samples, -1 means all
        n_val_samples: Number of final validation samples, -1 means all
        n_test_samples: Number of final test samples, -1 means all
        n_additional_test_samples: Number of additional test samples to create from training data
    """
    random_state = 24
    subset_df = create_subset_from_df(
        metadata_df,
        n_train_samples=n_train_samples,
        n_val_samples=n_val_samples,
        n_test_samples=n_test_samples,
        n_additional_test_samples=n_additional_test_samples,
        random_state=random_state,
    )

    return subset_df


def load_random_s1_image(metadata_df, root_dir):
    """Load a random S1 image from the metadata DataFrame.

    Args:
        metadata_df: DataFrame containing metadata with s1_name column
        root_dir: Root directory containing the BigEarthNet dataset
        random_seed: Random seed for reproducibility

    Returns:
        tuple: (vh_array, vv_array, row_info) - VH and VV polarization arrays and metadata row
    """
    valid_rows = metadata_df[metadata_df["s1_name"].notna()]

    random_idx = random.randint(0, len(valid_rows) - 1)
    row = valid_rows.iloc[random_idx]

    s1_patch_id = row["s1_name"]
    s1_patch_dir = "_".join(s1_patch_id.split("_")[0:-3])
    s1_dir = os.path.join(root_dir, "BigEarthNet-S1", s1_patch_dir, s1_patch_id)

    vh_path = os.path.join(s1_dir, f"{s1_patch_id}_VH.tif")
    vv_path = os.path.join(s1_dir, f"{s1_patch_id}_VV.tif")

    with rasterio.open(vh_path) as src:
        vh_array = src.read(1)

    with rasterio.open(vv_path) as src:
        vv_array = src.read(1)

    return np.stack([vh_array, vv_array])


def main():
    """Generate BigEarthNet Benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for BigEarthNet dataset"
    )
    parser.add_argument(
        "--save_dir",
        default="geobenchV2/benv2",
        help="Output directory for the benchmark",
    )
    args = parser.parse_args()

    new_metadata_path = os.path.join(args.save_dir, "geobench_benv2.parquet")

    os.makedirs(args.save_dir, exist_ok=True)

    if not os.path.exists(new_metadata_path):
        metadata_df = generate_metadata_df(args.root)
        metadata_df.to_parquet(new_metadata_path)
    else:
        metadata_df = pd.read_parquet(new_metadata_path)

    result_df_path = os.path.join(args.save_dir, "geobench_benv2_optimized.parquet")
    if os.path.exists(result_df_path):
        result_df = pd.read_parquet(result_df_path)
    else:
        result_df = create_geobench_version(
            metadata_df=metadata_df,
            n_train_samples=20000,
            n_val_samples=4000,
            n_test_samples=4000,
            n_additional_test_samples=0,
        )
        result_df = create_optimized_geotiffs(
            metadata_df=result_df,
            root_dir=args.root,
            save_dir=args.save_dir,
            num_workers=8,
        )
        result_df.to_parquet(result_df_path)

    tortilla_name = "geobench_benv2.tortilla"
    create_tortilla(
        args.save_dir, result_df, args.save_dir, tortilla_name=tortilla_name
    )

    create_unittest_subset(
        data_dir=args.save_dir,
        tortilla_pattern=tortilla_name,
        test_dir_name="benv2",
        n_train_samples=4,
        n_val_samples=2,
        n_test_samples=2,
        n_additional_test_samples=0,
    )


if __name__ == "__main__":
    main()
