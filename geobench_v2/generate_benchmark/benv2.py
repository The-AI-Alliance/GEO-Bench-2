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
import tacotoolbox
import tacoreader
from tqdm import tqdm

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from geobench_v2.generate_benchmark.utils import (
    plot_sample_locations,
    create_subset_from_tortilla,
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

    # Convert YYYYMMDD to YYYY-MM-DD
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
    row, root, dir_file_names = args
    patch_id = row["patch_id"]
    date = extract_date_from_patch_id(patch_id)
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


def generate_metadata_df(root_dir, num_workers: int = 8) -> pd.DataFrame:
    """Generate metadata DataFrame for BigEarthNet dataset with parallel processing.

    Args:
        ds: BigEarthNet dataset
        num_workers: Number of parallel workers to use

    Returns:
        DataFrame with metadata including geolocation for each patch
    """
    full_metadata_df = pd.read_parquet(os.path.join(root_dir, "metadata.parquet"))
    print(
        f"Generating metadata for {len(full_metadata_df)} patches using {num_workers} workers..."
    )

    # Prepare arguments for parallel processing
    args_list = [
        (row, ds.root, ds.dir_file_names) for _, row in full_metadata_df.iterrows()
    ]

    # Process in parallel with progress bar
    metadata = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Use tqdm to show progress
        for result in tqdm(
            executor.map(process_row, args_list),
            total=len(args_list),
            desc="Processing patches",
        ):
            metadata.append(result)

    # Create DataFrame from results
    metadata_df = pd.DataFrame(metadata)

    metadata_df = pd.merge(full_metadata_df, metadata_df, how="left", on="patch_id")

    return metadata_df


def create_tortilla(root_dir, metadata_df, save_dir):
    """Create a tortilla version of the dataset."""

    tortilla_dir = os.path.join(save_dir, "tortilla")
    os.makedirs(tortilla_dir, exist_ok=True)

    for idx, row in tqdm(
        metadata_df.iterrows(), total=len(metadata_df), desc="Creating tortilla"
    ):
        modality_samples = []

        # S1 band modalities
        s1_bands = ["VH", "VV"]

        patch_id = row["s1_name"]
        patch_dir = "_".join(patch_id.split("_")[0:-3])

        s1_dir = os.path.join(root_dir, "BigEarthNet-S1", patch_dir, patch_id)
        s1_paths = [os.path.join(s1_dir, f"{patch_id}_{band}.tif") for band in s1_bands]

        # S2 band modalities
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

        patch_id = row["patch_id"]
        patch_dir = "_".join(patch_id.split("_")[0:-2])
        s2_dir = os.path.join(root_dir, "BigEarthNet-S2", patch_dir, patch_id)

        s2_paths = [os.path.join(s2_dir, f"{patch_id}_{band}.tif") for band in s2_bands]

        # for path in [*s1_paths, *s2_paths]:
        #     assert os.path.exists(path), f"{path} does not exist"

        for band_name, path in zip(s1_bands + s2_bands, s1_paths + s2_paths):
            with rasterio.open(path) as src:
                profile = src.profile

            sample = tacotoolbox.tortilla.datamodel.Sample(
                id=band_name,
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
                "time_end": sample_data["stac:time_end"],
            },
            data_split=sample_data["tortilla:data_split"],
            lon=sample_data["lon"],
            lat=sample_data["lat"],
            country=sample_data["country"],
            contains_seasonal_snow=sample_data["contains_seasonal_snow"],
            contains_cloud_or_shadow=sample_data["contains_cloud_or_shadow"],
            labels=sample_data["labels"],
            patch_id=sample_data["patch_id"],
        )
        samples.append(sample_tortilla)

    # create final taco file
    final_samples = tacotoolbox.tortilla.datamodel.Samples(samples=samples)
    tacotoolbox.tortilla.create(
        final_samples, os.path.join(save_dir, "FullBenV2.tortilla"), quiet=True
    )


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

    # create_tortilla(args.root, metadata_df, args.save_dir)

    taco_glob = sorted(
        glob.glob(os.path.join(args.save_dir, "FullBenV2.*.part.tortilla"))
    )
    taco_ben = tacoreader.load(taco_glob)

    # create unit test subset
    unit_test_taco = create_subset_from_tortilla(
        taco_ben, n_train_samples=4, n_val_samples=2, n_test_samples=2
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(script_dir))
    test_data_dir = os.path.join(repo_root, "tests", "data", "benv2")
    os.makedirs(test_data_dir, exist_ok=True)
    tacoreader.compile(
        dataframe=unit_test_taco, output=os.path.join(test_data_dir, "benv2.tortilla")
    )


if __name__ == "__main__":
    # command:
    main()
