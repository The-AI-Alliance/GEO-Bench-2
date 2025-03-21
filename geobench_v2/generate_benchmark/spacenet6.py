# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of SpaceNet6 dataset."""

from torchgeo.datasets import SpaceNet6
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

from typing import List, Tuple, Dict, Any, Optional, Union
import os
import re
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from pathlib import Path
from tqdm import tqdm


def create_geobench_ds(
    ds: SpaceNet6, modalities: list[str], metadata_df: pd.DataFrame, save_dir: str
) -> None:
    """Create a subset of SpaceNet6 dataset."""
    os.makedirs(save_dir, exist_ok=True)

    modal_path_dict = {}
    for modality in modalities:
        os.makedirs(os.path.join(save_dir, modality), exist_ok=True)

        ds.image = modality
        images, masks = ds._list_files(ds.aois[0])
        modal_path_dict[modality] = images

    modal_path_dict["mask"] = masks

    patch_size = (450, 450)
    stride = (449, 449)

    patches_df = split_geospatial_tiles_into_patches(
        modal_path_dict=modal_path_dict,
        output_dir=save_dir,
        patch_size=patch_size,
        stride=stride,
        min_valid_data_ratio=0.7,
        min_positive_pixels_ratio=0.01,
    )


def generate_metadata_df(ds: SpaceNet6) -> pd.DataFrame:
    """Generate metadata DataFrame for SpaceNet6 dataset."""
    metadata: list[dict[str, str]] = []
    for path in tqdm(ds.images):
        filename = os.path.basename(path)
        parts = filename.split("_")
        date_match = re.match(r"(\d{4})(\d{2})(\d{2})(\d{6})", parts[6])
        if date_match:
            year, month, day, _ = date_match.groups()
            date = f"{year}-{month}-{day}"

        with rasterio.open(path) as src:
            lng, lat = src.lnglat()
            height_px, width_px = src.height, src.width

        metadata.append(
            {
                "path": filename,
                "longitude": lng,
                "latitude": lat,
                "date": date,
                "height_px": height_px,
                "width_px": width_px,
            }
        )

    metadata_df = pd.DataFrame(metadata)
    metadata_df["split"] = "train"

    return metadata_df


def create_tortilla(root_dir, df, save_dir):
    """Create a tortilla version of the dataset."""

    # filter by valid_ratio, which is the percent of valid number of pixels in an image
    # df = df[df["valid_ratio"] > 0.4]

    tortilla_dir = os.path.join(save_dir, "tortilla")
    os.makedirs(tortilla_dir, exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating tortilla"):
        modalities = ["PS-RGBNIR", "SAR-Intensity", "mask"]
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
                },
                lon=row["lon"],
                lat=row["lat"],
                source_img_file=row["source_img_file"],
                source_mask_file=row["source_mask_file"],
                patch_id=row["patch_id"],
            )

            modality_samples.append(sample)

        taco_samples = tacotoolbox.tortilla.datamodel.Samples(samples=modality_samples)
        samples_path = os.path.join(tortilla_dir, f"sample_{idx}.tortilla")
        tacotoolbox.tortilla.create(taco_samples, samples_path, quiet=True, nworkers=4)

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
            lon=sample_data["lon"],
            lat=sample_data["lat"],
            source_img_file=sample_data["source_img_file"],
            source_mask_file=sample_data["source_mask_file"],
            patch_id=sample_data["patch_id"],
        )
        samples.append(sample_tortilla)

    # create final taco file
    final_samples = tacotoolbox.tortilla.datamodel.Samples(samples=samples)
    tacotoolbox.tortilla.create(
        final_samples,
        os.path.join(save_dir, "SpaceNet6.tortilla"),
        quiet=True,
        nworkers=4,
    )


def main():
    """Generate SpaceNet6 Benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for SpaceNet6 dataset"
    )
    parser.add_argument(
        "--save_dir",
        default="geobenchV2/spacenet6",
        help="Directory to save the subset",
    )
    args = parser.parse_args()

    metadata_path = os.path.join(args.save_dir, "geobench_metadata.parquet")

    orig_dataset = SpaceNet6(root=args.root, download=False, image="SAR-Intensity")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if os.path.exists(metadata_path):
        metadata_df = pd.read_parquet(metadata_path)
    else:
        metadata_df = generate_metadata_df(orig_dataset)
        metadata_df.to_parquet(metadata_path)

    plot_sample_locations(
        metadata_df,
        os.path.join(args.save_dir, "sample_locations.png"),
        buffer_degrees=1.0,
    )

    path = "/mnt/rg_climate_benchmark/data/geobenchV2/SpaceNet6/patch_metadata.parquet"

    df = pd.read_parquet(path)

    # filter by valid ratio to remove some images with lots of no-data regions
    df = df[df["valid_ratio"] > 0.4].reset_index(drop=True)

    checker_split_df = checkerboard_split(
        df,
        n_blocks_x=13,
        n_blocks_y=13,
        pattern="balanced",
        random_state=42,
        ratio_tolerance=0.02,
    )

    visualize_geospatial_split(
        checker_split_df,
        title="Checkerboard Split",
        output_path=os.path.join(args.save_dir, "checker_split.png"),
        buffer_degrees=0.05,
    )

    create_tortilla(args.save_dir, checker_split_df, args.save_dir)


if __name__ == "__main__":
    main()
