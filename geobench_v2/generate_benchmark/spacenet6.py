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
from geobench_v2.generate_benchmark.utils import (
    plot_sample_locations,
    split_geospatial_tiles_into_patches,
    show_samples_per_valid_ratio,
)

from geobench_v2.generate_benchmark.geospatial_split_utils import (
    visualize_checkerboard_pattern,
    visualize_geospatial_split,
    checkerboard_split,
    geographic_buffer_split,
    geographic_distance_split,
    visualize_distance_clusters
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


def create_unit_test_subset() -> None:
    """Create a subset of SpaceNet6 dataset for GeoBench unit tests."""
    pass


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

    # df contains a vali_ratio column, make a plot that on the xaxis has the valid_ratio in steps of 0.05 (0-1 range overall) and on the yaxis the number of patches
    df = pd.read_parquet(path)
    # show_samples_per_valid_ratio(
    #     df, os.path.join(args.save_dir, "valid_ratio.png"), dataset_name="SpaceNet6"
    # )
    
    distance_df = geographic_distance_split(
        df,
        n_clusters=8,
        random_state=42
    )

    visualize_distance_clusters(
        distance_df,
        title='Distance Split',
        output_path=os.path.join(args.save_dir, 'distance_split.png'),
        buffer_degrees=0.05
    )

    checker_split_df = checkerboard_split(
        df,
        n_blocks_x=10,
        n_blocks_y=10,
        pattern="other",
        random_state=42,
    )

    visualize_geospatial_split(
        checker_split_df,
        title='Checkerboard Split',
        output_path=os.path.join(args.save_dir, 'checker_split.png'),
        buffer_degrees=0.05
    )
    


if __name__ == "__main__":
    main()
