# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of SpaceNet8 dataset."""

from torchgeo.datasets import SpaceNet8
import geopandas as gpd
import pandas as pd
import os
import argparse
import rasterio
from tqdm import tqdm
import re
from geobench_v2.generate_benchmark.utils import plot_sample_locations

from geobench_v2.generate_benchmark.utils import (
    plot_sample_locations,
)

from geobench_v2.generate_benchmark.geospatial_split_utils import (
    visualize_checkerboard_pattern,
    split_geospatial_tiles_into_patches,
    visualize_geospatial_split,
    checkerboard_split,
    geographic_buffer_split,
    geographic_distance_split,
    visualize_distance_clusters
)


def create_geobench_ds(
    metadata_df: pd.DataFrame, save_dir: str
) -> None:
    """Create a subset of SpaceNet6 dataset."""
    os.makedirs(save_dir, exist_ok=True)



    modal_path_dict = {}
    modal_path_dict["PRE-event"] = metadata_df["pre-path"].tolist()
    modal_path_dict["POST-event"] = metadata_df["post-path"].tolist()
    modal_path_dict["mask"] = metadata_df["label-path"].tolist()

    patch_size = (512, 512)
    stride = (511, 511)

    patches_df = split_geospatial_tiles_into_patches(
        modal_path_dict=modal_path_dict,
        output_dir=save_dir,
        patch_size=patch_size,
        stride=stride,
        buffer_top=64,
        buffer_bottom=64,
        buffer_left=64,
        buffer_right=64,
    )
    return patches_df


def generate_metadata_df(root_dir) -> pd.DataFrame:
    """Generate metadata DataFrame for SpaceNet8 dataset.

    Args:
        ds: SpaceNet8 dataset.
    """
    # metadata_file = os.path.join(ds.root, ds.dataset_id, "train", "train", "AOI_11_Rotterdam", "SummaryData", "SN6_Train_AOI_11_Rotterdam_Buildings.csv")

    paths = [
        "/mnt/rg_climate_benchmark/data/datasets_segmentation/SpaceNet8/Germany_Training_Public_label_image_mapping.csv",
        "/mnt/rg_climate_benchmark/data/datasets_segmentation/SpaceNet8/Louisiana-East_Training_Public_label_image_mapping.csv"
    ]

    df = pd.concat([pd.read_csv(path) for path in paths])

    metadata: list[dict[str, str]] = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        

        pre_event_path = os.path.join(root_dir, "SN8_floods",  "train", "PRE-event", row["pre-event image"])
        post_event_path = os.path.join(root_dir, "SN8_floods",  "train", "POST-event", row["post-event image 1"])
        label_path = os.path.join(root_dir, "SN8_floods",  "train", "annotations", row["label"])

        assert os.path.exists(pre_event_path)
        assert os.path.exists(post_event_path)
        assert os.path.exists(label_path)
        

        with rasterio.open(pre_event_path) as src:
            lng, lat = src.lnglat()
            height_px, width_px = src.height, src.width
        metadata.append(
            {"pre-path": pre_event_path, "post-path": post_event_path, "label-path": label_path, "longitude": lng, "latitude": lat, "height_px": height_px, "width_px": width_px}
        )


    metadata_df = pd.DataFrame(metadata)

    metadata_df["split"] = "train"

    regions = [
        {
            'name': 'Louisiana, USA',
            'bounds': {'min_lat': 29, 'max_lat': 33, 'min_lon': -94, 'max_lon': -89}
        },
        {
            'name': 'Germany',
            'bounds': {'min_lat': 47.5, 'max_lat': 54.5, 'min_lon': 6.5, 'max_lon': 14.5}
        }
    ]

    # match region to each sample
    metadata_df['region'] = 'unknown'  # Default value
    
    for region in regions:
        bounds = region["bounds"]
        metadata_df.loc[
            (metadata_df["latitude"] >= bounds["min_lat"])
            & (metadata_df["latitude"] <= bounds["max_lat"])
            & (metadata_df["longitude"] >= bounds["min_lon"])
            & (metadata_df["longitude"] <= bounds["max_lon"]),
            "region",
        ] = region["name"]


    return metadata_df


def create_unit_test_subset() -> None:
    """Create a subset of SpaceNet8 dataset for GeoBench unit tests."""

    # create random images etc that respect the structure of the dataset in minimal format
    pass


def main():
    """Generate SpaceNet8 Benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for SpaceNet8 dataset"
    )
    parser.add_argument(
        "--save_dir",
        default="geobenchV2/SpaceNet8",
        help="Directory to save the subset",
    )
    args = parser.parse_args()

    metadata_path = os.path.join(args.save_dir, "geobench_metadata.parquet")

    orig_dataset = SpaceNet8(root=args.root, download=False)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    metadata_df = generate_metadata_df(args.root)
    metadata_df.to_parquet(metadata_path)

    # create_geobench_ds(metadata_df, save_dir=args.save_dir)
    path = "/mnt/rg_climate_benchmark/data/geobenchV2/spacenet8/patch_metadata.parquet"

    df = pd.read_parquet(path)

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


    plot_sample_locations(
        df,
        output_path=os.path.join(args.save_dir, "sample_locations.png"),
        buffer_degrees=2,
    )


if __name__ == "__main__":
    main()