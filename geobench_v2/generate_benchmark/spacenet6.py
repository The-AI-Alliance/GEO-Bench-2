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


def create_subset(ds: SpaceNet6, metadata_df: pd.DataFrame, save_dir: str) -> None:
    """Create a subset of SpaceNet6 dataset.

    Args:
        ds: SpaceNet6 dataset.
        metadata_df: Metadata DataFrame.
        save_dir: Directory to save the subset.
    """
    # based on the metadata_df create a subset of the dataset and copy it
    # with the same structure to the save_dir
    pass


def generate_metadata_df(ds: SpaceNet6) -> pd.DataFrame:
    """Generate metadata DataFrame for SpaceNet6 dataset.

    Args:
        ds: SpaceNet6 dataset.
    """
    # metadata_file = os.path.join(ds.root, ds.dataset_id, "train", "train", "AOI_11_Rotterdam", "SummaryData", "SN6_Train_AOI_11_Rotterdam_Buildings.csv")

    # metadata_df = gpd.read_file(metadata_file)
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

        metadata.append(
            {"path": filename, "longitude": lng, "latitude": lat, "date": date}
        )

    metadata_df = pd.DataFrame(metadata)

    metadata_df["split"] = "train"

    return metadata_df


def create_unit_test_subset() -> None:
    """Create a subset of SpaceNet6 dataset for GeoBench unit tests."""

    # create random images etc that respect the structure of the dataset in minimal format
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

    orig_dataset = SpaceNet6(root=args.root, download=False)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # if os.path.exists(metadata_path):
    #     metadata_df = pd.read_parquet(metadata_path)
    # else:
    metadata_df = generate_metadata_df(orig_dataset)
    metadata_df.to_parquet(metadata_path)

    plot_sample_locations(
        metadata_df,
        output_path=os.path.join(args.save_dir, "sample_locations.png"),
        buffer_degrees=0.2,
    )


if __name__ == "__main__":
    main()
