# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of Fields of the World dataset."""

import argparse
import os
import json
import pandas as pd
from torchgeo.datasets import FieldsOfTheWorld
import shapely.wkb
from tqdm import tqdm


def create_subset(
    ds: FieldsOfTheWorld, metadata_df: pd.DataFrame, save_dir: str
) -> None:
    """Create a subset of Fields of the World dataset.

    Args:
        ds: Fields of the World dataset.
        metadata_df: Metadata DataFrame.
        save_dir: Directory to save the subset.
    """
    # based on the metadata_df create a subset of the dataset and copy it
    # with the same structure to the save_dir
    pass


def generate_metadata_df(ds: FieldsOfTheWorld) -> pd.DataFrame:
    """Generate metadata DataFrame for Fields of the World Benchmark.

    Args:
        ds: Fields of the World dataset.

    Returns:
        Metadata DataFrame.
    """

    overall_df = pd.DataFrame()
    selected_countries = ds.countries
    # df = pd.read_parquet("/mnt/rg_climate_benchmark/data/datasets_segmentation/FieldsOfTheWorld/austria/chips_austria.parquet")

    for country in tqdm(selected_countries, desc="Collecting metadata"):
        country_df = pd.read_parquet(f"{ds.root}/{country}/chips_{country}.parquet")

        with open(f"{ds.root}/{country}/data_config_{country}.json", "r") as f:
            data_config = json.load(f)

        country_df["year_of_collection"] = data_config["year_of_collection"]
        country_df["geometry_obj"] = country_df["geometry"].apply(
            lambda x: shapely.wkb.loads(x)
        )
        country_df["lon"] = country_df["geometry_obj"].apply(lambda g: g.centroid.x)
        country_df["lat"] = country_df["geometry_obj"].apply(lambda g: g.centroid.y)

        country_df.drop(columns=["geometry", "geometry_obj"], inplace=True)

        country_df["country"] = country

        overall_df = pd.concat([overall_df, country_df], ignore_index=True)

    overall_df["aoi_id"] = overall_df["aoi_id"].astype(str)

    return overall_df


def main():
    """Generate Fields of the World Benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for Fields of the World dataset"
    )
    args = parser.parse_args()

    # orig_dataset = FieldsOfTheWorld(root=args.root, download=False)
    orig_dataset = FieldsOfTheWorld(
        root="/mnt/rg_climate_benchmark/data/datasets_segmentation/FieldsOfTheWorld",
        countries=FieldsOfTheWorld.valid_countries,
        download=False,
    )

    metadata_df = generate_metadata_df(orig_dataset)

    metadata_df.to_parquet(f"{ds.root}/metadata.parquet")


if __name__ == "__main__":
    main()
