# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate GeoBenchV2 version of Fields of the World dataset."""

import argparse
import os
import json
import shutil
import pandas as pd
from torchgeo.datasets import FieldsOfTheWorld
import shapely.wkb
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


TOTAL_N = 20000

def create_subset(
    ds: FieldsOfTheWorld, df: pd.DataFrame, save_dir: str, random_state: int = 42
) -> None:
    """Create a subset of Fields of the World dataset. Creates a stratified
    subset that maintains the train/val/test and country distribtion for a new TOTAL_N.

    Args:
        ds: Fields of the World dataset.
        df: Metadata DataFrame.
        save_dir: Directory to save the subset.
    """
    # based on the metadata_df create a subset of the dataset and copy it
    # with the same structure to the save_dir
    total_orig = len(df)
    
    def sample_group(group):
        """Determine how many samples to draw from this group."""
        fraction = len(group) / total_orig
        n_samples = int(round(fraction * TOTAL_N))
        n_samples = min(n_samples, len(group))
        if n_samples == 0 and len(group) > 0:
            n_samples = 1
        return group.sample(n=n_samples, random_state=random_state)
    
    # Group by 'split' and 'country', then sample from each group.
    subset = df.groupby(['split', 'country'], group_keys=False).apply(sample_group)
    
    # In case due to rounding the total number of samples is not exactly TOTAL_N,
    # one might need to adjust. Here we reset index and trim or pad with extra samples:
    subset = subset.reset_index(drop=True)
    current_n = len(subset)
    if current_n > TOTAL_N:
        # Remove extra rows randomly
        subset = subset.sample(n=TOTAL_N, random_state=random_state).reset_index(drop=True)
    elif current_n < TOTAL_N:
        # Optionally, add extra rows by oversampling from some groups (if desired)
        extra_needed = TOTAL_N - current_n
        # Here, we simply sample extra rows from the original dataset.
        extra_samples = df.sample(n=extra_needed, random_state=random_state)
        subset = pd.concat([subset, extra_samples]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return subset
    

def copy_subset_files(ds, subset: pd.DataFrame, save_dir: str) -> None:
    """
    Copy files from the original dataset (ds.root) to the new directory (save_dir)
    following the same directory structure and naming convention.

    For each sample (identified by its country and aoi_id) the following files are
    copied:
      - Sentinel-2 images in two windows:
          {ds.root}/{country}/s2_images/window_a/{aoi_id}.tif
          {ds.root}/{country}/s2_images/window_b/{aoi_id}.tif
      - Label masks in three categories:
          {ds.root}/{country}/label_masks/instance/{aoi_id}.tif
          {ds.root}/{country}/label_masks/semantic_2class/{aoi_id}.tif
          {ds.root}/{country}/label_masks/semantic_3class/{aoi_id}.tif

    Args:
        ds: The original dataset object. It is expected to have an attribute `root`
            that points to the base directory of the dataset.
        subset: A pandas DataFrame with at least the columns "country" and "aoi_id".
        save_dir: Destination directory in which the subset should be created.
    """
    for _, row in tqdm(subset.iterrows(), total=len(subset), desc="Copying sample files"):
        country = row["country"]
        aoi_id = row["aoi_id"]

        # Copy Sentinel-2 images: window_a and window_b
        for window in ["window_a", "window_b"]:
            src_path = os.path.join(ds.root, country, "s2_images", window, f"{aoi_id}.tif")
            if not os.path.exists(src_path):
                continue
            dst_path = os.path.join(save_dir, country, "s2_images", window, f"{aoi_id}.tif")
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)

        # Copy label masks: instance, semantic_2class, semantic_3class
        for subdir in ["instance", "semantic_2class", "semantic_3class"]:
            src_path = os.path.join(ds.root, country, "label_masks", subdir, f"{aoi_id}.tif")
            if not os.path.exists(src_path):
                continue
            dst_path = os.path.join(save_dir, country, "label_masks", subdir, f"{aoi_id}.tif")
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)

def generate_metadata_df(ds: FieldsOfTheWorld) -> pd.DataFrame:
    """Generate metadata DataFrame for Fields of the World Benchmark.

    Args:
        ds: Fields of the World dataset.

    Returns:
        Metadata DataFrame.
    """
    overall_df = pd.DataFrame()
    selected_countries = ds.countries

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

    # country of india has some samples that are 'none' for the split, so drop them
    overall_df = overall_df[overall_df["split"]!="none"]
    return overall_df


def create_unit_test_subset() -> None:
    """Create a subset of Fields of the World dataset for GeoBench unit tests."""

    # create random images etc that respect the structure of the dataset in minimal format
    pass


def generate_benchmark(ds: FieldsOfTheWorld, metadata_df: pd.DataFrame, save_dir: str) -> None:
    """Generate Fields of the World Benchmark.

    Args:
        ds: Fields of the World dataset.
        save_dir: Directory to save the subset benchmark data.
    """
    subset = create_subset(ds, metadata_df, save_dir)
    copy_subset_files(ds, subset, save_dir)


def main():
    """Generate Fields of the World Benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for Fields of the World dataset"
    )
    parser.add_argument(
        "--save_dir", default="data_benchmark", help="Directory to save the subset benchmark data"
    )
    args = parser.parse_args()

    metadata_path = f"{args.root}/metadata.parquet"

    orig_dataset = FieldsOfTheWorld(
            root=args.root,
            countries=FieldsOfTheWorld.valid_countries,
            download=False,
        )

    metadata_df = generate_metadata_df(orig_dataset)

    os.makedirs(args.save_dir, exist_ok=True)

    metadata_df.to_parquet(f"{args.save_dir}/metadata.parquet")

    generate_benchmark(orig_dataset, metadata_df, args.save_dir)


if __name__ == "__main__":
    main()
