# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of BRIGHT dataset."""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import rasterio
import tacoreader
import tacotoolbox
from tqdm import tqdm

from geobench_v2.generate_benchmark.geospatial_split_utils import create_bright_patches
from geobench_v2.generate_benchmark.utils import (
    create_unittest_subset,
    plot_sample_locations,
)


def create_bright_dataset_splits(
    metadata_df: pd.DataFrame, random_seed: int = 42
) -> pd.DataFrame:
    """Create train/val/test splits for BRIGHT dataset.

    Args:
        metadata_df: DataFrame with BRIGHT dataset metadata
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with updated split assignments
    """
    df = metadata_df.copy()

    df.loc[df["split"] == "validation", "split"] = "test"

    train_events = df[df["split"] == "train"]["event_id"].unique()

    np.random.seed(random_seed)

    np.random.shuffle(train_events)

    n_val_events = int(len(train_events) * 0.15)

    val_events = train_events[:n_val_events]
    new_train_events = train_events[n_val_events:]

    df.loc[df["event_id"].isin(val_events), "split"] = "validation"

    # Count samples in each split
    split_counts = df["split"].value_counts()

    train_event_count = len(new_train_events)
    val_event_count = len(val_events)
    test_event_count = len(df[df["split"] == "test"]["event_id"].unique())

    print("Split statistics:")
    print(f"Train: {split_counts.get('train', 0)} samples ({train_event_count} events)")
    print(
        f"Validation: {split_counts.get('validation', 0)} samples ({val_event_count} events)"
    )
    print(f"Test: {split_counts.get('test', 0)} samples ({test_event_count} events)")

    total_samples = len(df)
    print("\nSplit percentages:")
    print(f"Train: {100 * split_counts.get('train', 0) / total_samples:.1f}%")
    print(f"Validation: {100 * split_counts.get('validation', 0) / total_samples:.1f}%")
    print(f"Test: {100 * split_counts.get('test', 0) / total_samples:.1f}%")

    event_splits = df.groupby("event_id")["split"].nunique()
    if (event_splits > 1).any():
        mixed_events = event_splits[event_splits > 1].index.tolist()
        print(f"Warning: {len(mixed_events)} events have samples in multiple splits!")
    else:
        print("All events are properly assigned to a single split.")

    return df


def generate_metadata_df(root_dir: str) -> pd.DataFrame:
    """Generate metadata DataFrame for SpaceNet8 dataset.

    Args:
        root_dir: Root directory of the SpaceNet8 dataset

    Returns:
        DataFrame with metadata for each sample
    """
    target_paths = glob.glob(
        os.path.join(root_dir, "**", "target", "*.tif"), recursive=True
    )

    df = pd.DataFrame(target_paths, columns=["target_path"])
    df["pre_event_path"] = (
        df["target_path"]
        .str.replace("target", "pre-event")
        .str.replace("_building_damage.tif", "_pre_disaster.tif")
    )
    df["post_event_path"] = (
        df["target_path"]
        .str.replace("target", "post-event")
        .str.replace("_building_damage.tif", "_post_disaster.tif")
    )

    metadata: list[dict[str, str]] = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        pre_event_path = row["pre_event_path"]
        with rasterio.open(pre_event_path) as src:
            lng, lat = src.lnglat()
            height_px, width_px = src.height, src.width

        metadata.append(
            {
                "pre_event_path": pre_event_path,
                "longitude": lng,
                "latitude": lat,
                "height_px": height_px,
                "width_px": width_px,
            }
        )

    metadata_df = pd.DataFrame(metadata)

    full_df = pd.merge(df, metadata_df, how="left", on="pre_event_path")

    # make the paths relative
    full_df["pre_event_path"] = full_df["pre_event_path"].str.replace(root_dir, "")
    full_df["post_event_path"] = full_df["post_event_path"].str.replace(root_dir, "")
    full_df["target_path"] = full_df["target_path"].str.replace(root_dir, "")

    full_df["event_id"] = full_df["target_path"].apply(
        lambda x: os.path.basename(x).replace("_building_damage.tif", "")
    )

    # read split text files
    train_split_path = os.path.join(root_dir, "train_setlevel.txt")

    with open(train_split_path) as f:
        train_ids = f.read().splitlines()

    # assign split based on event_ids
    full_df["split"] = full_df["event_id"].apply(
        lambda x: "train" if x in train_ids else "validation"
    )

    return full_df


def create_tortilla(root_dir, df, save_dir):
    """Create a tortilla version of the dataset."""
    tortilla_dir = os.path.join(save_dir, "tortilla")
    os.makedirs(tortilla_dir, exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating tortilla"):
        modalities = ["pre_event", "post_event", "target"]
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
                    "time_start": 0,
                },
                lon=row["lon"],
                lat=row["lat"],
                source_file=row[modality + "_path"],
                patch_id=row["patch_id"],
                event_id=row["event_id"],
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
            lon=sample_data["lon"],
            lat=sample_data["lat"],
            source_file=sample_data["source_file"],
            event_id=sample_data["event_id"],
            patch_id=sample_data["patch_id"],
        )
        samples.append(sample_tortilla)

    # create final taco file
    final_samples = tacotoolbox.tortilla.datamodel.Samples(samples=samples)
    tacotoolbox.tortilla.create(
        final_samples, os.path.join(save_dir, "SpaceNet8.tortilla"), quiet=True
    )


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

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    metadata_df = generate_metadata_df(args.root)
    metadata_df.to_parquet(metadata_path)

    df_with_splits = create_bright_dataset_splits(metadata_df)

    plot_sample_locations(
        df_with_splits,
        os.path.join(args.save_dir, "sample_locations.png"),
        buffer_degrees=1.0,
    )

    patches_path = os.path.join(args.save_dir, "patch_metadata.parquet")
    if os.path.exists(patches_path):
        patches_df = pd.read_parquet(patches_path)
    else:
        patches_df = create_bright_patches(
            df_with_splits,
            root_dir=args.root,
            output_dir=args.save_dir,
            visualize=False,
        )
        patches_df.to_parquet(patches_path)

    create_unittest_subset(
        data_dir=args.save_dir,
        tortilla_pattern="BRIGHT.tortilla",
        test_dir_name="bright",
        n_train_samples=2,
        n_val_samples=1,
        n_test_samples=1,
    )
    # create_tortilla(args.save_dir, patches_df, args.save_dir)


if __name__ == "__main__":
    main()
