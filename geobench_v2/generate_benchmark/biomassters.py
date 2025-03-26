# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of BioMassters dataset."""

from torchgeo.datasets import BioMassters
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
from sklearn.model_selection import train_test_split


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

from sklearn.model_selection import train_test_split


def consolidate_bio_meta_df(df):
    """
    Consolidate BioMassters dataset by chip_id to create one row per chip
    with columns for S1 and S2 paths and months.

    Args:
        df: DataFrame with BioMassters metadata

    Returns:
        Consolidated DataFrame with one row per chip_id
    """
    # Create an empty list to store new records
    consolidated_records = []

    unique_chip_ids = df["chip_id"].unique()

    for chip_id in tqdm(unique_chip_ids, desc="Consolidating chip data"):
        chip_data = df[df["chip_id"] == chip_id]

        s1_data = chip_data[chip_data["satellite"] == "S1"]
        s2_data = chip_data[chip_data["satellite"] == "S2"]

        split = chip_data["split"].iloc[0]

        features_dir = "test_features" if split == "test" else "train_features"
        agbm_dir = "test_agbm" if split == "test" else "train_agbm"

        record = {
            "chip_id": chip_id,
            "split": split,
            "S1_paths": [
                os.path.join(features_dir, filename)
                for filename in s1_data["filename"].tolist()
            ],
            "S1_months": s1_data["month"].tolist(),
            "S2_paths": [
                os.path.join(features_dir, filename)
                for filename in s2_data["filename"].tolist()
            ],
            "S2_months": s2_data["month"].tolist(),
            "agbm_path": os.path.join(
                agbm_dir, chip_data["corresponding_agbm"].iloc[0]
            ),
        }

        consolidated_records.append(record)

    consolidated_df = pd.DataFrame(consolidated_records)

    consolidated_df["num_S1_images"] = consolidated_df["S1_paths"].apply(len)
    consolidated_df["num_S2_images"] = consolidated_df["S2_paths"].apply(len)

    print(
        f"Consolidated {len(unique_chip_ids)} chips to {len(consolidated_df)} rows with both S1 and S2 data"
    )
    return consolidated_df


def generate_metadata_df(root: str) -> pd.DataFrame:
    """Generate metadata DataFrame for BioMassters dataset."""
    feature_df = pd.read_csv(os.path.join(root, "biomassters_features_metadata.csv"))
    consolidated_df = consolidate_bio_meta_df(feature_df)

    # there is no geospatial metadata included, also not in the tif files

    # create random train/val split from train entries
    train_df = consolidated_df[consolidated_df["split"] == "train"]
    test_df = consolidated_df[consolidated_df["split"] == "test"]

    train_indices, val_indices = train_test_split(
        train_df.index, test_size=0.2, random_state=42
    )
    consolidated_df.loc[val_indices, "split"] = "validation"

    split_counts = consolidated_df["split"].value_counts()
    print(f"Split distribution: {split_counts.to_dict()}")

    return consolidated_df


def create_tortilla(root_dir, df, save_dir):
    """Create a tortilla version of the dataset."""

    # filter by valid_ratio, which is the percent of valid number of pixels in an image
    # df = df[df["valid_ratio"] > 0.4]

    tortilla_dir = os.path.join(save_dir, "tortilla")
    os.makedirs(tortilla_dir, exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating tortilla"):
        modalities = ["S1", "S2"]
        modality_samples = []

        for modality in modalities:
            # for timesteps
            for path, month in zip(row[f"{modality}_paths"], row[f"{modality}_months"]):
                sample = tacotoolbox.tortilla.datamodel.Sample(
                    id=f"{row['chip_id']}_{modality}_{month}",
                    path=os.path.join(root_dir, path),
                    file_format="GTiff",
                    data_split=row["split"],
                    month=month,
                    source_img_file=path,
                    modality=modality,
                )

                modality_samples.append(sample)
        # add AGBM

        sample = tacotoolbox.tortilla.datamodel.Sample(
            id=f"{row['chip_id']}_AGBM",
            path=os.path.join(root_dir, row["agbm_path"]),
            file_format="GTiff",
            data_split=row["split"],
            modality="AGBM",
            source_img_file=row["agbm_path"],
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
            data_split=sample_data["tortilla:data_split"],
        )
        samples.append(sample_tortilla)

    # create final taco file
    final_samples = tacotoolbox.tortilla.datamodel.Samples(samples=samples)
    tacotoolbox.tortilla.create(
        final_samples,
        os.path.join(save_dir, "BioMassters.tortilla"),
        quiet=True,
        nworkers=4,
    )


def main():
    """Generate BioMassters Benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for BioMassters dataset"
    )
    parser.add_argument(
        "--save_dir",
        default="geobenchV2/BioMassters",
        help="Directory to save the subset",
    )
    args = parser.parse_args()

    metadata_path = os.path.join(args.save_dir, "geobench_biomassters.parquet")

    # orig_dataset = BioMassters(root=args.root)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if os.path.exists(metadata_path):
        metadata_df = pd.read_parquet(metadata_path)
    else:
        metadata_df = generate_metadata_df(args.root)
        metadata_df.to_parquet(metadata_path)
    # def check_sentinel_paths(row, sensor):
    #     paths = row[f"{sensor}_paths"]
    #     for path in paths:
    #         if not os.path.exists(os.path.join(args.root, path)):
    #             print(f"Path {path} does not exist")
    #             return False
    #     return True

    # metadata_df["S1_paths_exist"] = metadata_df.apply(lambda x: check_sentinel_paths(x, "S1"), axis=1)
    # metadata_df["S2_paths_exist"] = metadata_df.apply(lambda x: check_sentinel_paths(x, "S2"), axis=1)

    create_tortilla(args.root, metadata_df, args.save_dir)

    taco = tacoreader.load(os.path.join(args.save_dir, "BioMassters.tortilla"))
    import pdb

    pdb.set_trace()

    print(0)


if __name__ == "__main__":
    main()
