# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of EverWatch dataset."""

import pandas as pd
import os
import argparse
from geobench_v2.generate_benchmark.object_detection_util import (
    process_dataset,
    verify_patching,
    compare_resize,
    resize_object_detection_dataset,
)


def create_subset(metadata_df: pd.DataFrame, save_dir: str) -> None:
    """Create a subset of EverWatch dataset.

    Args:
        ds: EverWatch dataset.
        metadata_df: Metadata DataFrame.
        save_dir: Directory to save the subset.
    """
    # based on the metadata_df create a subset of the dataset and copy it
    # with the same structure to the save_dir
    # TODO
    # image comes in large tiles of 1500x1500 pixels, pass window size of potentially 750x750 and stride of 750 and resize to 512x512?
    # only train/test split so need to create a validation split but no geospatial metadata available, separate train and test csv file
    pass


def generate_metadata_df(root) -> pd.DataFrame:
    """Generate metadata DataFrame for EverWatch dataset."""
    annot_df_train = pd.read_csv(os.path.join(root, "train.csv"))
    annot_df_train["split"] = "train"
    annot_df_test = pd.read_csv(os.path.join(root, "test.csv"))
    annot_df_test["split"] = "test"
    annot_df = pd.concat([annot_df_train, annot_df_test], ignore_index=True)

    annot_df = annot_df[
        (annot_df["xmin"] != annot_df["xmax"]) & (annot_df["ymin"] != annot_df["ymax"])
    ].reset_index(drop=True)
    return annot_df


def create_unit_test_subset() -> None:
    """Create a subset of EverWatch dataset for GeoBench unit tests."""

    # create random images etc that respect the structure of the dataset in minimal format
    pass


def main():
    """Generate EverWatch Benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for EverWatch dataset"
    )
    parser.add_argument(
        "--save_dir",
        default="geobenchV2/everwatch",
        help="Directory to save the subset",
    )
    args = parser.parse_args()

    metadata_df = generate_metadata_df(args.root)

    path = os.path.join(args.root, "geobench_everwatch.parquet")
    metadata_df.to_parquet(path)

    # metadata_df = metadata_df.iloc[:10]

    resize_object_detection_dataset(
        args.root, metadata_df, args.save_dir, target_size=512
    )


if __name__ == "__main__":
    main()
