# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of EverWatch dataset."""

from torchgeo.datasets import EverWatch
import pandas as pd
import os
import argparse



def create_subset(
    ds: EverWatch, metadata_df: pd.DataFrame, save_dir: str
) -> None:
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


def generate_metadata_df(ds: EverWatch) -> pd.DataFrame:
    """Generate metadata DataFrame for EverWatch dataset."""
    annot_df_train = pd.read_csv(
        os.path.join(ds.root, ds.dir, 'train.csv')
    )
    annot_df_train["split"] = "train"
    annot_df_test = pd.read_csv(
        os.path.join(ds.root, ds.dir, 'test.csv')
    )
    annot_df_test["split"] = "test"
    annot_df = pd.concat([annot_df_train, annot_df_test], ignore_index=True)
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
    args = parser.parse_args()

    orig_dataset = EverWatch(root=args.root, download=False)

    metadata_df = generate_metadata_df(orig_dataset)

    metadata_df.to_parquet(f"{ds.root}/geobench_metadata.parquet")


if __name__ == "__main__":
    main()
