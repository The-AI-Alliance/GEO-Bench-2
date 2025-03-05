# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of BenV2 dataset."""

from torchgeo.datasets import BigEarthNetV2


def create_subset(
    ds: BigEarthNetV2, metadata_df: pd.DataFrame, save_dir: str
) -> None:
    """Create a subset of BigEarthNet dataset.

    Args:
        ds: BigEarthNet dataset.
        metadata_df: Metadata DataFrame.
        save_dir: Directory to save the subset.
    """
    # based on the metadata_df create a subset of the dataset and copy it
    # with the same structure to the save_dir
    pass


def create_unit_test_subset() -> None:
    """Create a subset of BigEarthNet dataset for GeoBench unit tests."""

    # create random images etc that respect the structure of the dataset in minimal format
    pass


def main():
    """Generate BigEarthNet Benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for BigEarthNet dataset"
    )
    args = parser.parse_args()

    orig_dataset = BigEarthNetV2(
        root=args.root,
        download=False,
    )

    metadata_df = generate_metadata_df(orig_dataset)

    metadata_df.to_parquet(f"{ds.root}/metadata.parquet")


if __name__ == "__main__":
    main()

