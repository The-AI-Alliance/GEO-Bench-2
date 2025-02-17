# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of BenV2 dataset."""

from geobench_v2.datasets import GeoBenchBigEarthNetV2


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


def create_unit_test_subset() -> None:
    """Create a subset of Fields of the World dataset for GeoBench unit tests."""

    # create random images etc that respect the structure of the dataset in minimal format
    pass


def main():
    """Generate Fields of the World Benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for Fields of the World dataset"
    )
    args = parser.parse_args()

    # orig_dataset = FieldsOfTheWorld(root=args.root, download=False)
    orig_dataset = FieldsOfTheWorld(
        root="/mnt/rg_climate_benchmark/data/datasets_classification/benv2",
        countries=FieldsOfTheWorld.valid_countries,
        download=False,
    )

    metadata_df = generate_metadata_df(orig_dataset)

    metadata_df.to_parquet(f"{ds.root}/metadata.parquet")


if __name__ == "__main__":
    main()
