# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of CloudSen12 dataset."""

import tacoreader
from huggingface_hub import snapshot_download


def create_subset(
    root: str, metadata_df: pd.DataFrame, save_dir: str
) -> None:
    """Create a subset of CloudSen12 dataset.

    Args:
        root: Root directory for CloudSen12 dataset.
        save_dir: Directory to save the subset.
    """
    # based on the metadata_df create a subset of the dataset and copy it
    # with the same structure to the save_dir
    # basically create mini tacos based on the metadata_df
    taco_files = [
        "cloudsen12-l1c.0000.part.taco",
        "cloudsen12-l1c.0001.part.taco",
        "cloudsen12-l1c.0002.part.taco",
        "cloudsen12-l1c.0003.part.taco",
        "cloudsen12-l1c.0004.part.taco",
    ]
    paths = [os.path.join(root, f) for f in taco_files]
    if not all([os.path.exists(p) for p in paths]):
        snapshot_download(repo_id="tacofoundation/cloudsen12", local_dir=".", cache_dir=".", repo_type="dataset", pattern="cloudsen12-l1c.*.part.taco")
    
    metadata_df = tacoreader.load(paths)
    # only use the high quality labels and the 512x512 images and the split
    metadata_df = metadata_df[
        metadata_df["stac:raster_shape"].apply(lambda x: np.array_equal(x, np.array([512, 512])))
        & (metadata_df["label_type"] == "high")
    ].reset_index(drop=True)

    # check if a subset of the dataset can be created and saved
    pass


def create_unit_test_subset() -> None:
    """Create a subset of CloudSen12 dataset for GeoBench unit tests."""

    # create random images etc that respect the structure of the dataset in minimal format
    pass


def main():
    """Generate CloudSen12 Benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for CloudSen12 dataset"
    )
    args = parser.parse_args()

    orig_dataset = CloudSen12(root=args.root, download=False)

    metadata_df = generate_metadata_df(orig_dataset)

    metadata_df.to_parquet(f"{ds.root}/metadata.parquet")


if __name__ == "__main__":
    main()