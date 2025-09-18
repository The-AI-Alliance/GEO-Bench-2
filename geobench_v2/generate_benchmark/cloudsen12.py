# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of CloudSen12 dataset."""

import argparse
import os

import numpy as np
import pandas as pd
import tacoreader
from huggingface_hub import snapshot_download

from geobench_v2.generate_benchmark.utils import (
    create_subset_from_df,
    create_unittest_subset,
    plot_sample_locations,
)


def create_subset(root: str, save_dir: str) -> None:
    """Create a subset of CloudSen12 dataset.

    Args:
        root: Root directory for CloudSen12 dataset.
        save_dir: Directory to save the subset.
    """
    meta_dfs = []
    taco_files: dict[str, list[str]] = {
        "l1c": [
            "cloudsen12-l1c.0000.part.taco",
            "cloudsen12-l1c.0001.part.taco",
            "cloudsen12-l1c.0002.part.taco",
            "cloudsen12-l1c.0003.part.taco",
            "cloudsen12-l1c.0004.part.taco",
        ],
        "l2a": [
            "cloudsen12-l2a.0000.part.taco",
            "cloudsen12-l2a.0001.part.taco",
            "cloudsen12-l2a.0002.part.taco",
            "cloudsen12-l2a.0003.part.taco",
            "cloudsen12-l2a.0004.part.taco",
            "cloudsen12-l2a.0005.part.taco",
        ],
        "extra": [
            "cloudsen12-extra.0000.part.taco",
            "cloudsen12-extra.0001.part.taco",
            "cloudsen12-extra.0002.part.taco",
        ],
    }

    for key, value in taco_files.items():
        paths = [os.path.join(root, f) for f in value]
        if not all([os.path.exists(p) for p in paths]):
            snapshot_download(
                repo_id="tacofoundation/cloudsen12",
                local_dir=".",
                cache_dir=".",
                repo_type="dataset",
                pattern=f"cloudsen12-{key}.*.part.taco",
            )

        metadata_df = tacoreader.load(paths)
        metadata_df = metadata_df[
            metadata_df["stac:raster_shape"].apply(
                lambda x: np.array_equal(x, np.array([512, 512]))
            )
            & (metadata_df["label_type"] == "high")
        ].reset_index(drop=True)

        tacoreader.compile(
            dataframe=metadata_df,
            output=os.path.join(save_dir, f"geobench_cloudsen12-{key}.taco"),
            nworkers=4,
        )

        meta_dfs.append(metadata_df)

    full_metadata = pd.concat(meta_dfs)
    full_metadata.reset_index(drop=True, inplace=True)

    return full_metadata


def main():
    """Generate CloudSen12 Benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for CloudSen12 dataset"
    )
    parser.add_argument(
        "--save_dir",
        default="geobenchV2/cloudsen12",
        help="Directory to save the subset",
    )
    args = parser.parse_args()

    metadata_path = os.path.join(args.save_dir, "geobench_cloudsen12_metadata.parquet")
    if os.path.exists(metadata_path):
        metadata_df = pd.read_parquet(metadata_path)
    else:
        metadata_df = create_subset(args.root, save_dir=args.save_dir)
        metadata_df.to_parquet(metadata_path)

    plot_sample_locations(
        metadata_df=metadata_df,
        output_path=os.path.join(args.save_dir, "sample_locations.png"),
    )

    l2a_taco = tacoreader.load(
        os.path.join(args.save_dir, "geobench_cloudsen12-l2a.taco")
    )

    subset_taco = create_subset_from_df(
        l2a_taco,
        n_train_samples=4000,
        n_val_samples=1000,
        n_test_samples=2000,
        n_additional_test_samples=0,
        split_column="tortilla:data_split",
        random_state=42,
    )

    tacoreader.compile(
        dataframe=subset_taco,
        output=os.path.join(args.save_dir, "geobench_cloudsen12.tortilla"),
        nworkers=4,
    )

    create_unittest_subset(
        data_dir=args.save_dir,
        tortilla_pattern="geobench_cloudsen12.tortilla",
        test_dir_name="cloudsen12",
        n_train_samples=2,
        n_val_samples=1,
        n_test_samples=1,
        n_additional_test_samples=0,
    )


if __name__ == "__main__":
    main()
