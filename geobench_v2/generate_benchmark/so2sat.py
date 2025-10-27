# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of So2Sat dataset."""

import argparse
import ast
import glob
import json
import os
import pickle
import warnings

import h5py
import numpy as np
import pandas as pd
import rasterio
import tacoreader
import tacotoolbox
from tqdm import tqdm

from geobench_v2.generate_benchmark.utils import create_unittest_subset

warnings.filterwarnings("ignore")


def generate_metadata_df(root_dir: str) -> pd.DataFrame:
    """Generate metadata DataFrame for mSo2Sat dataset.

    Args:
        root_dir: root directory of the BigEarthNet dataset
        num_workers: Number of parallel workers to use

    Returns:
        DataFrame with metadata including geolocation for each patch
    """
    file_list = glob.glob(f"{root_dir}*hdf5")
    splits_file = root_dir + "default_partition.json"

    with open(splits_file) as f:
        splits = json.load(f)

    metadata_df = pd.DataFrame({"path": file_list})
    metadata_df["file"] = [x.split("/")[-1] for x in metadata_df["path"].values]
    metadata_df["id"] = [x.split(".")[0] for x in metadata_df["file"].values]
    metadata_df["split"] = "train"
    metadata_df["split"].values[metadata_df["id"].isin(splits["valid"])] = "validation"
    metadata_df["split"].values[metadata_df["id"].isin(splits["test"])] = "test"

    return metadata_df


def create_tortilla(root_dir, metadata_df, save_dir, tortilla_name):
    """Create a tortilla version of the dataset."""
    tortilla_dir = os.path.join(save_dir, "tortilla")
    os.makedirs(tortilla_dir, exist_ok=True)

    for idx, row in tqdm(
        metadata_df.iterrows(), total=len(metadata_df), desc="Creating tortilla"
    ):
        sample_id = row["id"]

        file_path = row["path"]
        with h5py.File(file_path, "r") as h5file:
            keys = sorted(h5file.keys())
            keys = np.array([key for key in keys if key != "label"])
            bands = [np.array(h5file[key]) for key in keys]

            image = np.stack(bands, axis=-1).transpose(2, 0, 1)
            attr_dict = pickle.loads(ast.literal_eval(h5file.attrs["pickle"]))
            class_index = attr_dict["label"]

        modalities = {
            "s1": ["03 - VV.Real", "01 - VH.Real"],
            "s2": [
                "02 - Blue",
                "03 - Green",
                "04 - Red",
                "05 - Vegetation Red Edge",
                "06 - Vegetation Red Edge",
                "07 - Vegetation Red Edge",
                "08 - NIR",
                "08A - Vegetation Red Edge",
                "11 - SWIR",
                "12 - SWIR",
            ],
        }

        modality_samples = []

        for modality in ["s1", "s2"]:
            indexes = [i for i, val in enumerate(keys) if val in modalities[modality]]
            data = image[indexes]

            profile = {
                "driver": "GTiff",
                "height": data.shape[1],
                "width": data.shape[2],
                "count": data.shape[0],
                "dtype": data.dtype.name,
                "nodata": None,  # optional
            }

            tmp_file_path = save_dir + row["id"] + "_" + modality + ".tiff"
            with rasterio.open(tmp_file_path, "w", **profile) as dst:
                dst.write(data)

            sample = tacotoolbox.tortilla.datamodel.Sample(
                id=modality,
                path=tmp_file_path,
                file_format="GTiff",
                data_split=row["split"],
                stac_data=None,
                labels=class_index,
                patch_id=sample_id,
            )

            modality_samples.append(sample)

        taco_samples = tacotoolbox.tortilla.datamodel.Samples(samples=modality_samples)
        samples_path = os.path.join(tortilla_dir, f"sample_{sample_id}.tortilla")
        tacotoolbox.tortilla.create(taco_samples, samples_path, quiet=True)

    all_tortilla_files = sorted(glob.glob(os.path.join(tortilla_dir, "*.tortilla")))

    classes = [
        "Compact high-rise",
        "Compact middle-rise",
        "Compact low-rise",
        "Open high-rise",
        "Open middle-rise",
        "Open low-rise",
        "Lightweight low-rise",
        "Large low-rise",
        "Sparsely built",
        "Heavy industry",
        "Dense Trees",
        "Scattered trees",
        "Bush, scrub",
        "Low plants",
        "Bare rock or paved",
        "Bare soil or sand",
        "Water",
    ]

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
            labels=classes[sample_data["labels"]],
            patch_id=sample_data["patch_id"],
        )
        samples.append(sample_tortilla)

    final_samples = tacotoolbox.tortilla.datamodel.Samples(samples=samples)
    tacotoolbox.tortilla.create(
        final_samples, os.path.join(save_dir, tortilla_name), quiet=True
    )


def main():
    """Generate mSo2Sat Benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for Benchmark dataset"
    )
    parser.add_argument(
        "--save_dir",
        default="geobenchV2/mso2sat",
        help="Output directory for the benchmark",
    )
    args = parser.parse_args()

    tortilla_name = "geobench_so2sat.tortilla"

    #### download data using https://github.com/ServiceNow/geo-bench
    metadata_df = generate_metadata_df(args.root)

    create_tortilla(args.root, metadata_df, args.save_dir, tortilla_name)

    create_unittest_subset(
        data_dir=args.save_dir,
        tortilla_pattern=tortilla_name,
        test_dir_name="so2sat",
        n_train_samples=4,
        n_val_samples=2,
        n_test_samples=2,
        n_additional_test_samples=0,
    )


if __name__ == "__main__":
    main()
