# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of Forestnet dataset."""

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
from rasterio.transform import Affine
from tqdm import tqdm

from geobench_v2.generate_benchmark.utils import create_unittest_subset

warnings.filterwarnings("ignore")


def generate_metadata_df(root_dir: str) -> pd.DataFrame:
    """Generate metadata DataFrame for the Forestnet dataset.

    Args:
        root_dir: root directory of the Forestnet dataset
        num_workers: Number of parallel workers to use

    Returns:
        DataFrame with metadata including geolocation for each patch
    """
    file_list = glob.glob(f"{root_dir}*hdf5")
    file_list.sort()
    splits_file = root_dir + "default_partition.json"

    with open(splits_file) as f:
        splits = json.load(f)

    metadata_df = pd.DataFrame({"path": file_list})
    metadata_df["file"] = [x.split("/")[-1] for x in metadata_df["path"].values]
    metadata_df["id"] = [x.replace(".hdf5", "") for x in metadata_df["file"].values]
    metadata_df["split"] = None
    metadata_df["split"].values[metadata_df["id"].isin(splits["train"])] = "train"
    metadata_df["split"].values[metadata_df["id"].isin(splits["valid"])] = "validation"
    metadata_df["split"].values[metadata_df["id"].isin(splits["test"])] = "test"
    metadata_df["id"] = [
        str(i) + "_" + x for i, x in enumerate(metadata_df["id"].values)
    ]

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

            data = np.stack(bands, axis=-1).transpose(2, 1, 0)
            attr_dict = pickle.loads(ast.literal_eval(h5file.attrs["pickle"]))  # noqa: S301
            class_index = attr_dict["label"]

        id_, lat_str, lon_str, time = sample_id.split("_", 3)
        lat = float(lat_str)
        lon = float(lon_str)
        time = time.replace("_", "-")

        crs = "EPSG:4326"

        spatial_res_degrees = 15 / 111320  # approximation

        transform = Affine(
            spatial_res_degrees, 0.0, lon, 0.0, -spatial_res_degrees, lat
        )

        profile = {
            "driver": "GTiff",
            "height": data.shape[1],
            "width": data.shape[2],
            "count": data.shape[0],
            "dtype": data.dtype.name,
            "nodata": None,  # optional
            "crs": crs,
            "transform": transform,
        }

        tmp_file_path = save_dir + row["id"] + ".tiff"

        with rasterio.open(tmp_file_path, "w", **profile) as dst:
            dst.write(data)

        sample = tacotoolbox.tortilla.datamodel.Sample(
            id="image",
            path=tmp_file_path,
            file_format="GTiff",
            data_split=row["split"],
            stac_data={
                "crs": crs,
                "geotransform": profile["transform"].to_gdal(),
                "raster_shape": (profile["height"], profile["width"]),
                "time_start": time,
            },
            labels=class_index,
            patch_id=sample_id,
            lat=lat,
            lon=lon,
        )

        taco_samples = tacotoolbox.tortilla.datamodel.Samples(samples=[sample])
        samples_path = os.path.join(tortilla_dir, f"sample_{sample_id}.tortilla")
        tacotoolbox.tortilla.create(taco_samples, samples_path, quiet=True)

    all_tortilla_files = sorted(glob.glob(os.path.join(tortilla_dir, "*.tortilla")))

    samples = []

    classes = [
        "Oil palm plantation",
        "Timber plantation",
        "Other large-scale plantations",
        "Grassland/shrubland",
        "Small-scale agriculture",
        "Small-scale mixed plantation",
        "Small-scale oil palm plantation",
        "Mining",
        "Fish pond",
        "Logging road",
        "Secondary forest",
        "Other",
    ]

    for idx, tortilla_file in tqdm(
        enumerate(all_tortilla_files),
        total=len(all_tortilla_files),
        desc="Building taco",
    ):
        sample_data = tacoreader.load(tortilla_file).iloc[0]
        # import pdb
        # pdb.set_trace()
        sample_tortilla = tacotoolbox.tortilla.datamodel.Sample(
            id=os.path.basename(tortilla_file).split(".")[0],
            path=tortilla_file,
            file_format="TORTILLA",
            data_split=sample_data["tortilla:data_split"],
            labels=classes[sample_data["labels"]],
            patch_id=sample_data["patch_id"],
            stac_data={
                "crs": sample_data.get("stac:crs"),
                "geotransform": sample_data.get("stac:geotransform"),
                "raster_shape": sample_data.get("stac:raster_shape"),
                "time_start": sample_data.get("stac:time_start"),
            },
            lon=sample_data.get("lon"),
            lat=sample_data.get("lat"),
        )
        samples.append(sample_tortilla)

    final_samples = tacotoolbox.tortilla.datamodel.Samples(samples=samples)
    tacotoolbox.tortilla.create(
        final_samples, os.path.join(save_dir, tortilla_name), quiet=True
    )


def main():
    """Generate Forestnet Benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for Benchmark dataset"
    )
    parser.add_argument(
        "--save_dir",
        default="geobenchV2/forestnet",
        help="Output directory for the benchmark",
    )
    args = parser.parse_args()

    tortilla_name = "geobench_forestnet.tortilla"

    # download data using https://github.com/ServiceNow/geo-bench

    metadata_df = generate_metadata_df(args.root)

    create_tortilla(args.root, metadata_df, args.save_dir, tortilla_name)

    create_unittest_subset(
        data_dir=args.save_dir,
        tortilla_pattern=tortilla_name,
        test_dir_name="forestnet",
        n_train_samples=4,
        n_val_samples=2,
        n_test_samples=2,
        n_additional_test_samples=0,
    )


if __name__ == "__main__":
    main()
