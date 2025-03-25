# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of TreeSatAI dataset."""

from torchgeo.datasets import TreeSatAI
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
import json


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


def generate_metadata_df(root: str) -> pd.DataFrame:
    """Generate metadata DataFrame for TreeSatAI dataset."""

    df = pd.DataFrame()
    path = os.path.join(root, f"train_filenames.lst")
    with open(path) as f:
        train_files = f.read().strip().split("\n")

    path = os.path.join(root, f"test_filenames.lst")
    with open(path) as f:
        test_files = f.read().strip().split("\n")

    df["path"] = train_files + test_files
    df["split"] = ["train"] * len(train_files) + ["test"] * len(test_files)

    df["IMG_ID"] = df["path"].apply(lambda x: x.strip(".tif"))

    path = os.path.join(root, "geojson", "bb_60m.GeoJSON")

    gdf = gpd.read_file(path)

    df = df.merge(gdf, on="IMG_ID")

    df.drop(columns="geometry", inplace=True)

    path = os.path.join(root, "labels", "TreeSatBA_v9_60m_multi_labels.json")
    with open(path) as f:
        labels = json.load(f)

    def extract_labels(path):
        row_labels: list[list[str, float]] = labels[path]
        species = [label[0] for label in row_labels]
        dist = [label[1] for label in row_labels]
        return species, dist

    df["species"], df["dist"] = zip(*df["path"].apply(extract_labels))

    df["aerial_path"] = df["path"].apply(lambda x: os.path.join("aerial", "60m", x))
    df["s1_path"] = df["path"].apply(lambda x: os.path.join("s1", "60m", x))
    df["s2_path"] = df["path"].apply(lambda x: os.path.join("s2", "60m", x))

    # sentinel 2 ts paths are different
    # find all paths in dir
    ts_paths = glob.glob(os.path.join(root, "sentinel-ts", "*.h5"))
    ts_paths = [os.path.basename(path) for path in ts_paths]
    ts_img_ids = [path.strip(".h5")[:-5] for path in ts_paths]
    ts_path_df = pd.DataFrame({"IMG_ID": ts_img_ids, "sentinel-ts_path": ts_paths})
    ts_path_df["sentinel-ts_path"] = ts_path_df["sentinel-ts_path"].apply(
        lambda x: os.path.join("sentinel-ts", x)
    )
    df = df.merge(ts_path_df, on="IMG_ID", how="left")

    aerial_path = df["aerial_path"].iloc[0]

    def extract_lat_lng(aerial_path):
        with rasterio.open(os.path.join(root, aerial_path)) as src:
            lng, lat = src.lnglat()
        return lng, lat

    df["lon"], df["lat"] = zip(*df["aerial_path"].apply(extract_lat_lng))

    return df


def create_tortilla(root_dir, df, save_dir):
    """Create a tortilla version of the dataset."""

    tortilla_dir = os.path.join(save_dir, "tortilla")
    os.makedirs(tortilla_dir, exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating tortilla"):
        modalities = ["aerial", "s1", "s2"]  # "sentinel-ts"]
        modality_samples = []

        for modality in modalities:
            path = os.path.join(root_dir, row[modality + "_path"])

            if modality == "sentinel-ts":
                with rasterio.open(os.path.join(root_dir, row["aerial_path"])) as src:
                    profile = src.profile

                # # create samples for sen1_acs_data, sen1_des_data, sen2_data
                sample = tacotoolbox.tortilla.datamodel.Sample(
                    id=modality,
                    path=path,
                    file_format="HDF5",
                    data_split=row["split"],
                    year=row["YEAR"],
                    lon=row["lon"],
                    lat=row["lat"],
                    species_labels=row["species"],
                    dist_labels=row["dist"],
                    source_path=row["path"],
                )
            else:
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
                        "time_start": row["YEAR"],
                    },
                    year=row["YEAR"],
                    lon=row["lon"],
                    lat=row["lat"],
                    species_labels=row["species"],
                    dist_labels=row["dist"],
                    source_path=row["path"],
                    ts_path=row["sentinel-ts_path"],
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
            stac_data={
                "crs": sample_data["stac:crs"],
                "geotransform": sample_data["stac:geotransform"],
                "raster_shape": sample_data["stac:raster_shape"],
                "time_start": sample_data["stac:time_start"],
            },
            year=sample_data["year"],
            data_split=sample_data["tortilla:data_split"],
            lon=sample_data["lon"],
            lat=sample_data["lat"],
            species_labels=sample_data["species_labels"],
            dist_labels=sample_data["dist_labels"],
            source_path=sample_data["source_path"],
            ts_path=sample_data["ts_path"],
        )
        samples.append(sample_tortilla)

    # create final taco file
    final_samples = tacotoolbox.tortilla.datamodel.Samples(samples=samples)
    tacotoolbox.tortilla.create(
        final_samples,
        os.path.join(save_dir, "TreeSatAI.tortilla"),
        quiet=True,
        nworkers=4,
    )


def main():
    """Generate TreeSatAI Benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for TreeSatAI dataset"
    )
    parser.add_argument(
        "--save_dir",
        default="geobenchV2/TreeSatAI",
        help="Directory to save the subset",
    )
    args = parser.parse_args()

    metadata_path = os.path.join(args.save_dir, "geobench_treesatai.parquet")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if os.path.exists(metadata_path):
        metadata_df = pd.read_parquet(metadata_path)
    else:
        metadata_df = generate_metadata_df(args.root)
        metadata_df.to_parquet(metadata_path)

    # plot_sample_locations(
    #     metadata_df,
    #     os.path.join(args.save_dir, "sample_locations.png"),
    #     dataset_name="TreeSatAI",
    # )

    metadata_df.drop(columns="split", inplace=True)

    checker_split_df = checkerboard_split(
        df=metadata_df,
        n_blocks_x=10,
        n_blocks_y=10,
        pattern="balanced",
        random_state=42,
    )

    visualize_geospatial_split(
        checker_split_df,
        output_path=os.path.join(args.save_dir, "checkerboard_split.png"),
    )

    create_tortilla(args.root, checker_split_df, args.save_dir)

    taco = tacoreader.load(
        "/mnt/rg_climate_benchmark/data/geobenchV2/treesatai/TreeSatAI.tortilla"
    )

    sample = taco.read(0)


if __name__ == "__main__":
    main()
