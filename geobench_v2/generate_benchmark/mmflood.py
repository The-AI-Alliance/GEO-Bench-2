# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of MMFlood dataset."""

from torchgeo.datasets import MMFlood
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


from geobench_v2.generate_benchmark.geospatial_split_utils import create_mmflood_patches

from typing import List, Tuple, Dict, Any, Optional, Union
import os
import re
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from pathlib import Path
from tqdm import tqdm
import glob
import pandas as pd


def generate_metadata_df(root) -> pd.DataFrame:
    """Generate metadata DataFrame for MMFlood dataset."""
    metadata: list[dict[str, str]] = []

    paths = glob.glob(
        os.path.join(root, "activations", "**", "mask", "*.tif"), recursive=True
    )

    df = pd.DataFrame(paths, columns=["mask_path"])

    meta_data = pd.read_json(os.path.join(root, "activations.json"), orient="index")
    meta_data.reset_index(inplace=True)
    meta_data.rename(columns={"index": "region_id"}, inplace=True)

    df["s1_path"] = df["mask_path"].str.replace("/mask/", "/s1_raw/")
    df["hydro_path"] = df["mask_path"].str.replace("/mask/", "/hydro/")
    df["dem_path"] = df["mask_path"].str.replace("/mask/", "/DEM/")

    # only keep rows where hydro_path exists
    df["hydro_path_exist"] = df["hydro_path"].apply(
        lambda x: True if os.path.exists(x) else False
    )
    df = df[df["hydro_path_exist"] == True].reset_index(drop=True)
    df.drop(columns=["hydro_path_exist"], inplace=True)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating metadata"):
        mask_path = row["mask_path"]

        with rasterio.open(mask_path) as src:
            lng, lat = src.lnglat()
            height_px, width_px = src.height, src.width

            tags = src.tags()
            event_date = tags["event_date"]

        metadata.append(
            {
                "mask_path": mask_path,
                "longitude": lng,
                "latitude": lat,
                "date": event_date,
                "height_px": height_px,
                "width_px": width_px,
            }
        )

    metadata_df = pd.DataFrame(metadata)

    full_df = pd.merge(df, metadata_df, on="mask_path", how="left")

    # make all paths relative
    for col in ["mask_path", "s1_path", "hydro_path", "dem_path"]:
        full_df[col] = full_df[col].str.replace(root, "")

    full_df["aoi"] = full_df["mask_path"].str.split(os.sep, expand=True)[1]
    full_df["region_id"] = full_df["aoi"].str.split("-", expand=True)[0]

    full_df = pd.merge(
        full_df,
        meta_data[["region_id", "country", "start", "end", "subset"]],
        on="region_id",
        how="left",
    )
    # rename the column correctly
    full_df.rename(columns={"subset": "split"}, inplace=True)
    full_df["split"] = full_df["split"].replace({"val": "validation"})

    return full_df


def create_tortilla(root_dir, df, save_dir):
    """Create a tortilla version of the dataset."""

    # filter by valid_ratio, which is the percent of valid number of pixels in an image
    # df = df[df["valid_ratio"] > 0.4]

    tortilla_dir = os.path.join(save_dir, "tortilla")
    os.makedirs(tortilla_dir, exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating tortilla"):
        modalities = ["s1", "dem", "hydro", "mask"]
        modality_samples = []

        for modality in modalities:
            path = os.path.join(root_dir, row[modality + "_path"])
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
                    "time_start": row["start"],
                    "time_end": row["end"],
                },
                lon=row["lon"],
                lat=row["lat"],
                source_s1_file=row["source_s1_file"],
                source_mask_file=row["source_mask_file"],
                region_id=row["region_id"],
                aoi=row["aoi"],
                country=row["country"],
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
            data_split=sample_data["tortilla:data_split"],
            lon=sample_data["lon"],
            lat=sample_data["lat"],
            source_s1_file=sample_data["source_s1_file"],
            source_mask_file=sample_data["source_mask_file"],
            region_id=sample_data["region_id"],
            aoi=sample_data["aoi"],
            country=sample_data["country"],
        )
        samples.append(sample_tortilla)

    # create final taco file
    final_samples = tacotoolbox.tortilla.datamodel.Samples(samples=samples)
    tacotoolbox.tortilla.create(
        final_samples,
        os.path.join(save_dir, "MMFlood.tortilla"),
        quiet=True,
        nworkers=4,
    )


def main():
    """Generate MMFlood Benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for MMFlood dataset"
    )
    parser.add_argument(
        "--save_dir", default="geobenchV2/MMFlood", help="Directory to save the subset"
    )
    args = parser.parse_args()

    metadata_path = os.path.join(args.save_dir, "geobench_metadata.parquet")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if os.path.exists(metadata_path):
        metadata_df = pd.read_parquet(metadata_path)
    else:
        metadata_df = generate_metadata_df(args.root)
        metadata_df.to_parquet(metadata_path)

    plot_sample_locations(
        metadata_df,
        os.path.join(args.save_dir, "sample_locations.png"),
        buffer_degrees=1.0,
        dataset_name="MMFlood",
    )

    path = os.path.join(args.root, "patch_metadata.parquet")

    if os.path.exists(path):
        patch_metadata_df = pd.read_parquet(path)
    else:
        patch_metadata_df = create_mmflood_patches(
            metadata_df, args.root, os.path.join(args.root, "patches"), patch_size=512
        )

        patch_metadata_df.to_parquet(path)

    create_tortilla(
        os.path.join(args.root, "patches"), patch_metadata_df, args.save_dir
    )


if __name__ == "__main__":
    main()
