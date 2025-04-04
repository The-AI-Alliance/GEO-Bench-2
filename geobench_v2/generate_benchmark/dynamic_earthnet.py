# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate GeoBenchV2 version of DynamicEarthNet dataset."""

from torchvision.datasets.utils import download_url
import pandas as pd
import os
import argparse
import pyproj
import tacotoolbox
import tacoreader
import glob
from tqdm import tqdm
import rasterio


from geobench_v2.generate_benchmark.utils import (
    plot_sample_locations,
    visualize_dynamic_earthnet_sample,
)
from geobench_v2.generate_benchmark.geospatial_split_utils import (
    create_geospatial_temporal_split,
)


# TODO add automatic download of dataset to have a starting point for benchmark generation


def generate_metadata_df(root: str) -> pd.DataFrame:
    """Generate Metadata DataFrame for DynamicEarthNet dataset.

    Args:
        root: Directory to save the metadata file

    Returns:
        Metadata DataFrame for DynamicEarthNet
    """
    metadata_df = pd.read_csv(os.path.join(root, "split_info", "splits.csv"))
    original_sample_count = len(metadata_df)
    print(f"Original metadata contains {original_sample_count} monthly samples")

    metadata_df["area_id"] = metadata_df["planet_path"].apply(lambda x: x.split("/")[3])
    metadata_df["range_id"] = metadata_df["planet_path"].apply(
        lambda x: x.split("/")[2]
    )
    metadata_df["lat_id"] = metadata_df["planet_path"].apply(lambda x: x.split("/")[1])

    metadata_df["new_id"] = (
        metadata_df["area_id"]
        + "_"
        + metadata_df["range_id"]
        + "_"
        + metadata_df["lat_id"]
        + "_"
        + metadata_df["year_month"]
    )
    original_unique_ids = set(metadata_df["new_id"].unique())
    print(
        f"Original metadata contains {len(original_unique_ids)} unique location-month combinations"
    )

    expanded_rows = []

    for _, row in tqdm(
        metadata_df.iterrows(),
        total=len(metadata_df),
        desc="Expanding daily Planet data",
    ):
        year, month = row["year_month"].split("-")

        planet_base_path = os.path.join(root, row["planet_path"])
        daily_files = glob.glob(os.path.join(planet_base_path, f"{year}-{month}-*.tif"))

        # planet data for 1417_3281_13 is missing from mediaTUM dataset
        if not daily_files:
            print(
                f"WARNING: No daily files found for {row['planet_path']} in {row['year_month']}"
            )
            new_row = row.to_dict()
            new_row["aerial_path"] = row["planet_path"]
            new_row["planet_date"] = f"{year}-{month}-01"
            new_row["new_id"] = row["new_id"]
            new_row["area_id"] = row["area_id"]
            new_row["range_id"] = row["range_id"]
            new_row["lat_id"] = row["lat_id"]

            new_row["label_path"] = row["label_path"]
            new_row["s1_path"] = row["s1_path"]
            new_row["s2_path"] = row["s2_path"]
            new_row["year_month"] = row["year_month"]
            new_row["s1_missing"] = row["missing_s1"]
            new_row["s2_missing"] = row["missing_s2"]
            new_row["planet_missing"] = True

            expanded_rows.append(new_row)
        else:
            for daily_file in daily_files:
                filename = os.path.basename(daily_file)
                day = filename.split("-")[2].split(".")[0]

                new_row = row.to_dict()
                new_row["aerial_path"] = os.path.relpath(daily_file, root)
                new_row["planet_date"] = f"{year}-{month}-{day}"
                new_row["new_id"] = row["new_id"]
                new_row["area_id"] = row["area_id"]
                new_row["range_id"] = row["range_id"]
                new_row["lat_id"] = row["lat_id"]

                new_row["label_path"] = row["label_path"]
                new_row["s1_path"] = row["s1_path"]
                new_row["s2_path"] = row["s2_path"]
                new_row["year_month"] = row["year_month"]
                new_row["s1_missing"] = row["missing_s1"]
                new_row["s2_missing"] = row["missing_s2"]
                new_row["planet_missing"] = False

                expanded_rows.append(new_row)

    df = pd.DataFrame(expanded_rows)

    df["date"] = pd.to_datetime(df["planet_date"])
    df = df.sort_values(by=["new_id", "date"]).reset_index(drop=True)

    expanded_unique_ids = set(df["new_id"].unique())
    print(
        f"Expanded dataframe contains {len(expanded_unique_ids)} unique location-month combinations"
    )

    df = df[
        [
            "split",
            "aerial_path",
            "label_path",
            "s1_path",
            "s2_path",
            "year_month",
            "planet_date",
            "date",
            "s1_missing",
            "s2_missing",
            "planet_missing",
            "new_id",
            "area_id",
            "range_id",
            "lat_id",
        ]
    ]
    df.rename(columns={"aerial_path": "planet_path"}, inplace=True)

    df = df[~df["planet_missing"]].reset_index(drop=True)

    print("Extracting coordinates from raster files...")

    def extract_lng_lat(row):
        with rasterio.open(os.path.join(root, row["planet_path"])) as src:
            lon, lat = src.lnglat()
            return lon, lat

    coords = df.apply(extract_lng_lat, axis=1)
    df["lon"], df["lat"] = zip(*coords)

    return df


def create_tortilla(root_dir, df, save_dir):
    """Create a tortilla version of the dataset."""

    tortilla_dir = os.path.join(save_dir, "tortilla")
    os.makedirs(tortilla_dir, exist_ok=True)

    df["sample_idx"] = pd.factorize(df["new_id"])[0]

    unique_ts_samples = df["sample_idx"].unique()

    # rename val to validation
    df["split"] = df["split"].replace("val", "validation")

    for idx, row in tqdm(
        enumerate(unique_ts_samples),
        total=len(unique_ts_samples),
        desc="Creating tortilla",
    ):
        modalities = ["planet", "s1", "s2", "label"]
        modality_samples = []

        modality_df = df[df["sample_idx"] == row].reset_index(drop=True)

        for modality in modalities:
            if modality == "planet":
                # process the entire month time-series
                for planet_id, row in modality_df.iterrows():
                    path = os.path.join(root_dir, row[modality + "_path"])

                    with rasterio.open(path) as src:
                        profile = src.profile

                    sample = tacotoolbox.tortilla.datamodel.Sample(
                        id=f"{modality}_{planet_id}",
                        path=path,
                        file_format="GTiff",
                        data_split=row["split"],
                        stac_data={
                            "crs": "EPSG:" + str(profile["crs"].to_epsg()),
                            "geotransform": profile["transform"].to_gdal(),
                            "raster_shape": (profile["height"], profile["width"]),
                            "time_start": row["date"],
                        },
                        lon=row["lon"],
                        lat=row["lat"],
                        area_id=row["area_id"],
                        range_id=row["range_id"],
                        lat_id=row["lat_id"],
                        year_month=row["year_month"],
                        modality=modality,
                    )

                    modality_samples.append(sample)

            elif modality == "s1":
                # check for missing
                row = modality_df.iloc[0]
                path = os.path.join(root_dir, row[modality + "_path"])
                if not row["s1_missing"]:
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
                            "time_start": row["date"],
                        },
                        lon=row["lon"],
                        lat=row["lat"],
                        area_id=row["area_id"],
                        range_id=row["range_id"],
                        lat_id=row["lat_id"],
                        year_month=row["year_month"],
                        modality=modality,
                    )
                    modality_samples.append(sample)

            elif modality == "s2":
                # check for missing
                row = modality_df.iloc[0]
                path = os.path.join(root_dir, row[modality + "_path"])
                if not row["s2_missing"]:
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
                            "time_start": row["date"],
                        },
                        lon=row["lon"],
                        lat=row["lat"],
                        area_id=row["area_id"],
                        range_id=row["range_id"],
                        lat_id=row["lat_id"],
                        year_month=row["year_month"],
                        modality=modality,
                    )
                    modality_samples.append(sample)

            elif modality == "label":
                row = modality_df.iloc[0]
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
                        "time_start": row["date"],
                    },
                    lon=row["lon"],
                    lat=row["lat"],
                    area_id=row["area_id"],
                    range_id=row["range_id"],
                    lat_id=row["lat_id"],
                    year_month=row["year_month"],
                    modality=modality,
                )
                modality_samples.append(sample)

        taco_samples = tacotoolbox.tortilla.datamodel.Samples(samples=modality_samples)
        samples_path = os.path.join(tortilla_dir, f"sample_{idx}.tortilla")
        tacotoolbox.tortilla.create(taco_samples, samples_path, quiet=True)

    # merge tortillas into a single dataset
    all_tortilla_files = sorted(glob.glob(os.path.join(tortilla_dir, "*.tortilla")))

    samples = []

    for idx, tortilla_file in tqdm(
        enumerate(all_tortilla_files),
        total=len(all_tortilla_files),
        desc="Building final tortilla",
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
            area_id=sample_data["area_id"],
            range_id=sample_data["range_id"],
            lat_id=sample_data["lat_id"],
            year_month=sample_data["year_month"],
        )
        samples.append(sample_tortilla)

    final_samples = tacotoolbox.tortilla.datamodel.Samples(samples=samples)

    tacotoolbox.tortilla.create(
        final_samples,
        os.path.join(save_dir, "FullDynamicEarthNet.tortilla"),
        quiet=True,
    )


def main():
    """Generate DynamicEarthNet Benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for DynamicEarthNet dataset"
    )
    parser.add_argument(
        "--save_dir",
        default="geobenchV2/DynamicEarthNet",
        help="Directory to save the subset benchmark data",
    )

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    metadata_path = os.path.join(args.save_dir, "geobench_dynamic_earthnet.parquet")
    if os.path.exists(metadata_path):
        metadata_df = pd.read_parquet(metadata_path)
    else:
        metadata_df = generate_metadata_df(args.root)
        metadata_df.to_parquet(metadata_path)

    # validate_metadata_with_geo(metadata_df)
    df = create_geospatial_temporal_split(
        metadata_df, val_ratio=0.1, test_ratio=0.2, random_seed=42
    )

    # plot_sample_locations(
    #     # treat each time-series as single sample
    #     df.drop_duplicates(subset="new_id", keep="first"),
    #     output_path=os.path.join(args.save_dir, "sample_locations.png"),
    #     buffer_degrees=4,
    #     s=2.0,
    #     dataset_name="DynamicEarthNet",
    # )
    # create a plot for each split separately to see if they are geographically balanced
    for split in ["train", "val", "test"]:
        plot_sample_locations(
            df[df["split"] == split].drop_duplicates(subset="new_id", keep="first"),
            output_path=os.path.join(args.save_dir, f"sample_locations_{split}.png"),
            buffer_degrees=4,
            s=2.0,
            dataset_name="DynamicEarthNet",
        )

    # df["sample_idx"] = pd.factorize(df["new_id"])[0]
    # import numpy as np
    # for i in [645, 766, 815]:
    #     random_idx = i
    #     visualize_dynamic_earthnet_sample(
    #         args.root,
    #         df,
    #         sample_idx=random_idx,
    #         output_path=os.path.join(args.save_dir, f"sample_visualization_{random_idx}.png"),
    #         resize_factor=0.5
    #     )

    create_tortilla(args.root, df, args.save_dir)

    taco = tacoreader.load(os.path.join(args.save_dir, "FullDynamicEarthNet.tortilla"))
    sample = taco.read(1)


if __name__ == "__main__":
    main()
