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


from geobench_v2.generate_benchmark.utils import plot_sample_locations
from geobench_v2.generate_benchmark.geospatial_split_utils import (
    create_geospatial_temporal_split,
)

from skimage.transform import resize
import shutil
import numpy as np


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


def create_test_subset(
    root_dir: str,
    df: pd.DataFrame,
    save_dir: str,
    num_train_samples: int = 2,
    num_val_samples: int = 1,
    num_test_samples: int = 1,
    target_size: int = 32,
) -> None:
    """Create a test subset of the DynamicEarthNet dataset with downsampled images.

    Args:
        root_dir: Root directory containing original DynamicEarthNet data
        df: DataFrame with DynamicEarthNet metadata
        save_dir: Directory to save the downsampled test subset
        num_train_samples: Number of training samples to include
        num_val_samples: Number of validation samples to include
        num_test_samples: Number of test samples to include
        target_size: Size of the downsampled images (target_size x target_size)
    """
    test_dir = os.path.join(save_dir, "unittest")
    os.makedirs(test_dir, exist_ok=True)

    df_unique = df.drop_duplicates(subset="new_id", keep="first")
    train_samples = df_unique[df_unique["split"] == "train"].sample(
        num_train_samples, random_state=42
    )
    val_samples = df_unique[df_unique["split"] == "validation"].sample(
        num_val_samples, random_state=42
    )
    test_samples = df_unique[df_unique["split"] == "test"].sample(
        num_test_samples, random_state=42
    )

    selected_ids = (
        list(train_samples["new_id"])
        + list(val_samples["new_id"])
        + list(test_samples["new_id"])
    )
    subset_df = df[df["new_id"].isin(selected_ids)].copy()

    print(
        f"Creating test subset with {len(subset_df)} images from {len(selected_ids)} unique time-series"
    )

    modalities = ["planet", "s1", "s2", "label"]
    modality_dirs = {
        modality: os.path.join(test_dir, f"test_{modality}") for modality in modalities
    }

    for directory in modality_dirs.values():
        os.makedirs(directory, exist_ok=True)

    for idx, row in tqdm(
        subset_df.iterrows(), total=len(subset_df), desc="Creating downsampled images"
    ):
        for modality in modalities:
            if modality == "planet":
                path_key = "planet_path"
                is_missing = "planet_missing"
            else:
                path_key = f"{modality}_path"
                is_missing = (
                    f"{modality}_missing" if f"{modality}_missing" in row else False
                )

            # Skip if the modality is missing for this sample
            if is_missing in row and row[is_missing]:
                continue

            source_path = os.path.join(root_dir, row[path_key])
            if not os.path.exists(source_path):
                print(f"Warning: File not found - {source_path}")
                continue

            try:
                with rasterio.open(source_path) as src:
                    profile = src.profile.copy()
                    data = src.read()

                    data_small = np.zeros(
                        (data.shape[0], target_size, target_size), dtype=data.dtype
                    )

                    for band_idx in range(data.shape[0]):
                        data_small[band_idx] = resize(
                            data[band_idx],
                            (target_size, target_size),
                            preserve_range=True,
                        ).astype(data.dtype)

                    profile.update(height=target_size, width=target_size)

                    filename = (
                        f"small_{row['area_id']}_{row['range_id']}_{row['lat_id']}"
                    )
                    if modality == "planet":
                        planet_date = row["planet_date"].replace("-", "_")
                        filename += f"_{planet_date}"

                    filename += ".tif"
                    new_path = os.path.join(modality_dirs[modality], filename)

                    with rasterio.open(new_path, "w", **profile) as dst:
                        dst.write(data_small)

                    subset_df.loc[idx, path_key] = os.path.relpath(new_path, test_dir)

            except Exception as e:
                print(f"Error processing {source_path}: {e}")

    subset_df.to_parquet(os.path.join(test_dir, "subset_metadata.parquet"))

    create_tortilla(test_dir, subset_df, os.path.join(save_dir, "unittest"))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(script_dir))
    test_data_dir = os.path.join(repo_root, "tests", "data", "dynamic_earthnet")
    os.makedirs(test_data_dir, exist_ok=True)

    tortilla_path = os.path.join(save_dir, "unittest", "FullDynamicEarthNet.tortilla")

    tortilla_size_mb = os.path.getsize(tortilla_path) / (1024 * 1024)
    print(f"Tortilla file size: {tortilla_size_mb:.2f} MB")
    shutil.copy(tortilla_path, os.path.join(test_data_dir, "dynamic_earthnet.tortilla"))

    print(f"Test subset created successfully at {test_dir}")
    print(
        f"Tortilla file copied to {os.path.join(test_data_dir, 'dynamic_earthnet.tortilla')}"
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
    # for split in ["train", "val", "test"]:
    #     plot_sample_locations(
    #         df[df["split"] == split].drop_duplicates(subset="new_id", keep="first"),
    #         output_path=os.path.join(args.save_dir, f"sample_locations_{split}.png"),
    #         buffer_degrees=4,
    #         s=2.0,
    #         dataset_name="DynamicEarthNet",
    #     )

    create_test_subset(
        args.root,
        df,
        args.save_dir,
        num_train_samples=2,
        num_val_samples=1,
        num_test_samples=1,
        target_size=16,
    )

    # create_tortilla(args.root, df, args.save_dir)

    # taco = tacoreader.load(os.path.join(args.save_dir, "FullDynamicEarthNet.tortilla"))
    # sample = taco.read(1)


if __name__ == "__main__":
    main()
