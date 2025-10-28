# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate GeoBenchV2 version of DynamicEarthNet dataset."""

import argparse
import glob
import multiprocessing
import os
import random
import shutil
from functools import partial
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import tacoreader
import tacotoolbox
from matplotlib.colors import ListedColormap
from rasterio.windows import Window
from skimage.transform import resize
from tqdm import tqdm


def create_geospatial_temporal_split(
    metadata_df, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, random_seed=42
):
    """Create geospatial and temporal split for DynamicEarthNet, ensuring global coverage in each split."""
    np.random.seed(random_seed)
    location_time_df = metadata_df.drop_duplicates(subset=["new_id"]).copy()
    print(f"Found {len(location_time_df)} unique location-time combinations")

    location_groups = location_time_df.groupby("area_id")

    unique_locations = np.array(sorted(location_time_df["area_id"].unique()))
    np.random.shuffle(unique_locations)

    n_locations = len(unique_locations)
    n_train_loc = int(n_locations * 0.6)
    n_val_loc = int(n_locations * 0.2)

    train_locations = unique_locations[:n_train_loc]
    val_locations = unique_locations[n_train_loc : n_train_loc + n_val_loc]
    test_locations = unique_locations[n_train_loc + n_val_loc :]

    print(
        f"Split locations: Train={len(train_locations)}, Val={len(val_locations)}, Test={len(test_locations)}"
    )

    location_time_df["split_prelim"] = "unknown"
    location_time_df.loc[
        location_time_df["area_id"].isin(train_locations), "split_prelim"
    ] = "train"
    location_time_df.loc[
        location_time_df["area_id"].isin(val_locations), "split_prelim"
    ] = "validation"
    location_time_df.loc[
        location_time_df["area_id"].isin(test_locations), "split_prelim"
    ] = "test"

    location_time_df["lon_bin"] = pd.cut(location_time_df["lon"], bins=8, labels=False)
    location_time_df["lat_bin"] = pd.cut(location_time_df["lat"], bins=8, labels=False)
    location_time_df["geo_bin"] = (
        location_time_df["lon_bin"].astype(str)
        + "_"
        + location_time_df["lat_bin"].astype(str)
    )

    # compute global bins
    geo_bins = location_time_df["geo_bin"].unique()

    for geo_bin in geo_bins:
        bin_series = location_time_df[location_time_df["geo_bin"] == geo_bin]
        bin_locations = bin_series["area_id"].unique()

        if len(bin_locations) >= 3:
            # If there are at least 3 locations in this bin, ensure each split has one
            bin_splits = bin_series["split_prelim"].unique()
            missing_splits = set(["train", "validation", "test"]) - set(bin_splits)

            # For each missing split, reassign one location from train (or the largest split)
            for missing_split in missing_splits:
                source_split = "train" if "train" in bin_splits else bin_splits[0]
                source_locs = bin_series[bin_series["split_prelim"] == source_split][
                    "area_id"
                ].unique()
                loc_to_move = np.random.choice(source_locs)

                location_time_df.loc[
                    location_time_df["area_id"] == loc_to_move, "split_prelim"
                ] = missing_split

                bin_splits = np.append(bin_splits, missing_split)
                if source_split == "train":
                    train_locations = train_locations[train_locations != loc_to_move]
                elif source_split == "validation":
                    val_locations = val_locations[val_locations != loc_to_move]
                else:
                    test_locations = test_locations[test_locations != loc_to_move]

                if missing_split == "train":
                    train_locations = np.append(train_locations, loc_to_move)
                elif missing_split == "validation":
                    val_locations = np.append(val_locations, loc_to_move)
                else:
                    test_locations = np.append(test_locations, loc_to_move)

    final_split_map = {}

    for area_id, group in location_groups:
        if area_id in train_locations:
            split = "train"
        elif area_id in val_locations:
            split = "validation"
        else:
            split = "test"

        for _, row in group.iterrows():
            final_split_map[row["new_id"]] = split

    metadata_df["split"] = metadata_df["new_id"].map(final_split_map)

    TRAIN_AREA_IDS = sorted(train_locations.tolist())
    VALIDATION_AREA_IDS = sorted(val_locations.tolist())
    TEST_AREA_IDS = sorted(test_locations.tolist())

    print(
        f"Final splits: Train={TRAIN_AREA_IDS},/n Val={VALIDATION_AREA_IDS},/n Test={TEST_AREA_IDS}"
    )

    timeseries_counts = location_time_df.groupby("split_prelim").size()
    print(f"Split timeseries counts: {timeseries_counts}")

    sample_counts = metadata_df.groupby("split").size()
    print(f"Split sample counts: {sample_counts}")

    metadata_df["lon_bin"] = pd.cut(
        metadata_df["lon"], bins=4, labels=["West", "Mid-West", "Mid-East", "East"]
    )
    metadata_df["lat_bin"] = pd.cut(
        metadata_df["lat"], bins=4, labels=["South", "Mid-South", "Mid-North", "North"]
    )

    geo_distribution = pd.crosstab(
        [metadata_df["lon_bin"], metadata_df["lat_bin"]],
        metadata_df["split"],
        normalize="index",
    )
    print("\nGeographical distribution by region:")
    print(geo_distribution)

    return metadata_df, TRAIN_AREA_IDS, VALIDATION_AREA_IDS, TEST_AREA_IDS


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

    # exclude rows where grouped by new_id, not all s1_missing or s2_missing or planet_missing are False
    df = df[
        (~df["planet_missing"]) & (~df["s1_missing"]) & (~df["s2_missing"])
    ].reset_index(drop=True)

    print("Extracting coordinates from raster files...")

    def extract_lng_lat(row):
        with rasterio.open(os.path.join(root, row["planet_path"])) as src:
            lon, lat = src.lnglat()
            return lon, lat

    coords = df.apply(extract_lng_lat, axis=1)
    df["lon"], df["lat"] = zip(*coords)

    # based on area id assign split
    df, train_ids, val_ids, test_ids = create_geospatial_temporal_split(
        df, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, random_seed=42
    )

    return df


def create_tortilla(root_dir, df, save_dir, tortilla_name):
    """Create a tortilla version of the dataset."""
    tortilla_dir = os.path.join(save_dir, "tortilla")
    os.makedirs(tortilla_dir, exist_ok=True)

    df["sample_idx"] = pd.factorize(df["patch_id"])[0]

    unique_ts_samples = df["sample_idx"].unique()

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
                            "time_start": row["planet_date"],
                        },
                        add_test_split=row["is_additional_test"],
                        lon=row["lon"],
                        lat=row["lat"],
                        area_id=row["area_id"],
                        original_id=row["original_id"],
                        modality=modality,
                    )

                    modality_samples.append(sample)

            elif modality == "s1":
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
                        "time_start": row["planet_date"],
                    },
                    add_test_split=row["is_additional_test"],
                    lon=row["lon"],
                    lat=row["lat"],
                    area_id=row["area_id"],
                    original_id=row["original_id"],
                    modality=modality,
                )
                modality_samples.append(sample)

            elif modality == "s2":
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
                        "time_start": row["planet_date"],
                    },
                    add_test_split=row["is_additional_test"],
                    lon=row["lon"],
                    lat=row["lat"],
                    area_id=row["area_id"],
                    original_id=row["original_id"],
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
                        "time_start": row["planet_date"],
                    },
                    add_test_split=row["is_additional_test"],
                    lon=row["lon"],
                    lat=row["lat"],
                    area_id=row["area_id"],
                    original_id=row["original_id"],
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
            add_test_split=sample_data["add_test_split"],
            data_split=sample_data["tortilla:data_split"],
            lon=sample_data["lon"],
            lat=sample_data["lat"],
            area_id=sample_data["area_id"],
            original_id=sample_data["original_id"],
        )
        samples.append(sample_tortilla)

    final_samples = tacotoolbox.tortilla.datamodel.Samples(samples=samples)

    tacotoolbox.tortilla.create(
        final_samples, os.path.join(save_dir, tortilla_name), quiet=True
    )


def process_dynamic_earthnet(metadata_df, input_root, output_root, num_workers=8):
    """Process DynamicEarthNet dataset - first unique S1/S2/labels, then Planet images."""
    os.makedirs(output_root, exist_ok=True)

    modality_dirs = {
        "planet": os.path.join(output_root, "planet"),
        "s1": os.path.join(output_root, "s1"),
        "s2": os.path.join(output_root, "s2"),
        "label": os.path.join(output_root, "labels"),
    }

    for directory in modality_dirs.values():
        os.makedirs(directory, exist_ok=True)

    patch_positions = [(0, 0), (0, 512), (512, 0), (512, 512)]

    unique_ids = metadata_df.drop_duplicates(subset=["new_id"]).reset_index(drop=True)
    print(f"Found {len(unique_ids)} unique location-month combinations")

    patch_mappings = {"s1": {}, "s2": {}, "label": {}}

    with multiprocessing.Pool(num_workers) as pool:
        for modality in ["s1", "s2", "label"]:
            print(f"Processing unique {modality} files...")

            process_func = partial(
                process_single_modality,
                input_root=input_root,
                output_dir=modality_dirs[modality],
                patch_positions=patch_positions,
                modality=modality,
            )

            results = list(
                tqdm(
                    pool.imap(process_func, unique_ids.to_dict("records")),
                    total=len(unique_ids),
                    desc=f"Processing {modality} files",
                )
            )

            for result in results:
                if result and not result.get("error"):
                    patch_mappings[modality][result["original_path"]] = result[
                        "patches"
                    ]

    print("Stage 2: Processing Planet images...")

    with multiprocessing.Pool(num_workers) as pool:
        process_func = partial(
            process_planet,
            input_root=input_root,
            output_dir=modality_dirs["planet"],
            patch_mappings=patch_mappings,
            patch_positions=patch_positions,
        )

        planet_results = list(
            tqdm(
                pool.imap(process_func, metadata_df.to_dict("records")),
                total=len(metadata_df),
                desc="Processing Planet images",
            )
        )

    flat_results = []
    for result in planet_results:
        if result and not isinstance(result, list):
            flat_results.append(result)
        elif result and isinstance(result, list):
            flat_results.extend(result)

    updated_df = pd.DataFrame(flat_results)

    updated_df["area_id"] = updated_df["patch_id"].apply(lambda x: x.split("-")[0][:-4])

    updated_df["date"] = updated_df["planet_path"].apply(
        lambda x: os.path.basename(x).split("_")[1].replace("-", "_")
    )

    return updated_df


def process_single_modality(record, input_root, output_dir, patch_positions, modality):
    """Process a single modality file (S1, S2, or Label) and create patches."""
    sample_id = f"{record['area_id']}_{record['range_id']}_{record['lat_id']}_{record['year_month']}"

    path_key = f"{modality}_path"
    input_path = os.path.join(input_root, record[path_key])

    with rasterio.open(input_path) as src:
        crs = src.crs
        transform = src.transform
        dtype = src.dtypes[0]
        count = src.count

        patch_results = []

        for patch_idx, (row_off, col_off) in enumerate(patch_positions):
            window = Window(col_off, row_off, 512, 512)
            window_transform = rasterio.windows.transform(window, transform)

            patch_data = src.read(window=window)

            if modality == "label" and patch_data.shape[0] > 1:
                # create single channel mask
                # https://github.com/aysim/dynnet/blob/1e7d90294b54f52744ae2b35db10b4d0a48d093d/data/utae_dynamicen.py#L119
                single_channel_mask = np.zeros(
                    (patch_data.shape[1], patch_data.shape[2]), dtype=np.uint8
                )

                for i in range(7):
                    single_channel_mask[patch_data[i] == 255] = i

                patch_data = single_channel_mask[np.newaxis, :, :]
                count = 1

            output_filename = f"{sample_id}_patch{patch_idx}.tif"
            output_path = os.path.join(output_dir, output_filename)

            profile = {
                "driver": "GTiff",
                "height": 512,
                "width": 512,
                "count": count if modality != "label" else 1,
                "dtype": dtype if modality != "label" else "uint8",
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
                "interleave": "pixel",
                "compress": "zstd",
                "zstd_level": 13,
                "predictor": 2,
                "crs": crs,
                "transform": window_transform,
            }

            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(patch_data)

            rel_path = os.path.relpath(output_path, os.path.dirname(output_dir))
            patch_results.append({"patch_idx": patch_idx, "path": rel_path})

    return {"original_path": record[path_key], "patches": patch_results}


def process_planet(record, input_root, output_dir, patch_mappings, patch_positions):
    """Process a Planet image and associate with pre-processed S1, S2, and label patches."""
    sample_id = f"{record['area_id']}_{record['range_id']}_{record['lat_id']}_{record['year_month']}"
    planet_date = record.get("planet_date", "")

    s1_patches = patch_mappings["s1"].get(record["s1_path"], [])
    s2_patches = patch_mappings["s2"].get(record["s2_path"], [])
    label_patches = patch_mappings["label"].get(record["label_path"], [])

    s1_indices = {p["patch_idx"] for p in s1_patches}
    s2_indices = {p["patch_idx"] for p in s2_patches}
    label_indices = {p["patch_idx"] for p in label_patches}

    common_indices = s1_indices & s2_indices & label_indices

    s1_map = {
        p["patch_idx"]: p["path"]
        for p in s1_patches
        if p["patch_idx"] in common_indices
    }
    s2_map = {
        p["patch_idx"]: p["path"]
        for p in s2_patches
        if p["patch_idx"] in common_indices
    }
    label_map = {
        p["patch_idx"]: p["path"]
        for p in label_patches
        if p["patch_idx"] in common_indices
    }

    input_path = os.path.join(input_root, record["planet_path"])
    if not os.path.exists(input_path):
        return None

    results = []

    with rasterio.open(input_path) as src:
        transform = src.transform

        for patch_idx in common_indices:
            row_off, col_off = patch_positions[patch_idx]
            window = Window(col_off, row_off, 512, 512)
            window_transform = rasterio.windows.transform(window, transform)

            patch_data = src.read(window=window)

            date_str = planet_date.replace("-", "_")
            output_filename = f"{sample_id}_{date_str}_patch{patch_idx}.tif"
            output_path = os.path.join(output_dir, output_filename)

            profile = {
                "driver": "GTiff",
                "height": 512,
                "width": 512,
                "count": patch_data.shape[0],
                "dtype": patch_data.dtype,
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
                "interleave": "pixel",
                "compress": "zstd",
                "zstd_level": 13,
                "predictor": 2,
                "crs": src.crs,
                "transform": window_transform,
            }

            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(patch_data)

            rel_path = os.path.relpath(output_path, os.path.dirname(output_dir))
            result = {
                "original_id": sample_id,
                "patch_id": f"{sample_id}_patch{patch_idx}",
                "patch_position": patch_idx,
                "split": record["split"],
                "planet_path": rel_path,
                "s1_path": s1_map[patch_idx],
                "s2_path": s2_map[patch_idx],
                "label_path": label_map[patch_idx],
                "planet_date": planet_date,
            }

            results.append(result)

    return results


def create_dynamic_earthnet_patches(
    input_root, output_root, metadata_df, num_workers=8
):
    """Main function to create GeoBench version of DynamicEarthNet."""
    os.makedirs(output_root, exist_ok=True)

    updated_df = process_dynamic_earthnet(
        metadata_df, input_root, output_root, num_workers=num_workers
    )

    print(
        f"Created {len(updated_df)} patch records from {len(metadata_df.drop_duplicates(subset=['new_id']))} unique time-series"
    )

    return updated_df


def create_test_subset(
    root_dir: str,
    df: pd.DataFrame,
    save_dir: str,
    num_train_samples: int = 2,
    num_val_samples: int = 1,
    num_test_samples: int = 1,
    n_additional_test_samples: int = 1,
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
        n_additional_test_samples: Number of additional test samples from train split
        target_size: Size of the downsampled images (target_size x target_size)
    """
    test_dir = os.path.join(save_dir, "unittest")
    os.makedirs(test_dir, exist_ok=True)

    df_unique = df.drop_duplicates(subset="patch_id", keep="first")

    train_unique = df_unique[df_unique["split"] == "train"]
    val_unique = df_unique[df_unique["split"] == "validation"]
    test_unique = df_unique[df_unique["split"] == "test"]

    # Validate we have enough training samples for both train and additional test
    total_train_needed = num_train_samples + n_additional_test_samples
    if n_additional_test_samples > 0 and total_train_needed > len(train_unique):
        raise ValueError(
            f"Not enough training samples available. Need {total_train_needed} "
            f"({num_train_samples} train + {n_additional_test_samples} additional test) "
            f"but only {len(train_unique)} available."
        )

    # Sample training data ensuring disjoint sets
    if n_additional_test_samples > 0:
        # Sample all needed training samples at once
        total_train_sample = train_unique.sample(total_train_needed, random_state=42)

        # Split into actual train and additional test
        train_samples = total_train_sample.iloc[:num_train_samples]
        additional_test_samples = total_train_sample.iloc[num_train_samples:]

        # Mark additional test samples
        additional_test_samples = additional_test_samples.copy()
        additional_test_samples["split"] = "train"
        additional_test_samples["is_additional_test"] = True
    else:
        train_samples = train_unique.sample(num_train_samples, random_state=42)
        additional_test_samples = pd.DataFrame()

    # Sample validation and test as usual
    val_samples = val_unique.sample(num_val_samples, random_state=42)
    test_samples = test_unique.sample(num_test_samples, random_state=42)

    # Combine all selected samples
    selected_samples = [train_samples, val_samples, test_samples]
    if len(additional_test_samples) > 0:
        selected_samples.append(additional_test_samples)

    selected_df = pd.concat(selected_samples, ignore_index=True)
    selected_ids = selected_df["patch_id"].tolist()

    # Get all records for the selected patch IDs (including time series)
    subset_df = df[df["patch_id"].isin(selected_ids)].copy()

    subset_df["is_additional_test"] = False

    # Apply the split changes to the full subset
    if len(additional_test_samples) > 0:
        additional_test_ids = additional_test_samples["patch_id"].tolist()
        subset_df.loc[
            subset_df["patch_id"].isin(additional_test_ids), "is_additional_test"
        ] = True

    print(
        f"Creating test subset with {len(subset_df)} images from {len(selected_ids)} unique time-series"
    )
    print("Split distribution:")
    split_counts = subset_df["split"].value_counts()
    for split, count in split_counts.items():
        print(f"  {split}: {count} samples")

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
            source_path = os.path.join(root_dir, row[f"{modality}_path"])
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

                    filename = f"small_{row['patch_id']}"
                    if modality == "planet":
                        planet_date = row["planet_date"].replace("-", "_")
                        filename += f"_{planet_date}"

                    filename += ".tif"
                    new_path = os.path.join(modality_dirs[modality], filename)

                    with rasterio.open(new_path, "w", **profile) as dst:
                        dst.write(data_small)

                    subset_df.loc[idx, f"{modality}_path"] = os.path.relpath(
                        new_path, test_dir
                    )

            except Exception as e:
                print(f"Error processing {source_path}: {e}")

    subset_df.to_parquet(os.path.join(test_dir, "subset_metadata.parquet"))

    tortilla_name = "dynamic_earthnet.tortilla"
    create_tortilla(
        test_dir,
        subset_df,
        os.path.join(save_dir, "unittest"),
        tortilla_name=tortilla_name,
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(script_dir))
    test_data_dir = os.path.join(repo_root, "tests", "data", "dynamic_earthnet")
    os.makedirs(test_data_dir, exist_ok=True)

    tortilla_path = os.path.join(save_dir, "unittest", tortilla_name)

    tortilla_size_mb = os.path.getsize(tortilla_path) / (1024 * 1024)
    print(f"Tortilla file size: {tortilla_size_mb:.2f} MB")
    shutil.copy(tortilla_path, os.path.join(test_data_dir, tortilla_name))

    print(f"Test subset created successfully at {test_dir}")
    print(f"Tortilla file copied to {os.path.join(test_data_dir, tortilla_name)}")


def visualize_dynamic_earthnet_patches(
    patches_df, output_root, num_samples=3, save_dir=None
):
    """Visualize patches from the DynamicEarthNet dataset to check correctness."""
    if save_dir and os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    sample_ids = random.sample(
        list(patches_df["patch_id"].unique()),
        min(num_samples, len(patches_df["patch_id"].unique())),
    )

    label_colors = [
        (0.3, 0.3, 0.3),  # Impervious surfaces - dark gray
        (1.0, 1.0, 0.0),  # Agriculture - yellow
        (0.0, 0.8, 0.0),  # Forest & other vegetation - green
        (0.0, 0.8, 0.8),  # Wetlands - cyan
        (0.8, 0.7, 0.5),  # Soil - brown
        (0.0, 0.5, 1.0),  # Water - blue
        (1.0, 1.0, 1.0),  # Snow & ice - white
    ]
    label_cmap = ListedColormap(label_colors)

    # Use the provided class names
    class_names = [
        "Impervious surfaces",
        "Agriculture",
        "Forest & other vegetation",
        "Wetlands",
        "Soil",
        "Water",
        "Snow & ice",
    ]

    for sample_id in sample_ids:
        sample = patches_df[patches_df["patch_id"] == sample_id].iloc[0]

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f"Sample: {sample_id} (Split: {sample['split']})", fontsize=14)

        with rasterio.open(os.path.join(output_root, sample["planet_path"])) as src:
            planet_data = src.read()
        rgb_bands = (
            planet_data[[2, 1, 0], :, :] if planet_data.shape[0] >= 3 else planet_data
        )
        rgb = np.transpose(rgb_bands, (1, 2, 0)).astype(np.float32)
        p2, p98 = np.percentile(rgb, (2, 98))
        rgb_norm = np.clip((rgb - p2) / (p98 - p2), 0, 1)

        axes[0].imshow(rgb_norm)
        axes[0].set_title("Planet (RGB)")
        axes[0].axis("off")
        with rasterio.open(os.path.join(output_root, sample["s1_path"])) as src:
            s1_data = src.read(1)

        s1_norm = np.clip(
            (s1_data - np.percentile(s1_data, 2))
            / (np.percentile(s1_data, 98) - np.percentile(s1_data, 2)),
            0,
            1,
        )
        axes[1].imshow(s1_norm, cmap="gray")
        axes[1].set_title("Sentinel-1 (Band 1)")
        axes[1].axis("off")

        with rasterio.open(os.path.join(output_root, sample["s2_path"])) as src:
            s2_data = src.read()

        s2_rgb = np.transpose(s2_data[[3, 2, 1], :, :], (1, 2, 0)).astype(np.float32)
        p2, p98 = np.percentile(s2_rgb, (2, 98))
        s2_norm = np.clip((s2_rgb - p2) / (p98 - p2), 0, 1)
        axes[2].imshow(s2_norm)
        axes[2].set_title("Sentinel-2 (RGB)")

        axes[2].axis("off")

        with rasterio.open(os.path.join(output_root, sample["label_path"])) as src:
            label_data = src.read(1)

        unique_values = np.unique(label_data)

        print(unique_values)

        axes[3].imshow(label_data, cmap=label_cmap, vmin=0, vmax=6)
        axes[3].set_title("Land Cover Label")
        axes[3].axis("off")
        if len(unique_values) > 0:
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(
                    facecolor=label_colors[val],
                    edgecolor="black",
                    label=f"{val}: {class_names[val]}",
                )
                for val in unique_values
            ]

            axes[3].legend(handles=legend_elements, loc="lower right", fontsize=8)

        plt.tight_layout()

        if save_dir:
            plt.savefig(
                os.path.join(save_dir, f"sample_{sample_id}.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()
        else:
            plt.show()

    print(f"Visualized {len(sample_ids)} samples")


def create_geobench_subset(
    patches_df, train_series=10, val_series=5, test_series=5, additional_test_series=0
):
    """Create a subset of the dataset with specified number of unique time-series per split.

    Args:
        patches_df: DataFrame with patch information
        train_series: Number of unique time-series to select from training set
        val_series: Number of unique time-series to select from validation set
        test_series: Number of unique time-series to select from test set
        additional_test_series: Number of additional test samples from train split

    Returns:
        DataFrame containing the selected subset
    """
    train_ids = patches_df[patches_df["split"] == "train"]["patch_id"].unique()
    val_ids = patches_df[patches_df["split"] == "validation"]["patch_id"].unique()
    test_ids = patches_df[patches_df["split"] == "test"]["patch_id"].unique()

    # Validate we have enough training samples for both train and additional test
    total_train_needed = train_series + additional_test_series
    if additional_test_series > 0 and total_train_needed > len(train_ids):
        raise ValueError(
            f"Not enough training samples available. Need {total_train_needed} "
            f"({train_series} train + {additional_test_series} additional test) "
            f"but only {len(train_ids)} available."
        )

    # set a random generator to be reproducible
    rng = np.random.default_rng(42)

    if additional_test_series > 0:
        # Sample all needed training samples (train + additional test) at once
        total_train_sample = rng.choice(train_ids, total_train_needed, replace=False)

        # Split into actual train and additional test
        selected_train = total_train_sample[:train_series]
        selected_additional_test = total_train_sample[train_series:]
    else:
        selected_train = rng.choice(
            train_ids, min(train_series, len(train_ids)), replace=False
        )
        selected_additional_test = np.array([])

    selected_val = rng.choice(val_ids, min(val_series, len(val_ids)), replace=False)
    selected_test = rng.choice(test_ids, min(test_series, len(test_ids)), replace=False)

    # Create subset with original splits
    selected_ids = np.concatenate(
        [selected_train, selected_val, selected_test, selected_additional_test]
    )
    subset_df = patches_df[patches_df["patch_id"].isin(selected_ids)].copy()

    subset_df["is_additional_test"] = False

    # Mark additional test samples
    if len(selected_additional_test) > 0:
        subset_df.loc[subset_df["patch_id"].isin(selected_additional_test), "split"] = (
            "train"
        )
        subset_df.loc[
            subset_df["patch_id"].isin(selected_additional_test), "is_additional_test"
        ] = True

    print("Selected subset contains:")
    print(
        f"- {len(selected_train)} training time-series with {len(subset_df[subset_df['split'] == 'train'])} total samples"
    )
    print(
        f"- {len(selected_val)} validation time-series with {len(subset_df[subset_df['split'] == 'validation'])} total samples"
    )
    print(
        f"- {len(selected_test)} test time-series with {len(subset_df[subset_df['split'] == 'test'])} total samples"
    )

    if additional_test_series > 0:
        print(
            f"- {len(selected_additional_test)} additional test time-series with {len(subset_df[subset_df['is_additional_test']])} total samples"
        )

    return subset_df


def verify_split_disjointness(metadata_df):
    """Verify that train, validation, and test splits are disjoint in space-time.

    For proper disjointedness:
    1. A location can appear in multiple splits only if the time periods are different
    2. For any specific location, all observations in a given time period must be in the same split

    Args:
        metadata_df: DataFrame with metadata including lat, lon, date and split information

    Returns:
        True if splits are properly disjoint, False otherwise
    """
    if not pd.api.types.is_datetime64_any_dtype(metadata_df["planet_date"]):
        metadata_df = metadata_df.copy()
        metadata_df["planet_date"] = pd.to_datetime(metadata_df["planet_date"])

    metadata_df["location_id"] = metadata_df["area_id"]
    metadata_df["year_month"] = metadata_df["planet_date"].dt.strftime("%Y-%m")
    metadata_df["location_time_id"] = (
        metadata_df["location_id"] + "_" + metadata_df["year_month"]
    )

    split_locations = {}
    for split in ["train", "validation", "test"]:
        split_data = metadata_df[metadata_df["split"] == split]
        split_locations[split] = set(split_data["location_id"].unique())

    all_disjoint = True
    location_temporal_violations = []

    for split1, split2 in combinations(["train", "validation", "test"], 2):
        location_overlap = split_locations[split1].intersection(split_locations[split2])
        if location_overlap:
            print(
                f"NOTE: {split1} and {split2} share {len(location_overlap)} locations."
            )
    location_time_splits = {}

    for _, row in metadata_df.iterrows():
        loc_id = row["location_id"]
        ym = row["year_month"]
        split = row["split"]

        if loc_id not in location_time_splits:
            location_time_splits[loc_id] = {}

        if ym not in location_time_splits[loc_id]:
            location_time_splits[loc_id][ym] = set()

        location_time_splits[loc_id][ym].add(split)

    violations = 0
    for loc_id, time_splits in location_time_splits.items():
        for ym, splits in time_splits.items():
            if len(splits) > 1:
                violations += 1
                if len(location_temporal_violations) < 5:
                    location_temporal_violations.append((loc_id, ym, list(splits)))
                all_disjoint = False

    if violations > 0:
        print(
            f"WARNING: Found {violations} location-time combinations that appear in multiple splits!"
        )
        print("Examples of violations:")
        for loc, ym, splits in location_temporal_violations:
            print(f"  Location {loc}, time {ym} appears in splits: {', '.join(splits)}")
    else:
        print("All location-time combinations appear in only one split.")

    if all_disjoint:
        print(
            "SUCCESS: Space-time disjointedness verified! Each location's time series is in only one split."
        )
    else:
        print("FAILURE: Space-time disjointedness violated.")

    loc_counts = {split: len(locs) for split, locs in split_locations.items()}
    print("\nSplit distribution summary:")
    for split, count in loc_counts.items():
        times = len(metadata_df[metadata_df["split"] == split]["year_month"].unique())
        samples = len(metadata_df[metadata_df["split"] == split])
        print(
            f"{split}: {count} locations, {times} time periods, {samples} total samples"
        )

    return all_disjoint


def add_coordinates_to_subset(subset_df, metadata_df):
    """Add lat/lon coordinates from metadata_df to subset_df based on matching location identifiers.

    Args:
        subset_df: DataFrame with patch information (output from create_geobench_subset)
        metadata_df: Original metadata DataFrame with lat/lon information

    Returns:
        subset_df with added lat/lon columns
    """
    subset_df["location_id"] = subset_df["original_id"].apply(lambda x: x)

    coord_map = {}
    for _, row in metadata_df.iterrows():
        if (
            row["new_id"] not in coord_map
            and not pd.isna(row["lon"])
            and not pd.isna(row["lat"])
        ):
            coord_map[row["new_id"]] = (row["lon"], row["lat"])

    def get_coords(location_id):
        if location_id in coord_map:
            return coord_map[location_id]
        return (None, None)

    coords = subset_df["location_id"].apply(get_coords)
    subset_df["lon"], subset_df["lat"] = zip(*coords)

    coord_count = subset_df.dropna(subset=["lon", "lat"]).shape[0]
    print(
        f"Added coordinates to {coord_count} out of {len(subset_df)} records ({coord_count / len(subset_df) * 100:.1f}%)"
    )

    subset_df = subset_df.drop(columns=["location_id"])

    return subset_df


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

    metadata_path = os.path.join(args.save_dir, "geobench_metadata.parquet")
    if os.path.exists(metadata_path):
        metadata_df = pd.read_parquet(metadata_path)
    else:
        metadata_df = generate_metadata_df(args.root)
        metadata_df.to_parquet(metadata_path)

    patches_path = os.path.join(
        args.save_dir, "geobench_dynamic_earthnet_patches.parquet"
    )

    if os.path.exists(patches_path):
        patches_df = pd.read_parquet(patches_path)
        patches_df = add_coordinates_to_subset(patches_df, metadata_df)
        patches_df.to_parquet(patches_path)
    else:
        patches_df = create_dynamic_earthnet_patches(
            args.root, args.save_dir, metadata_df, num_workers=16
        )
        patches_df = add_coordinates_to_subset(patches_df, metadata_df)
        patches_df.to_parquet(patches_path)

    subset_path = os.path.join(args.save_dir, "geobench_dynamic_earthnet.parquet")
    if os.path.exists(subset_path):
        subset_df = pd.read_parquet(subset_path)
    else:
        subset_df = create_geobench_subset(
            patches_df,
            train_series=700,
            val_series=100,
            test_series=200,
            additional_test_series=0,
        )
        subset_df.to_parquet(subset_path)

    verify_split_disjointness(subset_df)

    tortilla_name = "geobench_dynamic_earthnet.tortilla"
    create_tortilla(args.save_dir, subset_df, args.save_dir, tortilla_name)

    create_test_subset(
        args.save_dir,
        subset_df,
        args.save_dir,
        num_train_samples=2,
        num_val_samples=1,
        num_test_samples=1,
        n_additional_test_samples=0,
        target_size=16,
    )


if __name__ == "__main__":
    main()
