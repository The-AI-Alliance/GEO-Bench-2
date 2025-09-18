# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of TreeSatAI dataset."""

import argparse
import concurrent
import glob
import json
import os

import geopandas as gpd
import pandas as pd
import rasterio
import tacoreader
import tacotoolbox
from tqdm import tqdm

from geobench_v2.generate_benchmark.geospatial_split_utils import (
    checkerboard_split,
    visualize_geospatial_split,
)
from geobench_v2.generate_benchmark.utils import (
    create_subset_from_df,
    create_unittest_subset,
)


def generate_metadata_df(root: str) -> pd.DataFrame:
    """Generate metadata DataFrame for TreeSatAI dataset."""
    df = pd.DataFrame()
    path = os.path.join(root, "train_filenames.lst")
    with open(path) as f:
        train_files = f.read().strip().split("\n")

    path = os.path.join(root, "test_filenames.lst")
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

    def extract_lat_lng(aerial_path):
        with rasterio.open(os.path.join(root, aerial_path)) as src:
            lng, lat = src.lnglat()
        return lng, lat

    df["lon"], df["lat"] = zip(*df["aerial_path"].apply(extract_lat_lng))

    return df


def create_tortilla(root_dir, df, save_dir, tortilla_name):
    """Create a tortilla version of the dataset."""
    tortilla_dir = os.path.join(save_dir, "tortilla")
    os.makedirs(tortilla_dir, exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating tortilla"):
        modalities = ["aerial", "s1", "s2"]
        modality_samples = []

        for modality in modalities:
            path = os.path.join(root_dir, row[modality + "_path"])

            if modality == "sentinel-ts":
                with rasterio.open(os.path.join(root_dir, row["aerial_path"])) as src:
                    profile = src.profile

                sample = tacotoolbox.tortilla.datamodel.Sample(
                    id=modality,
                    path=path,
                    file_format="HDF5",
                    data_split=row["split"],
                    add_test_split=row["is_additional_test"],
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
                    add_test_split=row["is_additional_test"],
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
                    source_path=row["original_path"],
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
            add_test_split=sample_data["add_test_split"],
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
        final_samples, os.path.join(save_dir, tortilla_name), quiet=True, nworkers=4
    )


def process_treesatai_sample(args):
    """Process a single TreeSatAI sample by rewriting with optimized profile."""
    idx, row, root_dir, save_dir = args

    try:
        modalities = ["aerial", "s1", "s2"]
        result = {
            "IMG_ID": row["IMG_ID"],
            "species": row["species"],
            "dist": row["dist"],
            "YEAR": row["YEAR"],
            "lon": row["lon"],
            "lat": row["lat"],
            "original_path": row["path"],
            "split": row["split"],
            "sentinel-ts_path": row["sentinel-ts_path"],
            "is_additional_test": row["is_additional_test"],
        }

        for modality in modalities:
            src_path = os.path.join(root_dir, row[f"{modality}_path"])
            if not os.path.exists(src_path):
                continue

            modality_dir = os.path.join(save_dir, modality)
            os.makedirs(modality_dir, exist_ok=True)

            dst_filename = f"{row['IMG_ID']}_{modality}.tif"
            dst_path = os.path.join(modality_dir, dst_filename)

            with rasterio.open(src_path) as src:
                data = src.read()
                count = data.shape[0]
                crs = src.crs
                transform = src.transform

                optimized_profile = {
                    "driver": "GTiff",
                    "height": 304,
                    "width": 304,
                    "count": count,
                    "dtype": "uint8" if modality == "aerial" else "uint16",
                    "tiled": True,
                    "blockxsize": 304,
                    "blockysize": 304,
                    "interleave": "pixel",
                    "compress": "zstd",
                    "zstd_level": 13,
                    "predictor": 2,
                    "crs": crs,
                    "transform": transform,
                }

                with rasterio.open(dst_path, "w", **optimized_profile) as dst:
                    dst.write(data)

            result[f"{modality}_path"] = os.path.relpath(dst_path, start=save_dir)

        return result

    except Exception as e:
        print(f"Error processing sample {idx} (IMG_ID: {row.get('IMG_ID', '')}: {e}")
        import traceback

        traceback.print_exc()
        return None


def optimize_treesatai_dataset(metadata_df, root_dir, save_dir, num_workers=8):
    """Rewrite TreeSatAI dataset with optimized GeoTIFF profiles."""
    os.makedirs(save_dir, exist_ok=True)

    tasks = [(idx, row, root_dir, save_dir) for idx, row in metadata_df.iterrows()]
    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        list_of_results = list(
            tqdm(
                executor.map(process_treesatai_sample, tasks),
                total=len(tasks),
                desc="Optimizing TreeSatAI samples",
            )
        )

        for result in list_of_results:
            if result is not None:
                results.append(result)

    optimized_df = pd.DataFrame(results)

    metadata_path = os.path.join(save_dir, "optimized_treesatai_metadata.parquet")
    optimized_df.to_parquet(metadata_path)

    print(f"Processed {len(optimized_df)} samples successfully")

    return optimized_df


def create_geobench_version(
    metadata_df: pd.DataFrame,
    n_train_samples: int,
    n_val_samples: int,
    n_test_samples: int,
    n_additional_test_samples: int,
) -> None:
    """Create a GeoBench version of the dataset.

    Args:
        metadata_df: DataFrame with metadata including geolocation for each patch
        n_train_samples: Number of final training samples, -1 means all
        n_val_samples: Number of final validation samples, -1 means all
        n_test_samples: Number of final test samples, -1 means all
        n_additional_test_samples: Number of extra test samples to append.
    """
    subset_df = create_subset_from_df(
        metadata_df,
        n_train_samples=n_train_samples,
        n_val_samples=n_val_samples,
        n_test_samples=n_test_samples,
        n_additional_test_samples=n_additional_test_samples,
        random_state=24,
    )

    return subset_df


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

    metadata_path = os.path.join(args.save_dir, "geobench_treesatai_metadata.parquet")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if os.path.exists(metadata_path):
        metadata_df = pd.read_parquet(metadata_path)
    else:
        metadata_df = generate_metadata_df(args.root)
        metadata_df.to_parquet(metadata_path)

    metadata_df.drop(columns="split", inplace=True)

    final_metadata_path = os.path.join(
        args.save_dir, "geobench_treesatai_split.parquet"
    )
    if os.path.exists(final_metadata_path):
        final_metadata_df = pd.read_parquet(final_metadata_path)
    else:
        final_metadata_df = checkerboard_split(
            df=metadata_df,
            n_blocks_x=10,
            n_blocks_y=10,
            pattern="balanced",
            random_state=42,
        )
        final_metadata_df.to_parquet(final_metadata_path)

    visualize_geospatial_split(
        final_metadata_df,
        output_path=os.path.join(args.save_dir, "checkerboard_split.png"),
    )

    # optimized dataset
    optimized_path = os.path.join(args.save_dir, "geobench_treesatai.parquet")
    if os.path.exists(optimized_path):
        optimized_df = pd.read_parquet(optimized_path)
    else:
        subset_df = create_geobench_version(
            metadata_df=final_metadata_df,
            n_train_samples=4000,
            n_val_samples=1000,
            n_test_samples=2000,
            n_additional_test_samples=0,
        )
        optimized_df = optimize_treesatai_dataset(
            metadata_df=subset_df, root_dir=args.root, save_dir=args.save_dir
        )
        optimized_df.to_parquet(optimized_path)

    tortilla_name = "geobench_treesatai.tortilla"

    create_tortilla(
        root_dir=args.save_dir,
        df=optimized_df,
        save_dir=args.save_dir,
        tortilla_name=tortilla_name,
    )

    create_unittest_subset(
        data_dir=args.save_dir,
        tortilla_pattern=tortilla_name,
        test_dir_name="treesatai",
        n_train_samples=4,
        n_val_samples=2,
        n_test_samples=2,
        n_additional_test_samples=0,
    )


if __name__ == "__main__":
    main()
