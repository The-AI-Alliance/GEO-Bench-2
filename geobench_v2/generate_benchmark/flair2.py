# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate GeoBenchV2 version of FLAIR2 dataset."""

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import pyproj
import rasterio
import tacoreader
import tacotoolbox
from torchvision.datasets.utils import download_url
from tqdm import tqdm

from geobench_v2.generate_benchmark.utils import (
    create_subset_from_df,
    create_unittest_subset,
    plot_sample_locations,
)


def generate_metadata_df(save_dir: str, root: str) -> pd.DataFrame:
    """Generate Metadata DataFrame for FLAIR2 dataset.

    Args:
        save_dir: Directory to save the metadata file
        root: Root directory for FLAIR2 dataset

    Returns:
        Metadata DataFrame for flair2
    """
    metadata_link = "https://huggingface.co/datasets/IGNF/FLAIR/resolve/main/aux-data/flair_aerial_metadata.json"
    download_url(metadata_link, save_dir)
    metadata_df = pd.read_json(metadata_link, orient="index")

    # Create coordinate transformer from Lambert-93 (EPSG:2154) to WGS84 (EPSG:4326)
    transformer = pyproj.Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)

    # Convert patch centroids to lat/lon
    lon_lat_coords = [
        transformer.transform(row.patch_centroid_x, row.patch_centroid_y)
        for _, row in metadata_df.iterrows()
    ]
    metadata_df["lon"] = [coord[0] for coord in lon_lat_coords]
    metadata_df["lat"] = [coord[1] for coord in lon_lat_coords]

    print("Coordinate conversion example:")
    for i, (_, row) in enumerate(metadata_df.head().iterrows()):
        print(
            f"  {row.name}: Lambert-93 ({row.patch_centroid_x}, {row.patch_centroid_y}) -> "
            f"WGS84 ({row.lon:.6f}, {row.lat:.6f})"
        )
        if i >= 4:
            break

    # from https://huggingface.co/datasets/IGNF/FLAIR#data-splits
    train_ids = (
        "D006",
        "D007",
        "D008",
        "D009",
        "D013",
        "D016",
        "D017",
        "D021",
        "D023",
        "D030",
        "D032",
        "D033",
        "D034",
        "D035",
        "D038",
        "D041",
        "D044",
        "D046",
        "D049",
        "D051",
        "D052",
        "D055",
        "D060",
        "D063",
        "D070",
        "D072",
        "D074",
        "D078",
        "D080",
        "D081",
        "D086",
        "D091",
    )
    val_ids = ("D004", "D014", "D029", "D031", "D058", "D066", "D067", "D077")
    test_ids = (
        "D015",
        "D022",
        "D026",
        "D036",
        "D061",
        "D064",
        "D068",
        "D069",
        "D071",
        "D084",
    )
    # find match in the domain column which has values of id_year
    metadata_df["split"] = (
        metadata_df["domain"]
        .apply(lambda x: x.split("_")[0])
        .replace({train_id: "train" for train_id in train_ids})
        .replace({val_id: "validation" for val_id in val_ids})
        .replace({test_id: "test" for test_id in test_ids})
    )

    metadata_df = metadata_df.reset_index().rename(columns={"index": "image_id"})

    metadata_df["aerial_path"] = (
        "aerial"
        + os.sep
        + metadata_df["zone"].astype(str)
        + os.sep
        + metadata_df["image_id"]
        + ".tif"
    )
    metadata_df["mask_path"] = (
        metadata_df["aerial_path"]
        .str.replace("aerial", "labels")
        .str.replace("IMG_", "MSK_")
    )

    metadata_df["aerial_path"] = metadata_df.apply(
        lambda x: "data" + os.sep + "train-val" + os.sep + x["aerial_path"]
        if x["split"] in ["train", "validation"]
        else "data" + os.sep + "flair#2-test" + os.sep + x["aerial_path"],
        axis=1,
    )
    metadata_df["mask_path"] = metadata_df.apply(
        lambda x: "data" + os.sep + "train-val" + os.sep + x["mask_path"]
        if x["split"] in ["train", "validation"]
        else "data" + os.sep + "flair#2-test" + os.sep + x["mask_path"],
        axis=1,
    )

    print(f"\nTotal patches: {len(metadata_df)}")
    print("Split distribution:")
    split_counts = metadata_df["split"].value_counts()
    for split, count in split_counts.items():
        print(f"  {split}: {count} ({100 * count / len(metadata_df):.1f}%)")

    metadata_df = metadata_df[
        metadata_df["split"].isin(["train", "validation", "test"])
    ].reset_index(drop=True)

    # These images are listed in the json but not in the actual data from HF
    metadata_df["img_path_exists"] = metadata_df.apply(
        lambda x: os.path.exists(os.path.join(root, x["aerial_path"])), axis=1
    )
    metadata_df = metadata_df[metadata_df["img_path_exists"]].reset_index(drop=True)

    return metadata_df


def create_tortilla(root_dir, df, save_dir, tortilla_name):
    """Create a tortilla version of the dataset."""
    tortilla_dir = os.path.join(save_dir, "tortilla")
    os.makedirs(tortilla_dir, exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating tortilla"):
        modalities = ["aerial", "mask"]
        modality_samples = []

        for modality in modalities:
            path = os.path.join(root_dir, row[modality + "_path"])

            with rasterio.open(path) as src:
                profile = src.profile

            if modality == "aerial":
                crs = "EPSG:" + str(profile["crs"].to_epsg())

            sample = tacotoolbox.tortilla.datamodel.Sample(
                id=modality,
                path=path,
                file_format="GTiff",
                data_split=row["split"],
                stac_data={
                    "crs": crs,
                    "geotransform": profile["transform"].to_gdal(),
                    "raster_shape": (profile["height"], profile["width"]),
                    "time_start": row["date"],
                },
                add_test_split=row["is_additional_test"],
                lon=row["lon"],
                lat=row["lat"],
                image_id=row["image_id"],
                domain=row["domain"],
                zone=row["zone"],
                camera=row["camera"],
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
            add_test_split=sample_data["add_test_split"],
            lon=sample_data["lon"],
            lat=sample_data["lat"],
            image_id=sample_data["image_id"],
            domain=sample_data["domain"],
            zone=sample_data["zone"],
            camera=sample_data["camera"],
        )
        samples.append(sample_tortilla)

    # create final taco file
    final_samples = tacotoolbox.tortilla.datamodel.Samples(samples=samples)
    tacotoolbox.tortilla.create(
        final_samples, os.path.join(save_dir, tortilla_name), quiet=True
    )


def process_flair2_sample(args):
    """Process a single FLAIR2 sample by reading, optimizing, and writing with improved profile."""
    idx, row, root_dir, save_dir, blockxsize, blockysize = args

    try:
        save_aerial_dir = os.path.join(save_dir, "aerial")
        save_mask_dir = os.path.join(save_dir, "labels")

        os.makedirs(save_aerial_dir, exist_ok=True)
        os.makedirs(save_mask_dir, exist_ok=True)

        aerial_in_path = os.path.join(root_dir, row["aerial_path"])
        mask_in_path = os.path.join(root_dir, row["mask_path"])
        aerial_out_path = os.path.join(save_aerial_dir, f"{row['image_id']}.tif")
        mask_out_path = os.path.join(save_mask_dir, f"MSK_{row['image_id'][4:]}.tif")

        with rasterio.open(aerial_in_path) as src:
            data = src.read()
            aerial_profile = src.profile.copy()

            aerial_optimized_profile = {
                "driver": "GTiff",
                "height": aerial_profile["height"],
                "width": aerial_profile["width"],
                "count": data.shape[0],
                "dtype": data.dtype,
                "tiled": True,
                "blockxsize": blockxsize,
                "blockysize": blockysize,
                "interleave": "pixel",
                "compress": "zstd",
                "zstd_level": 22,
                "predictor": 2,
                "crs": aerial_profile["crs"],
                "transform": aerial_profile["transform"],
            }

            if "nodata" in aerial_profile and aerial_profile["nodata"] is not None:
                aerial_optimized_profile["nodata"] = aerial_profile["nodata"]

            with rasterio.open(aerial_out_path, "w", **aerial_optimized_profile) as dst:
                dst.write(data)

        with rasterio.open(mask_in_path) as src:
            mask_data = src.read()
            mask_profile = src.profile.copy()

            mask_optimized_profile = {
                "driver": "GTiff",
                "height": mask_profile["height"],
                "width": mask_profile["width"],
                "count": mask_data.shape[0],
                "dtype": mask_data.dtype,
                "tiled": True,
                "blockxsize": blockxsize,
                "blockysize": blockysize,
                "interleave": "pixel",
                "compress": "zstd",
                "zstd_level": 22,
                "predictor": 2,
                "crs": mask_profile["crs"],
                "transform": mask_profile["transform"],
            }

            with rasterio.open(mask_out_path, "w", **mask_optimized_profile) as dst:
                dst.write(mask_data)

        return (aerial_out_path, mask_out_path)

    except Exception as e:
        import traceback

        return (
            False,
            f"Error processing {row['image_id']}: {str(e)}\n{traceback.format_exc()}",
        )


def store_dataset_under_new_profile(
    df, root_dir, save_dir, block_size=(512, 512), num_workers=8
):
    """Store dataset under new profile.

    Args:
        df: DataFrame with metadata including geolocation for each patch
        root_dir: Root directory for FLAIR2 dataset
        save_dir: Directory to save the subset benchmark data
        block_size: Size of blocks for optimized GeoTIFF writing
        num_workers: Number of parallel workers
    """
    blockxsize, blockysize = block_size
    blockxsize = blockxsize - (blockxsize % 16) if blockxsize % 16 != 0 else blockxsize
    blockysize = blockysize - (blockysize % 16) if blockysize % 16 != 0 else blockysize

    print(f"Optimizing {len(df)} FLAIR2 images with {num_workers} workers")
    print(f"Block size: {blockxsize}x{blockysize}")

    os.makedirs(os.path.join(save_dir, "aerial"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "labels"), exist_ok=True)

    tasks = [
        (idx, row, root_dir, save_dir, blockxsize, blockysize)
        for idx, row in df.iterrows()
    ]

    results = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        all_results = list(
            tqdm(
                executor.map(process_flair2_sample, tasks),
                total=len(tasks),
                desc="Processing FLAIR2 samples",
            )
        )

        for result in all_results:
            if result is not None:
                aerial_path, mask_path = result
                results.append(
                    {
                        "new_aerial_path": os.path.relpath(aerial_path, start=save_dir),
                        "new_mask_path": os.path.relpath(mask_path, start=save_dir),
                    }
                )

    results_df = pd.DataFrame(results)
    results_df["image_id"] = results_df["new_aerial_path"].apply(
        lambda x: os.path.basename(x).split(".")[0]
    )

    optimized_df = df.copy()
    optimized_df = optimized_df.merge(
        results_df[["image_id", "new_aerial_path", "new_mask_path"]],
        on="image_id",
        how="left",
    )

    optimized_df = optimized_df.drop(columns=["aerial_path", "mask_path"])
    optimized_df = optimized_df.rename(
        columns={"new_aerial_path": "aerial_path", "new_mask_path": "mask_path"}
    )

    metadata_path = os.path.join(save_dir, "optimized_metadata.parquet")
    optimized_df.to_parquet(metadata_path)
    print(f"Saved optimized metadata to {metadata_path}")

    return optimized_df


def create_geobench_version(
    metadata_df: pd.DataFrame,
    n_train_samples: int,
    n_val_samples: int,
    n_test_samples: int,
    n_additional_test_samples: int = 0,
) -> pd.DataFrame:
    """Create a GeoBench version of the dataset.

    Args:
        metadata_df: DataFrame with metadata including geolocation for each patch
        n_train_samples: Number of final training samples, -1 means all
        n_val_samples: Number of final validation samples, -1 means all
        n_test_samples: Number of final test samples, -1 means all
        n_additional_test_samples: Number of additional test samples from train split
        root_dir: Root directory for FLAIR2 dataset
        save_dir: Directory to save the subset benchmark data
        block_size: Size of blocks for optimized GeoTIFF writing
        num_workers: Number of parallel workers
    """
    random_state = 24

    subset_df = create_subset_from_df(
        metadata_df,
        n_train_samples=n_train_samples,
        n_val_samples=n_val_samples,
        n_test_samples=n_test_samples,
        n_additional_test_samples=n_additional_test_samples,
        random_state=random_state,
    )

    return subset_df


def main():
    """Generate FLAIR2 Benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for FLAIR2 dataset"
    )
    parser.add_argument(
        "--save_dir",
        default="geobenchV2/flair2",
        help="Directory to save the subset benchmark data",
    )

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    metadata_path = os.path.join(args.save_dir, "geobench_metadata.parquet")
    if os.path.exists(metadata_path):
        metadata_df = pd.read_parquet(metadata_path)
    else:
        metadata_df = generate_metadata_df(save_dir=args.save_dir, root=args.root)
        metadata_df.to_parquet(metadata_path)

    plot_sample_locations(
        metadata_df,
        output_path=os.path.join(args.save_dir, "sample_locations.png"),
        buffer_degrees=1.5,
        split_column="split",
        s=2.0,
    )

    result_df_path = os.path.join(args.save_dir, "geobench_flair2.parquet")
    if os.path.exists(result_df_path):
        result_df = pd.read_parquet(result_df_path)
    else:
        result_df = create_geobench_version(
            metadata_df,
            n_train_samples=4000,
            n_val_samples=1000,
            n_test_samples=2000,
            n_additional_test_samples=0,
        )
        result_df.to_parquet(result_df_path)

    optimized_path = os.path.join(args.save_dir, "optimized_metadata.parquet")
    if os.path.exists(optimized_path):
        optimized_df = pd.read_parquet(optimized_path)
    else:
        optimized_df = store_dataset_under_new_profile(
            result_df,
            root_dir=args.root,
            save_dir=args.save_dir,
            block_size=(512, 512),
            num_workers=8,
        )
        optimized_df.to_parquet(optimized_path)

    tortilla_name = "geobench_flair2.tortilla"

    create_tortilla(args.save_dir, optimized_df, args.save_dir, tortilla_name)

    create_unittest_subset(
        data_dir=args.save_dir,
        tortilla_pattern=tortilla_name,
        test_dir_name="flair2",
        n_train_samples=2,
        n_val_samples=1,
        n_test_samples=1,
        n_additional_test_samples=0,
    )


if __name__ == "__main__":
    main()
