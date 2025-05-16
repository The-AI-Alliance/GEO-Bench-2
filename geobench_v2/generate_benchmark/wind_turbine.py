# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of Windturbine dataset."""

import argparse
import glob
import os
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import tacoreader
import tacotoolbox
from PIL import Image
from rasterio.errors import NotGeoreferencedWarning
from shapely.geometry import box
from tqdm import tqdm

from geobench_v2.generate_benchmark.object_detection_util import (
    convert_pngs_to_geotiffs,
)
from geobench_v2.generate_benchmark.utils import create_unittest_subset


def resize_and_save_dataset(df, root_dir, save_dir, target_size=512):
    """Resize images and save them to the specified directory.

    Args:
        df (pd.DataFrame): DataFrame containing metadata for the dataset.
        root_dir (str): Root directory of the original dataset.
        save_dir (str): Directory to save the resized dataset.
        target_size (int): Target size for resizing images.
    """
    for split in ["train", "validation", "test"]:
        os.makedirs(os.path.join(save_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "labels", split), exist_ok=True)

    resized_metadata = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Resizing images"):
        img_path = os.path.join(root_dir, row["image_path"])
        label_path = os.path.join(root_dir, row["label_path"])

        img = Image.open(img_path)
        orig_width, orig_height = img.size

        resized_img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)

        split = row["split"]
        filename = row["filename"]

        new_img_path = os.path.join(save_dir, "images", split, f"{filename}.png")
        new_label_path = os.path.join(save_dir, "labels", split, f"{filename}.txt")

        resized_img.save(new_img_path)

        width_ratio = target_size / orig_width
        height_ratio = target_size / orig_height

        with open(label_path) as f:
            annotations = f.readlines()

        new_annotations = []
        for ann in annotations:
            parts = ann.strip().split()
            if len(parts) >= 5:
                class_id = parts[0]
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                new_width = width * orig_width * width_ratio / target_size
                new_height = height * orig_height * height_ratio / target_size

                new_annotations.append(
                    f"{class_id} {x_center} {y_center} {new_width} {new_height}\n"
                )

        with open(new_label_path, "w") as f:
            f.writelines(new_annotations)

        resized_metadata.append(
            {
                "image_path": os.path.relpath(new_img_path, save_dir),
                "label_path": os.path.relpath(new_label_path, save_dir),
                "width": target_size,
                "height": target_size,
                "filename": filename,
                "region_name": row["region_name"],
                "split": split,
                "original_width": orig_width,
                "original_height": orig_height,
            }
        )

    return pd.DataFrame(resized_metadata)


def create_region_based_splits(
    df, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, random_seed=42
):
    np.random.seed(random_seed)

    regions = df["region_name"].unique()
    np.random.shuffle(regions)

    n_regions = len(regions)
    train_idx = int(n_regions * train_ratio)
    val_idx = int(n_regions * (train_ratio + val_ratio))

    train_regions = regions[:train_idx]
    val_regions = regions[train_idx:val_idx]
    test_regions = regions[val_idx:]

    df_with_splits = df.copy()
    df_with_splits["split"] = "unknown"
    df_with_splits.loc[df_with_splits["region_name"].isin(train_regions), "split"] = (
        "train"
    )
    df_with_splits.loc[df_with_splits["region_name"].isin(val_regions), "split"] = (
        "validation"
    )
    df_with_splits.loc[df_with_splits["region_name"].isin(test_regions), "split"] = (
        "test"
    )

    split_counts = df_with_splits["split"].value_counts()
    total = len(df_with_splits)
    print(
        f"Train: {split_counts['train']} ({100 * split_counts['train'] / total:.1f}%)"
    )
    print(
        f"Validation: {split_counts['validation']} ({100 * split_counts['validation'] / total:.1f}%)"
    )
    print(f"Test: {split_counts['test']} ({100 * split_counts['test'] / total:.1f}%)")

    return df_with_splits


def convert_annotations_to_geoparquet(metadata_df, save_dir):
    """Convert annotations to GeoParquet format.

    Args:
        metadata_df: DataFrame with image and annotation paths (one row per sample)
        save_dir: Directory to save GeoParquet files

    Returns:
        DataFrame with paths to GeoParquet annotation files
    """
    geoparquet_dir = os.path.join(save_dir, "annotations_geoparquet")
    os.makedirs(geoparquet_dir, exist_ok=True)

    results = []

    for _, row in tqdm(
        metadata_df.iterrows(),
        total=len(metadata_df),
        desc="Converting annotations to GeoParquet",
    ):
        try:
            image_path = os.path.join(save_dir, row["image_path"])
            label_path = os.path.join(save_dir, row["label_path"])
            geotiff_path = row.get("geotiff_path", "")
            filename = row["filename"]
            split = row["split"]

            output_file = os.path.join(geoparquet_dir, f"{filename}_annotations.gpq")

            objects = []

            with open(label_path) as f:
                lines = f.readlines()

            width = row["width"]
            height = row["height"]

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * width
                    y_center = float(parts[2]) * height
                    box_width = float(parts[3]) * width
                    box_height = float(parts[4]) * height

                    # Convert center and width/height to xmin, ymin, xmax, ymax
                    xmin = max(0, int(x_center - (box_width / 2)))
                    ymin = max(0, int(y_center - (box_height / 2)))
                    xmax = min(width, int(x_center + (box_width / 2)))
                    ymax = min(height, int(y_center + (box_height / 2)))

                    # Create bbox record
                    objects.append(
                        {
                            "filename": filename,
                            "image_path": row["image_path"],
                            "geotiff_path": geotiff_path,
                            "label": "windTurbine",
                            "class_id": class_id,
                            "xmin": xmin,
                            "ymin": ymin,
                            "xmax": xmax,
                            "ymax": ymax,
                            "width": width,
                            "height": height,
                            "region_name": row["region_name"],
                            "split": split,
                        }
                    )

            # Skip if no objects found
            if not objects:
                results.append(
                    {
                        "filename": filename,
                        "geotiff_path": geotiff_path,
                        "status": "empty",
                        "error": "No objects found in annotation file",
                    }
                )
                continue

            bbox_df = pd.DataFrame(objects)
            geometries = [
                box(row.xmin, row.ymin, row.xmax, row.ymax)
                for _, row in bbox_df.iterrows()
            ]

            gdf = gpd.GeoDataFrame(bbox_df, geometry=geometries, crs="EPSG:4326")

            gdf.to_parquet(output_file)

            results.append(
                {
                    "filename": filename,
                    "geotiff_path": geotiff_path,
                    "annotation_path": output_file,
                    "num_annotations": len(gdf),
                    "status": "success",
                }
            )

        except Exception as e:
            results.append(
                {
                    "filename": row.get("filename", "unknown"),
                    "geotiff_path": row.get("geotiff_path", "unknown"),
                    "status": "error",
                    "error": str(e),
                }
            )

    results_df = pd.DataFrame(results)

    success_count = len(results_df[results_df["status"] == "success"])
    error_count = len(results_df[results_df["status"] == "error"])
    empty_count = len(results_df[results_df["status"] == "empty"])

    print(f"Successfully created {success_count} GeoParquet files")
    print(f"Empty annotations: {empty_count}")

    if error_count > 0:
        print(f"Failed to create {error_count} files")
        for _, row in results_df[results_df["status"] == "error"].head(5).iterrows():
            print(f"  - {row['filename']}: {row['error']}")

    results_success = results_df[results_df["status"] == "success"]

    results_success["annotation_path"] = results_success["annotation_path"].str.replace(
        save_dir, ""
    )

    final_df = metadata_df.merge(
        results_success[["filename", "annotation_path", "num_annotations"]],
        on="filename",
        how="left",
    )

    return final_df


def generate_metadata_df(root_dir: str) -> pd.DataFrame:
    """Generate metadata DataFrame for Windturbine dataset."""
    image_paths = glob.glob(
        os.path.join(root_dir, "windTurbineDataSet", "JPEGImages", "*.png")
    )

    df = pd.DataFrame(image_paths, columns=["image_path"])
    df["label_path"] = (
        df["image_path"].str.replace("JPEGImages", "labels").str.replace(".png", ".txt")
    )

    # find image sizes
    def extract_image_size(image_path):
        with open(image_path, "rb") as f:
            img = Image.open(f)
            img.load()
            return img.size

    df["image_size"] = df["image_path"].apply(extract_image_size)
    df["width"] = df["image_size"].apply(lambda x: x[0])
    df["height"] = df["image_size"].apply(lambda x: x[1])
    df["filename"] = df["image_path"].apply(lambda x: os.path.basename(x).split(".")[0])
    df["region_name"] = df["filename"].str.replace(r"\d+$", "", regex=True)

    # make paths relative
    df["image_path"] = df["image_path"].str.replace(root_dir + os.sep, "")
    df["label_path"] = df["label_path"].str.replace(root_dir + os.sep, "")

    df["lat"] = None
    df["lon"] = None

    # create splits
    df = create_region_based_splits(df)

    df["image_path"] = df["image_path"].str.replace(root_dir, "")
    df["label_path"] = df["label_path"].str.replace(root_dir, "")

    return df


def create_tortilla(annotations_df, root_dir, save_dir, tortilla_name):
    """Create a tortilla version of an object detection dataset.

    Args:
        annotations_df: DataFrame with annotations including image_path, label, bbox coordinates
        image_dir: Directory containing the GeoTIFF images
        save_dir: Directory to save the tortilla files
        tortilla_name: Name of the final tortilla file
    """
    tortilla_dir = os.path.join(root_dir, "tortilla")
    os.makedirs(tortilla_dir, exist_ok=True)

    for idx, row in tqdm(
        annotations_df.iterrows(), total=len(annotations_df), desc="Creating tortillas"
    ):
        geotiff_path = os.path.join(root_dir, row["geotiff_path"])

        annotation_path = os.path.join(root_dir, row["annotation_path"])

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
            with rasterio.open(geotiff_path) as src:
                profile = src.profile
                height, width = profile["height"], profile["width"]
                crs = (
                    "EPSG:" + str(profile["crs"].to_epsg())
                    if profile["crs"]
                    else "EPSG:4326"
                )
                transform = (
                    profile["transform"].to_gdal() if profile["transform"] else None
                )

        split = row["split"]
        lon = row["lon"] if not pd.isna(row["lon"]) else None
        lat = row["lat"] if not pd.isna(row["lat"]) else None

        # create image
        image_sample = tacotoolbox.tortilla.datamodel.Sample(
            id="image",
            path=geotiff_path,
            file_format="GTiff",
            data_split=split,
            stac_data={
                "crs": crs,
                "geotransform": transform,
                "raster_shape": (height, width),
                "time_start": "2020",
            },
            region_name=row["region_name"],
            lon=lon,
            lat=lat,
        )

        # Create annotation part
        annotations_sample = tacotoolbox.tortilla.datamodel.Sample(
            id="annotations",
            path=annotation_path,
            file_format="GeoParquet",
            data_split=split,
        )

        taco_samples = tacotoolbox.tortilla.datamodel.Samples(
            samples=[image_sample, annotations_sample]
        )

        sample_path = os.path.join(tortilla_dir, f"sample_{idx}.tortilla")
        tacotoolbox.tortilla.create(taco_samples, sample_path, quiet=True)

    # Merge all individual tortillas into one dataset
    all_tortilla_files = sorted(glob.glob(os.path.join(tortilla_dir, "*.tortilla")))

    samples = []
    for tortilla_file in tqdm(all_tortilla_files, desc="Building final tortilla"):
        sample_data = tacoreader.load(tortilla_file).iloc[0]

        sample = tacotoolbox.tortilla.datamodel.Sample(
            id=os.path.basename(tortilla_file).split(".")[0],
            path=tortilla_file,
            file_format="TORTILLA",
            stac_data={
                "crs": sample_data.get("stac:crs"),
                "geotransform": sample_data.get("stac:geotransform"),
                "raster_shape": sample_data.get("stac:raster_shape"),
                "time_start": sample_data.get("stac:time_start"),
            },
            data_split=sample_data["tortilla:data_split"],
            region_name=sample_data["region_name"],
            lon=sample_data.get("lon"),
            lat=sample_data.get("lat"),
        )
        samples.append(sample)

    final_samples = tacotoolbox.tortilla.datamodel.Samples(samples=samples)
    final_path = os.path.join(save_dir, tortilla_name)
    tacotoolbox.tortilla.create(final_samples, final_path, quiet=True)


def main():
    """Generate Windturbine Benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for Windturbine dataset"
    )
    parser.add_argument(
        "--save_dir",
        default="geobenchV2/wind_turbine",
        help="Directory to save the subset",
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

    resized_path = os.path.join(args.save_dir, "geobench_wind_turbine_resized.parquet")
    if os.path.exists(resized_path):
        resized_metadata_df = pd.read_parquet(resized_path)
    else:
        resized_metadata_df = resize_and_save_dataset(
            metadata_df, args.root, args.save_dir, target_size=512
        )
        resized_metadata_df.to_parquet(resized_path)

    final_path = os.path.join(args.save_dir, "geobench_wind_turbine.parquet")
    if os.path.exists(final_path):
        final_df = pd.read_parquet(final_path)
    else:
        final_df = convert_pngs_to_geotiffs(
            resized_metadata_df, args.save_dir, args.save_dir, num_workers=16
        )
        final_df = convert_annotations_to_geoparquet(final_df, args.save_dir)
        final_df.to_parquet(final_path)

    tortilla_name = "geobench_wind_turbine.tortilla"
    create_tortilla(final_df, args.save_dir, args.save_dir, tortilla_name)

    create_unittest_subset(
        data_dir=args.save_dir,
        tortilla_pattern=tortilla_name,
        test_dir_name="wind_turbine",
        n_train_samples=2,
        n_val_samples=1,
        n_test_samples=1,
    )


if __name__ == "__main__":
    main()
