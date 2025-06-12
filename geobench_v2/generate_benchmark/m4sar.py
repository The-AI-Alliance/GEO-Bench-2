# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of M4SAR dataset."""

import argparse
import glob
import os
import warnings

import geopandas as gpd
import pandas as pd
import rasterio
import tacoreader
import tacotoolbox
from rasterio.errors import NotGeoreferencedWarning
from shapely.geometry import box
from tqdm import tqdm

from geobench_v2.generate_benchmark.object_detection_util import (
    convert_pngs_to_geotiffs,
)
from geobench_v2.generate_benchmark.utils import (
    create_subset_from_df,
    create_unittest_subset,
)


def generate_metadata_df(root_dir: str) -> pd.DataFrame:
    """Generate metadata DataFrame for M4SAR dataset.

    Args:
        root_dir: Root directory of the M4SAR dataset.
    """
    optical_image_paths = glob.glob(
        os.path.join(root_dir, "M4-SAR", "optical", "images", "*", "*.jpg")
    )

    # /mnt/rg_climate_benchmark/data/datasets_object_detection/m4_sar/M4-SAR/M4-SAR/optical/images/test/39115.jpg

    df = pd.DataFrame({"optical_path": optical_image_paths})

    df["sar_path"] = df["optical_path"].apply(lambda x: x.replace("/optical/", "/sar/"))

    df["optical_label_path"] = df["optical_path"].apply(
        lambda x: x.replace("/images/", "/labels/").replace(".jpg", ".txt")
    )
    df["sar_label_path"] = df["sar_path"].apply(
        lambda x: x.replace("/images/", "/labels/").replace(".jpg", ".txt")
    )

    df["split"] = df["optical_path"].apply(lambda x: x.split("/")[-2])
    df["split"] = df["split"].replace("val", "validation")

    df["optical_path"] = df["optical_path"].str.replace(root_dir, "").str.lstrip("/")
    df["sar_path"] = df["sar_path"].str.replace(root_dir, "").str.lstrip("/")
    df["optical_label_path"] = df["optical_label_path"].str.replace(root_dir, "").str.lstrip("/")
    df["sar_label_path"] = df["sar_label_path"].str.replace(root_dir, "").str.lstrip("/")

    df["label_path"] = df["optical_label_path"]

    # For optical images:
    # Files named 1.jpg to 56087.jpg have a resolution of 10 meters.
    # Files named 56088.jpg to 112174.jpg have a resolution of 60 meters.
    # only pick the high-resolution images
    df["filename"] = df["optical_path"].apply(lambda x: os.path.basename(x).replace(".jpg", ""))
    df["file_number"] = df["filename"].astype(int)
    
    # Filter for high-resolution images (1.jpg to 56087.jpg)
    df = df[df["file_number"] <= 56087].copy()
    
    # Drop temporary columns
    df = df.drop(columns=["filename", "file_number"])

    return df


def convert_annotations_to_geoparquet(metadata_df, root_dir, save_dir):
    """Convert annotations to GeoParquet format.

    Args:
        metadata_df: DataFrame with image and annotation paths (one row per sample)
        root_dir: Root directory of the M4SAR dataset
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
        label_path = os.path.join(root_dir, row["label_path"])
        geotiff_path = row.get("optical_geotiff_path", "")
        split = row["split"]

        output_file = os.path.join(
            geoparquet_dir,
            f"{os.path.basename(geotiff_path).replace('.tif', '')}_annotations.gpq",
        )

        objects = []

        with open(label_path) as f:
            lines = f.readlines()

        width = 512
        height = 512

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])

                # Extract the four corner points of the quadrilateral
                x1 = float(parts[1]) * width
                y1 = float(parts[2]) * height
                x2 = float(parts[3]) * width
                y2 = float(parts[4]) * height
                x3 = float(parts[5]) * width
                y3 = float(parts[6]) * height
                x4 = float(parts[7]) * width
                y4 = float(parts[8]) * height

                # Create bbox record
                objects.append(
                    {
                        "optical_geotiff_path": row["optical_geotiff_path"],
                        "class_id": class_id,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "x3": x3,
                        "y3": y3,
                        "x4": x4,
                        "y4": y4,
                        "xmin": min(x1, x2, x3, x4),
                        "ymin": min(y1, y2, y3, y4),
                        "xmax": max(x1, x2, x3, x4),
                        "ymax": max(y1, y2, y3, y4),
                        "width": width,
                        "height": height,
                        "split": split,
                    }
                )

        bbox_df = pd.DataFrame(objects)
        geometries = [
            box(row.xmin, row.ymin, row.xmax, row.ymax) for _, row in bbox_df.iterrows()
        ]

        gdf = gpd.GeoDataFrame(bbox_df, geometry=geometries, crs="EPSG:4326")

        gdf.to_parquet(output_file)

        results.append(
            {
                "optical_geotiff_path": geotiff_path,
                "annotation_path": output_file,
                "num_annotations": len(gdf),
            }
        )

    results_df = pd.DataFrame(results)

    results_df["annotation_path"] = results_df["annotation_path"].str.replace(
        save_dir, ""
    )

    final_df = metadata_df.merge(results_df, on="optical_geotiff_path", how="left")

    return final_df


def create_tortilla(annotations_df, root_dir, save_dir, tortilla_name):
    """Create a tortilla version of an object detection dataset.

    Args:
        annotations_df: DataFrame with annotations including image_path, label, bbox coordinates
        root_dir: Directory containing the GeoTIFF images
        save_dir: Directory to save the tortilla files
        tortilla_name: Name of the final tortilla file
    """
    tortilla_dir = os.path.join(root_dir, "tortilla")
    os.makedirs(tortilla_dir, exist_ok=True)

    for idx, row in tqdm(
        annotations_df.iterrows(), total=len(annotations_df), desc="Creating tortillas"
    ):
        modalities = ["optical", "sar"]
        samples = []
        for modality in modalities:
            geotiff_path = os.path.join(
                root_dir, "tif_images", row[modality + "_geotiff_path"]
            )

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

            image_sample = tacotoolbox.tortilla.datamodel.Sample(
                id=modality,
                path=geotiff_path,
                file_format="GTiff",
                data_split=split,
                stac_data={
                    "crs": crs,
                    "geotransform": transform,
                    "raster_shape": (height, width),
                    "time_start": "2020",
                    "time_end": "2022",
                },
                lon=lon,
                lat=lat,
            )
            samples.append(image_sample)

        annotation_path = os.path.join(root_dir, row["annotation_path"])

        # Create annotation part
        annotations_sample = tacotoolbox.tortilla.datamodel.Sample(
            id="annotations",
            path=annotation_path,
            file_format="GeoParquet",
            data_split=split,
        )
        samples.append(annotations_sample)

        taco_samples = tacotoolbox.tortilla.datamodel.Samples(samples=samples)

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
                "time_end": sample_data.get("stac:time_end"),
            },
            data_split=sample_data["tortilla:data_split"],
            lon=sample_data.get("lon"),
            lat=sample_data.get("lat"),
        )
        samples.append(sample)

    final_samples = tacotoolbox.tortilla.datamodel.Samples(samples=samples)
    final_path = os.path.join(save_dir, tortilla_name)
    tacotoolbox.tortilla.create(final_samples, final_path, quiet=True)


def main():
    """Generate M4SAR Benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for M4SAR dataset"
    )
    parser.add_argument(
        "--save_dir", default="geobenchV2/m4sar", help="Directory to save the subset"
    )
    args = parser.parse_args()

    metadata_df = generate_metadata_df(args.root)

    path = os.path.join(args.root, "geobench_metadata_df.parquet")
    metadata_df.to_parquet(path)


    # create a subset
    subset_df = create_subset_from_df(
        metadata_df,
        n_train_samples=4000,
        n_val_samples=1000,
        n_test_samples=2000,
        random_state=42,
    )

    final_path = os.path.join(args.save_dir, "geobench_m4sar.parquet")
    tif_image_dir = os.path.join(args.save_dir, "tif_images")
    # if os.path.exists(final_path):
    #     final_df = pd.read_parquet(final_path)
    # else:
    final_df = convert_pngs_to_geotiffs(
        subset_df,
        args.root,
        target_dir=tif_image_dir,
        image_columns=["optical_path", "sar_path"],
        num_workers=16,
    )
    final_df = convert_annotations_to_geoparquet(final_df, args.root, args.save_dir)
    final_df.to_parquet(final_path)

    tortilla_name = "geobench_m4sar.tortilla"
    create_tortilla(final_df, args.save_dir, args.save_dir, tortilla_name=tortilla_name)

    create_unittest_subset(
        data_dir=args.save_dir,
        tortilla_pattern=tortilla_name,
        test_dir_name="m4sar",
        n_train_samples=2,
        n_val_samples=1,
        n_test_samples=1,
    )


if __name__ == "__main__":
    main()
