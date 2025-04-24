# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of QFabric dataset."""

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
import json
import pandas as pd
import os
from datetime import datetime

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
from shapely.geometry import shape, box
import json
import concurrent.futures
from tqdm import tqdm
import affine
import geopandas as gpd
from shapely.geometry import Polygon


def parse_qfabric_annotation(json_path: str, sample_idx: int) -> pd.DataFrame:
    """Parse QFabric annotation file and extract image metadata.

    Args:
        json_path: Path to the QFabric COCO annotation file
        sample_idx: Sample index to assign to this annotation

    Returns:
        DataFrame containing image metadata
    """
    with open(json_path, "r") as f:
        annotation_data = json.load(f)

    info = annotation_data.get("info", {})
    annotation_id = info.get("id", "")

    if len(annotation_data.get("images", [])) != 5:
        return None

    images_data = []
    for img_idx, img_info in enumerate(annotation_data.get("images", [])):
        img_name = img_info.get("name", "")
        img_width = img_info.get("width", 0)
        img_height = img_info.get("height", 0)
        img_file = img_info.get("file_name", "")
        img_date = img_info.get("date_captured", "")

        row_data = {
            "sample_idx": sample_idx,
            "annotation_id": annotation_id,
            "json_file": os.path.basename(json_path),
            f"img_{img_idx}_file": img_file,
            f"img_{img_idx}_date": img_date,
            f"img_{img_idx}_width": img_width,
            f"img_{img_idx}_height": img_height,
        }

        images_data.append(row_data)

    merged_data = {}
    for d in images_data:
        merged_data.update(d)
    return pd.DataFrame([merged_data])


def generate_metadata_df(root_dir: str) -> pd.DataFrame:
    """Generate metadata DataFrame for QFabric dataset."""

    vector_paths = glob.glob(
        os.path.join(
            root_dir,
            "dataset/vectors/random-split-1_2023_02_05-11_47_30/COCO",
            "*.json",
        )
    )

    full_df = pd.DataFrame()

    for idx, json_path in enumerate(tqdm(vector_paths)):
        if os.path.basename(json_path) == "metadata.json":
            continue
        coco_data = parse_qfabric_annotation(json_path, idx)

        if coco_data is None:
            continue
        full_df = pd.concat([full_df, coco_data], ignore_index=True)

    unique_files = full_df["img_4_file"].unique()
    for file_name in tqdm(unique_files, desc="Extracting lat/long"):
        with rasterio.open(os.path.join(root_dir, "dataset", file_name), "r") as src:
            lon, lat = src.lnglat()
            width = src.width
            height = src.height
        full_df.loc[full_df["img_4_file"] == file_name, "lon"] = lon
        full_df.loc[full_df["img_4_file"] == file_name, "lat"] = lat
        full_df.loc[full_df["img_4_file"] == file_name, "width"] = width
        full_df.loc[full_df["img_4_file"] == file_name, "height"] = height

    # randomly split the dataset into train/val/test (70, 10, 20) by sample_idx which denotes a unique location
    full_df["split"] = "train"
    np.random.seed(42)
    unique_sample_idx = full_df["sample_idx"].unique()
    np.random.shuffle(unique_sample_idx)

    n_train = int(len(unique_sample_idx) * 0.7)
    n_val = int(len(unique_sample_idx) * 0.1)
    n_test = len(unique_sample_idx) - n_train - n_val
    train_sample_idx = unique_sample_idx[:n_train]
    val_sample_idx = unique_sample_idx[n_train : n_train + n_val]
    test_sample_idx = unique_sample_idx[n_train + n_val :]

    full_df.loc[full_df["sample_idx"].isin(train_sample_idx), "split"] = "train"
    full_df.loc[full_df["sample_idx"].isin(val_sample_idx), "split"] = "validation"
    full_df.loc[full_df["sample_idx"].isin(test_sample_idx), "split"] = "test"

    return full_df


def transform_polygon_to_patch(poly, window_bounds, image_shape):
    """Transform polygon coordinates from original image to patch coordinates."""
    window_minx, window_miny, window_maxx, window_maxy = window_bounds.bounds

    def transform_point(x, y):
        rel_x = (x - window_minx) / (window_maxx - window_minx)
        rel_y = (y - window_miny) / (window_maxy - window_miny)

        patch_x = rel_x * image_shape[1]
        patch_y = rel_y * image_shape[0]

        return patch_x, patch_y

    polygon_coords = list(poly.exterior.coords)
    transformed_coords = [transform_point(x, y) for x, y in polygon_coords]

    return Polygon(transformed_coords)


def window_to_bounds(window, transform):
    """Convert rasterio window to bounds."""
    left, top = transform * (window.col_off, window.row_off)
    right, bottom = transform * (
        window.col_off + window.width,
        window.row_off + window.height,
    )
    return left, bottom, right, top


def create_patch_windows(height, width, patch_size=1024, overlap=0):
    """Create sliding windows of specified size with optional overlap."""
    windows = []
    positions = []

    for y in range(0, height, patch_size - overlap):
        if y + patch_size > height:
            y = max(0, height - patch_size)

        for x in range(0, width, patch_size - overlap):
            if x + patch_size > width:
                x = max(0, width - patch_size)

            windows.append(Window(x, y, patch_size, patch_size))
            positions.append((y // (patch_size - overlap), x // (patch_size - overlap)))

            if x + patch_size >= width:
                break

        if y + patch_size >= height:
            break

    return windows, positions


def parse_annotations(annotation_data, image_transform, patch_window, image_shape):
    """Extract and transform annotations for a specific patch."""
    change_type_polys = []
    change_status_polys = {i: [] for i in range(5)}

    window_bounds = box(*window_to_bounds(patch_window, image_transform))

    for annotation in annotation_data["annotations"]:
        if (
            "segmentation" in annotation
            and annotation["segmentation"]
            and len(annotation["segmentation"]) > 0
        ):
            polygon_coords = annotation["segmentation"][0]

            points = [
                (polygon_coords[i], polygon_coords[i + 1])
                for i in range(0, len(polygon_coords), 2)
            ]

            poly = Polygon(points)

            if poly.intersects(window_bounds):
                if "properties" in annotation:
                    change_type = None
                    change_status = {}

                    for prop in annotation["properties"]:
                        if prop.get("type") == "Change Type" and "labels" in prop:
                            if (
                                isinstance(prop["labels"], list)
                                and len(prop["labels"]) > 0
                            ):
                                change_type = int(prop["labels"][0])

                        if prop.get("type") == "Change Status" and "labels" in prop:
                            if isinstance(prop["labels"], dict):
                                for img_idx, status in prop["labels"].items():
                                    if img_idx.isdigit():
                                        if isinstance(status, list) and len(status) > 0:
                                            change_status[int(img_idx)] = int(status[0])

                    if change_type is not None:
                        poly_patch = transform_polygon_to_patch(
                            poly, window_bounds, image_shape
                        )
                        if poly_patch.is_valid and not poly_patch.is_empty:
                            change_type_polys.append((poly_patch, change_type))

                    for img_idx, status in change_status.items():
                        if img_idx < 5:
                            poly_patch = transform_polygon_to_patch(
                                poly, window_bounds, image_shape
                            )
                            if poly_patch.is_valid and not poly_patch.is_empty:
                                change_status_polys[img_idx].append(
                                    (poly_patch, status)
                                )

    change_type_mask = None
    if change_type_polys:
        change_type_mask = rasterize(
            shapes=[(poly, value) for poly, value in change_type_polys],
            out_shape=image_shape,
            transform=rasterio.transform.from_bounds(
                0, 0, image_shape[1], image_shape[0], image_shape[1], image_shape[0]
            ),
            fill=0,
            all_touched=False,
            dtype=np.uint8,
        )

    change_status_masks = {}
    for img_idx, polys in change_status_polys.items():
        if polys:
            change_status_masks[img_idx] = rasterize(
                shapes=[(poly, value) for poly, value in polys],
                out_shape=image_shape,
                transform=rasterio.transform.from_bounds(
                    0, 0, image_shape[1], image_shape[0], image_shape[1], image_shape[0]
                ),
                fill=0,
                all_touched=False,
                dtype=np.uint8,
            )
        else:
            change_status_masks[img_idx] = np.zeros(image_shape, dtype=np.uint8)

    return change_type_mask, change_status_masks


def process_sample(args):
    """Process a single QFabric sample, creating patches for all images and masks."""
    sample_idx, row, root_dir, output_dir, patch_size = (
        args["sample_idx"],
        args["row"],
        args["root_dir"],
        args["output_dir"],
        args["patch_size"],
    )

    json_path = os.path.join(
        root_dir,
        "dataset/vectors/random-split-1_2023_02_05-11_47_30/COCO",
        row["json_file"],
    )

    with open(json_path, "r") as f:
        annotation_data = json.load(f)

    image_paths = []
    for img_info in annotation_data["images"]:
        img_path = os.path.join(root_dir, "dataset", img_info["file_name"])
        image_paths.append(img_path)

    if len(image_paths) != 5:
        return {
            "sample_idx": sample_idx,
            "status": "error",
            "message": "Invalid image count",
        }

    # Open first image to get metadata
    with rasterio.open(image_paths[0]) as src:
        height = src.height
        width = src.width
        crs = src.crs
        transform = src.transform

        # precompute the windows
        is_training = row.get("split") == "train"
        overlap = 0.1 if is_training else 0.0
        windows, positions = create_centered_patch_windows(
            height, width, patch_size, overlap=overlap
        )

    patches_info = []

    for window, position in zip(windows, positions):
        patch_id = f"{sample_idx}_{position[0]}_{position[1]}"

        window_transform = rasterio.windows.transform(window, transform)

        change_type_mask, change_status_masks = parse_annotations(
            annotation_data, transform, window, (window.height, window.width)
        )

        if change_type_mask is None:
            change_type_mask = np.zeros((window.height, window.width), dtype=np.uint8)
            change_type_ratio = 0.0
        else:
            change_type_ratio = np.count_nonzero(change_type_mask) / (
                window.height * window.width
            )

        status_ratios = {}
        for img_idx, status_mask in change_status_masks.items():
            status_ratios[img_idx] = np.count_nonzero(status_mask) / (
                window.height * window.width
            )

        images_data = []
        for i, img_path in enumerate(image_paths):
            with rasterio.open(img_path) as src:
                patch_data = src.read(window=window)
                images_data.append(patch_data)

        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

        # input images
        image_patch_paths = []
        for i, img_data in enumerate(images_data):
            img_patch_path = os.path.join(
                output_dir, "images", f"{patch_id}_img{i}.tif"
            )

            profile = {
                "driver": "GTiff",
                "height": window.height,
                "width": window.width,
                "count": img_data.shape[0],
                "dtype": img_data.dtype,
                "tiled": True,
                "blockxsize": window.width,
                "blockysize": window.height,
                "interleave": "pixel",
                "compress": "zstd",
                "zstd_level": 13,
                "predictor": 2,
                "crs": crs,
                "transform": window_transform,
            }

            with rasterio.open(img_patch_path, "w", **profile) as dst:
                dst.write(img_data)

            image_patch_paths.append(img_patch_path)

        # Write change type mask overall for the series
        change_type_path = os.path.join(
            output_dir, "masks", f"{patch_id}_change_type.tif"
        )

        mask_profile = {
            "driver": "GTiff",
            "height": window.height,
            "width": window.width,
            "count": 1,
            "dtype": "uint8",
            "tiled": True,
            "blockxsize": window.width,
            "blockysize": window.height,
            "interleave": "pixel",
            "compress": "zstd",
            "zstd_level": 13,
            "predictor": 2,
            "crs": crs,
            "transform": window_transform,
        }

        with rasterio.open(change_type_path, "w", **mask_profile) as dst:
            dst.write(change_type_mask[np.newaxis, :, :])

        # change status masks
        status_mask_paths = []
        for img_idx, status_mask in change_status_masks.items():
            status_path = os.path.join(
                output_dir, "masks", f"{patch_id}_status_{img_idx}.tif"
            )

            with rasterio.open(status_path, "w", **mask_profile) as dst:
                dst.write(status_mask[np.newaxis, :, :])

            status_mask_paths.append(status_path)

        patches_info.append(
            {
                "patch_id": patch_id,
                "window": str(window),
                "position": str(position),
                "image_paths": image_patch_paths,
                "change_type_path": change_type_path,
                "status_paths": status_mask_paths,
                "change_type_ratio": change_type_ratio,
                "status_ratios": status_ratios,
            }
        )

    return {
        "sample_idx": sample_idx,
        "status": "success",
        "patches": patches_info,
        "lon": row.get("lon"),
        "lat": row.get("lat"),
    }


def create_centered_patch_windows(height, width, patch_size, overlap=0.0):
    """Create sliding windows centered on the image with optional overlap.

    Args:
        height: Image height
        width: Image width
        patch_size: Size of each patch
        overlap: Overlap fraction between patches (0.0 to 1.0)

    Returns:
        List of windows and their position identifiers
    """
    windows = []
    positions = []

    overlap_px = int(patch_size * overlap)
    step = patch_size - overlap_px

    n_patches_h = max(1, (height - overlap_px) // step)
    n_patches_w = max(1, (width - overlap_px) // step)

    border_h = (
        height - ((n_patches_h * patch_size) - (n_patches_h - 1) * overlap_px)
    ) // 2
    border_w = (
        width - ((n_patches_w * patch_size) - (n_patches_w - 1) * overlap_px)
    ) // 2

    border_h = max(0, border_h)
    border_w = max(0, border_w)

    for i in range(n_patches_h):
        for j in range(n_patches_w):
            y = border_h + i * step
            x = border_w + j * step

            if y + patch_size > height:
                y = height - patch_size
            if x + patch_size > width:
                x = width - patch_size

            windows.append(Window(x, y, patch_size, patch_size))
            positions.append((i, j))

    return windows, positions


def process_qfabric_dataset(
    metadata_df, root_dir, output_dir, patch_size=1024, num_workers=8
):
    """Process entire QFabric dataset with parallel execution."""
    os.makedirs(output_dir, exist_ok=True)

    tasks = []
    for idx, row in metadata_df.iterrows():
        tasks.append(
            {
                "sample_idx": idx,
                "row": row,
                "root_dir": root_dir,
                "output_dir": output_dir,
                "patch_size": patch_size,
            }
        )

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for result in tqdm(
            executor.map(process_sample, tasks),
            total=len(tasks),
            desc="Processing QFabric samples",
        ):
            results.append(result)

    patches_data = []
    for result in results:
        if result["status"] == "success" and "patches" in result:
            for patch in result["patches"]:
                patch_info = {
                    "sample_idx": result["sample_idx"],
                    "patch_id": patch["patch_id"],
                    "lon": result.get("lon"),
                    "lat": result.get("lat"),
                    "change_type_path": patch["change_type_path"],
                    "change_type_ratio": patch["change_type_ratio"],
                    "status_ratios": patch["status_ratios"],
                }

                for i, img_path in enumerate(patch["image_paths"]):
                    patch_info[f"image_{i}_path"] = img_path

                for i, status_path in enumerate(patch["status_paths"]):
                    patch_info[f"status_{i}_path"] = status_path

                patches_data.append(patch_info)

    patches_df = pd.DataFrame(patches_data)
    return patches_df


def create_tortilla(root_dir, df, save_dir, tortilla_name):
    """Create a tortilla version of the dataset."""

    tortilla_dir = os.path.join(save_dir, "tortilla")
    os.makedirs(tortilla_dir, exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating tortilla"):
        modalities = ["PS-RGBNIR", "SAR-Intensity", "mask"]
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
                    "time_start": row["date"],
                },
                lon=row["lon"],
                lat=row["lat"],
                source_img_file=row["source_img_file"],
                source_mask_file=row["source_mask_file"],
                patch_id=row["patch_id"],
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
            source_img_file=sample_data["source_img_file"],
            source_mask_file=sample_data["source_mask_file"],
            patch_id=sample_data["patch_id"],
        )
        samples.append(sample_tortilla)

    final_samples = tacotoolbox.tortilla.datamodel.Samples(samples=samples)
    tacotoolbox.tortilla.create(
        final_samples, os.path.join(save_dir, tortilla_name), quiet=True, nworkers=4
    )


# def create_geobench_version(
#     metadata_df: pd.DataFrame,
#     n_train_samples: int,
#     n_val_samples: int,
#     n_test_samples: int,
# ) -> None:
#     """Create a GeoBench version of the dataset.
#     Args:
#         metadata_df: DataFrame with metadata including geolocation for each patch
#         n_train_samples: Number of final training samples, -1 means all
#         n_val_samples: Number of final validation samples, -1 means all
#         n_test_samples: Number of final test samples, -1 means all
#     """
#     random_state = 24

#     subset_df = create_subset_from_df(
#         metadata_df,
#         n_train_samples=n_train_samples,
#         n_val_samples=n_val_samples,
#         n_test_samples=n_test_samples,
#         random_state=random_state,
#     )

#     return subset_df


def main():
    """Generate QFabric Benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for QFabric dataset"
    )
    parser.add_argument(
        "--save_dir", default="geobenchV2/QFabric", help="Directory to save the subset"
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

    patches_path = os.path.join(args.save_dir, "geobench_qfabric_patches.parquet")
    # if os.path.exists(patches_path):
    #     patches_df = pd.read_parquet(patches_path)
    # else:
    patches_df = process_qfabric_dataset(
        metadata_df, args.root, args.save_dir, patch_size=2048, num_workers=2
    )
    patches_df.to_parquet(patches_path)

    import pdb

    pdb.set_trace()
    # create geobench_version

    tortilla_name = "geobench_qfabric.tortilla"
    create_tortilla(args.save_dir, patches_df, args.save_dir, tortilla_name)

    import pdb

    pdb.set_trace()

    print(0)


if __name__ == "__main__":
    main()
