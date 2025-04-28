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

from geobench_v2.generate_benchmark.utils import (
    plot_sample_locations,
    create_unittest_subset,
    create_subset_from_df,
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

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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


COCO_PATH = "dataset/vectors/random-split-1_2023_04_18-11_51_35/COCO"


def generate_metadata_df(root_dir: str) -> pd.DataFrame:
    """Generate metadata DataFrame for QFabric dataset."""

    vector_paths = glob.glob(os.path.join(root_dir, COCO_PATH, "*.json"))

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


def transform_polygon_to_patch_absolute(poly, patch_window, image_shape):
    """Transform polygon coordinates from absolute image coordinates to patch coordinates.
    Uses absolute pixel coordinates instead of relative coordinates."""

    def transform_point(x, y):
        # Convert from absolute image coords to patch-local coords
        patch_x = x - patch_window.col_off
        patch_y = y - patch_window.row_off
        return patch_x, patch_y

    if poly.geom_type == "Polygon":
        exterior_coords = [transform_point(x, y) for x, y in poly.exterior.coords]
        interior_rings = []

        for interior in poly.interiors:
            interior_coords = [transform_point(x, y) for x, y in interior.coords]
            interior_rings.append(interior_coords)

        return Polygon(exterior_coords, interior_rings)

    elif poly.geom_type == "MultiPolygon":
        parts = []
        for geom in poly.geoms:
            transformed_poly = transform_polygon_to_patch_absolute(
                geom, patch_window, image_shape
            )
            if transformed_poly.is_valid and not transformed_poly.is_empty:
                parts.append(transformed_poly)

        if parts:
            from shapely.ops import unary_union

            return unary_union(parts)
        else:
            return Polygon()

    else:
        return Polygon()


def parse_annotations(annotation_data, image_transform, patch_window, image_shape):
    """Extract and transform annotations for a specific patch."""
    change_type_polys = []
    change_status_polys = {i: [] for i in range(5)}

    # Get patch bounds in absolute pixel coordinates
    window_bounds = box(
        patch_window.col_off,
        patch_window.row_off,
        patch_window.col_off + patch_window.width,
        patch_window.row_off + patch_window.height,
    )

    for annotation in annotation_data["annotations"]:
        if not annotation.get("segmentation") or not annotation["segmentation"]:
            continue

        if (
            not isinstance(annotation["segmentation"], list)
            or not annotation["segmentation"]
        ):
            continue

        polygon_coords = annotation["segmentation"][0]

        if len(polygon_coords) < 6:  # Need at least 3 points (x,y pairs)
            continue

        points = [
            (polygon_coords[i], polygon_coords[i + 1])
            for i in range(0, len(polygon_coords), 2)
        ]

        poly = Polygon(points)
        if not poly.is_valid:
            poly = poly.buffer(0)

        # Check intersection with patch window in absolute coordinates
        if poly.intersects(window_bounds):
            intersection = poly.intersection(window_bounds)

            if intersection.is_empty:
                continue

            # Transform to patch-local coordinates
            patch_poly = transform_polygon_to_patch_absolute(
                intersection, patch_window, image_shape
            )

            if not patch_poly.is_valid or patch_poly.is_empty:
                continue

            # Extract properties
            change_type = None
            change_status = {}

            if "properties" in annotation:
                for prop in annotation["properties"]:
                    # Extract change type
                    if prop.get("type") == "Change Type" and "labels" in prop:
                        labels = prop.get("labels", [])
                        if isinstance(labels, list) and labels:
                            try:
                                change_type = int(labels[0])
                            except (ValueError, TypeError, IndexError):
                                pass

                    # Extract change status for each timepoint
                    if prop.get("type") == "Change Status" and "labels" in prop:
                        status_labels = prop.get("labels", {})
                        if isinstance(status_labels, dict):
                            for img_idx, status in status_labels.items():
                                if img_idx.isdigit() and int(img_idx) < 5:
                                    if isinstance(status, list) and status:
                                        try:
                                            change_status[int(img_idx)] = int(status[0])
                                        except (ValueError, TypeError, IndexError):
                                            pass

            # Add to result collections
            if change_type is not None:
                change_type_polys.append((patch_poly, change_type))

            for img_idx, status in change_status.items():
                change_status_polys[img_idx].append((patch_poly, status))

        else:
            continue
            # print(f"Polygon does not intersect with patch window: {poly}")

    # Rasterize change type mask
    change_type_mask = np.zeros(image_shape, dtype=np.uint8)
    if change_type_polys:
        valid_shapes = [
            (poly, val)
            for poly, val in change_type_polys
            if poly.is_valid and not poly.is_empty
        ]

        if valid_shapes:
            transform = rasterio.transform.from_bounds(
                0, 0, image_shape[1], image_shape[0], image_shape[1], image_shape[0]
            )

            change_type_mask = rasterize(
                shapes=valid_shapes,
                out_shape=image_shape,
                transform=transform,
                fill=0,
                all_touched=True,
                dtype=np.uint8,
            )

    # Rasterize change status masks for each timepoint
    change_status_masks = {}
    for img_idx in range(5):
        mask = np.zeros(image_shape, dtype=np.uint8)

        if img_idx in change_status_polys and change_status_polys[img_idx]:
            valid_shapes = [
                (poly, val)
                for poly, val in change_status_polys[img_idx]
                if poly.is_valid and not poly.is_empty
            ]

            if valid_shapes:
                transform = rasterio.transform.from_bounds(
                    0, 0, image_shape[1], image_shape[0], image_shape[1], image_shape[0]
                )

                mask = rasterize(
                    shapes=valid_shapes,
                    out_shape=image_shape,
                    transform=transform,
                    fill=0,
                    all_touched=True,
                    dtype=np.uint8,
                )

        change_status_masks[img_idx] = mask

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

    json_path = os.path.join(root_dir, COCO_PATH, row["json_file"])

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

        patch_info = {
            "patch_id": patch_id,
            "window": str(window),
            "position": str(position),
            "image_paths": image_patch_paths,
            "change_type_path": change_type_path,
            "status_paths": status_mask_paths,
            "change_type_ratio": change_type_ratio,
            "split": row["split"],
        }

        # Add individual status ratio columns
        for img_idx, ratio in status_ratios.items():
            patch_info[f"status_ratio_{img_idx}"] = ratio

        patches_info.append(patch_info)

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


def create_full_masks(sample_idx, row, root_dir, output_dir):
    """Create full resolution rasterized masks for a sample."""
    json_path = os.path.join(root_dir, COCO_PATH, row["json_file"])

    # Load annotation data
    with open(json_path, "r") as f:
        annotation_data = json.load(f)

    # Get the first image to use as reference for dimensions and transform
    img_path = os.path.join(root_dir, "dataset", row["img_0_file"])
    with rasterio.open(img_path) as src:
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs

    # Prepare to collect annotations
    change_type_polys = []
    change_status_polys = {i: [] for i in range(5)}

    # Process all annotations
    for annotation in annotation_data["annotations"]:
        if not annotation.get("segmentation") or not annotation["segmentation"]:
            continue

        if (
            not isinstance(annotation["segmentation"], list)
            or not annotation["segmentation"]
        ):
            continue

        polygon_coords = annotation["segmentation"][0]

        if len(polygon_coords) < 6:
            continue

        # Create polygon directly from COCO coordinates (already in pixel space)
        points = [
            (polygon_coords[i], polygon_coords[i + 1])
            for i in range(0, len(polygon_coords), 2)
        ]

        poly = Polygon(points)
        if not poly.is_valid:
            poly = poly.buffer(0)
            if not poly.is_valid:
                continue

        change_type = None
        change_status = {}

        if "properties" in annotation:
            for prop in annotation["properties"]:
                if prop.get("type") == "Change Type" and "labels" in prop:
                    labels = prop.get("labels", [])
                    if isinstance(labels, list) and labels:
                        try:
                            change_type = int(labels[0])
                        except (ValueError, TypeError, IndexError):
                            pass

                if prop.get("type") == "Change Status" and "labels" in prop:
                    status_labels = prop.get("labels", {})
                    if isinstance(status_labels, dict):
                        for img_idx, status in status_labels.items():
                            if img_idx.isdigit() and int(img_idx) < 5:
                                if isinstance(status, list) and status:
                                    try:
                                        change_status[int(img_idx)] = int(status[0])
                                    except (ValueError, TypeError, IndexError):
                                        pass

        if change_type is not None:
            change_type_polys.append((poly, change_type))

        for img_idx, status in change_status.items():
            change_status_polys[img_idx].append((poly, status))

    sample_dir = os.path.join(output_dir, f"sample_{sample_idx}")
    os.makedirs(sample_dir, exist_ok=True)

    change_type_path = os.path.join(sample_dir, "change_type.tif")
    change_type_mask = np.zeros((height, width), dtype=np.uint8)

    if change_type_polys:
        # features = []
        # for poly, val in change_type_polys:
        #     if poly.is_valid and not poly.is_empty:
        #         features.append((poly, int(val)))

        # if features:
        # Rasterize WITHOUT providing a transform
        # This is critical - polygons are already in pixel space
        change_type_mask = rasterio.features.rasterize(
            change_type_polys,
            out_shape=(height, width),
            fill=0,
            dtype=np.uint8,
            all_touched=True,
        )

    # Use transform only when writing to file
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,  # Use transform here for georeferencing the output file
    }

    with rasterio.open(change_type_path, "w", **profile) as dst:
        dst.write(change_type_mask[np.newaxis, :, :])

    status_paths = []
    for img_idx in range(5):
        status_path = os.path.join(sample_dir, f"status_{img_idx}.tif")
        status_mask = np.zeros((height, width), dtype=np.uint8)

        if img_idx in change_status_polys and change_status_polys[img_idx]:
            features = []
            # for poly, val in change_status_polys[img_idx]:
            #     if poly.is_valid and not poly.is_empty:
            #         features.append((poly, int(val)))

            # if features:
            # Rasterize WITHOUT providing a transform
            status_mask = rasterio.features.rasterize(
                change_status_polys[img_idx],
                out_shape=(height, width),
                fill=0,
                dtype=np.uint8,
                all_touched=True,
            )

        with rasterio.open(status_path, "w", **profile) as dst:
            dst.write(status_mask[np.newaxis, :, :])

        status_paths.append(status_path)

    return {"change_type_path": change_type_path, "status_paths": status_paths}


def process_sample_from_masks(args):
    """Process a single QFabric sample from pre-rasterized masks."""
    sample_idx = args["sample_idx"]
    row = args["row"]
    root_dir = args["root_dir"]
    output_dir = args["output_dir"]
    full_masks_dir = args["full_masks_dir"]
    patch_size = args["patch_size"]

    # Paths to full masks
    sample_masks_dir = os.path.join(full_masks_dir, f"sample_{sample_idx}")
    change_type_path = os.path.join(sample_masks_dir, "change_type.tif")
    status_paths = [os.path.join(sample_masks_dir, f"status_{i}.tif") for i in range(5)]

    # Paths to images
    image_paths = []
    for i in range(5):
        img_path = os.path.join(root_dir, "dataset", row[f"img_{i}_file"])
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

        # Precompute the windows
        is_training = row.get("split") == "train"
        overlap = 0.1 if is_training else 0.0
        windows, positions = create_centered_patch_windows(
            height, width, patch_size, overlap=overlap
        )

    patches_info = []

    # Create output directories
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

    for window_idx, (window, position) in enumerate(zip(windows, positions)):
        patch_id = f"{sample_idx}_{position[0]}_{position[1]}"
        window_transform = rasterio.windows.transform(window, transform)

        # Read mask patches
        with rasterio.open(change_type_path) as src:
            change_type_mask = src.read(1, window=window)
            change_type_ratio = np.count_nonzero(change_type_mask) / (
                window.height * window.width
            )

        status_masks = []
        status_ratios = {}
        for img_idx, status_path in enumerate(status_paths):
            with rasterio.open(status_path) as src:
                status_mask = src.read(1, window=window)
                status_masks.append(status_mask)
                status_ratios[img_idx] = np.count_nonzero(status_mask) / (
                    window.height * window.width
                )

        # Read image patches
        images_data = []
        for img_path in image_paths:
            with rasterio.open(img_path) as src:
                patch_data = src.read(window=window)
                images_data.append(patch_data)

        # Save image patches
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

        # Save change type mask
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

        # Save status masks
        status_mask_paths = []
        for img_idx, status_mask in enumerate(status_masks):
            status_path = os.path.join(
                output_dir, "masks", f"{patch_id}_status_{img_idx}.tif"
            )

            with rasterio.open(status_path, "w", **mask_profile) as dst:
                dst.write(status_mask[np.newaxis, :, :])

            status_mask_paths.append(status_path)

        patch_info = {
            "patch_id": patch_id,
            "window": str(window),
            "position": str(position),
            "image_paths": image_patch_paths,
            "change_type_path": change_type_path,
            "status_paths": status_mask_paths,
            "change_type_ratio": change_type_ratio,
            "split": row["split"],
        }

        # Add individual status ratio columns
        for img_idx, ratio in status_ratios.items():
            patch_info[f"status_ratio_{img_idx}"] = ratio

        patches_info.append(patch_info)

    return {
        "sample_idx": sample_idx,
        "status": "success",
        "patches": patches_info,
        "lon": row.get("lon"),
        "lat": row.get("lat"),
    }


def process_qfabric_dataset(
    metadata_df, root_dir, output_dir, patch_size=1024, num_workers=8
):
    """Process entire QFabric dataset with parallel execution."""
    os.makedirs(output_dir, exist_ok=True)

    # First create full-resolution masks for each sample
    full_masks_dir = os.path.join(output_dir, "full_masks")
    os.makedirs(full_masks_dir, exist_ok=True)

    # Create tasks for full mask creation
    mask_tasks = []
    for idx, row in metadata_df.iterrows():
        mask_tasks.append(
            {
                "sample_idx": idx,
                "row": row,
                "root_dir": root_dir,
                "output_dir": full_masks_dir,
            }
        )

    # create_full_masks(**mask_tasks[0])

    # import pdb
    # pdb.set_trace()

    # Parallelize the full mask rasterization process
    print(f"Creating full masks using {num_workers} processes")
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(
                executor.map(create_full_masks_parallel, mask_tasks),
                total=len(mask_tasks),
                desc="Creating full masks",
            )
        )

    # Now process patches using the pre-rasterized masks
    patch_tasks = []
    for idx, row in metadata_df.iterrows():
        patch_tasks.append(
            {
                "sample_idx": idx,
                "row": row,
                "root_dir": root_dir,
                "output_dir": output_dir,
                "full_masks_dir": full_masks_dir,
                "patch_size": patch_size,
            }
        )

    # Parallelize the patch creation process
    print(f"Processing patches using {num_workers} processes")
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for result in tqdm(
            executor.map(process_sample_from_masks, patch_tasks),
            total=len(patch_tasks),
            desc="Processing QFabric patches",
        ):
            results.append(result)

    # Process results into a DataFrame
    patches_data = []
    for result in results:
        if result["status"] == "success" and "patches" in result:
            metadata_row = metadata_df.loc[result["sample_idx"]]

            for patch in result["patches"]:
                patch_info = {
                    "sample_idx": result["sample_idx"],
                    "patch_id": patch["patch_id"],
                    "lon": result.get("lon"),
                    "lat": result.get("lat"),
                    "change_type_path": patch["change_type_path"],
                    "change_type_ratio": patch["change_type_ratio"],
                    "split": patch["split"],
                }
                patch_info["status_ratio"] = {
                    f"status_ratio_{i}": patch.get(f"status_ratio_{i}", 0.0)
                    for i in range(5)
                }

                patch_info["orig_annotation_path"] = metadata_row["json_file"]

                for i in range(5):
                    date_key = f"img_{i}_date"
                    if date_key in metadata_row:
                        patch_info[f"image_{i}_date"] = metadata_row[date_key]

                for i, img_path in enumerate(patch["image_paths"]):
                    patch_info[f"image_{i}_path"] = img_path

                for i, status_path in enumerate(patch["status_paths"]):
                    patch_info[f"status_{i}_path"] = status_path

                patches_data.append(patch_info)

    patches_df = pd.DataFrame(patches_data)

    # Fix the paths to be relative
    patches_df["change_type_path"] = patches_df["change_type_path"].apply(
        lambda x: x.replace(output_dir, "").lstrip("/")
    )

    for i in range(5):
        patches_df[f"status_{i}_path"] = patches_df[f"status_{i}_path"].apply(
            lambda x: x.replace(output_dir, "").lstrip("/")
        )
        patches_df[f"image_{i}_path"] = patches_df[f"image_{i}_path"].apply(
            lambda x: x.replace(output_dir, "").lstrip("/")
        )

    return patches_df


def create_full_masks_parallel(args):
    """Wrapper for create_full_masks to use with concurrent.futures."""
    return create_full_masks(
        args["sample_idx"], args["row"], args["root_dir"], args["output_dir"]
    )


def create_tortilla(root_dir, df, save_dir, tortilla_name):
    """Create a tortilla version of the dataset."""

    tortilla_dir = os.path.join(save_dir, "tortilla")
    os.makedirs(tortilla_dir, exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating tortilla"):
        modalities = [
            "image_0",
            "image_1",
            "image_2",
            "image_3",
            "image_4",
            "status_0",
            "status_1",
            "status_2",
            "status_3",
            "status_4",
            "change_type",
        ]
        modality_samples = []

        for modality in modalities:
            path = os.path.join(root_dir, row[modality + "_path"])
            with rasterio.open(path) as src:
                profile = src.profile

            if modality.startswith("image_"):
                date = row[modality + "_date"].split(" ")[0]
            else:
                date = row["image_4_date"].split(" ")[0]

            sample = tacotoolbox.tortilla.datamodel.Sample(
                id=modality,
                path=path,
                file_format="GTiff",
                data_split=row["split"],
                stac_data={
                    "crs": "EPSG:" + str(profile["crs"].to_epsg()),
                    "geotransform": profile["transform"].to_gdal(),
                    "raster_shape": (profile["height"], profile["width"]),
                    "time_start": date,
                },
                lon=row["lon"],
                lat=row["lat"],
                patch_id=row["patch_id"],
                orig_sample_idx=row["sample_idx"],
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
            patch_id=sample_data["patch_id"],
        )
        samples.append(sample_tortilla)

    final_samples = tacotoolbox.tortilla.datamodel.Samples(samples=samples)
    tacotoolbox.tortilla.create(
        final_samples, os.path.join(save_dir, tortilla_name), quiet=True, nworkers=4
    )


def create_geobench_version(
    metadata_df: pd.DataFrame,
    n_train_samples: int,
    n_val_samples: int,
    n_test_samples: int,
) -> None:
    """Create a GeoBench version of the dataset.
    Args:
        metadata_df: DataFrame with metadata including geolocation for each patch
        n_train_samples: Number of final training samples, -1 means all
        n_val_samples: Number of final validation samples, -1 means all
        n_test_samples: Number of final test samples, -1 means all
    """
    random_state = 24

    subset_df = create_subset_from_df(
        metadata_df,
        n_train_samples=n_train_samples,
        n_val_samples=n_val_samples,
        n_test_samples=n_test_samples,
        random_state=random_state,
    )

    return subset_df


def visualize_qfabric_patch(
    patches_df, sample_idx, patch_position, output_dir, save_path=None
):
    """Visualize a QFabric patch with three rows: images, status overlays, and change type overlays."""
    import matplotlib.pyplot as plt
    import numpy as np
    import rasterio
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    change_classes = (
        "industrial",
        "commercial",
        "road",
        "residential",
        "demolition",
        "mega projects",
    )
    status_classes = (
        "no change",
        "prior construction",
        "greenland",
        "land cleared",
        "excavation",
        "materials dumped",
        "construction started",
        "construction midway",
        "construction done",
        "operational",
    )

    patch_id = f"{sample_idx}_{patch_position[0]}_{patch_position[1]}"
    patch = patches_df[patches_df["patch_id"] == patch_id].iloc[0]

    fig, axes = plt.subplots(3, 5, figsize=(25, 15))

    # Include transparent color for background (0 class)
    status_colors = [
        "white",
        "yellow",
        "red",
        "green",
        "blue",
        "cyan",
        "magenta",
        "gray",
        "orange",
        "purple",
    ]
    change_colors = ["white", "red", "green", "blue", "yellow", "cyan", "magenta"]

    status_cmap = ListedColormap(status_colors)
    change_cmap = ListedColormap(change_colors)

    images = []
    status_masks = []

    for i in range(5):
        with rasterio.open(os.path.join(output_dir, patch[f"image_{i}_path"])) as src:
            img = src.read([1, 2, 3])
            img = np.transpose(img, (1, 2, 0)).astype(np.float32)

            p2, p98 = np.percentile(img, (2, 98))
            img = np.clip((img - p2) / (p98 - p2), 0, 1)
            images.append(img)

        with rasterio.open(os.path.join(output_dir, patch[f"status_{i}_path"])) as src:
            status_masks.append(src.read(1))

    with rasterio.open(os.path.join(output_dir, patch["change_type_path"])) as src:
        change_mask = src.read(1)

    all_status_values = set()
    for mask in status_masks:
        all_status_values.update(np.unique(mask))

    change_values = set(np.unique(change_mask))

    # Row 1: Display original images
    for i in range(5):
        axes[0, i].imshow(images[i])
        img_date = patch.get(f"image_{i}_date", f"Time {i}")
        if isinstance(img_date, str) and len(img_date) > 10:
            img_date = img_date.split(" ")[0]
        axes[0, i].set_title(f"Image {i} ({img_date})", fontsize=12)
        axes[0, i].axis("off")

    # Row 2: Status masks overlaid on images (one mask per timepoint)
    for i in range(5):
        overlay = images[i].copy()

        # Include all classes including 0 (background) for visualization completeness
        for val in np.unique(status_masks[i]):
            mask = status_masks[i] == val
            # if val == 0:  # Background - use lower alpha for better visibility
            #     color = np.array(status_cmap(val)[0:3])
            #     for c in range(3):
            #         overlay[mask, c] = 0. * overlay[mask, c] + 0.05 * color[c]
            # else:  # Other classes
            color = np.array(status_cmap(val)[0:3])
            for c in range(3):
                overlay[mask, c] = 0.6 * overlay[mask, c] + 0.4 * color[c]

        axes[1, i].imshow(overlay)
        axes[1, i].set_title(f"Status Overlay {i}", fontsize=12)
        axes[1, i].axis("off")

    # Row 3: Change type mask overlaid on all images (same mask for all timepoints)
    for i in range(5):
        overlay = images[i].copy()

        # Include all classes including 0 (background) for visualization completeness
        for val in change_values:
            mask = change_mask == val
            # if val == 0:  # Background - use lower alpha for better visibility
            #     color = np.array(change_cmap(val)[0:3])
            #     for c in range(3):
            #         overlay[mask, c] = 0.95 * overlay[mask, c] + 0.05 * color[c]
            # else:  # Other classes
            color = np.array(change_cmap(val)[0:3])
            for c in range(3):
                overlay[mask, c] = 0.6 * overlay[mask, c] + 0.4 * color[c]

        axes[2, i].imshow(overlay)
        axes[2, i].set_title(f"Change Type Overlay {i}", fontsize=12)
        axes[2, i].axis("off")

    # Create legends including the background (0) class
    status_legend = []
    for val in sorted(all_status_values):
        if val < len(status_classes):
            status_legend.append(
                Patch(
                    facecolor=status_colors[val],
                    edgecolor="black",
                    label=f"{val}: {status_classes[val]}",
                )
            )

    if status_legend:
        axes[1, 4].legend(
            handles=status_legend, loc="center", fontsize=10, title="Status Classes"
        )

    change_legend = []
    for val in sorted(change_values):
        if val < len(change_classes):
            change_legend.append(
                Patch(
                    facecolor=change_colors[val],
                    edgecolor="black",
                    label=f"{val}: {change_classes[val]}",
                )
            )

    if change_legend:
        axes[2, 4].legend(
            handles=change_legend,
            loc="center",
            fontsize=10,
            title="Change Type Classes",
        )

    plt.suptitle(f"QFabric Sample {sample_idx}, Patch {patch_position}", fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    return fig


def visualize_qfabric_patches(
    patches_df, sample_indices, patch_positions, output_dir, save_dir=None
):
    """
    Visualize multiple QFabric patches with proper alignment of images and masks.

    Args:
        patches_df: DataFrame containing patch information
        sample_indices: List of sample indices to visualize
        patch_positions: List of patch positions (list of tuples) to visualize
        output_dir: Directory where processed data is stored
        save_dir: Directory to save visualizations (if None, just displays)
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for sample_idx in sample_indices:
        for position in patch_positions:
            try:
                # Create aligned visualization
                fig = visualize_qfabric_patch(
                    patches_df, sample_idx, position, output_dir
                )

                if save_dir:
                    save_path = os.path.join(
                        save_dir,
                        f"sample_{sample_idx}_patch_{position[0]}_{position[1]}.png",
                    )
                    plt.savefig(save_path, dpi=150, bbox_inches="tight")
                    plt.close(fig)

            except Exception as e:
                print(f"Error visualizing sample {sample_idx}, patch {position}: {e}")


def visualize_qfabric_patching(
    metadata_df, patches_df, sample_idx, root_dir, output_dir, save_path=None
):
    """Visualize how a QFabric tile is split into patches."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import rasterio
    from rasterio.windows import Window
    import numpy as np
    from matplotlib.colors import ListedColormap

    sample_meta = metadata_df[metadata_df["sample_idx"] == sample_idx].iloc[0]
    sample_patches = patches_df[patches_df["sample_idx"] == sample_idx]

    if len(sample_patches) == 0:
        raise ValueError(f"No patches found for sample_idx {sample_idx}")

    fig, axes = plt.subplots(
        3, 3, figsize=(20, 18), gridspec_kw={"width_ratios": [1, 1, 0.8]}
    )

    row_titles = ["Original Tile", "Patches", "Change Masks"]
    time_indices = [0, 2, 4]

    # Get paths to original images
    original_image_paths = []
    for t in time_indices:
        img_file = sample_meta[f"img_{t}_file"]
        img_path = os.path.join(root_dir, "dataset", img_file)
        original_image_paths.append(img_path)

    # Extract original image size to compute windows
    with rasterio.open(original_image_paths[0]) as src:
        height = src.height
        width = src.width
        transform = src.transform

    # Determine patch size from one of the patch files
    patch_size = None
    for _, patch in sample_patches.iterrows():
        img_path = os.path.join(output_dir, patch["image_0_path"])
        with rasterio.open(img_path) as src:
            patch_size = src.height  # Assuming square patches
            break

    if patch_size is None:
        raise ValueError("Could not determine patch size")

    # Extract position info from patch_id (format: "{sample_idx}_{row}_{col}")
    patch_positions = []
    for _, patch in sample_patches.iterrows():
        patch_id_parts = patch["patch_id"].split("_")
        row = int(patch_id_parts[1])
        col = int(patch_id_parts[2])
        patch_positions.append((row, col))

    # Recreate windows based on patch positions
    patch_windows = []
    for position in patch_positions:
        # Compute window parameters based on position
        # This assumes a regular grid with consistent patch size and overlap
        # We'll compute positions using centered approach similar to create_centered_patch_windows
        max_row = max(pos[0] for pos in patch_positions) + 1
        max_col = max(pos[1] for pos in patch_positions) + 1

        # Calculate the effective step size based on image dimensions and patch positions
        step_y = (height - patch_size) / max(1, max_row - 1) if max_row > 1 else 0
        step_x = (width - patch_size) / max(1, max_col - 1) if max_col > 1 else 0

        # Calculate borders (padding)
        border_h = (
            (height - (max_row * patch_size - (max_row - 1) * (patch_size - step_y)))
            // 2
            if step_y > 0
            else 0
        )
        border_w = (
            (width - (max_col * patch_size - (max_col - 1) * (patch_size - step_x)))
            // 2
            if step_x > 0
            else 0
        )

        # Position offsets
        row, col = position
        y = int(border_h + row * step_y)
        x = int(border_w + col * step_x)

        # Adjust if window would go out of bounds
        if y + patch_size > height:
            y = height - patch_size
        if x + patch_size > width:
            x = width - patch_size

        window = Window(x, y, patch_size, patch_size)
        patch_windows.append(window)

    # Get paths to patch images and masks
    patch_image_paths = []
    patch_status_mask_paths = []
    patch_change_type_paths = []

    for _, patch in sample_patches.iterrows():
        paths = []
        for t in time_indices:
            img_path = os.path.join(output_dir, patch[f"image_{t}_path"])
            paths.append(img_path)
        patch_image_paths.append(paths)

        status_paths = []
        for t in time_indices:
            status_path = os.path.join(output_dir, patch[f"status_{t}_path"])
            status_paths.append(status_path)
        patch_status_mask_paths.append(status_paths)

        change_type_path = os.path.join(output_dir, patch["change_type_path"])
        patch_change_type_paths.append(change_type_path)

    # Plot original images with patch outlines
    for i, img_path in enumerate(original_image_paths):
        with rasterio.open(img_path) as src:
            img_data = src.read([1, 2, 3])
            height, width = src.height, src.width

            img_data = np.transpose(img_data, (1, 2, 0))
            img_data = img_data.astype(np.float32)

            p2, p98 = np.percentile(img_data, (2, 98))
            img_data = np.clip((img_data - p2) / (p98 - p2), 0, 1)

            axes[0, i].imshow(img_data)
            axes[0, i].set_title(f"Time {time_indices[i]}", fontsize=14)

            # Draw patch boundaries
            for window, position in zip(patch_windows, patch_positions):
                rect = mpatches.Rectangle(
                    (window.col_off, window.row_off),
                    window.width,
                    window.height,
                    linewidth=2,
                    edgecolor="red",
                    facecolor="none",
                    alpha=0.7,
                )
                axes[0, i].add_patch(rect)

                # Add position label
                axes[0, i].text(
                    window.col_off + window.width // 2,
                    window.row_off + window.height // 2,
                    f"({position[0]}, {position[1]})",
                    color="white",
                    fontsize=10,
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="red", alpha=0.7, pad=2),
                )

            axes[0, i].set_xlim(0, width)
            axes[0, i].set_ylim(height, 0)  # Invert y-axis for correct orientation
            axes[0, i].axis("off")

    # Create mosaic of patches
    max_row = max(pos[0] for pos in patch_positions) + 1
    max_col = max(pos[1] for pos in patch_positions) + 1

    # For each timepoint, create a mosaic of patches
    for t_idx, t in enumerate(range(3)):
        mosaic = np.zeros(
            (max_row * patch_size, max_col * patch_size, 3), dtype=np.float32
        )

        for idx, (paths, position) in enumerate(
            zip(patch_image_paths, patch_positions)
        ):
            img_path = paths[t_idx]  # Get the right timepoint image

            if os.path.exists(img_path):
                with rasterio.open(img_path) as src:
                    patch_data = src.read([1, 2, 3])

                    patch_data = np.transpose(patch_data, (1, 2, 0))
                    p2, p98 = np.percentile(patch_data, (2, 98))
                    patch_data = np.clip((patch_data - p2) / (p98 - p2), 0, 1)

                    row, col = position
                    row_start = row * patch_size
                    col_start = col * patch_size

                    if (
                        row_start + patch_size <= mosaic.shape[0]
                        and col_start + patch_size <= mosaic.shape[1]
                    ):
                        mosaic[
                            row_start : row_start + patch_size,
                            col_start : col_start + patch_size,
                            :,
                        ] = patch_data

        axes[1, t_idx].imshow(mosaic)
        axes[1, t_idx].set_title(
            f"Patch Mosaic - Time {time_indices[t_idx]}", fontsize=14
        )

        # Add grid lines to show patch boundaries
        for r in range(1, max_row):
            axes[1, t_idx].axhline(
                y=r * patch_size, color="white", linestyle="-", alpha=0.5
            )
        for c in range(1, max_col):
            axes[1, t_idx].axvline(
                x=c * patch_size, color="white", linestyle="-", alpha=0.5
            )

        # Add position labels
        for position in patch_positions:
            row, col = position
            axes[1, t_idx].text(
                col * patch_size + patch_size // 2,
                row * patch_size + patch_size // 2,
                f"({row}, {col})",
                color="white",
                fontsize=10,
                ha="center",
                va="center",
                bbox=dict(facecolor="black", alpha=0.5, pad=2),
            )

        axes[1, t_idx].axis("off")

    # Plot change status masks for two timepoints
    for t_idx in range(2):
        mask_mosaic = np.zeros(
            (max_row * patch_size, max_col * patch_size), dtype=np.uint8
        )

        for idx, (paths, position) in enumerate(
            zip(patch_status_mask_paths, patch_positions)
        ):
            status_path = paths[t_idx]

            if os.path.exists(status_path):
                with rasterio.open(status_path) as src:
                    mask_data = src.read(1)

                    row, col = position
                    row_start = row * patch_size
                    col_start = col * patch_size

                    if (
                        row_start + patch_size <= mask_mosaic.shape[0]
                        and col_start + patch_size <= mask_mosaic.shape[1]
                    ):
                        mask_mosaic[
                            row_start : row_start + patch_size,
                            col_start : col_start + patch_size,
                        ] = mask_data

        # Create colormap for status masks
        status_cmap = ListedColormap(
            [
                "brown",
                "yellow",
                "red",
                "green",
                "blue",
                "cyan",
                "magenta",
                "gray",
                "orange",
                "purple",
            ]
        )

        im = axes[2, t_idx].imshow(mask_mosaic, cmap=status_cmap, vmin=0, vmax=9)
        axes[2, t_idx].set_title(
            f"Status Masks - Time {time_indices[t_idx]}", fontsize=14
        )

        # Add grid lines
        for r in range(1, max_row):
            axes[2, t_idx].axhline(
                y=r * patch_size, color="white", linestyle="-", alpha=0.5
            )
        for c in range(1, max_col):
            axes[2, t_idx].axvline(
                x=c * patch_size, color="white", linestyle="-", alpha=0.5
            )

        axes[2, t_idx].axis("off")

    # Plot change type mask
    change_mosaic = np.zeros(
        (max_row * patch_size, max_col * patch_size), dtype=np.uint8
    )

    for idx, (change_path, position) in enumerate(
        zip(patch_change_type_paths, patch_positions)
    ):
        if os.path.exists(change_path):
            with rasterio.open(change_path) as src:
                change_data = src.read(1)

                row, col = position
                row_start = row * patch_size
                col_start = col * patch_size

                if (
                    row_start + patch_size <= change_mosaic.shape[0]
                    and col_start + patch_size <= change_mosaic.shape[1]
                ):
                    change_mosaic[
                        row_start : row_start + patch_size,
                        col_start : col_start + patch_size,
                    ] = change_data

    # Create colormap for change type
    change_cmap = ListedColormap(
        ["brown", "red", "green", "blue", "yellow", "cyan", "magenta"]
    )

    im = axes[2, 2].imshow(change_mosaic, cmap=change_cmap, vmin=0, vmax=6)
    axes[2, 2].set_title("Change Type Masks", fontsize=14)

    # Add grid lines
    for r in range(1, max_row):
        axes[2, 2].axhline(y=r * patch_size, color="white", linestyle="-", alpha=0.5)
    for c in range(1, max_col):
        axes[2, 2].axvline(x=c * patch_size, color="white", linestyle="-", alpha=0.5)

    axes[2, 2].axis("off")

    # Add row titles
    for i, title in enumerate(row_titles):
        fig.text(
            0.01,
            0.75 - i * 0.3,
            title,
            rotation=90,
            fontsize=16,
            ha="center",
            va="center",
            fontweight="bold",
        )

    # Add title
    fig.suptitle(
        f"QFabric Tile Splitting Visualization - Sample {sample_idx}",
        fontsize=18,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0.02, 0, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to {save_path}")

    return fig


def collect_unique_classes(patches_df, output_dir):
    """Collect all unique classes present in the change type and status masks.

    Args:
        patches_df: DataFrame containing patch information
        output_dir: Directory where the processed patches are stored

    Returns:
        Dictionary with unique classes for change type and status masks
    """
    change_type_classes = set()
    status_classes = {i: set() for i in range(5)}  # One set for each timepoint

    # Sample a subset of patches for efficiency (adjust if needed)
    sample_size = min(200, len(patches_df))
    sampled_patches = patches_df.sample(sample_size, random_state=42)

    for _, patch in tqdm(
        sampled_patches.iterrows(),
        total=len(sampled_patches),
        desc="Analyzing mask classes",
    ):
        # Check change type classes
        change_type_path = os.path.join(output_dir, patch["change_type_path"])
        if os.path.exists(change_type_path):
            with rasterio.open(change_type_path) as src:
                change_data = src.read(1)
                unique_values = np.unique(change_data)
                change_type_classes.update(unique_values)

        # Check status classes for each timepoint
        for t in range(5):
            status_path = os.path.join(output_dir, patch[f"status_{t}_path"])
            if os.path.exists(status_path):
                with rasterio.open(status_path) as src:
                    status_data = src.read(1)
                    unique_values = np.unique(status_data)
                    status_classes[t].update(unique_values)

    # Convert sets to sorted lists for better readability
    result = {
        "change_type_classes": sorted(list(change_type_classes)),
        "status_classes": {
            t: sorted(list(classes)) for t, classes in status_classes.items()
        },
    }

    return result


def collect_unique_classes_from_originals(metadata_df, root_dir):
    """Collect all unique classes present in the original annotation files.

    Args:
        metadata_df: DataFrame containing original tile metadata
        root_dir: Root directory for the QFabric dataset

    Returns:
        Dictionary with unique classes for change type and status masks
    """
    change_type_classes = set()
    status_classes = {i: set() for i in range(5)}

    # Sample a subset for efficiency
    sample_size = min(300, len(metadata_df))
    sampled_tiles = metadata_df.sample(sample_size, random_state=42)

    for _, tile in tqdm(
        sampled_tiles.iterrows(),
        total=len(sampled_tiles),
        desc="Analyzing original annotations",
    ):
        json_path = os.path.join(root_dir, COCO_PATH, tile["json_file"])

        # Load and parse annotations
        with open(json_path, "r") as f:
            annotation_data = json.load(f)

        # Extract class information from annotations
        for annotation in annotation_data.get("annotations", []):
            if "properties" in annotation:
                for prop in annotation["properties"]:
                    # Extract change type classes
                    if prop.get("type") == "Change Type" and "labels" in prop:
                        if isinstance(prop["labels"], list) and len(prop["labels"]) > 0:
                            change_type = int(prop["labels"][0])
                            change_type_classes.add(change_type)

                    # Extract status classes for each timepoint
                    if prop.get("type") == "Change Status" and "labels" in prop:
                        if isinstance(prop["labels"], dict):
                            for img_idx, status in prop["labels"].items():
                                if img_idx.isdigit() and int(img_idx) < 5:
                                    if isinstance(status, list) and len(status) > 0:
                                        status_val = int(status[0])
                                        status_classes[int(img_idx)].add(status_val)

    # Convert sets to sorted lists for better readability
    result = {
        "change_type_classes": sorted(list(change_type_classes)),
        "status_classes": {
            t: sorted(list(classes)) for t, classes in status_classes.items()
        },
    }

    return result


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
    if os.path.exists(patches_path):
        patches_df = pd.read_parquet(patches_path)
    else:
        patches_df = process_qfabric_dataset(
            metadata_df.iloc[37:38],
            args.root,
            args.save_dir,
            patch_size=2048,
            num_workers=1,
        )
    patches_df.to_parquet(patches_path)
    class_info_patches = collect_unique_classes(patches_df, args.save_dir)
    print("Unique classes in patches:", class_info_patches)
    class_info_orig = collect_unique_classes_from_originals(
        metadata_df.iloc[37:38], args.root
    )
    print("Unique classes in originals:", class_info_orig)

    samples_to_validate = [37]  # List of sample indices to check
    patch_positions_to_validate = [
        (0, 0),
        (0, 1),
        (3, 4),
        (2, 3),
    ]  # List of (row, col) positions to check

    validation_dir = os.path.join(args.save_dir, "mask_validation")
    visualize_qfabric_patches(
        patches_df=patches_df,
        sample_indices=samples_to_validate,
        patch_positions=patch_positions_to_validate,
        output_dir=args.save_dir,
        save_dir=validation_dir,
    )
    import pdb

    pdb.set_trace()

    # subset_path = os.path.join(args.save_dir, "geobench_qfabric.parquet")
    # if os.path.exists(subset_path):
    #     subset_df = pd.read_parquet(subset_path)
    # else:
    #     subset_df = create_geobench_version(
    #         patches_df, n_train_samples=4000, n_val_samples=1000, n_test_samples=2000
    #     )
    #     subset_df.to_parquet(subset_path)

    sample_to_visualize = 3  # Choose a sample with multiple patches
    viz_path = os.path.join(
        args.save_dir, f"qfabric_patching_sample_{sample_to_visualize}.png"
    )

    visualize_qfabric_patching(
        metadata_df=metadata_df,
        patches_df=patches_df,
        sample_idx=sample_to_visualize,
        root_dir=args.root,
        output_dir=args.save_dir,
        save_path=viz_path,
    )

    tortilla_name = "geobench_qfabric.tortilla"
    create_tortilla(args.save_dir, patches_df, args.save_dir, tortilla_name)

    create_unittest_subset(
        data_dir=args.save_dir,
        tortilla_pattern=tortilla_name,
        test_dir_name="qfabric",
        n_train_samples=2,
        n_val_samples=1,
        n_test_samples=1,
    )
    tort_paths = glob.glob(
        os.path.join(
            "/mnt/rg_climate_benchmark/data/geobenchV2/q_fabric_full", "*.part.tortilla"
        ),
        recursive=True,
    )

    taco = tacoreader.load(tort_paths)


if __name__ == "__main__":
    main()
