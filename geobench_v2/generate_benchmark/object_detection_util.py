# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Utility functions for processing and visualizing object detection data."""

import multiprocessing
import os
import random
import shutil

import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw
from rasterio.transform import from_origin
from tqdm import tqdm


def process_single_image(args):
    """Process a single image for conversion to GeoTIFF.

    Args:
        args: Tuple containing (img_name, source_dir, target_dir, coords_mapping)

    Returns:
        Dictionary with processing results
    """
    img_name, source_dir, target_dir, coords_mapping = args

    png_path = os.path.join(source_dir, img_name)

    if not os.path.exists(png_path):
        print(f"Warning: PNG file not found for {img_name} at {png_path}")
        return {"img_name": img_name, "success": False}

    base_name = os.path.splitext(img_name)[0]
    geotiff_name = f"{base_name}.tif"
    geotiff_path = os.path.join(target_dir, geotiff_name)

    os.makedirs(os.path.dirname(geotiff_path), exist_ok=True)

    with Image.open(png_path) as img:
        img_array = np.array(img)

        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            count = 3  # RGB
            img_array = img_array.transpose(2, 0, 1)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
            count = 4  # RGBA
            img_array = img_array.transpose(2, 0, 1)
        else:
            count = 1  # Grayscale
            img_array = img_array[np.newaxis, :, :]

        transform = None
        crs = None

        if img_name in coords_mapping:
            lon, lat = coords_mapping[img_name]
            pixel_size = 1.0
            transform = from_origin(lon, lat, pixel_size, pixel_size)
            crs = "EPSG:4326"

        height = img_array.shape[1]
        width = img_array.shape[2]
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": count,
            "dtype": img_array.dtype,
            "interleave": "pixel",
            "blockxsize": width,
            "blockysize": height,
            "compress": "zstd",
            "zstd_level": 13,
            "predictor": 2,
        }

        if transform is not None and crs is not None:
            profile.update({"transform": transform, "crs": crs})

        with rasterio.open(geotiff_path, "w", **profile) as dst:
            dst.write(img_array)

            if img_name in coords_mapping:
                dst.update_tags(lat=lat, lon=lon)

        return {"img_name": img_name, "success": True, "geotiff_path": geotiff_name}


def convert_pngs_to_geotiffs(
    metadata_df: pd.DataFrame,
    source_dir: str,
    target_dir: str,
    image_columns=["image_path"],
    num_workers=8,
):
    """Convert PNG images from multiple modalities to GeoTIFF format in parallel.

    Args:
        metadata_df: DataFrame with annotation data including colony_name, lon, lat if available
        source_dir: Directory containing the resized PNG images
        target_dir: Directory where GeoTIFF files will be saved
        image_columns: List of column names containing image paths to convert (e.g., ["optical_path", "sar_path"])
        num_workers: Number of parallel workers for processing

    Returns:
        Updated DataFrame with paths to GeoTIFF files
    """
    source_dir = os.path.join(source_dir)
    os.makedirs(target_dir, exist_ok=True)

    missing_columns = [col for col in image_columns if col not in metadata_df.columns]
    if missing_columns:
        raise ValueError(f"Image columns not found in metadata: {missing_columns}")

    metadata_df = metadata_df.copy()

    for col in image_columns:
        geotiff_col = f"{col.replace('_path', '')}_geotiff_path"
        if geotiff_col not in metadata_df.columns:
            metadata_df[geotiff_col] = None

    if "lat" not in metadata_df.columns or "lon" not in metadata_df.columns:
        print("No coordinates found in metadata. Skipping geospatial information.")
        metadata_df["lon"] = None
        metadata_df["lat"] = None

    # Process each image column separately to organize by modality
    for col in image_columns:
        print(f"Processing {col} images...")

        modality_name = col.replace("_path", "")
        modality_target_dir = os.path.join(target_dir, modality_name)
        os.makedirs(modality_target_dir, exist_ok=True)

        unique_images = metadata_df[col].dropna().unique()

        image_coords = metadata_df.groupby(col)[["lon", "lat"]].first().reset_index()
        coords_mapping = {
            row[col]: (row["lon"], row["lat"])
            for _, row in image_coords.iterrows()
            if not pd.isna(row["lon"]) and not pd.isna(row["lat"])
        }

        print(
            f"Found coordinates for {len(coords_mapping)} out of {len(unique_images)} {col} images"
        )

        tasks = [
            (img_name, source_dir, modality_target_dir, coords_mapping)
            for img_name in unique_images
        ]

        results = []
        with multiprocessing.Pool(processes=num_workers) as pool:
            for result in tqdm(
                pool.imap_unordered(process_single_image, tasks),
                total=len(unique_images),
                desc=f"Converting {col} to GeoTIFF",
            ):
                results.append(result)

        geotiff_col = f"{modality_name}_geotiff_path"
        for result in results:
            img_name = result["img_name"]
            rel_geotiff_path = os.path.join(modality_name, result["geotiff_path"])

            metadata_df.loc[metadata_df[col] == img_name, geotiff_col] = (
                rel_geotiff_path
            )

    return metadata_df


def visualize_processing_results(df, input_dir, output_dir, num_samples=20, seed=42):
    """Visualize the processing results by showing before and after images with bounding boxes.

    Args:
        df: DataFrame with processing metadata
        input_dir: Root directory of the original dataset
        output_dir: Directory where processed images are saved
        num_samples: Number of random samples to visualize
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    vis_dir = os.path.join(output_dir, "visualizations")
    if os.path.exists(vis_dir):
        shutil.rmtree(vis_dir)
    os.makedirs(vis_dir, exist_ok=True)

    image_groups = df.groupby("original_image")

    unique_images = list(image_groups.groups.keys())

    multi_patch_images = [
        img for img in unique_images if len(image_groups.get_group(img)) > 1
    ]
    single_patch_images = [
        img for img in unique_images if len(image_groups.get_group(img)) == 1
    ]

    num_multi = min(num_samples // 2, len(multi_patch_images))
    num_single = min(num_samples - num_multi, len(single_patch_images))

    selected_multi = (
        random.sample(multi_patch_images, num_multi) if multi_patch_images else []
    )
    selected_single = (
        random.sample(single_patch_images, num_single) if single_patch_images else []
    )

    remaining = num_samples - (num_multi + num_single)
    if remaining > 0:
        if len(multi_patch_images) > num_multi:
            remaining_multi = random.sample(
                [img for img in multi_patch_images if img not in selected_multi],
                min(remaining, len(multi_patch_images) - num_multi),
            )
            selected_multi.extend(remaining_multi)
            remaining -= len(remaining_multi)

        if remaining > 0 and len(single_patch_images) > num_single:
            remaining_single = random.sample(
                [img for img in single_patch_images if img not in selected_single],
                min(remaining, len(single_patch_images) - num_single),
            )
            selected_single.extend(remaining_single)

    selected_images = selected_multi + selected_single

    class_colors = {
        "small-vehicle": (255, 0, 0),  # Red
        "large-vehicle": (0, 255, 0),  # Green
        "ship": (0, 0, 255),  # Blue
        "plane": (255, 255, 0),  # Yellow
        "storage-tank": (255, 0, 255),  # Magenta
        "harbor": (0, 255, 255),  # Cyan
        "bridge": (128, 0, 0),  # Dark Red
        "helicopter": (0, 128, 0),  # Da    # Load annotationsrk Green
        "soccer-ball-field": (0, 0, 128),  # Dark Blue
        "swimming-pool": (128, 128, 0),  # Olive
        "roundabout": (128, 0, 128),  # Purple
        "tennis-court": (0, 128, 128),  # Teal
        "baseball-diamond": (128, 128, 128),  # Gray
        "ground-track-field": (64, 0, 0),  # Brown
        "basketball-court": (0, 64, 0),  # Forest Green
        "container-crane": (0, 0, 64),  # Navy
    }

    default_color = (200, 200, 200)

    for i, orig_img_path in enumerate(selected_images):
        print(f"Visualizing example {i + 1}/{len(selected_images)}: {orig_img_path}")
        patches_df = image_groups.get_group(orig_img_path)
        orig_img_full_path = os.path.join(input_dir, orig_img_path)
        original_image = Image.open(orig_img_full_path).convert("RGB")
        first_record = patches_df.iloc[0]

        annotation_path = os.path.join(
            input_dir,
            first_record["original_image"]
            .replace("/images/", "/annotations/version2.0/")
            .replace(".png", ".txt"),
        )

        orig_img_with_boxes = original_image.copy()
        draw = ImageDraw.Draw(orig_img_with_boxes)

        if os.path.exists(annotation_path):
            with open(annotation_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 9:
                        # get coords and class
                        x1, y1 = float(parts[0]), float(parts[1])
                        x2, y2 = float(parts[2]), float(parts[3])
                        x3, y3 = float(parts[4]), float(parts[5])
                        x4, y4 = float(parts[6]), float(parts[7])
                        class_name = parts[8]
                        color = class_colors.get(class_name, default_color)

                        draw.polygon(
                            [(x1, y1), (x2, y2), (x3, y3), (x4, y4)],
                            outline=color,
                            width=3,
                        )
                        text_position = (min(x1, x2, x3, x4), min(y1, y2, y3, y4) - 10)
                        draw.text(text_position, class_name, fill=color)

        num_patches = len(patches_df)

        if num_patches <= 1:
            vis_width = original_image.width + 512 + 30
            vis_height = max(original_image.height, 512) + 20
            grid_cols = 2
            grid_rows = 1
        elif num_patches <= 4:
            patch_display_size = 512
            vis_width = original_image.width + (2 * patch_display_size) + 40
            vis_height = max(original_image.height, 2 * patch_display_size + 30)
            grid_cols = 3
            grid_rows = 2
        else:
            patch_display_size = 384
            grid_cols = min(4, num_patches)
            grid_rows = (num_patches + grid_cols - 1) // grid_cols + 1
            vis_width = max(original_image.width, grid_cols * patch_display_size + 30)
            vis_height = (
                original_image.height + ((grid_rows - 1) * patch_display_size) + 40
            )

        vis_img = Image.new("RGB", (vis_width, vis_height), (240, 240, 240))

        if num_patches <= 4:
            vis_img.paste(orig_img_with_boxes, (10, 10))
            draw = ImageDraw.Draw(vis_img)
            draw.text(
                (10, 5), f"Original: {os.path.basename(orig_img_path)}", fill=(0, 0, 0)
            )
            if num_patches <= 1:
                patch_x = original_image.width + 20
                patch_y = 10
            else:
                patch_display_size = 512
                start_x = original_image.width + 20
                start_y = 10
                col_spacing = patch_display_size + 10
                row_spacing = patch_display_size + 10
        else:
            orig_width = min(vis_width - 20, original_image.width)
            orig_height = int(
                original_image.height * (orig_width / original_image.width)
            )
            orig_img_with_boxes = orig_img_with_boxes.resize(
                (orig_width, orig_height), Image.Resampling.LANCZOS
            )
            vis_img.paste(orig_img_with_boxes, (10, 10))

            draw = ImageDraw.Draw(vis_img)
            draw.text(
                (10, 5), f"Original: {os.path.basename(orig_img_path)}", fill=(0, 0, 0)
            )

            patch_display_size = 384
            start_x = 10
            start_y = orig_height + 30
            col_spacing = patch_display_size + 10
            row_spacing = patch_display_size + 10

        for j, (_, patch_row) in enumerate(patches_df.iterrows()):
            if num_patches <= 1:
                patch_x = original_image.width + 20
                patch_y = 10
            elif num_patches <= 4:
                patch_x = start_x + (j % 2) * col_spacing
                patch_y = start_y + (j // 2) * row_spacing
            else:
                patch_x = start_x + (j % grid_cols) * col_spacing
                patch_y = start_y + (j // grid_cols) * row_spacing

            processed_img_path = os.path.join(output_dir, patch_row["processed_image"])
            if not os.path.exists(processed_img_path):
                print(f"Warning: Processed image not found: {processed_img_path}")
                continue

            processed_img = Image.open(processed_img_path).convert("RGB")

            processed_label_path = os.path.join(
                output_dir, patch_row["processed_label"]
            )

            if os.path.exists(processed_label_path):
                draw = ImageDraw.Draw(processed_img)
                with open(processed_label_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 9:
                            # coords and class
                            x1, y1 = float(parts[0]), float(parts[1])
                            x2, y2 = float(parts[2]), float(parts[3])
                            x3, y3 = float(parts[4]), float(parts[5])
                            x4, y4 = float(parts[6]), float(parts[7])
                            class_name = parts[8]

                            color = class_colors.get(class_name, default_color)
                            draw.polygon(
                                [(x1, y1), (x2, y2), (x3, y3), (x4, y4)],
                                outline=color,
                                width=3,
                            )
                            text_position = (
                                min(x1, x2, x3, x4),
                                min(y1, y2, y3, y4) - 10,
                            )
                            draw.text(text_position, class_name, fill=color)

            display_patch_size = patch_display_size if num_patches > 1 else 512
            if processed_img.width != display_patch_size:
                processed_img = processed_img.resize(
                    (display_patch_size, display_patch_size), Image.Resampling.LANCZOS
                )

            vis_img.paste(processed_img, (patch_x, patch_y))

            draw = ImageDraw.Draw(vis_img)
            patch_title = f"Patch {patch_row['patch_id']} ({patch_row['strategy']})"
            draw.text((patch_x, patch_y - 15), patch_title, fill=(0, 0, 0))

        vis_filename = f"visualization_{i + 1:02d}_{os.path.basename(orig_img_path).replace('.png', '.jpg')}"
        vis_path = os.path.join(vis_dir, vis_filename)
        vis_img.save(vis_path, quality=90)
        vis_img.close()

    print(f"Saved {len(selected_images)} visualizations to {vis_dir}")
    return vis_dir
