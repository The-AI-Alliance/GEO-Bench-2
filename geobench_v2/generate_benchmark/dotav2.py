# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of DOTAV2 dataset."""

import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from geobench_v2.generate_benchmark.object_detection_util import (
    convert_pngs_to_geotiffs,
    visualize_processing_results,
)

Image.MAX_IMAGE_PIXELS = None

# For eliminating near-duplicate patches:

# Increase stride factors (0.8 → 0.9) for less overlap in slightly larger images
# Make the special case condition more aggressive (width_ratio < 1.3 instead of 1.5)
# For better handling of dense annotation images:

# Lower HIGH_DENSITY_THRESHOLD (120 → 80) to trigger the dense annotation case more often
# Increase width_ratio < 1.8 to something like 2.0 to capture more medium-sized images
# For maximizing annotation coverage:

# Lower visibility threshold (0.5 → 0.4) to include more partially visible annotations
# Adjust points_inside >= 3 to points_inside >= 2 for more forgiving inclusion
# For reducing redundant patches:

# Add a filter after window generation to remove highly similar patches (comparing overlap areas)
# Add a max_patches parameter to limit patches per image regardless of size


def generate_metadata_df(root: str) -> pd.DataFrame:
    """Generate Metadata DataFrame for DOTAV2 dataset with improved processing strategy."""
    # Load sample data
    samples_df = pd.read_csv(os.path.join(root, "samples.csv"))
    samples_df = samples_df[samples_df["version"] == 2.0].reset_index(drop=True)

    image_metadata = []
    for idx, row in tqdm(
        samples_df.iterrows(), total=len(samples_df), desc="Extracting image metadata"
    ):
        image_path = os.path.join(root, row["image_path"])
        annotation_path = os.path.join(root, row["annotation_path"])

        # Load annotations to count objects
        annotations = []
        if os.path.exists(annotation_path):
            with open(annotation_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if (
                        len(parts) >= 9
                    ):  # Format: x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult
                        x1, y1 = float(parts[0]), float(parts[1])
                        x2, y2 = float(parts[2]), float(parts[3])
                        x3, y3 = float(parts[4]), float(parts[5])
                        x4, y4 = float(parts[6]), float(parts[7])
                        class_name = parts[8]
                        difficult = int(parts[9]) if len(parts) > 9 else 0

                        annotations.append(
                            {
                                "points": [(x1, y1), (x2, y2), (x3, y3), (x4, y4)],
                                "class_name": class_name,
                                "difficult": difficult,
                            }
                        )

        with Image.open(image_path) as img:
            width, height = img.size

        image_metadata.append(
            {
                "image_path": row["image_path"],
                "annotation_path": row["annotation_path"],
                "split": row["split"],
                "width": width,
                "height": height,
                "num_annotations": len(annotations),
                "annotation_density": len(annotations) / (width * height) * 1e6,
                "aspect_ratio": width / height,
                "annotations": annotations,
            }
        )

    metadata_df = pd.DataFrame(image_metadata)

    patch_size = 1024  # Standard window size

    expanded_records = []

    for idx, row in tqdm(
        metadata_df.iterrows(),
        total=len(metadata_df),
        desc="Generating sliding window patches",
    ):
        width, height = row["width"], row["height"]

        annotation_density = row["annotation_density"]  # Annotations per million pixels
        annotation_count = row["num_annotations"]

        # Define density thresholds
        HIGH_DENSITY_THRESHOLD = 80  # Annotations per million pixels
        MEDIUM_DENSITY_THRESHOLD = 30  # Annotations per million pixels
        HIGH_COUNT_THRESHOLD = 20  # Total number of annotations

        # Decide whether to use sliding window or simple resize
        if (
            max(width, height) <= patch_size
            and annotation_density < MEDIUM_DENSITY_THRESHOLD
            and annotation_count < HIGH_COUNT_THRESHOLD
        ):
            # Small image with low/medium annotation density - just use as is (will be resized later)
            record = row.to_dict()
            record["patch_id"] = 0
            record["patch_coords"] = [0, 0, width, height]
            record["patch_width"] = width
            record["patch_height"] = height
            record["strategy"] = "resize"
            record["keep_aspect_ratio"] = True

            # Original annotations in normalized coordinates
            patch_annotations = []
            for ann in row["annotations"]:
                patch_ann = ann.copy()
                normalized_points = [
                    (px / width, py / height) for px, py in ann["points"]
                ]
                patch_ann["points"] = normalized_points
                patch_annotations.append(patch_ann)

            record["patch_annotations"] = patch_annotations
            record["patch_annotation_count"] = len(patch_annotations)
            expanded_records.append(record)
        else:
            # Either:
            # 1. Image is larger than patch_size, OR
            # 2. Image has high annotation density/count

            # Calculate width/height ratios relative to patch size
            width_ratio = width / patch_size
            height_ratio = height / patch_size

            # For images just slightly larger than patch_size or with high annotation density,
            # use special handling
            if (
                width_ratio < 1.5
                and height_ratio < 1.5
                and annotation_density < HIGH_DENSITY_THRESHOLD
                and annotation_count < HIGH_COUNT_THRESHOLD
            ):
                # Image just slightly larger than patch_size with moderate annotation density
                # If annotations are all concentrated in one area, just take one patch there
                if row["num_annotations"] > 0:
                    # Calculate centroid of all annotations
                    # [existing centroid calculation code]
                    all_points = [
                        point for ann in row["annotations"] for point in ann["points"]
                    ]
                    center_x = sum(p[0] for p in all_points) / len(all_points)
                    center_y = sum(p[1] for p in all_points) / len(all_points)

                    # If annotations are concentrated, use a single patch centered on them
                    # Check concentration by standard deviation
                    std_x = np.std([p[0] for p in all_points])
                    std_y = np.std([p[1] for p in all_points])

                    if std_x < patch_size / 2 and std_y < patch_size / 2:
                        # Annotations are concentrated - use single patch centered on annotations
                        x1 = max(
                            0, min(width - patch_size, int(center_x - patch_size / 2))
                        )
                        y1 = max(
                            0, min(height - patch_size, int(center_y - patch_size / 2))
                        )
                        x2 = min(width, x1 + patch_size)
                        y2 = min(height, y1 + patch_size)

                        window_positions = [(x1, y1)]
                    else:
                        # Annotations are spread out - use corner patches
                        window_positions = [
                            (0, 0),  # Top-left
                            (max(0, width - patch_size), 0),  # Top-right
                            (0, max(0, height - patch_size)),  # Bottom-left
                            (
                                max(0, width - patch_size),
                                max(0, height - patch_size),
                            ),  # Bottom-right
                        ]
                else:
                    # No annotations - just take central patch
                    x1 = max(0, (width - patch_size) // 2)
                    y1 = max(0, (height - patch_size) // 2)
                    window_positions = [(x1, y1)]
            elif (
                width_ratio < 1.8
                and height_ratio < 1.8
                and (
                    annotation_density >= HIGH_DENSITY_THRESHOLD
                    or annotation_count >= HIGH_COUNT_THRESHOLD
                )
            ):
                # For dense annotations on smaller images, use a grid of 512x512 patches
                small_patch_size = 512

                # Calculate grid size based on image dimensions
                grid_width = max(1, (width + small_patch_size - 1) // small_patch_size)
                grid_height = max(
                    1, (height + small_patch_size - 1) // small_patch_size
                )

                # Calculate patch positions with minimal overlap
                window_positions = []

                # Special case: If we need a 2x2 grid but the image is just slightly larger than
                # 2*small_patch_size, use strategic positions instead of a strict grid
                if (
                    grid_width <= 2
                    and grid_height <= 2
                    and width < 2.1 * small_patch_size
                    and height < 2.1 * small_patch_size
                ):
                    # Use strategic positions at the corners
                    window_positions = [
                        (0, 0),  # Top-left
                        (max(0, width - small_patch_size), 0),  # Top-right
                        (0, max(0, height - small_patch_size)),  # Bottom-left
                        (
                            max(0, width - small_patch_size),
                            max(0, height - small_patch_size),
                        ),  # Bottom-right
                    ]
                else:
                    # Use standard grid approach
                    for y_idx in range(grid_height):
                        for x_idx in range(grid_width):
                            # Calculate positions with evenly distributed patches
                            x_stride = (
                                max(
                                    1,
                                    (width - small_patch_size)
                                    // max(1, grid_width - 1),
                                )
                                if grid_width > 1
                                else 0
                            )
                            y_stride = (
                                max(
                                    1,
                                    (height - small_patch_size)
                                    // max(1, grid_height - 1),
                                )
                                if grid_height > 1
                                else 0
                            )

                            # Position the patch
                            if grid_width == 1:
                                x1 = max(
                                    0, (width - small_patch_size) // 2
                                )  # Center horizontally
                            elif x_idx == grid_width - 1:
                                x1 = max(0, width - small_patch_size)  # Rightmost patch
                            else:
                                x1 = min(width - small_patch_size, x_idx * x_stride)

                            if grid_height == 1:
                                y1 = max(
                                    0, (height - small_patch_size) // 2
                                )  # Center vertically
                            elif y_idx == grid_height - 1:
                                y1 = max(0, height - small_patch_size)  # Bottom patch
                            else:
                                y1 = min(height - small_patch_size, y_idx * y_stride)

                            window_positions.append((x1, y1))

                # Override the patch size for this special case
                patch_size = small_patch_size
            else:
                # For larger images or moderate annotation density, use adaptive stride based on image size
                if max(width_ratio, height_ratio) < 2:
                    # Images less than 2x patch size - use minimal overlap (20%)
                    stride_factor = 0.9
                elif max(width_ratio, height_ratio) < 3:
                    # Images 2x-3x patch size - use moderate overlap (33%)
                    stride_factor = 0.8
                else:
                    # Very large images - keep the original 50% overlap
                    stride_factor = 0.75

                stride = int(patch_size * stride_factor)

                # Calculate optimal number of windows in each dimension
                num_windows_x = max(1, int(np.ceil((width - patch_size) / stride)) + 1)
                num_windows_y = max(1, int(np.ceil((height - patch_size) / stride)) + 1)

                # For very large images with many windows, ensure we have enough coverage at the edges
                if num_windows_x > 3 or num_windows_y > 3:
                    # For the last column/row, ensure we reach the edge
                    window_positions = []
                    for y_idx in range(num_windows_y):
                        for x_idx in range(num_windows_x):
                            if x_idx == num_windows_x - 1:
                                x1 = max(0, width - patch_size)
                            else:
                                x1 = min(width - patch_size, x_idx * stride)

                            if y_idx == num_windows_y - 1:
                                y1 = max(0, height - patch_size)
                            else:
                                y1 = min(height - patch_size, y_idx * stride)

                            window_positions.append((x1, y1))
                else:
                    # For moderately sized images, use evenly spaced windows
                    window_positions = []
                    for y_idx in range(num_windows_y):
                        for x_idx in range(num_windows_x):
                            x1 = min(width - patch_size, int(x_idx * stride))
                            y1 = min(height - patch_size, int(y_idx * stride))
                            window_positions.append((x1, y1))

            # Process each window position
            patches_with_annotations = 0
            window_records = []

            for idx, (x1, y1) in enumerate(window_positions):
                x2 = min(width, x1 + patch_size)
                y2 = min(height, y1 + patch_size)

                # Find annotations in this window
                window_annotations = []

                for ann in row["annotations"]:
                    points = ann["points"]

                    # Calculate the bounding box of the oriented box
                    min_x = min(p[0] for p in points)
                    max_x = max(p[0] for p in points)
                    min_y = min(p[1] for p in points)
                    max_y = max(p[1] for p in points)

                    # Check for intersection
                    if not (max_x < x1 or min_x > x2 or max_y < y1 or min_y > y2):
                        # Calculate centroid of the polygon
                        centroid_x = sum(p[0] for p in points) / len(points)
                        centroid_y = sum(p[1] for p in points) / len(points)

                        # Count points inside the window
                        points_inside = sum(
                            1 for px, py in points if x1 <= px <= x2 and y1 <= py <= y2
                        )

                        # Calculate visibility ratio (rough estimate)
                        ann_width = max_x - min_x
                        ann_height = max_y - min_y

                        # Calculate intersection area
                        x_overlap = max(0, min(x2, max_x) - max(x1, min_x))
                        y_overlap = max(0, min(y2, max_y) - max(y1, min_y))
                        overlap_area = x_overlap * y_overlap
                        ann_area = ann_width * ann_height

                        visibility = overlap_area / ann_area if ann_area > 0 else 0

                        # Include annotation if it meets visibility criteria
                        if (
                            visibility >= 0.5
                            or (x1 <= centroid_x <= x2 and y1 <= centroid_y <= y2)
                            or points_inside >= 3
                        ):
                            # Clone the annotation and adjust coordinates
                            patch_ann = ann.copy()

                            # Create normalized points
                            normalized_points = []
                            for px, py in points:
                                rel_px = px - x1
                                rel_py = py - y1

                                norm_px = rel_px / (x2 - x1)
                                norm_py = rel_py / (y2 - y1)

                                normalized_points.append((norm_px, norm_py))

                            patch_ann["points"] = normalized_points
                            patch_ann["visibility"] = visibility
                            window_annotations.append(patch_ann)

                record = row.copy().to_dict()
                record["patch_id"] = idx
                record["patch_coords"] = [x1, y1, x2, y2]
                record["patch_width"] = x2 - x1
                record["patch_height"] = y2 - y1
                record["patch_annotations"] = window_annotations
                record["patch_annotation_count"] = len(window_annotations)
                record["strategy"] = "sliding_window"
                record["keep_aspect_ratio"] = True
                record["total_windows"] = len(window_positions)
                window_records.append((record, len(window_annotations)))

                if len(window_annotations) > 0:
                    patches_with_annotations += 1

            # Remove redundant windows if we have enough with annotations
            if patches_with_annotations >= 2:
                # Sort windows by annotation count (descending)
                window_records.sort(key=lambda x: x[1], reverse=True)

                # Take windows with the most annotations
                for record, _ in window_records:
                    if record["patch_annotation_count"] > 0:
                        expanded_records.append(record)
            else:
                # Keep all windows, including those without annotations
                for record, _ in window_records:
                    expanded_records.append(record)

            # If we didn't find ANY annotations in ANY windows, include at least one window
            if patches_with_annotations == 0 and len(window_records) == 0:
                # Take central window
                x1 = max(0, (width - patch_size) // 2)
                y1 = max(0, (height - patch_size) // 2)
                x2 = min(width, x1 + patch_size)
                y2 = min(height, y1 + patch_size)

                record = row.copy().to_dict()
                record["patch_id"] = 0
                record["patch_coords"] = [x1, y1, x2, y2]
                record["patch_width"] = x2 - x1
                record["patch_height"] = y2 - y1
                record["patch_annotations"] = []
                record["patch_annotation_count"] = 0
                record["strategy"] = "sliding_window"
                record["keep_aspect_ratio"] = True
                record["total_windows"] = 1
                expanded_records.append(record)

    expanded_df = pd.DataFrame(expanded_records)

    # Calculate statistics
    total_patches = len(expanded_df)
    total_annotations = expanded_df["patch_annotation_count"].sum()
    empty_patches = len(expanded_df[expanded_df["patch_annotation_count"] == 0])
    sliding_window_patches = len(
        expanded_df[expanded_df["strategy"] == "sliding_window"]
    )
    resize_patches = len(expanded_df[expanded_df["strategy"] == "resize"])

    print(f"Generated {total_patches} patches from {len(metadata_df)} images")
    print(f"- Sliding window patches: {sliding_window_patches}")
    print(f"- Simple resize patches: {resize_patches}")
    print(f"Total annotations across all patches: {total_annotations}")
    print(f"Empty patches: {empty_patches} ({empty_patches / total_patches:.1%})")

    return expanded_df


def process_dotav2_dataset(df, input_dir, output_dir, target_size=512, num_workers=8):
    """Process DOTAV2 dataset according to the determined strategies with parallel processing.

    Args:
        df: DataFrame with processing strategy information
        input_dir: Path to original dataset
        output_dir: Path to save processed dataset
        target_size: Target patch size (default: 512)
        num_workers: Number of parallel workers (default: 8)
    """
    df.loc[df["split"] == "val", "split"] = "test"

    total_samples = len(df)
    train_samples = int(0.7 * total_samples)
    val_samples = int(0.1 * total_samples)

    df["original_image_base"] = df["image_path"].apply(
        lambda x: os.path.basename(x).split(".")[0]
    )

    total_samples = len(df)
    target_val_ratio = 0.10

    train_source_images = df[df["split"] == "train"]["original_image_base"].unique()
    np.random.seed(42)
    np.random.shuffle(train_source_images)

    source_counts = df[df["split"] == "train"].groupby("original_image_base").size()
    total_train_samples = source_counts.sum()

    target_val_samples = int(total_samples * target_val_ratio)

    val_sources = []
    current_val_samples = 0

    for source in train_source_images:
        if current_val_samples < target_val_samples:
            val_sources.append(source)
            current_val_samples += source_counts.get(source, 0)
        else:
            break

    df.loc[
        df["original_image_base"].isin(val_sources) & (df["split"] == "train"), "split"
    ] = "validation"

    split_counts = df["split"].value_counts()
    print("\nSplit distribution:")
    print(
        f"Train: {split_counts.get('train', 0)} samples ({100 * split_counts.get('train', 0) / total_samples:.1f}%)"
    )
    print(
        f"Validation: {split_counts.get('validation', 0)} samples ({100 * split_counts.get('validation', 0) / total_samples:.1f}%)"
    )
    print(
        f"Test: {split_counts.get('test', 0)} samples ({100 * split_counts.get('test', 0) / total_samples:.1f}%)"
    )

    source_split_check = df.groupby("original_image_base")["split"].nunique()
    mixed_sources = source_split_check[source_split_check > 1]
    if len(mixed_sources) > 0:
        print(
            f"Warning: {len(mixed_sources)} source images have patches in multiple splits!"
        )
    else:
        print("All patches from the same source image are in the same split.")

    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)

    def process_row(row_tuple):
        idx, row = row_tuple
        img_path = os.path.join(input_dir, row["image_path"])
        img = Image.open(img_path)
        base_filename = os.path.splitext(os.path.basename(row["image_path"]))[0]

        if row["strategy"] == "resize":
            output_filename = f"{base_filename}.png"
        else:
            output_filename = f"{base_filename}_patch{row['patch_id']:02d}.png"

        output_img_path = os.path.join(output_dir, "images", output_filename)
        output_label_path = os.path.join(
            output_dir, "annotations", f"{os.path.splitext(output_filename)[0]}.txt"
        )

        x1, y1, x2, y2 = row["patch_coords"]
        patch_img = img.crop((x1, y1, x2, y2))

        orig_width, orig_height = patch_img.size

        patch_img = patch_img.resize(
            (target_size, target_size), Image.Resampling.LANCZOS
        )
        patch_img.save(output_img_path, format="PNG", optimize=True)

        with open(output_label_path, "w") as f:
            for ann in row["patch_annotations"]:
                class_name = ann["class_name"]
                difficult = ann.get("difficult", 0)
                target_points = []
                for px_rel, py_rel in ann["points"]:
                    px_abs = px_rel * target_size
                    py_abs = py_rel * target_size
                    target_points.append((px_abs, py_abs))

                # DOTAV2 format: x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult
                coord_str = " ".join([f"{px:.1f} {py:.1f}" for px, py in target_points])
                f.write(f"{coord_str} {class_name} {difficult}\n")

        return {
            "original_image": row["image_path"],
            "processed_image": os.path.join("images", output_filename),
            "processed_label": os.path.join(
                "annotations", f"{os.path.splitext(output_filename)[0]}.txt"
            ),
            "strategy": row["strategy"],
            "patch_id": row["patch_id"],
            "annotation_count": row.get("patch_annotation_count", 0),
            "split": row["split"],
            "original_width": row["width"],
            "original_height": row["height"],
            "patch_width": x2 - x1,
            "patch_height": y2 - y1,
            "scale_factor_x": target_size / (x2 - x1),
            "scale_factor_y": target_size / (y2 - y1),
        }

    total_items = len(df)
    processed_records = []

    print(f"Processing {total_items} images with {num_workers} workers...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_row, (idx, row)): idx for idx, row in df.iterrows()
        }
        with tqdm(total=total_items, desc="Processing images") as pbar:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    processed_records.append(result)
                pbar.update(1)

    processed_df = pd.DataFrame(processed_records)
    processed_df.to_csv(os.path.join(output_dir, "processed_metadata.csv"), index=False)

    print(f"Processed {len(processed_df)} images/patches:")
    print(f"  Train: {len(processed_df[processed_df['split'] == 'train'])}")
    print(f"  Val: {len(processed_df[processed_df['split'] == 'validation'])}")
    print(f"  Test: {len(processed_df[processed_df['split'] == 'test'])}")

    return processed_df


def main():
    """Generate DOTAV2 Benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for DOTAV2 dataset"
    )
    parser.add_argument(
        "--save_dir", default="geobenchV2/DOTAV2", help="Directory to save the subset"
    )
    args = parser.parse_args()

    path = os.path.join(args.root, "geobench_metadata.parquet")
    if os.path.exists(path):
        metadata_df = pd.read_parquet(path)
    else:
        metadata_df = generate_metadata_df(args.root)
        metadata_df.to_parquet(path)

    processed_path = os.path.join(args.save_dir, "geobench_dotav2_processed.parquet")
    if os.path.exists(processed_path):
        processed_df = pd.read_parquet(processed_path)
    else:
        processed_df = process_dotav2_dataset(
            metadata_df, args.root, args.save_dir, target_size=512, num_workers=6
        )
        processed_df.to_parquet(processed_path)

    converted_path = os.path.join(args.save_dir, "geobench_dotav2.parquet")
    if os.path.exists(converted_path):
        converted_df = pd.read_parquet(converted_path)
    else:
        print("Converting PNGs to GeoTIFFs...")
        converted_df = convert_pngs_to_geotiffs(
            processed_df, args.save_dir, args.save_dir, num_workers=16
        )
        subset_df = create_subset_df(
            converted_df,
            n_train_samples=7000,
            n_val_samples=1000,
            n_test_samples=2000,
            random_state=42,
        )
        converted_df.to_parquet(converted_path)

    vis_dir = visualize_processing_results(
        processed_df, args.root, args.save_dir, num_samples=25
    )


if __name__ == "__main__":
    main()
