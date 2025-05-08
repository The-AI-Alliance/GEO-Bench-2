import concurrent.futures
import os
import random
import shutil

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm
from tqdm.auto import tqdm


def process_everwatch_dataset(image_dir, annotations_df, output_dir, target_size=512):
    """Resize all images in the dataset to a target size and adapt annotations.

    Args:
        image_dir (str): Directory containing original images
        annotations_df (pd.DataFrame): DataFrame with annotations
        output_dir (str): Directory to save resized images and annotations
        target_size (int): Target size for both width and height
    """
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    unique_images = annotations_df["image_path"].unique()
    resized_annotations = []
    corrupted_images = []

    for img_name in tqdm(unique_images, desc="Resizing images"):
        image_path = os.path.join(image_dir, img_name)
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found, skipping.")
            continue

        try:
            # Open the image with PIL with error handling for truncated files
            from PIL import ImageFile

            ImageFile.LOAD_TRUNCATED_IMAGES = True

            img = Image.open(image_path)
            # Force load to identify potential issues early
            img.load()

            orig_width, orig_height = img.size
            img_resized = img.resize(
                (target_size, target_size), Image.Resampling.LANCZOS
            )

            output_path = os.path.join(output_dir, "images", img_name)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            img_resized.save(output_path)

            scale_x = target_size / orig_width
            scale_y = target_size / orig_height

            img_annotations = annotations_df[annotations_df["image_path"] == img_name]
            for _, ann in img_annotations.iterrows():
                xmin_scaled = int(ann["xmin"] * scale_x)
                ymin_scaled = int(ann["ymin"] * scale_y)
                xmax_scaled = int(ann["xmax"] * scale_x)
                ymax_scaled = int(ann["ymax"] * scale_y)

                if (xmax_scaled - xmin_scaled < 3) or (ymax_scaled - ymin_scaled < 3):
                    continue

                resized_annotations.append(
                    {
                        "image_path": img_name,
                        "label": ann["label"],
                        "xmin": xmin_scaled,
                        "ymin": ymin_scaled,
                        "xmax": xmax_scaled,
                        "ymax": ymax_scaled,
                        "split": ann["split"] if "split" in ann else "unknown",
                    }
                )
        except (OSError, SyntaxError) as e:
            print(f"Error processing image {image_path}: {str(e)}")
            corrupted_images.append(img_name)
            continue

    resized_df = pd.DataFrame(resized_annotations)
    resized_df.to_csv(os.path.join(output_dir, "resized_annotations.csv"), index=False)

    # Save a list of corrupted images for reference
    if corrupted_images:
        with open(os.path.join(output_dir, "corrupted_images.txt"), "w") as f:
            for img_name in corrupted_images:
                f.write(f"{img_name}\n")
        print(
            f"Warning: {len(corrupted_images)} corrupted images found. List saved to corrupted_images.txt"
        )

    print(
        f"Resized {len(unique_images) - len(corrupted_images)} images to {target_size}x{target_size}"
    )
    print(
        f"Created {len(resized_df)} annotations across {len(resized_df['image_path'].unique())} images"
    )
    return resized_df


def process_dotav2_dataset(df, input_dir, output_dir, target_size=512, num_workers=8):
    """Process DOTAV2 dataset according to the determined strategies with parallel processing.

    Args:
        df: DataFrame with processing strategy information
        input_dir: Path to original dataset
        output_dir: Path to save processed dataset
        target_size: Target patch size (default: 512)
        num_workers: Number of parallel workers (default: 8)
    """
    # datsat only has train/val split
    # rename val to test and shift validation samples from remaining train
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

    # for split in df["split"].unique():
    #     if os.path.exists(os.path.join(output_dir, split)):
    #         shutil.rmtree(os.path.join(output_dir, split))
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
        "helicopter": (0, 128, 0),  # Dark Green
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
