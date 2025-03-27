import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import argparse
from pathlib import Path


def resize_object_detection_dataset(
    image_dir, annotations_df, output_dir, target_size=512
):
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
        except (OSError, IOError, SyntaxError) as e:
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
