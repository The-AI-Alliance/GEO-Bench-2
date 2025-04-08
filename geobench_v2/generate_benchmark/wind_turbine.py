# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of Windturbine dataset."""

import os
import argparse
import pandas as pd
from tqdm import tqdm
import glob
from PIL import Image
import numpy as np


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

        with open(label_path, "r") as f:
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


def generate_metadata_df(root_dir: str) -> pd.DataFrame:
    """Generate metadata DataFrame for Windturbine dataset."""

    image_paths = glob.glob(os.path.join(root_dir, "JPEGImages", "*.png"))

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

    # create splits
    df = create_region_based_splits(df)

    return df


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

    # if os.path.exists(metadata_path):
    #     metadata_df = pd.read_parquet(metadata_path)
    # else:
    metadata_df = generate_metadata_df(args.root)
    metadata_df.to_parquet(metadata_path)

    resized_metadata_df = resize_and_save_dataset(metadata_df, args.root, args.save_dir)
    resized_metadata_df.to_parquet(
        os.path.join(args.save_dir, "geobench_wind_turbine.parquet")
    )


if __name__ == "__main__":
    main()
