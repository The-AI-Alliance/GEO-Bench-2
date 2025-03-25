# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of EverWatch dataset."""

import pandas as pd
import os
import argparse
from geobench_v2.generate_benchmark.object_detection_util import (
    process_dataset,
    verify_patching,
    compare_resize,
    resize_object_detection_dataset,
)
import numpy as np


def create_subset(metadata_df: pd.DataFrame, save_dir: str) -> None:
    """Create a subset of EverWatch dataset.

    Args:
        ds: EverWatch dataset.
        metadata_df: Metadata DataFrame.
        save_dir: Directory to save the subset.
    """
    # based on the metadata_df create a subset of the dataset and copy it
    # with the same structure to the save_dir
    # TODO
    # image comes in large tiles of 1500x1500 pixels, pass window size of potentially 750x750 and stride of 750 and resize to 512x512?
    # only train/test split so need to create a validation split but no geospatial metadata available, separate train and test csv file
    pass


def generate_metadata_df(root: str) -> pd.DataFrame:
    """Generate metadata DataFrame for EverWatch dataset with geolocation from colonies.geojson.

    Args:
        root: Root directory for EverWatch dataset

    Returns:
        DataFrame with annotations and geo-information
    """
    import json
    import re
    from shapely.geometry import shape, Point

    # Load annotations
    annot_df_train = pd.read_csv(os.path.join(root, "train.csv"))
    annot_df_train["split"] = "train"
    annot_df_test = pd.read_csv(os.path.join(root, "test.csv"))
    annot_df_test["split"] = "test"
    annot_df = pd.concat([annot_df_train, annot_df_test], ignore_index=True)

    # Filter out invalid boxes
    annot_df = annot_df[
        (annot_df["xmin"] != annot_df["xmax"]) & (annot_df["ymin"] != annot_df["ymax"])
    ].reset_index(drop=True)

    # Load colonies GeoJSON
    with open(os.path.join(root, "colonies.geojson"), "r") as f:
        colony_data = json.load(f)

    # Create dictionary mapping colony names to their geometries and centroid coordinates
    colony_dict = {}
    for feature in colony_data["features"]:
        name = feature["properties"]["Name"].lower()
        geom = shape(feature["geometry"])
        centroid = geom.centroid
        colony_dict[name] = {"geometry": geom, "lon": centroid.x, "lat": centroid.y}

    # Function to match image names to colony names
    def match_colony(image_name):
        """Match image names to colony names in geojson.

        Handles both named colonies (e.g., "horus_04_27_2022_361.png")
        and numeric filenames (e.g., "46552351.png").

        Args:
            image_name: Image filename to match

        Returns:
            Colony name or None if no match found
        """
        # Convert to lowercase and remove file extension
        basename = os.path.splitext(image_name.lower())[0]

        # Try direct match with colony names
        for colony_name in colony_dict:
            if basename.startswith(colony_name):
                return colony_name

        # Try pattern matching for common variations
        pattern_mappings = {
            "3bramp": "3b_boat_ramp",
            "6thbridge": "6th_bridge",
            "jupiter": "jupiter",
            "juno": "juno",
            "shamash": "shamash",
            "jetport": "jetport",
            "horus": "horus",
            "lostmans": "lostmans_creek",
        }

        for pattern, colony in pattern_mappings.items():
            if pattern in basename:
                return colony

        # For numeric-only filenames, map the prefix to colony names
        numeric_colonies = {
            "10": "10",
            "1351": "1351",
            "1573": "1573",
            "1824": "1824",
            "1844": "1844",
            "1882": "1882",
            "1888": "1888",
            "2282": "2282",
            "2307": "2307",
            "2309": "2309",
            "2418": "2418",
            "2419": "2419",
            "2647": "2647",
            "2968": "2968",
            "3134": "3134",
            "3235": "3235",
            "3702": "3702",
        }

        # For numeric prefixes like '52928xxx'
        for prefix, colony in numeric_colonies.items():
            if basename.startswith(prefix):
                return colony

        # For exact numeric matches (when the entire filename without extension matches a colony)
        if basename in numeric_colonies:
            return numeric_colonies[basename]

        return None

    # Create a set of unique image names
    unique_images = set(annot_df["image_path"])

    # Create a mapping of image to colony
    image_to_colony = {}
    images_without_match = []

    for img in unique_images:
        colony = match_colony(img)
        if colony:
            image_to_colony[img] = colony
        else:
            images_without_match.append(img)

    if images_without_match:
        print(
            f"Warning: Could not match {len(images_without_match)} images to colony names."
        )
        print("First 5 unmatched images:", images_without_match[:5])

    # Add colony information to the annotations DataFrame
    def get_colony_info(img_name):
        colony = image_to_colony.get(img_name)
        if colony:
            return pd.Series(
                {
                    "colony_name": colony,
                    "lon": colony_dict[colony]["lon"],
                    "lat": colony_dict[colony]["lat"],
                }
            )
        return pd.Series({"colony_name": None, "lon": None, "lat": None})

    # Apply the function to add colony information
    colony_info = annot_df["image_path"].apply(get_colony_info)
    annot_df = pd.concat([annot_df, colony_info], axis=1)

    # Create validation split - stratify by colony if possible
    train_indices = annot_df[annot_df["split"] == "train"].index

    # Use colony_name for stratification if available, otherwise use a simpler approach
    if annot_df["colony_name"].notna().all():
        from sklearn.model_selection import train_test_split

        train_idx, val_idx = train_test_split(
            train_indices,
            test_size=0.1,
            random_state=42,
            stratify=annot_df.loc[train_indices, "colony_name"],
        )
        annot_df.loc[val_idx, "split"] = "val"
    else:
        # Simple approach: take 20% of training data for validation
        val_size = int(len(train_indices) * 0.1)
        val_indices = np.random.choice(train_indices, val_size, replace=False)
        annot_df.loc[val_indices, "split"] = "val"

    print(f"Split counts: {annot_df['split'].value_counts().to_dict()}")
    print(
        f"Colonies matched: {annot_df['colony_name'].notna().sum()} of {len(annot_df)} annotations"
    )

    return annot_df


def create_unit_test_subset() -> None:
    """Create a subset of EverWatch dataset for GeoBench unit tests."""

    # create random images etc that respect the structure of the dataset in minimal format
    pass


def main():
    """Generate EverWatch Benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for EverWatch dataset"
    )
    parser.add_argument(
        "--save_dir",
        default="geobenchV2/everwatch",
        help="Directory to save the subset",
    )
    args = parser.parse_args()

    metadata_df = generate_metadata_df(args.root)

    path = os.path.join(args.root, "geobench_everwatch.parquet")
    metadata_df.to_parquet(path)

    # metadata_df = metadata_df.iloc[:10]
    resize_object_detection_dataset(
        args.root, metadata_df, args.save_dir, target_size=512
    )


if __name__ == "__main__":
    main()
