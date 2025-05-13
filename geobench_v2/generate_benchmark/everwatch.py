# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of EverWatch dataset."""

import argparse
import os

import numpy as np
import pandas as pd

import json
from tqdm import tqdm
from PIL import ImageFile
from PIL import Image
import geopandas as gpd
from shapely.geometry import shape

import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
from PIL import Image
from tqdm import tqdm
import multiprocessing
from functools import partial

from geobench_v2.generate_benchmark.object_detection_util import (
    convert_pngs_to_geotiffs,
)

from geobench_v2.generate_benchmark.utils import create_subset_from_df


import os
import json
import glob
import rasterio
import tacoreader
import tacotoolbox


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
    annot_df_train = pd.read_csv(os.path.join(root, "train.csv"))
    annot_df_train["split"] = "train"
    annot_df_test = pd.read_csv(os.path.join(root, "test.csv"))
    annot_df_test["split"] = "test"
    annot_df = pd.concat([annot_df_train, annot_df_test], ignore_index=True)

    # Filter out invalid boxes
    annot_df = annot_df[
        (annot_df["xmin"] != annot_df["xmax"]) & (annot_df["ymin"] != annot_df["ymax"])
    ].reset_index(drop=True)

    with open(os.path.join(root, "colonies.geojson")) as f:
        colony_data = json.load(f)

    colony_dict = {}
    for feature in colony_data["features"]:
        name = feature["properties"]["Name"].lower()
        geom = shape(feature["geometry"])
        centroid = geom.centroid
        colony_dict[name] = {"geometry": geom, "lon": centroid.x, "lat": centroid.y}

    def match_colony(image_name):
        """Match image names to colony names in geojson.

        Handles both named colonies (e.g., "horus_04_27_2022_361.png")
        and numeric filenames (e.g., "46552351.png").

        Args:
            image_name: Image filename to match

        Returns:
            Colony name or None if no match found
        """
        basename = os.path.splitext(image_name.lower())[0]

        for colony_name in colony_dict:
            if basename.startswith(colony_name):
                return colony_name

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

        for prefix, colony in numeric_colonies.items():
            if basename.startswith(prefix):
                return colony

        if basename in numeric_colonies:
            return numeric_colonies[basename]

        return None

    unique_images = set(annot_df["image_path"])

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

    colony_info = annot_df["image_path"].apply(get_colony_info)
    annot_df = pd.concat([annot_df, colony_info], axis=1)

    train_indices = annot_df[annot_df["split"] == "train"].index

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
        val_size = int(len(train_indices) * 0.1)
        val_indices = np.random.choice(train_indices, val_size, replace=False)
        annot_df.loc[val_indices, "split"] = "val"

    print(f"Split counts: {annot_df['split'].value_counts().to_dict()}")
    print(
        f"Colonies matched: {annot_df['colony_name'].notna().sum()} of {len(annot_df)} annotations"
    )

    return annot_df


def process_image(args):
    """Process a single image for parallel processing.

    Args:
        args: Tuple containing (img_name, image_dir, output_dir, target_size, annotations_subset)

    Returns:
        Dictionary with processing results
    """
    img_name, image_dir, output_dir, target_size, annotations_subset = args

    image_path = os.path.join(image_dir, img_name)
    if not os.path.exists(image_path):
        return {"status": "missing", "img_name": img_name}

    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img = Image.open(image_path)
        img.load()

        orig_width, orig_height = img.size
        img_resized = img.resize((target_size, target_size), Image.Resampling.LANCZOS)

        output_path = os.path.join(output_dir, "images", img_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img_resized.save(output_path)

        scale_x = target_size / orig_width
        scale_y = target_size / orig_height

        img_annotations = []

        for _, ann in annotations_subset.iterrows():
            xmin_scaled = int(ann["xmin"] * scale_x)
            ymin_scaled = int(ann["ymin"] * scale_y)
            xmax_scaled = int(ann["xmax"] * scale_x)
            ymax_scaled = int(ann["ymax"] * scale_y)

            if (xmax_scaled - xmin_scaled < 3) or (ymax_scaled - ymin_scaled < 3):
                continue

            img_annotations.append(
                {
                    "image_path": img_name,
                    "label": ann["label"],
                    "xmin": xmin_scaled,
                    "ymin": ymin_scaled,
                    "xmax": xmax_scaled,
                    "ymax": ymax_scaled,
                    "colony_name": ann["colony_name"],
                    "lon": ann["lon"],
                    "lat": ann["lat"],
                    "split": ann["split"] if "split" in ann else "unknown",
                }
            )

        return {
            "status": "success",
            "img_name": img_name,
            "annotations": img_annotations,
        }

    except (OSError, SyntaxError) as e:
        return {"status": "corrupted", "img_name": img_name, "error": str(e)}


def process_everwatch_dataset(
    image_dir, annotations_df, output_dir, target_size=512, num_workers=None
):
    """Resize all images in the dataset to a target size and adapt annotations (Parallelized).

    Args:
        image_dir (str): Directory containing original images
        annotations_df (pd.DataFrame): DataFrame with annotations
        output_dir (str): Directory to save resized images and annotations
        target_size (int): Target size for both width and height
        num_workers (int): Number of worker processes (defaults to CPU count)
    """
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    unique_images = annotations_df["image_path"].unique()

    # Create task arguments for parallel processing
    tasks = []
    for img_name in unique_images:
        # Extract annotations for this image
        img_annotations = annotations_df[annotations_df["image_path"] == img_name]
        tasks.append((img_name, image_dir, output_dir, target_size, img_annotations))

    resized_annotations = []
    corrupted_images = []

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(process_image, tasks),
                total=len(tasks),
                desc="Resizing images",
            )
        )

    for result in results:
        if result["status"] == "success":
            resized_annotations.extend(result["annotations"])
        elif result["status"] == "corrupted":
            corrupted_images.append(result["img_name"])

    resized_df = pd.DataFrame(resized_annotations)
    resized_df.to_csv(os.path.join(output_dir, "resized_annotations.csv"), index=False)

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


def create_tortilla(annotations_df, image_dir, save_dir, tortilla_name):
    """Create a tortilla version of an object detection dataset.

    Args:
        annotations_df: DataFrame with annotations including image_path, label, bbox coordinates
        image_dir: Directory containing the GeoTIFF images
        save_dir: Directory to save the tortilla files
        tortilla_name: Name of the final tortilla file
    """

    tortilla_dir = os.path.join(save_dir, "tortilla")
    os.makedirs(tortilla_dir, exist_ok=True)

    unique_images = annotations_df["image_path"].unique()

    for idx, img_name in enumerate(tqdm(unique_images, desc="Creating tortillas")):
        img_annotations = annotations_df[annotations_df["image_path"] == img_name]

        geotiff_path = os.path.join(image_dir, img_annotations.iloc[0]["geotiff_path"])

        # Create annotations dictionary in COCO-like format
        boxes = []
        for _, ann in img_annotations.iterrows():
            boxes.append(
                {
                    "category_id": ann["label"],
                    "bbox": [
                        ann["xmin"],
                        ann["ymin"],
                        ann["xmax"] - ann["xmin"],
                        ann["ymax"] - ann["ymin"],
                    ],
                    "bbox_mode": "xywh",
                }
            )

        with rasterio.open(geotiff_path) as src:
            profile = src.profile
            height, width = profile["height"], profile["width"]
            crs = "EPSG:" + str(profile["crs"].to_epsg()) if profile["crs"] else None
            transform = profile["transform"].to_gdal() if profile["transform"] else None

        first_row = img_annotations.iloc[0]
        split = first_row["split"]
        lon = first_row["lon"] if not pd.isna(first_row["lon"]) else None
        lat = first_row["lat"] if not pd.isna(first_row["lat"]) else None

        annotations_file = os.path.join(
            tortilla_dir, f"{os.path.splitext(img_name)[0]}_annotations.json"
        )
        with open(annotations_file, "w") as f:
            json.dump({"boxes": boxes, "image_size": (height, width)}, f)

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
            lon=lon,
            lat=lat,
        )

        # Create annotation part
        annotations_sample = tacotoolbox.tortilla.datamodel.Sample(
            id="annotations",
            path=annotations_file,
            file_format="JSON",
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

    path = os.path.join(args.root, "geobench_metadata_df.parquet")
    metadata_df.to_parquet(path)

    # metadata_df = metadata_df.iloc[:10]
    resized_path = os.path.join(args.save_dir, "geobench_everwatch_resized.parquet")
    if os.path.exists(resized_path):
        resized_df = pd.read_parquet(resized_path)
    else:
        resized_df = process_everwatch_dataset(
            args.root, metadata_df, args.save_dir, target_size=512, num_workers=16
        )
        resized_df.to_parquet(resized_path)

    final_path = os.path.join(args.save_dir, "geobench_everwatch.parquet")
    if os.path.exists(final_path):
        final_df = pd.read_parquet(final_path)
    else:
        final_df = convert_pngs_to_geotiffs(
            resized_df, args.save_dir, args.save_dir, num_workers=16
        )
        final_df.to_parquet(final_path)

    import pdb

    pdb.set_trace()

    tortilla_name = "geobench_everwatch.tortilla"
    create_tortilla(final_df, args.save_dir, args.save_dir, tortilla_name=tortilla_name)


if __name__ == "__main__":
    main()
