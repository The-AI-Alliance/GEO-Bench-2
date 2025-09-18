# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate GeoBench Version."""

import argparse
import glob
import multiprocessing as mp
import os
import pickle
import re
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyproj
import rasterio
import tacoreader
import tacotoolbox
from PIL import Image
from rasterio.transform import from_bounds
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from geobench_v2.generate_benchmark.utils import (
    create_subset_from_df,
    create_unittest_subset,
)


def load_metadata(metadata_path: str):
    """Load metadata from CSV file.

    Args:
        metadata_path: Path to the metadata CSV file
    """
    try:
        metadata_df = pd.read_csv(metadata_path, delimiter=";", encoding="latin-1")
        metadata_df.columns = metadata_df.columns.str.strip()
        metadata_df["date"] = pd.to_datetime(
            metadata_df["date"], format="%d.%m.%Y", errors="coerce"
        )

        def parse_bbox(bbox_str):
            pattern = r"BoundingBox\(left=([-\d.]+),\s*bottom=([-\d.]+),\s*right=([-\d.]+),\s*top=([-\d.]+)\)"
            match = re.search(pattern, bbox_str)
            if match:
                return (
                    float(match.group(1)),
                    float(match.group(2)),
                    float(match.group(3)),
                    float(match.group(4)),
                )
            return None, None, None, None

        metadata_df[["left", "bottom", "right", "top"]] = metadata_df[
            "Bounding box coordinates"
        ].apply(lambda x: pd.Series(parse_bbox(x)))
        return metadata_df
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return pd.DataFrame()


def calculate_patch_coordinates(
    img_width,
    img_height,
    patch_x,
    patch_y,
    patch_size,
    bbox_left,
    bbox_bottom,
    bbox_right,
    bbox_top,
    coord_system,
):
    """Calculate the coordinates of the patch center in the image.

    Args:
        img_width: Width of the image
        img_height: Height of the image
        patch_x: X coordinate of the patch
        patch_y: Y coordinate of the patch
        patch_size: Size of the patch
        bbox_left: Left bounding box coordinate
        bbox_bottom: Bottom bounding box coordinate
        bbox_right: Right bounding box coordinate
        bbox_top: Top bounding box coordinate
        coord_system: Coordinate system
    """
    patch_center_x = patch_x + patch_size / 2
    patch_center_y = patch_y + patch_size / 2

    coord_x = bbox_left + (bbox_right - bbox_left) * (patch_center_x / img_width)
    coord_y = bbox_bottom + (bbox_top - bbox_bottom) * (
        1 - (patch_center_y / img_height)
    )

    lat, lon = None, None
    if coord_system.startswith("EPSG:"):
        try:
            import pyproj

            if coord_system != "EPSG:4326":
                projection = pyproj.Transformer.from_crs(
                    coord_system, "EPSG:4326", always_xy=True
                )
                lon, lat = projection.transform(coord_x, coord_y)
            else:
                lon, lat = coord_x, coord_y
        except ImportError:
            pass

    return coord_x, coord_y, lat, lon


def process_files_for_coordinates(
    files, data_split_dir, patch_size, overlap, metadata_df, patch_metadata
):
    """Process files to extract patch coordinates and metadata.

    Args:
        files: List of files to process
        data_split_dir: Directory for the data split (train/val/test)
        patch_size: Size of the patches
        overlap: Overlap between patches
        metadata_df: DataFrame containing metadata
        patch_metadata: Dictionary to store patch metadata
    """
    for file in files:
        file_basename = os.path.basename(file)
        img_name = os.path.splitext(file_basename)[0]

        original_name = img_name.split("__")[0] if "__" in img_name else img_name
        tif_match = metadata_df[metadata_df["image_name"].str.startswith(original_name)]

        if len(tif_match) == 0:
            print(f"WARNING: No metadata match found for {original_name}")
            continue

        img_metadata = tif_match.iloc[0]
        image_date = img_metadata["date"]
        bbox_left = img_metadata["left"]
        bbox_bottom = img_metadata["bottom"]
        bbox_right = img_metadata["right"]
        bbox_top = img_metadata["top"]
        coord_system = img_metadata["Coordinate system"]

        image = Image.open(file.__str__())
        orig_height, orig_width = image.shape

        bottom = patch_size - (orig_height % patch_size)
        bottom = bottom % patch_size
        right = patch_size - (orig_width % patch_size)
        right = right % patch_size

        if overlap > 0:
            bottom = (patch_size - overlap) - (
                (orig_height - patch_size) % (patch_size - overlap)
            )
            right = (patch_size - overlap) - (
                (orig_width - patch_size) % (patch_size - overlap)
            )

        stride = (patch_size - overlap, patch_size - overlap)
        padded_height = orig_height + bottom
        padded_width = orig_width + right

        x_tmp = np.arange(0, padded_height - patch_size + 1, stride[0])
        y_tmp = np.arange(0, padded_width - patch_size + 1, stride[1])

        x_coord, y_coord = np.meshgrid(x_tmp, y_tmp)
        x_coord = x_coord.ravel()
        y_coord = y_coord.ravel()

        for j in range(len(x_coord)):
            patch_x = x_coord[j]
            patch_y = y_coord[j]

            center_x, center_y, lat, lon = calculate_patch_coordinates(
                orig_width,
                orig_height,
                patch_x,
                patch_y,
                patch_size,
                bbox_left,
                bbox_bottom,
                bbox_right,
                bbox_top,
                coord_system,
            )

            add_to_name = f"__{bottom}_{right}_{j}_{patch_x}_{patch_y}.png"
            patch_filename = img_name + add_to_name

            patch_metadata[patch_filename] = {
                "timestamp": image_date.strftime("%Y-%m-%d")
                if not pd.isna(image_date)
                else None,
                "center_x": float(center_x),
                "center_y": float(center_y),
                "latitude": float(lat) if lat is not None else None,
                "longitude": float(lon) if lon is not None else None,
                "coordinate_system": coord_system,
                "original_image": img_metadata["image_name"],
                "glacier_name": img_metadata["glacier_name"].strip()
                if not pd.isna(img_metadata["glacier_name"])
                else None,
                "sensor": img_metadata["sensor"].strip()
                if not pd.isna(img_metadata["sensor"])
                else None,
                "resolution_m": float(img_metadata["resolution (m)"])
                if not pd.isna(img_metadata["resolution (m)"])
                else None,
                "polarization": img_metadata["polarization"].strip()
                if not pd.isna(img_metadata["polarization"])
                else None,
                "data_split": data_split_dir,
                "patch_x": int(patch_x),
                "patch_y": int(patch_y),
                "patch_idx": int(j),
            }


def save_patch_coordinates_only(
    raw_data_dir, patch_size, overlap, overlap_test, overlap_val
):
    """Save the patch coordinates for the images in the dataset.

    Args:
        raw_data_dir: Directory containing the raw data
        patch_size: Size of the patches
        overlap: Overlap between patches
        overlap_test: Overlap for test set
        overlap_val: Overlap for validation set
    """
    patch_metadata = {}

    metadata_df = load_metadata(os.path.join(raw_data_dir, "meta_data.csv"))
    if metadata_df.empty:
        print("ERROR: Failed to load metadata from meta_data.csv")
        return

    for modality_dir in ["sar_images"]:
        for data_split_dir in ["test", "train"]:
            raw_dir_path = os.path.join(raw_data_dir, modality_dir, data_split_dir)
            if not os.path.exists(raw_dir_path):
                print(f"Directory not found: {raw_dir_path}")
                continue

            folder = sorted(Path(raw_dir_path).rglob("*.png"))
            files = [x for x in folder]

            if data_split_dir == "train":
                if not os.path.exists("data_splits"):
                    os.makedirs("data_splits")

                data_idx = np.arange(len(files))
                train_idx, val_idx = train_test_split(
                    data_idx, test_size=0.1, random_state=1
                )

                with open(os.path.join("data_splits", "train_idx.txt"), "wb") as fp:
                    pickle.dump(train_idx, fp)

                with open(os.path.join("data_splits", "val_idx.txt"), "wb") as fp:
                    pickle.dump(val_idx, fp)

                process_files_for_coordinates(
                    [files[i] for i in train_idx],
                    modality_dir,
                    data_split_dir,
                    patch_size,
                    overlap,
                    metadata_df,
                    patch_metadata,
                )

                process_files_for_coordinates(
                    [files[i] for i in val_idx],
                    modality_dir,
                    "val",
                    patch_size,
                    overlap_val,
                    metadata_df,
                    patch_metadata,
                )
            else:
                process_files_for_coordinates(
                    files,
                    modality_dir,
                    data_split_dir,
                    patch_size,
                    overlap_test,
                    metadata_df,
                    patch_metadata,
                )

    patches_df = pd.DataFrame.from_dict(patch_metadata, orient="index").reset_index()
    patches_df.rename(columns={"index": "filename"}, inplace=True)
    return patches_df


def read_png_file(file_path: str) -> np.ndarray:
    """Read PNG file and return as numpy array.

    Args:
        file_path: Path to PNG file

    Returns:
        Numpy array of image data
    """
    with Image.open(file_path) as img:
        array = np.array(img)

        if len(array.shape) == 2:
            return array[np.newaxis, :, :]
        else:
            return np.transpose(array, (2, 0, 1))


def calculate_patch_bounds(
    row: pd.Series, patch_size: int
) -> tuple[float, float, float, float]:
    """Calculate bounds for the patch based on metadata.

    Args:
        row: Row from metadata DataFrame
        patch_size: Size of patch in pixels

    Returns:
        Tuple of (west, south, east, north) bounds
    """
    center_x = row["center_x"]
    center_y = row["center_y"]

    pixel_size = row["resolution_m"]

    half_width = (patch_size / 2) * pixel_size
    half_height = (patch_size / 2) * pixel_size

    west = center_x - half_width
    east = center_x + half_width
    south = center_y - half_height
    north = center_y + half_height

    return west, south, east, north


def remap_mask_values(mask_data: np.ndarray) -> np.ndarray:
    """Remap mask values to sequential class indices.

    Args:
        mask_data: Original mask data

    Returns:
        Remapped mask data
    """
    # Define class mapping
    px_class_values_zones = {
        0: 0,  # 'N/A' -> 0
        64: 1,  # 'rock' -> 1
        127: 2,  # 'glacier' -> 2
        254: 3,  # 'ocean/ice melange' -> 3
    }

    remapped_mask = np.zeros_like(mask_data)

    for orig_val, new_val in px_class_values_zones.items():
        remapped_mask[mask_data == orig_val] = new_val

    return remapped_mask


def process_patch(
    row: pd.Series, input_base_dir: str, output_base_dir: str
) -> dict[str, Any]:
    """Process a single patch from PNG to GeoTIFF.

    Args:
        row: Row from metadata DataFrame
        input_base_dir: Base directory for input PNG files
        output_base_dir: Base directory for output GeoTIFF files
        patch_size: Size of patch in pixels

    Returns:
        Dictionary with processing results and metadata
    """
    img_filename = row["filename"]
    mask_filename = img_filename.replace("__", "_zones__")

    data_split = row["split"]
    img_input_path = os.path.join(
        input_base_dir, "sar_images", data_split, img_filename
    )
    mask_input_path = os.path.join(input_base_dir, "zones", data_split, mask_filename)

    img_output_dir = os.path.join(output_base_dir, "sar_images", data_split)
    mask_output_dir = os.path.join(output_base_dir, "zones", data_split)

    os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(mask_output_dir, exist_ok=True)

    img_basename = os.path.splitext(img_filename)[0]
    mask_basename = os.path.splitext(mask_filename)[0]

    img_output_path = os.path.join(img_output_dir, f"{img_basename}.tif")
    mask_output_path = os.path.join(mask_output_dir, f"{mask_basename}.tif")

    img_data = read_png_file(img_input_path)
    patch_size = img_data.shape[1]
    mask_data = read_png_file(mask_input_path)

    mask_data = remap_mask_values(mask_data)

    west, south, east, north = calculate_patch_bounds(row, patch_size)
    transform = from_bounds(west, south, east, north, patch_size, patch_size)

    crs = row["coordinate_system"]

    img_profile = {
        "driver": "GTiff",
        "height": patch_size,
        "width": patch_size,
        "count": img_data.shape[0],
        "dtype": img_data.dtype,
        "tiled": True,
        "blockxsize": patch_size,
        "blockysize": patch_size,
        "interleave": "pixel",
        "compress": "zstd",
        "zstd_level": 13,
        "predictor": 2,
        "crs": crs,
        "transform": transform,
    }

    mask_profile = img_profile.copy()
    mask_profile["count"] = mask_data.shape[0]
    mask_profile["dtype"] = mask_data.dtype

    with rasterio.open(img_output_path, "w", **img_profile) as dst:
        dst.write(img_data)

    with rasterio.open(mask_output_path, "w", **mask_profile) as dst:
        dst.write(mask_data)

    valid_pixels = np.count_nonzero(mask_data != 0)
    total_pixels = mask_data.size
    valid_ratio = float(valid_pixels) / total_pixels

    lon, lat = row["longitude"], row["latitude"]
    if (
        crs != "EPSG:4326"
        and not pd.isna(row["center_x"])
        and not pd.isna(row["center_y"])
    ):
        if pd.isna(lon) or pd.isna(lat):
            transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(row["center_x"], row["center_y"])

    return {
        "id": img_basename,
        "status": "success",
        "img_output_path": img_output_path,
        "mask_output_path": mask_output_path,
        "longitude": lon,
        "latitude": lat,
        "valid_ratio": valid_ratio,
        "sensor": row["sensor"],
        "glacier_name": row["glacier_name"],
        "split": row["split"],
        "date": row["timestamp"],
    }


def process_patches_parallel(
    metadata_df: pd.DataFrame,
    input_base_dir: str,
    output_base_dir: str,
    num_workers: int = None,
) -> pd.DataFrame:
    """Process patches in parallel.

    Args:
        metadata_df: DataFrame with metadata
        input_base_dir: Base directory for input PNG files
        output_base_dir: Base directory for output GeoTIFF files
        num_workers: Number of workers for parallel processing

    Returns:
        DataFrame with processing results
    """
    os.makedirs(output_base_dir, exist_ok=True)

    process_func = partial(
        process_patch, input_base_dir=input_base_dir, output_base_dir=output_base_dir
    )

    print(f"Processing {len(metadata_df)} patches with {num_workers} workers")

    results = []
    with mp.Pool(num_workers) as pool:
        for result in tqdm(
            pool.imap(process_func, metadata_df.to_dict("records")),
            total=len(metadata_df),
            desc="Converting PNG to GeoTIFF",
        ):
            results.append(result)

    results_df = pd.DataFrame(results)

    print(f"Processed {len(results_df)} patches:")

    results_df["img_output_path"] = results_df["img_output_path"].replace(
        output_base_dir, ""
    )
    results_df["mask_output_path"] = results_df["mask_output_path"].replace(
        output_base_dir, ""
    )

    results_df.drop(columns=["status"], inplace=True)

    results_df["split"] = results_df["split"].replace({"val": "validation"})

    return results_df


def generate_metadata_df(root: str) -> pd.DataFrame:
    """Generate metadata DataFrame for CaFFe dataset with parallel processing.

    Args:
        root: Root directory for CaFFe dataset

    Returns:
        DataFrame with metadata including geolocation for each patch
    """
    df = save_patch_coordinates_only(
        raw_data_dir=os.path.join(root, "data_raw"),
        patch_size=512,
        overlap=0,
        overlap_test=128,
        overlap_val=128,
    )

    # remove the samples under the "test" split that are on the southern hemisphere
    df = df[~((df["data_split"] == "test") & (df["latitude"] < 0))]

    df.rename(columns={"data_split": "split"}, inplace=True)

    # The quality factor (with 1 being the best and 6 the worst)
    def extract_quality_factor(filename):
        try:
            parts = os.path.basename(filename).split("__")[0].split("_")
            if len(parts) >= 5:
                return str(parts[4])
            return None
        except (IndexError, ValueError):
            return None

    df["quality_factor"] = df["filename"].apply(extract_quality_factor)

    # check if the file names exist
    img_root = os.path.join(root, "caffe_processed", "sar_images")

    def check_exist(row):
        img_path = os.path.join(img_root, row["split"], row["filename"])
        if not os.path.exists(img_path):
            return False
        return True

    df["file_exists"] = df.apply(check_exist, axis=1)

    df = df[df["file_exists"]]

    return df


def create_tortilla(root_dir, df, save_dir, tortilla_name):
    """Create a tortilla version of the dataset."""
    tortilla_dir = os.path.join(save_dir, "tortilla")
    os.makedirs(tortilla_dir, exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating tortilla"):
        modalities = ["img", "mask"]
        modality_samples = []

        for modality in modalities:
            path = os.path.join(root_dir, row[modality + "_output_path"])
            with rasterio.open(path) as src:
                profile = src.profile

            crs_str = "EPSG:" + str(profile["crs"].to_epsg())

            stac_data = {
                "crs": crs_str,
                "geotransform": profile["transform"].to_gdal(),
                "raster_shape": (profile["height"], profile["width"]),
                "time_start": row["date"],
            }

            sample = tacotoolbox.tortilla.datamodel.Sample(
                id=modality,
                path=path,
                file_format="GTiff",
                data_split=row["split"],
                add_test_split=row["is_additional_test"],
                stac_data=stac_data,
                lat=row["latitude"],
                lon=row["longitude"],
                sensor=row["sensor"],
                glacier_name=row["glacier_name"],
            )

            modality_samples.append(sample)

        taco_samples = tacotoolbox.tortilla.datamodel.Samples(samples=modality_samples)
        samples_path = os.path.join(tortilla_dir, f"sample_{idx}.tortilla")
        tacotoolbox.tortilla.create(taco_samples, samples_path, quiet=True)

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
            add_test_split=sample_data["add_test_split"],
            lat=sample_data["lat"],
            lon=sample_data["lon"],
            glacier_name=sample_data["glacier_name"],
            sensor=sample_data["sensor"],
        )
        samples.append(sample_tortilla)

    # create final taco file
    final_samples = tacotoolbox.tortilla.datamodel.Samples(samples=samples)
    tacotoolbox.tortilla.create(
        final_samples, os.path.join(save_dir, tortilla_name), quiet=True
    )


def create_geobench_version(
    metadata_df: pd.DataFrame,
    n_train_samples: int,
    n_val_samples: int,
    n_test_samples: int,
    n_additional_test_samples: int = 0,
) -> None:
    """Create a GeoBench version of the dataset.

    Args:
        metadata_df: DataFrame with metadata including geolocation for each patch
        n_train_samples: Number of final training samples, -1 means all
        n_val_samples: Number of final validation samples, -1 means all
        n_test_samples: Number of final test samples, -1 means all
        n_additional_test_samples: Number of additional test samples from training set
    """
    random_state = 24

    subset_df = create_subset_from_df(
        metadata_df,
        n_train_samples=n_train_samples,
        n_val_samples=n_val_samples,
        n_test_samples=n_test_samples,
        n_additional_test_samples=n_additional_test_samples,
        random_state=random_state,
    )

    return subset_df


def main():
    """Generate CaFFe Benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for CaFFe dataset"
    )
    parser.add_argument(
        "--save_dir",
        default="geobenchBenV2/caffe",
        help="Output directory for the benchmark",
    )
    args = parser.parse_args()

    new_metadata_path = os.path.join(args.save_dir, "geobench_metadata.parquet")

    os.makedirs(args.save_dir, exist_ok=True)

    if os.path.exists(new_metadata_path):
        metadata_df = pd.read_parquet(new_metadata_path)
    else:
        metadata_df = generate_metadata_df(args.root)
        metadata_df.to_parquet(new_metadata_path)

    patches_path = os.path.join(args.save_dir, "geobench_caffe.parquet")

    if os.path.exists(patches_path):
        patches_df = pd.read_parquet(patches_path)
    else:
        patches_df = process_patches_parallel(
            metadata_df,
            os.path.join(args.root, "caffe_processed"),
            args.save_dir,
            num_workers=8,
        )
        patches_df = create_geobench_version(
            patches_df,
            n_train_samples=4000,
            n_val_samples=1000,
            n_test_samples=2000,
            n_additional_test_samples=0,
        )
        patches_df.to_parquet(patches_path)

    tortilla_name = "geobench_caffe.tortilla"
    create_tortilla(args.save_dir, patches_df, args.save_dir, tortilla_name)

    create_unittest_subset(
        data_dir=args.save_dir,
        tortilla_pattern=tortilla_name,
        test_dir_name="caffe",
        n_train_samples=4,
        n_val_samples=2,
        n_test_samples=2,
        n_additional_test_samples=0,
    )


if __name__ == "__main__":
    main()
