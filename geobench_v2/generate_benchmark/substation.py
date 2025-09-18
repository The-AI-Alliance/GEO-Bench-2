"""Generation script for Substation GeoBench subset.

Provides utilities to process raw substation annotations and imagery into the
standard GeoBench format.
"""

import argparse
import glob
import json
import os
import pdb

import h5py
import numpy as np
import pandas as pd
import rasterio
import tacoreader
import tacotoolbox
from pyproj import CRS, Transformer
from rasterio.transform import Affine
from tqdm import tqdm

from geobench_v2.generate_benchmark.utils import create_unittest_subset


def generate_metadata_df(root_dir: str) -> pd.DataFrame:
    """Generate a DataFrame containing metadata for the substation dataset.

    Args:
        root_dir: Root directory of the dataset.

    Returns:
        A DataFrame containing metadata information for images and annotations.
    """
    with open(root_dir + "annotations.json") as f:
        d = json.load(f)

    images_df = pd.DataFrame(d["images"])
    images_df = images_df.rename(columns={"id": "image_id"})

    splits_df = pd.read_csv(root_dir + "substation_meta_splits_full.csv")
    splits_df = splits_df.drop(columns=["id", "index_right"])
    splits_df = splits_df.rename(columns={"image": "file_name"})

    images_df = pd.merge(images_df, splits_df, on=["file_name", "lat", "lon"])

    annotations_df = pd.DataFrame(d["annotations"])

    metadata_df = pd.merge(images_df, annotations_df, on="image_id")

    metadata_df["file_name"] = [
        root_dir.replace("Substation/", "") + x for x in metadata_df["file_name"].values
    ]

    return metadata_df


def generate_random_subsample(metadata_df, n_splits):
    """Generate a random subsample of the metadata DataFrame.

    Args:
        metadata_df: DataFrame containing metadata for the entire dataset.
        n_splits: List containing the number of samples for each split.

    Returns:
        A subsampled DataFrame containing a random selection of images and annotations.
    """
    splits = ["train", "val", "test"]

    metadata_sub_df = pd.DataFrame()

    for n, split in zip(n_splits, splits):
        tmp = metadata_df[metadata_df["split"].values == split]
        image_ids = tmp["image_id"].unique()
        rng = np.random.RandomState(42)
        image_ids_sample = rng.choice(image_ids, size=n, replace=False)
        image_ids_sample = [int(x) for x in list(image_ids_sample)]
        filter = [
            True if x in image_ids_sample else False for x in tmp["image_id"].values
        ]
        tmp = tmp[filter]
        metadata_sub_df = pd.concat([metadata_sub_df, tmp], axis=0)

    metadata_sub_df = metadata_sub_df.reset_index(drop=True)

    metadata_sub_df["split"].values[metadata_sub_df["split"].values == "val"] = (
        "validation"
    )

    return metadata_sub_df


def save_image_tiff(image_path, lat, lon, output_folder):
    """Save image as GeoTIFF file.

    Args:
        image_path: Path to the input image file.
        lat: Latitude of the image centroid.
        lon: Longitude of the image centroid.
        output_folder: Directory to save the output GeoTIFF file.

    Returns:
        A tuple containing the TIFF profile and the path to the saved GeoTIFF file.
    """
    image_data = np.load(image_path)["arr_0"]
    image_data = np.mean(image_data, axis=0)
    utm_zone = int((lon + 180) / 6) + 1
    utm_crs = CRS.from_epsg(32600 + utm_zone)

    transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    centroid_x, centroid_y = transformer.transform(lon, lat)

    rows, cols = image_data.shape[-2:]
    resolution = 10

    x_top_left = centroid_x - (cols * resolution) / 2
    y_top_left = centroid_y + (rows * resolution) / 2

    transform = Affine(resolution, 0, x_top_left, 0, -resolution, y_top_left)

    profile = {
        "driver": "GTiff",
        "height": rows,
        "width": cols,
        "count": image_data.shape[0],
        "dtype": np.int16,
        "crs": utm_crs,
        "transform": transform,
        "nodata": None,
    }

    new_image_path = output_folder + image_path.split("/")[-1].replace(".npz", ".tiff")

    with rasterio.open(new_image_path, "w", **profile) as dst:
        for i in range(image_data.shape[0]):
            dst.write(image_data[i].astype(np.int16), i + 1)

    return profile, new_image_path


def create_tortilla(metadata_df, save_dir, tortilla_name):
    """Create a tortilla version of an object detection dataset.

    Args:
        metadata_df: DataFrame with annotations including image_path, label, bbox coordinates
        save_dir: Directory to save the tortilla files
        tortilla_name: Name of the final tortilla file
    """
    tortilla_dir = os.path.join(save_dir, "tortilla/")
    os.makedirs(tortilla_dir, exist_ok=True)

    unique_images = metadata_df["file_name"].unique()

    for idx, image_path in enumerate(tqdm(unique_images, desc="Creating tortillas")):
        tiff_profile, new_image_path = save_image_tiff(
            image_path,
            metadata_df["lat"].values[idx],
            metadata_df["lon"].values[idx],
            tortilla_dir,
        )

        img_annotations = metadata_df[metadata_df["file_name"] == image_path]

        boxes = []
        for _, ann in img_annotations.iterrows():
            boxes.append(
                {
                    "category_id": ann["category_id"],
                    "bbox": ann["bbox"],
                    "bbox_mode": "xywh",
                    "mask": ann["segmentation"],
                }
            )

        first_row = img_annotations.iloc[0]
        split = first_row["split"]
        if split == "val":
            split = "validation"
        lon = first_row["lon"] if not pd.isna(first_row["lon"]) else None
        lat = first_row["lat"] if not pd.isna(first_row["lat"]) else None

        annotations_file = os.path.join(
            tortilla_dir,
            f"{os.path.splitext(new_image_path.split('/')[-1])[0]}_annotations.HDF5",
        )

        with h5py.File(annotations_file, "w") as f:
            f.attrs["annotation"] = json.dumps(
                {
                    "sample_annotations": boxes,
                    "image_size": (tiff_profile["height"], tiff_profile["width"]),
                }
            )

        image_sample = tacotoolbox.tortilla.datamodel.Sample(
            id="image",
            path=new_image_path,
            file_format="GTiff",
            data_split=split,
            stac_data={
                "crs": "EPSG:" + str(tiff_profile["crs"].to_epsg())
                if tiff_profile["crs"]
                else None,
                "geotransform": tiff_profile["transform"].to_gdal()
                if tiff_profile["transform"]
                else None,
                "raster_shape": (tiff_profile["height"], tiff_profile["width"]),
                "time_start": "2020",
            },
            lon=lon,
            lat=lat,
        )

        annotations_sample = tacotoolbox.tortilla.datamodel.Sample(
            id="annotations",
            path=annotations_file,
            file_format="HDF5",
            data_split=split,
        )

        taco_samples = tacotoolbox.tortilla.datamodel.Samples(
            samples=[image_sample, annotations_sample]
        )

        sample_path = os.path.join(tortilla_dir, f"sample_{idx}.tortilla")
        tacotoolbox.tortilla.create(taco_samples, sample_path, quiet=True)

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
                "time_start": "2016",
            },
            data_split=sample_data["tortilla:data_split"],
            lon=sample_data.get("lon"),
            lat=sample_data.get("lat"),
        )
        samples.append(sample)

    final_samples = tacotoolbox.tortilla.datamodel.Samples(samples=samples)
    final_path = os.path.join(save_dir, tortilla_name)
    pdb.set_trace()
    tacotoolbox.tortilla.create(final_samples, final_path, quiet=False, nworkers=1)


if __name__ == "__main__":
    """Generate Substation Instance Segmentation Benchmark."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for nz-Cattle dataset"
    )
    parser.add_argument(
        "--save_dir", default="geobenchV2/nzcattle", help="Directory to save the subset"
    )

    args = parser.parse_args()

    metadata_df = generate_metadata_df(args.root)

    metadata_df = generate_random_subsample(metadata_df, [4000, 500, 500])

    tortilla_name = "geobench_substation.tortilla"

    create_tortilla(metadata_df, args.save_dir, tortilla_name=tortilla_name)

    create_unittest_subset(
        data_dir=args.save_dir,
        tortilla_pattern="geobench_substation*tortilla",
        test_dir_name="substation",
        n_train_samples=2,
        n_val_samples=1,
        n_test_samples=1,
    )
