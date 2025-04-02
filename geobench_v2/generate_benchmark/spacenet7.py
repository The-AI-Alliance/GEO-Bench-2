# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of SpaceNet7 dataset."""

from torchgeo.datasets import SpaceNet7
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
import glob
import matplotlib.pyplot as plt
from rasterio.features import rasterize


from geobench_v2.generate_benchmark.geospatial_split_utils import (
    show_samples_per_valid_ratio,
    split_geospatial_tiles_into_patches,
    visualize_checkerboard_pattern,
    visualize_geospatial_split,
    checkerboard_split,
    geographic_buffer_split,
    geographic_distance_split,
    visualize_distance_clusters,
    split_spacenet7_into_patches,
    create_geographic_splits_spacenet7,
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


def create_geobench_ds(
    ds: SpaceNet7, modalities: list[str], metadata_df: pd.DataFrame, save_dir: str
) -> None:
    """Create a subset of SpaceNet7 dataset."""
    os.makedirs(save_dir, exist_ok=True)

    modal_path_dict = {}
    for modality in modalities:
        os.makedirs(os.path.join(save_dir, modality), exist_ok=True)

        ds.image = modality
        images, masks = ds._list_files(ds.aois[0])
        modal_path_dict[modality] = images

    modal_path_dict["mask"] = masks

    patch_size = (512, 512)
    stride = (511, 511)

    patches_df = split_geospatial_tiles_into_patches(
        modal_path_dict=modal_path_dict,
        output_dir=save_dir,
        patch_size=patch_size,
        stride=stride,
        min_valid_data_ratio=0.7,
        min_positive_pixels_ratio=0.01,
    )


def generate_metadata_df(root: str) -> pd.DataFrame:
    """Generate metadata DataFrame for SpaceNet7 dataset."""
    metadata: list[dict[str, str]] = []

    image_paths = glob.glob(os.path.join(root, "**", "images", "*.tif"))

    df = pd.DataFrame(image_paths, columns=["image_path"])

    df["image_masked_path"] = df["image_path"].str.replace(
        "/images/", "/images_masked/"
    )
    df["labels_path"] = (
        df["image_path"]
        .str.replace("/images/", "/labels/")
        .str.replace(".tif", "_Buildings.geojson")
    )
    df["labels_match_path"] = df["labels_path"].str.replace(
        "/labels/", "/labels_match/"
    )
    df["labels_match_pix_path"] = df["labels_path"].str.replace(
        "/labels/", "/labels_match_pix/"
    )

    date_pattern = r"global_monthly_(\d{4})_(\d{2})_mosaic"

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating metadata"):
        image_path = row["image_path"]

        date_match = re.search(date_pattern, os.path.basename(image_path))
        year, month = date_match.groups()
        date = f"{year}-{month}"

        with rasterio.open(image_path) as src:
            lng, lat = src.lnglat()
            height_px, width_px = src.height, src.width

        metadata.append(
            {
                "image_path": image_path,
                "aoi": image_path.split("/")[7],
                "longitude": lng,
                "latitude": lat,
                "date": date,
                "year": year,
                "month": month,
                "height_px": height_px,
                "width_px": width_px,
            }
        )

    metadata_df = pd.DataFrame(metadata)
    full_df = pd.merge(df, metadata_df, on="image_path", how="left")

    # make all the paths relative
    full_df["image_path"] = full_df["image_path"].str.replace(root, "")
    full_df["image_masked_path"] = full_df["image_masked_path"].str.replace(root, "")
    full_df["labels_path"] = full_df["labels_path"].str.replace(root, "")
    full_df["labels_match_path"] = full_df["labels_match_path"].str.replace(root, "")
    full_df["labels_match_pix_path"] = full_df["labels_match_pix_path"].str.replace(
        root, ""
    )

    full_df["split"] = "train"

    return full_df


def create_tortilla(root_dir, df, save_dir):
    """Create a tortilla version of the dataset."""

    # filter by valid_ratio, which is the percent of valid number of pixels in an image
    # df = df[df["valid_ratio"] > 0.4]

    tortilla_dir = os.path.join(save_dir, "tortilla")
    os.makedirs(tortilla_dir, exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating tortilla"):
        modalities = ["images", "mask"]
        modality_samples = []

        for modality in modalities:
            path = os.path.join(root_dir, row[modality + "_path"])
            with rasterio.open(path) as src:
                profile = src.profile

            sample = tacotoolbox.tortilla.datamodel.Sample(
                id=modality,
                path=path,
                file_format="GTiff",
                data_split=row["split"],
                stac_data={
                    "crs": "EPSG:" + str(profile["crs"].to_epsg()),
                    "geotransform": profile["transform"].to_gdal(),
                    "raster_shape": (profile["height"], profile["width"]),
                    "time_start": row["date"],
                },
                lon=row["lon"],
                lat=row["lat"],
                source_img_file=row["source_img_file"],
                source_mask_file=row["source_mask_file"],
                patch_id=row["patch_id"],
                aoi=row["aoi"],
                year=row["year"],
                month=row["month"],
            )

            modality_samples.append(sample)

        taco_samples = tacotoolbox.tortilla.datamodel.Samples(samples=modality_samples)
        samples_path = os.path.join(tortilla_dir, f"sample_{idx}.tortilla")
        tacotoolbox.tortilla.create(taco_samples, samples_path, quiet=True, nworkers=4)

    # merge tortillas into a single dataset"
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
            source_img_file=sample_data["source_img_file"],
            source_mask_file=sample_data["source_mask_file"],
            patch_id=sample_data["patch_id"],
            aoi=sample_data["aoi"],
            year=sample_data["year"],
            month=sample_data["month"],
        )
        samples.append(sample_tortilla)

    # create final taco file
    final_samples = tacotoolbox.tortilla.datamodel.Samples(samples=samples)
    tacotoolbox.tortilla.create(
        final_samples,
        os.path.join(save_dir, "SpaceNet7.tortilla"),
        quiet=True,
        nworkers=4,
    )


def visualize_sample(row, root, output_path):
    """Visualize a sample from the dataset."""
    image_path = row["image_path"]
    image_masked_path = row["image_masked_path"]
    labels_path = row["labels_path"]
    labels_match_path = row["labels_match_path"]
    labels_match_pix_path = row["labels_match_pix_path"]

    with rasterio.open(os.path.join(root, image_path)) as src:
        image = src.read()
        tfm = src.transform
        src_crs = src.crs

    with rasterio.open(os.path.join(root, image_masked_path)) as src:
        image_masked = src.read()

    # labels need to be read with geopandas geojson
    labels = gpd.read_file(os.path.join(root, labels_path))

    if labels.crs != src_crs:
        labels = labels.to_crs(src_crs)

    labels_match = gpd.read_file(os.path.join(root, labels_match_path))
    labels_match_pix = gpd.read_file(os.path.join(root, labels_match_pix_path))

    # create mask from labels

    label_shapes = [(geom, 1) for geom in labels.geometry]
    # import pdb
    # pdb.set_trace()
    label_mask = rasterize(
        label_shapes,
        out_shape=(image.shape[1], image.shape[2]),
        fill=0,  # nodata value
        transform=tfm,
        all_touched=False,
        dtype=np.int64,
    )

    fig, axs = plt.subplots(1, 3, figsize=(20, 4))
    axs[0].imshow(image[:3].transpose(1, 2, 0))
    axs[0].set_title("Image")
    axs[0].axis("off")
    axs[1].imshow(image_masked[:3].transpose(1, 2, 0))
    axs[1].set_title("Masked Image")
    axs[1].axis("off")
    axs[2].imshow(label_mask, cmap="gray")
    axs[2].set_title("Label")
    axs[2].axis("off")
    # axs[3].set_title("Label Match")
    # axs[4].imshow(label_match_pix[0], cmap="gray")
    # axs[4].set_title("Label Match Pix")
    plt.tight_layout()
    plt.savefig(output_path)

    print(f"Saved sample visualization to {output_path}")
    plt.close(fig)


def main():
    """Generate SpaceNet7 Benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for SpaceNet7 dataset"
    )
    parser.add_argument(
        "--save_dir",
        default="geobenchV2/SpaceNet7",
        help="Directory to save the subset",
    )
    args = parser.parse_args()

    metadata_path = os.path.join(args.save_dir, "geobench_spacenet7.parquet")

    # orig_dataset = SpaceNet7(root=args.root, download=False, image="SAR-Intensity")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if os.path.exists(metadata_path):
        metadata_df = pd.read_parquet(metadata_path)
    else:
        metadata_df = generate_metadata_df(args.root)
        metadata_df.to_parquet(metadata_path)

    plot_sample_locations(
        metadata_df,
        os.path.join(args.save_dir, "sample_locations.png"),
        buffer_degrees=1.0,
    )

    visualize_sample(
        metadata_df.iloc[15],
        args.root,
        os.path.join(args.save_dir, "sample_visualization.png"),
    )

    patches_path = os.path.join(args.save_dir, "patch_metadata.parquet")
    if os.path.exists(patches_path):
        patches_df = pd.read_parquet(patches_path)
    else:
        patches_df = split_spacenet7_into_patches(
            args.root,
            metadata_df,
            os.path.join(os.path.dirname(args.root), "patches"),
            patch_size=(512, 512),
        )
        patches_df.to_parquet(os.path.join(args.save_dir, "patch_metadata.parquet"))

    patches_with_split_path = os.path.join(
        args.save_dir, "patch_metadata_split.parquet"
    )
    if os.path.exists(patches_with_split_path):
        patches_df_with_split = pd.read_parquet(patches_with_split_path)
    else:
        patches_df_with_split = create_geographic_splits_spacenet7(
            patches_df, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, random_state=0
        )
        patches_df_with_split.to_parquet(patches_with_split_path)

    plot_sample_locations(
        patches_df_with_split,
        os.path.join(args.save_dir, "sample_locations_split.png"),
        buffer_degrees=1.0,
    )

    # sort by aoi, year, month
    patches_df_with_split = patches_df_with_split.sort_values(
        by=["aoi", "year", "month"]
    ).reset_index(drop=True)

    # convert date to datetime column, assigning first day of month
    patches_df_with_split["date"] = pd.to_datetime(patches_df_with_split["date"])

    patches_df_with_split["split"] = patches_df_with_split["split"].replace(
        "val", "validation"
    )

    create_tortilla(
        os.path.join(os.path.dirname(args.root), "patches"),
        patches_df_with_split,
        os.path.join(args.save_dir, "tortilla"),
    )

    import pdb

    pdb.set_trace()

    taco = tacoreader.load(
        os.path.join(args.save_dir, "tortilla", "SpaceNet7.tortilla")
    )

    # create_tortilla(args.save_dir, checker_split_df, args.save_dir)


if __name__ == "__main__":
    main()
