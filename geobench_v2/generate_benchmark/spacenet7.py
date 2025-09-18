# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of SpaceNet7 dataset."""

import argparse
import glob
import os
import re
from concurrent.futures import ProcessPoolExecutor

import geopandas as gpd
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import tacoreader
import tacotoolbox
from matplotlib.patches import Rectangle
from rasterio.enums import Compression
from rasterio.features import rasterize
from tqdm import tqdm

from geobench_v2.generate_benchmark.utils import (
    create_subset_from_df,
    create_unittest_subset,
    plot_sample_locations,
)


def process_spacenet7_row(args):
    """Process a single SpaceNet7 row into patches.

    Args:
        args: Tuple containing:
            - idx: Row index
            - row: DataFrame row with metadata
            - root_dir: Root directory of the dataset
            - output_dir: Directory to save patches
            - img_dir: Directory for image patches
            - mask_dir: Directory for mask patches
            - patch_size: Size of patches (height, width)
            - blockxsize: Block width for GeoTIFF
            - blockysize: Block height for GeoTIFF

    Returns:
        List of patch metadata dictionaries
    """
    (
        idx,
        row,
        root_dir,
        output_dir,
        img_dir,
        mask_dir,
        patch_size,
        blockxsize,
        blockysize,
    ) = args

    result_metadata = []

    try:
        img_path = os.path.join(root_dir, row["image_path"])
        label_path = os.path.join(root_dir, row["labels_path"])

        img_filename = os.path.basename(img_path)
        img_basename = os.path.splitext(img_filename)[0]

        with rasterio.open(img_path) as img_src:
            image = img_src.read()

            src_height, src_width = img_src.height, img_src.width
            src_crs = img_src.crs
            src_transform = img_src.transform

            gdf = gpd.read_file(label_path)
            if len(gdf) > 0 and not gdf.empty:
                if gdf.crs is None:
                    gdf.set_crs(src_crs, inplace=True)
                elif gdf.crs != src_crs:
                    gdf = gdf.to_crs(src_crs)

                label_shapes = [
                    (geom, 1)
                    for geom in gdf.geometry
                    if geom is not None and not geom.is_empty
                ]

                label_mask = rasterize(
                    label_shapes,
                    out_shape=(src_height, src_width),
                    transform=src_transform,
                    fill=0,
                    dtype=np.uint8,
                    all_touched=False,
                )
                label_mask = label_mask[np.newaxis, :, :]
            else:
                label_mask = np.zeros((1, src_height, src_width), dtype=np.uint8)

        num_patches_h = 2
        num_patches_w = 2

        patch_info_list = []
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                # window position with possible overlap for odd-sized images
                row_start = i * (src_height // num_patches_h)
                if i == 1 and src_height < num_patches_h * patch_size[0]:
                    # second patch gets full size by overlapping
                    row_start = src_height - patch_size[0]

                col_start = j * (src_width // num_patches_w)
                if j == 1 and src_width < num_patches_w * patch_size[1]:
                    # second patch gets full size by overlapping
                    col_start = src_width - patch_size[1]

                # window is within bounds
                row_end = min(row_start + patch_size[0], src_height)
                col_end = min(col_start + patch_size[1], src_width)

                img_patch = image[:, row_start:row_end, col_start:col_end]
                mask_patch = label_mask[:, row_start:row_end, col_start:col_end]

                patch_transform = rasterio.transform.from_origin(
                    src_transform.c + col_start * src_transform.a,
                    src_transform.f + row_start * src_transform.e,
                    src_transform.a,
                    src_transform.e,
                )

                patch_id = f"{img_basename}_p{i}{j}"

                img_patch_path = os.path.join(img_dir, f"{patch_id}.tif")
                mask_patch_path = os.path.join(mask_dir, f"{patch_id}.tif")

                img_profile = {
                    "driver": "GTiff",
                    "compress": Compression.lzw,
                    "interleave": "pixel",
                    "tiled": True,
                    "blockxsize": blockxsize,
                    "blockysize": blockysize,
                    "predictor": 2,
                    "zlevel": 9,
                    "count": img_patch.shape[0],
                    "dtype": img_patch.dtype,
                    "crs": src_crs,
                    "transform": patch_transform,
                    "width": img_patch.shape[2],
                    "height": img_patch.shape[1],
                }

                mask_profile = {
                    "driver": "GTiff",
                    "compress": Compression.lzw,
                    "interleave": "pixel",
                    "tiled": True,
                    "blockxsize": blockxsize,
                    "blockysize": blockysize,
                    "predictor": 2,
                    "zlevel": 9,
                    "count": mask_patch.shape[0],
                    "dtype": mask_patch.dtype,
                    "crs": src_crs,
                    "transform": patch_transform,
                    "width": mask_patch.shape[2],
                    "height": mask_patch.shape[1],
                }

                with rasterio.open(img_patch_path, "w", **img_profile) as dst:
                    dst.write(img_patch)

                with rasterio.open(mask_patch_path, "w", **mask_profile) as dst:
                    dst.write(mask_patch)

                patch_bounds = rasterio.transform.array_bounds(
                    img_patch.shape[1], img_patch.shape[2], patch_transform
                )
                west, south, east, north = patch_bounds
                center_x = (west + east) / 2
                center_y = (north + south) / 2

                lon, lat = center_x, center_y
                if src_crs and not src_crs.is_geographic:
                    from pyproj import Transformer

                    transformer = Transformer.from_crs(
                        src_crs, "EPSG:4326", always_xy=True
                    )
                    lon, lat = transformer.transform(center_x, center_y)

                building_pixels = np.sum(mask_patch > 0)
                total_pixels = mask_patch.size
                building_ratio = building_pixels / total_pixels
                is_positive = building_ratio > 0

                patch_metadata = {
                    "source_img_file": img_filename,
                    "source_mask_file": os.path.basename(label_path),
                    "split": row["split"],
                    "patch_id": patch_id,
                    "images_path": os.path.relpath(img_patch_path, start=output_dir),
                    "mask_path": os.path.relpath(mask_patch_path, start=output_dir),
                    "lon": lon,
                    "lat": lat,
                    "height_px": img_patch.shape[1],
                    "width_px": img_patch.shape[2],
                    "crs": str(src_crs),
                    "row": i,
                    "col": j,
                    "row_px": row_start,
                    "col_px": col_start,
                    "date": row["date"],
                    "year": row["year"],
                    "month": row["month"],
                    "aoi": row["aoi"],
                    "building_ratio": float(building_ratio),
                    "is_positive": is_positive,
                }

                patch_info_list.append(
                    ((img_patch, mask_patch), i, j, row_start, col_start)
                )

                result_metadata.append(patch_metadata)

        return result_metadata

    except Exception as e:
        print(f"Error processing row {idx}: {str(e)}")
        return []


def split_spacenet7_into_patches(
    root_dir: str,
    metadata_df: pd.DataFrame,
    output_dir: str,
    patch_size: tuple[int, int] = (512, 512),
    block_size: tuple[int, int] = (512, 512),
    num_workers: int = 8,
) -> pd.DataFrame:
    """Split SpaceNet7 images and labels into patches of specified size.

    Args:
        root_dir: Root directory of SpaceNet7 dataset
        metadata_df: DataFrame with image/label paths and metadata
        output_dir: Directory to save patches and metadata
        patch_size: Size of patches to create (height, width)
        block_size: Size of the blocks for GeoTIFF tiling (height, width)
        num_workers: Number of parallel processes to use

    Returns:
        DataFrame with metadata for all created patches
    """
    blockxsize, blockysize = block_size
    blockxsize = blockxsize - (blockxsize % 16) if blockxsize % 16 != 0 else blockxsize
    blockysize = blockysize - (blockysize % 16) if blockysize % 16 != 0 else blockysize

    os.makedirs(output_dir, exist_ok=True)

    img_dir = os.path.join(output_dir, "images")
    mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    all_patch_metadata = []

    batch_size = max(1, min(100, len(metadata_df) // (num_workers * 2)))
    batches = [
        metadata_df.iloc[i : i + batch_size]
        for i in range(0, len(metadata_df), batch_size)
    ]

    for batch_idx, batch in enumerate(batches):
        print(f"Processing batch {batch_idx + 1}/{len(batches)}")

        tasks = [
            (
                idx,
                row,
                root_dir,
                output_dir,
                img_dir,
                mask_dir,
                patch_size,
                blockxsize,
                blockysize,
            )
            for idx, row in batch.iterrows()
        ]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(
                tqdm(
                    executor.map(process_spacenet7_row, tasks),
                    total=len(tasks),
                    desc=f"Processing batch {batch_idx + 1}/{len(batches)}",
                )
            )

            # Flatten results
            for result_list in results:
                all_patch_metadata.extend(result_list)

    patches_df = pd.DataFrame(all_patch_metadata)

    metadata_path = os.path.join(output_dir, "patch_metadata.parquet")
    patches_df.to_parquet(metadata_path, index=False)

    print(f"Created {len(patches_df)} patches from {len(metadata_df)} source images")
    print(f"Patch metadata saved to {metadata_path}")

    pos_patches = patches_df[patches_df["is_positive"]]
    neg_patches = patches_df[~patches_df["is_positive"]]
    pos_pct = len(pos_patches) / len(patches_df) * 100 if len(patches_df) > 0 else 0
    neg_pct = len(neg_patches) / len(patches_df) * 100 if len(patches_df) > 0 else 0
    print(f"Positive patches: {len(pos_patches)} ({pos_pct:.1f}%)")
    print(f"Negative patches: {len(neg_patches)} ({neg_pct:.1f}%)")

    return patches_df


def visualize_spacenet7_patches(
    image_data, mask_data, patches_info, output_path=None, figsize=(22, 8)
):
    """Visualize SpaceNet7 original images and their patches.

    Args:
        image_data: Full-sized RGB image data
        mask_data: Full-sized binary mask data
        patches_info: List of tuples (patch_id, row, col, row_px, col_px) for each patch
        output_path: Path to save the visualization (optional)
        figsize: Figure size (width, height)
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 5, figure=fig, wspace=0.05, hspace=0.2)

    patch_colors = ["r", "g", "b", "y"]

    ax_img = fig.add_subplot(gs[0, 0])
    if image_data.shape[0] >= 3:
        img_display = np.stack([image_data[i] for i in range(3)], axis=2)
        if img_display.dtype != np.uint8:
            img_display = np.clip(img_display / np.percentile(img_display, 99), 0, 1)
    else:
        img_display = image_data[0]
        if img_display.dtype != np.uint8:
            img_display = np.clip(img_display / np.percentile(img_display, 99), 0, 1)

    ax_img.imshow(img_display)
    ax_img.set_title("Original Image")
    ax_img.axis("off")

    ax_mask = fig.add_subplot(gs[1, 0])
    mask_display = mask_data[0] if mask_data.shape[0] == 1 else mask_data
    ax_mask.imshow(mask_display, cmap="gray")
    ax_mask.set_title("Original Building Mask")
    ax_mask.axis("off")

    building_pixels = np.sum(mask_display > 0)
    total_pixels = mask_display.size
    building_pct = 100 * building_pixels / total_pixels

    legend_patches = [
        mpatches.Patch(
            color="black",
            label=f"Background: {total_pixels - building_pixels} px ({100 - building_pct:.1f}%)",
        ),
        mpatches.Patch(
            color="white",
            label=f"Buildings: {building_pixels} px ({building_pct:.1f}%)",
        ),
    ]
    ax_mask.legend(
        handles=legend_patches, loc="lower right", fontsize=8, framealpha=0.7
    )

    for i, (patch_id, row, col, row_px, col_px) in enumerate(patches_info[:4]):
        if i == 0:
            patch_height, patch_width = patch_id[0].shape[1], patch_id[0].shape[2]

        ax_img_patch = fig.add_subplot(gs[0, i + 1])
        if patch_id[0].shape[0] >= 3:
            img_patch_display = np.stack([patch_id[0][j] for j in range(3)], axis=2)
            if img_patch_display.dtype != np.uint8:
                img_patch_display = np.clip(
                    img_patch_display / np.percentile(img_patch_display, 99), 0, 1
                )
        else:
            img_patch_display = patch_id[0][0]
            if img_patch_display.dtype != np.uint8:
                img_patch_display = np.clip(
                    img_patch_display / np.percentile(img_patch_display, 99), 0, 1
                )

        ax_img_patch.imshow(img_patch_display)
        ax_img_patch.set_title(
            f"Image Patch ({row},{col})", color=patch_colors[i % len(patch_colors)]
        )
        for spine in ax_img_patch.spines.values():
            spine.set_color(patch_colors[i % len(patch_colors)])
            spine.set_linewidth(3)
        ax_img_patch.axis("off")

        ax_mask_patch = fig.add_subplot(gs[1, i + 1])
        mask_patch_display = patch_id[1][0]
        ax_mask_patch.imshow(mask_patch_display, cmap="gray")
        ax_mask_patch.set_title(
            f"Mask Patch ({row},{col})", color=patch_colors[i % len(patch_colors)]
        )
        for spine in ax_mask_patch.spines.values():
            spine.set_color(patch_colors[i % len(patch_colors)])
            spine.set_linewidth(3)
        ax_mask_patch.axis("off")

        patch_building_pixels = np.sum(mask_patch_display > 0)
        patch_total_pixels = mask_patch_display.size
        patch_building_pct = 100 * patch_building_pixels / patch_total_pixels

        ax_mask_patch.text(
            0.98,
            0.02,
            f"Buildings: {patch_building_pct:.1f}%",
            transform=ax_mask_patch.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="white",
            bbox=dict(facecolor="black", alpha=0.7, boxstyle="round,pad=0.3"),
        )

        rect_img = Rectangle(
            (col_px, row_px),
            patch_width,
            patch_height,
            linewidth=2,
            edgecolor=patch_colors[i % len(patch_colors)],
            facecolor="none",
            alpha=0.8,
        )
        ax_img.add_patch(rect_img)

        ax_img.text(
            col_px + patch_width // 2,
            row_px + patch_height // 2,
            f"{row},{col}",
            color="white",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.5, pad=0.5, boxstyle="round"),
        )

        rect_mask = Rectangle(
            (col_px, row_px),
            patch_width,
            patch_height,
            linewidth=2,
            edgecolor=patch_colors[i % len(patch_colors)],
            facecolor="none",
            alpha=0.8,
        )
        ax_mask.add_patch(rect_mask)

        ax_mask.text(
            col_px + patch_width // 2,
            row_px + patch_height // 2,
            f"{row},{col}",
            color="white",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.5, pad=0.5, boxstyle="round"),
        )

    fig.text(
        0.01,
        0.01,
        f"Overall Building Coverage: {building_pct:.2f}%\nTotal Patches: {len(patches_info)}",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8, boxstyle="round"),
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()

    plt.close(fig)


def create_geographic_splits_spacenet7(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    aoi_col: str = "aoi",
    random_state: int = 42,
) -> pd.DataFrame:
    """Create train/val/test splits by assigning AOIs to maintain geographic separation.

    Args:
        df: DataFrame with patch metadata
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing
        aoi_col: Column name for area of interest
        random_state: Random seed

    Returns:
        DataFrame with added 'split' column
    """
    df_copy = df.copy()

    aoi_counts = df_copy.groupby(aoi_col).size().reset_index(name="count")
    total_samples = aoi_counts["count"].sum()

    target_test = int(total_samples * test_ratio)
    target_val = int(total_samples * val_ratio)

    np.random.seed(random_state)
    aoi_counts = aoi_counts.sample(frac=1, random_state=random_state)

    aoi_counts["split"] = "train"

    curr_test, curr_val = 0, 0

    for idx, row in aoi_counts.iterrows():
        if curr_test < target_test:
            aoi_counts.loc[idx, "split"] = "test"
            curr_test += row["count"]
        elif curr_val < target_val:
            aoi_counts.loc[idx, "split"] = "validation"
            curr_val += row["count"]
        else:
            break

    aoi_split_map = dict(zip(aoi_counts[aoi_col], aoi_counts["split"]))

    df_copy["split"] = df_copy[aoi_col].map(aoi_split_map)

    split_counts = df_copy["split"].value_counts()
    print("\nSplit statistics:")
    for split in ["train", "validation", "test"]:
        if split in split_counts:
            count = split_counts[split]
            pct = 100 * count / len(df_copy)
            print(f"{split}: {count} samples ({pct:.1f}%)")

    df_copy["date"] = pd.to_datetime(df_copy["date"])

    return df_copy


def generate_metadata_df(root: str) -> pd.DataFrame:
    """Generate metadata DataFrame for SpaceNet7 dataset."""
    metadata: list[dict[str, str]] = []

    image_paths = glob.glob(os.path.join(root, "train", "**", "images", "*.tif"))

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
                "aoi": image_path.split(os.sep)[-3],
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
    full_df["image_path"] = (
        full_df["image_path"].str.replace(root, "").str.lstrip(os.sep)
    )
    full_df["image_masked_path"] = (
        full_df["image_masked_path"].str.replace(root, "").str.lstrip(os.sep)
    )
    full_df["labels_path"] = (
        full_df["labels_path"].str.replace(root, "").str.lstrip(os.sep)
    )
    full_df["labels_match_path"] = (
        full_df["labels_match_path"].str.replace(root, "").str.lstrip(os.sep)
    )
    full_df["labels_match_pix_path"] = (
        full_df["labels_match_pix_path"].str.replace(root, "").str.lstrip(os.sep)
    )

    full_df["split"] = "train"

    return full_df


def create_tortilla(root_dir, df, save_dir, tortilla_name):
    """Create a tortilla version of the dataset."""
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
        final_samples, os.path.join(save_dir, tortilla_name), quiet=True, nworkers=4
    )


def visualize_sample(row, root, output_path):
    """Visualize a sample from the dataset.

    Args:
        row: DataFrame row with metadata
        root: Root directory of the dataset
        output_path: Path to save the visualization
    """
    image_path = row["image_path"]
    image_masked_path = row["image_masked_path"]
    labels_path = row["labels_path"]

    with rasterio.open(os.path.join(root, image_path)) as src:
        image = src.read()
        tfm = src.transform
        src_crs = src.crs

    with rasterio.open(os.path.join(root, image_masked_path)) as src:
        image_masked = src.read()

    labels = gpd.read_file(os.path.join(root, labels_path))

    if labels.crs != src_crs:
        labels = labels.to_crs(src_crs)

    label_shapes = [(geom, 1) for geom in labels.geometry]

    label_mask = rasterize(
        label_shapes,
        out_shape=(image.shape[1], image.shape[2]),
        fill=0,
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

    plt.tight_layout()
    plt.savefig(output_path)

    print(f"Saved sample visualization to {output_path}")
    plt.close(fig)


def create_geobench_version(
    metadata_df: pd.DataFrame,
    n_train_samples: int,
    n_val_samples: int,
    n_test_samples: int,
    n_additional_test_samples: int,
    root_dir: str,
    save_dir: str,
) -> None:
    """Create a GeoBench version of the dataset.

    Args:
        metadata_df: DataFrame with metadata including geolocation for each patch
        n_train_samples: Number of final training samples, -1 means all
        n_val_samples: Number of final validation samples, -1 means all
        n_test_samples: Number of final test samples, -1 means all
        n_additional_test_samples: Number of additional test samples, 0 means none
        root_dir: Root directory for the dataset
        save_dir: Directory to save the GeoBench version
    """
    random_state = 24

    patches_df = split_spacenet7_into_patches(
        root_dir,
        metadata_df,
        os.path.join(os.path.dirname(save_dir), "patches"),
        patch_size=(512, 512),
    )

    subset_df = create_subset_from_df(
        patches_df,
        n_train_samples=n_train_samples,
        n_val_samples=n_val_samples,
        n_test_samples=n_test_samples,
        n_additional_test_samples=n_additional_test_samples,
        random_state=random_state,
    )

    return subset_df


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

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if os.path.exists(metadata_path):
        metadata_df_with_split = pd.read_parquet(metadata_path)
    else:
        metadata_df = generate_metadata_df(args.root)

        metadata_df_with_split = create_geographic_splits_spacenet7(
            metadata_df, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, random_state=0
        )
        metadata_df_with_split.to_parquet(metadata_path)

    plot_sample_locations(
        metadata_df_with_split,
        os.path.join(args.save_dir, "sample_locations_split.png"),
        buffer_degrees=1.0,
        dataset_name="SpaceNet7",
        s=5,
    )

    geobench_df_path = os.path.join(args.save_dir, "geobench_spacenet7_patches.parquet")

    if os.path.exists(geobench_df_path):
        patches_df = pd.read_parquet(geobench_df_path)
    else:
        patches_df = create_geobench_version(
            metadata_df_with_split,
            n_train_samples=3500,
            n_val_samples=-1,
            n_test_samples=-1,
            n_additional_test_samples=0,
            root_dir=args.root,
            save_dir=args.save_dir,
        )
        patches_df.to_parquet(geobench_df_path)

    tortilla_name = "geobench_spacenet7.tortilla"
    create_tortilla(
        os.path.join(args.root, "patches"),
        patches_df,
        args.save_dir,
        tortilla_name=tortilla_name,
    )

    create_unittest_subset(
        data_dir=args.save_dir,
        tortilla_pattern=tortilla_name,
        test_dir_name="spacenet7",
        n_train_samples=2,
        n_val_samples=1,
        n_test_samples=1,
        n_additional_test_samples=0,
    )


if __name__ == "__main__":
    main()
