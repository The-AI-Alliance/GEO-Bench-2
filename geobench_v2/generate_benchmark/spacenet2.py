# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of SpaceNet2 dataset."""

from torchgeo.datasets import SpaceNet2
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


from geobench_v2.generate_benchmark.geospatial_split_utils import (
    show_samples_per_valid_ratio,
    split_geospatial_tiles_into_patches,
    visualize_checkerboard_pattern,
    visualize_geospatial_split,
    checkerboard_split,
    geographic_buffer_split,
    geographic_distance_split,
    visualize_distance_clusters,
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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from rasterio.features import rasterize


def generate_metadata_df(root: str) -> pd.DataFrame:
    """Generate metadata DataFrame for SpaceNet2 dataset."""
    label_paths = glob.glob(
        os.path.join(root, "**", "geojson", "buildings", "*.geojson"), recursive=True
    )

    df = pd.DataFrame(label_paths, columns=["label_path"])
    df["ps-ms_path"] = (
        df["label_path"]
        .str.replace(".geojson", ".tif")
        .str.replace("geojson/buildings/", "MUL-PanSharpen/")
        .str.replace("buildings_", "MUL-PanSharpen_")
    )
    df["pan_path"] = (
        df["label_path"]
        .str.replace(".geojson", ".tif")
        .str.replace("geojson/buildings/", "PAN/")
        .str.replace("buildings_", "PAN_")
    )
    df["ps-rgb_path"] = (
        df["label_path"]
        .str.replace(".geojson", ".tif")
        .str.replace("geojson/buildings/", "RGB-PanSharpen/")
        .str.replace("buildings_", "RGB-PanSharpen_")
    )

    def extract_lng_lat(path):
        with rasterio.open(path, "r") as src:
            lng, lat = src.lnglat()

        return lng, lat

    df["lon"], df["lat"] = zip(*df["pan_path"].apply(extract_lng_lat))

    # make path relative
    df["label_path"] = df["label_path"].str.replace(root, "")
    df["ps-ms_path"] = df["ps-ms_path"].str.replace(root, "")
    df["pan_path"] = df["pan_path"].str.replace(root, "")
    df["ps-rgb_path"] = df["ps-rgb_path"].str.replace(root, "")

    df["area"] = df["pan_path"].str.split("_").str[2]

    df["split"] = "train"

    return df


def create_city_based_checkerboard_splits(
    df: pd.DataFrame,
    city_col: str = "area",
    lon_col: str = "lon",
    lat_col: str = "lat",
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    n_blocks_x: int = 6,
    n_blocks_y: int = 6,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create train/val/test splits using checkerboard pattern separately for each city.

    Args:
        df: DataFrame with SpaceNet2 metadata
        city_col: Column name containing city/area information
        lon_col: Column name for longitude
        lat_col: Column name for latitude
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        n_blocks_x: Number of blocks along x-axis for checkerboard pattern
        n_blocks_y: Number of blocks along y-axis for checkerboard pattern
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with added 'split' column
    """
    result_df = df.copy()
    cities = df[city_col].unique()

    print(f"Creating checkerboard splits for {len(cities)} cities")

    for city in cities:
        city_df = df[df[city_col] == city].copy()
        print(f"\nProcessing {city} with {len(city_df)} samples")

        city_split = checkerboard_split(
            city_df,
            lon_col=lon_col,
            lat_col=lat_col,
            n_blocks_x=n_blocks_x,
            n_blocks_y=n_blocks_y,
            pattern="balanced",
            target_test_ratio=test_ratio,
            target_val_ratio=val_ratio,
            ratio_tolerance=0.02,
            random_state=random_state + cities.tolist().index(city),
        )

        result_df.loc[city_df.index, "split"] = city_split["split"]
        result_df.loc[city_df.index, "block_x"] = city_split["block_x"]
        result_df.loc[city_df.index, "block_y"] = city_split["block_y"]
        result_df.loc[city_df.index, "block_id"] = city_split["block_id"]

        split_counts = city_split["split"].value_counts()
        print(f"Split statistics for {city}:")
        for split_name in ["train", "validation", "test"]:
            if split_name in split_counts:
                count = split_counts[split_name]
                pct = 100 * count / len(city_df)
                print(f"  {split_name}: {count} samples ({pct:.1f}%)")

    print("\nOverall split statistics:")
    overall_counts = result_df["split"].value_counts()
    for split_name in ["train", "validation", "test"]:
        if split_name in overall_counts:
            count = overall_counts[split_name]
            pct = 100 * count / len(result_df)
            print(f"  {split_name}: {count} samples ({pct:.1f}%)")

    return result_df


def create_spacenet2_masks(
    df: pd.DataFrame,
    src_root: str,
    output_root: str,
    copy_originals: bool = False,
    visualize_samples: bool = True,
    num_vis_samples: int = 4,
) -> pd.DataFrame:
    """Create binary semantic and instance segmentation masks for SpaceNet2.

    Args:
        df: DataFrame with SpaceNet2 metadata
        src_root: Root directory of the source dataset
        output_root: Root directory to save the output dataset
        copy_originals: Whether to copy original images to the output directory
        visualize_samples: Whether to visualize some samples
        num_vis_samples: Number of samples to visualize

    Returns:
        Updated DataFrame with added mask paths
    """
    result_df = df.copy()
    os.makedirs(output_root, exist_ok=True)

    for split in df["split"].unique():
        mask_dir_semantic = os.path.join(output_root, split, "semantic_masks")
        mask_dir_instance = os.path.join(output_root, split, "instance_masks")
        os.makedirs(mask_dir_semantic, exist_ok=True)
        os.makedirs(mask_dir_instance, exist_ok=True)

    if visualize_samples:
        os.makedirs(os.path.join(output_root, "visualizations"), exist_ok=True)
        vis_indices = np.random.choice(
            len(df), min(num_vis_samples, len(df)), replace=False
        )
        vis_samples = []

    result_df["semantic_mask_path"] = None
    result_df["instance_mask_path"] = None

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating masks"):
        label_path = os.path.join(src_root, row["label_path"])
        pan_path = os.path.join(src_root, row["pan_path"])
        split = row["split"]

        label_dir = os.path.dirname(row["label_path"])
        parent_dir = os.path.dirname(label_dir)
        base_dir = os.path.dirname(parent_dir)

        img_id = (
            os.path.basename(row["label_path"])
            .replace("buildings_", "")
            .replace(".geojson", "")
        )

        semantic_mask_dir = os.path.join(base_dir, "semantic_masks")
        instance_mask_dir = os.path.join(base_dir, "instance_masks")
        os.makedirs(os.path.join(output_root, semantic_mask_dir), exist_ok=True)
        os.makedirs(os.path.join(output_root, instance_mask_dir), exist_ok=True)

        semantic_mask_path = os.path.join(semantic_mask_dir, f"semantic_{img_id}.tif")
        instance_mask_path = os.path.join(instance_mask_dir, f"instance_{img_id}.tif")

        with rasterio.open(os.path.join(src_root, row["pan_path"])) as src:
            height, width = src.height, src.width
            transform = src.transform
            profile = src.profile.copy()

        gdf = gpd.read_file(label_path)
        valid_geoms = [
            geom for geom in gdf.geometry if geom is not None and not geom.is_empty
        ]

        semantic_mask = rasterize(
            [(geom, 1) for geom in valid_geoms],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True,
        )

        instance_mask = np.zeros((height, width), dtype=np.uint16)

        for i, geom in enumerate(valid_geoms, start=1):
            building_mask = rasterize(
                [(geom, i)],
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype=np.uint16,
                all_touched=True,
            )
            instance_mask = np.maximum(instance_mask, building_mask)

        unique_instances = np.unique(instance_mask)
        if len(unique_instances) - 1 != len(valid_geoms):
            print(
                f"Warning: Instance count mismatch in {img_id}: {len(unique_instances) - 1} vs {len(valid_geoms)}"
            )

        profile.update(count=1, dtype="uint8", nodata=0)
        instance_profile = profile.copy()
        instance_profile.update(dtype="uint16")

        with rasterio.open(
            os.path.join(output_root, semantic_mask_path), "w", **profile
        ) as dst:
            dst.write(semantic_mask[np.newaxis, :, :])

        with rasterio.open(
            os.path.join(output_root, instance_mask_path), "w", **instance_profile
        ) as dst:
            dst.write(instance_mask[np.newaxis, :, :])

        result_df.at[idx, "semantic_mask_path"] = semantic_mask_path
        result_df.at[idx, "instance_mask_path"] = instance_mask_path

        if visualize_samples and idx in vis_indices:
            with rasterio.open(os.path.join(src_root, row["ps-rgb_path"])) as src:
                rgb_img = src.read() / 3000.0
                rgb_img = np.clip(rgb_img, 0, 1)
                rgb_img = rgb_img.transpose(1, 2, 0)

            vis_samples.append(
                {
                    "rgb_img": rgb_img,
                    "semantic_mask": semantic_mask,
                    "instance_mask": instance_mask,
                    "building_count": len(valid_geoms),
                    "sample_idx": idx,
                    "area": row["area"],
                    "split": row["split"],
                    "unique_ids": len(np.unique(instance_mask)) - 1,
                }
            )

    if visualize_samples and vis_samples:
        for i, sample in enumerate(vis_samples):
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            axes[0].imshow(sample["rgb_img"])
            axes[0].set_title(f"RGB Image - {sample['area']}")
            axes[0].axis("off")

            axes[1].imshow(sample["semantic_mask"], cmap="binary")
            axes[1].set_title(f"Semantic Mask - {sample['building_count']} buildings")
            axes[1].axis("off")

            from matplotlib import cm

            cmap = cm.get_cmap("tab20", np.max(sample["instance_mask"]) + 1)
            masked_instance = np.ma.masked_where(
                sample["instance_mask"] == 0, sample["instance_mask"]
            )
            axes[2].imshow(
                np.ones_like(sample["instance_mask"]), cmap="gray", vmin=0, vmax=1
            )
            axes[2].imshow(masked_instance, cmap=cmap, interpolation="none")
            axes[2].set_title(f"Instance Mask - {sample['unique_ids']} instances")
            axes[2].axis("off")

            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    output_root,
                    "visualizations",
                    f"sample_{i}_idx_{sample['sample_idx']}.png",
                ),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)

    result_df.to_parquet(
        os.path.join(output_root, "spacenet2_metadata.parquet"), index=False
    )

    semantic_count = len(result_df[~result_df["semantic_mask_path"].isna()])
    instance_count = len(result_df[~result_df["instance_mask_path"].isna()])
    print(
        f"Generated masks for {len(df)} samples: {semantic_count} semantic, {instance_count} instance"
    )

    return result_df


def visualize_samples(
    df: pd.DataFrame,
    root: str,
    num_samples: int = 8,
    output_path: str = "spacenet2_samples.png",
) -> None:
    """Visualize multiple random data samples from the SpaceNet2 dataset.

    Args:
        df: DataFrame with SpaceNet2 metadata
        root: Root directory of the dataset
        num_samples: Number of random samples to visualize
        output_path: Path to save the visualization
    """
    # set new random seed
    np.random.seed(np.random.randint(0, 10000))
    random_indices = np.random.choice(len(df), min(num_samples, len(df)), replace=False)
    random_rows = df.iloc[random_indices]

    fig, axs = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    for row_idx, (_, sample_row) in enumerate(random_rows.iterrows()):
        for col_idx, col in enumerate(
            ["ps-ms_path", "pan_path", "ps-rgb_path", "label_path"]
        ):
            ax = axs[row_idx, col_idx]

            if col == "label_path":
                gdf = gpd.read_file(os.path.join(root, sample_row[col]))

                with rasterio.open(
                    os.path.join(root, sample_row["pan_path"]), "r"
                ) as src:
                    src_height, src_width = src.shape
                    src_transform = src.transform

                buildings = [
                    (geom, 1)
                    for geom in gdf.geometry
                    if geom is not None and not geom.is_empty
                ]

                mask_full = rasterize(
                    buildings,
                    out_shape=(src_height, src_width),
                    transform=src_transform,
                    fill=0,
                    dtype=np.uint8,
                    all_touched=True,
                    merge_alg=rasterio.enums.MergeAlg.replace,
                )
                mask_full = mask_full[np.newaxis, :, :]

                ax.imshow(mask_full[0], cmap="gray", alpha=0.8)
                title = "Buildings"

            elif col == "ps-ms_path":
                with rasterio.open(os.path.join(root, sample_row[col]), "r") as src:
                    img = src.read()[[4, 2, 1], ...] / 3000.0
                img = np.clip(img, 0, 1)
                ax.imshow(img.transpose(1, 2, 0))
                title = "Multi-Spectral"

            elif col == "ps-rgb_path":
                with rasterio.open(os.path.join(root, sample_row[col]), "r") as src:
                    img = src.read() / 3000.0
                img = np.clip(img, 0, 1)
                ax.imshow(img.transpose(1, 2, 0))
                title = "RGB"

            else:
                # Handle panchromatic imagery
                with rasterio.open(os.path.join(root, sample_row[col]), "r") as src:
                    img = src.read()
                    if img.shape[0] == 1:
                        img = img[0]
                        ax.imshow(img, cmap="gray")
                    else:
                        ax.imshow(img.transpose(1, 2, 0))
                title = "Panchromatic"

            if col_idx == 0:
                title = f"{title}\nArea: {sample_row['area']}\nLat: {sample_row['lat']:.4f}, Lon: {sample_row['lon']:.4f}"

            if col_idx == 3:
                title = f"{title}\nSplit: {sample_row['split']}"

            ax.set_title(title, fontsize=10)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Visualization of {num_samples} random samples saved to {output_path}")


def create_tortilla(root_dir, df, save_dir):
    """Create a tortilla version of the dataset."""

    tortilla_dir = os.path.join(save_dir, "tortilla")
    os.makedirs(tortilla_dir, exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating tortilla"):
        modalities = ["ps-ms", "pan", "semantic_mask", "instance_mask"]
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
                    "time_start": 0,
                },
                lon=row["lon"],
                lat=row["lat"],
                source_label_file=row["label_path"],
                city=row["area"],
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
            lon=sample_data["lon"],
            lat=sample_data["lat"],
            source_label_file=sample_data["source_label_file"],
            city=sample_data["city"],
        )
        samples.append(sample_tortilla)

    # create final taco file
    final_samples = tacotoolbox.tortilla.datamodel.Samples(samples=samples)
    tacotoolbox.tortilla.create(
        final_samples, os.path.join(save_dir, "SpaceNet2.tortilla"), quiet=True
    )


def main():
    """Generate SpaceNet2 Benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for SpaceNet2 dataset"
    )
    parser.add_argument(
        "--save_dir",
        default="geobenchV2/SpaceNet2",
        help="Directory to save the subset",
    )
    args = parser.parse_args()

    metadata_path = os.path.join(args.save_dir, "geobench_metadata.parquet")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if os.path.exists(metadata_path):
        metadata_df = pd.read_parquet(metadata_path)
    else:
        metadata_df = generate_metadata_df(args.root)

    full_df = create_city_based_checkerboard_splits(metadata_df)

    result_path = os.path.join(args.save_dir, "geobench_spacenet2.parquet")
    if os.path.exists(result_path):
        result_df = pd.read_parquet(result_path)
    else:
        result_df = create_spacenet2_masks(
            full_df,
            src_root=args.root,
            output_root=args.root,
            copy_originals=False,
            visualize_samples=False,
            # num_vis_samples=4,
        )

        result_df.to_parquet(result_path, index=False)

    create_tortilla(args.root, result_df, args.save_dir)

    # for city in full_df["area"].unique():
    #     plot_sample_locations(
    #         full_df[full_df["area"] == city],
    #         os.path.join(args.save_dir, f"sample_locations_{city.lower()}.png"),
    #         buffer_degrees=0.2,
    #         dataset_name=f"SpaceNet2 {city}",
    #     )

    visualize_samples(full_df, args.root)

    full_df.to_parquet(metadata_path)


if __name__ == "__main__":
    main()
