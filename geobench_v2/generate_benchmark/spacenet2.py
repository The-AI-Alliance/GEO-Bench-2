# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of SpaceNet2 dataset."""

import argparse
import glob
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import tacoreader
import tacotoolbox
from rasterio.features import rasterize
from tqdm import tqdm

from geobench_v2.generate_benchmark.geospatial_split_utils import checkerboard_split
from geobench_v2.generate_benchmark.utils import (
    create_subset_from_df,
    create_unittest_subset,
)


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
    df["label_path"] = df["label_path"].str.replace(root, "").str.lstrip(os.sep)
    df["ps-ms_path"] = df["ps-ms_path"].str.replace(root, "").str.lstrip(os.sep)
    df["pan_path"] = df["pan_path"].str.replace(root, "").str.lstrip(os.sep)
    df["ps-rgb_path"] = df["ps-rgb_path"].str.replace(root, "").str.lstrip(os.sep)

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


def process_spacenet2_sample(args):
    """Process a single SpaceNet2 sample to create masks and copy images.

    Args:
        args: Tuple containing (idx, row, src_root, output_root, copy_originals)

    Returns:
        Dictionary with processing results
    """
    idx, row, src_root, output_root, copy_originals = args
    label_path = os.path.join(src_root, row["label_path"])
    pan_path = os.path.join(src_root, row["pan_path"])

    img_id = (
        os.path.basename(row["label_path"])
        .replace("buildings_", "")
        .replace(".geojson", "")
    )

    semantic_mask_name = f"semantic_{img_id}.tif"
    instance_mask_name = f"instance_{img_id}.tif"
    semantic_mask_path = os.path.join("semantic_masks", semantic_mask_name)
    instance_mask_path = os.path.join("instance_masks", instance_mask_name)

    with rasterio.open(pan_path) as src:
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs

    base_profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "interleave": "band",
        "compress": "lzw",
        "predictor": 2,
        "zlevel": 9,
        "crs": crs,
        "transform": transform,
    }

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

    semantic_profile = base_profile.copy()
    semantic_profile.update(count=1, dtype="uint8", nodata=0)

    instance_profile = base_profile.copy()
    instance_profile.update(count=1, dtype="uint16", nodata=0)

    with rasterio.open(
        os.path.join(output_root, semantic_mask_path), "w", **semantic_profile
    ) as dst:
        dst.write(semantic_mask[np.newaxis, :, :])

    with rasterio.open(
        os.path.join(output_root, instance_mask_path), "w", **instance_profile
    ) as dst:
        dst.write(instance_mask[np.newaxis, :, :])

    result = {
        "idx": idx,
        "semantic_mask_path": semantic_mask_path,
        "instance_mask_path": instance_mask_path,
    }

    if copy_originals:
        pan_img_name = os.path.basename(row["pan_path"])
        new_pan_path = os.path.join("PAN", pan_img_name)

        ps_ms_img_name = os.path.basename(row["ps-ms_path"])
        new_ps_ms_path = os.path.join("MUL-PanSharpen", ps_ms_img_name)

        ps_rgb_img_name = os.path.basename(row["ps-rgb_path"])
        new_ps_rgb_path = os.path.join("RGB-PanSharpen", ps_rgb_img_name)

        try:
            with rasterio.open(os.path.join(src_root, row["pan_path"])) as src:
                pan_data = src.read()
                pan_profile = base_profile.copy()
                pan_profile.update(count=pan_data.shape[0], dtype=pan_data.dtype)

                with rasterio.open(
                    os.path.join(output_root, new_pan_path), "w", **pan_profile
                ) as dst:
                    dst.write(pan_data)

            with rasterio.open(os.path.join(src_root, row["ps-ms_path"])) as src:
                ms_data = src.read()
                ms_profile = base_profile.copy()
                ms_profile.update(count=ms_data.shape[0], dtype=ms_data.dtype)

                with rasterio.open(
                    os.path.join(output_root, new_ps_ms_path), "w", **ms_profile
                ) as dst:
                    dst.write(ms_data)

            with rasterio.open(os.path.join(src_root, row["ps-rgb_path"])) as src:
                rgb_data = src.read()
                rgb_profile = base_profile.copy()
                rgb_profile.update(count=rgb_data.shape[0], dtype=rgb_data.dtype)

                with rasterio.open(
                    os.path.join(output_root, new_ps_rgb_path), "w", **rgb_profile
                ) as dst:
                    dst.write(rgb_data)

            result["pan_path_new"] = new_pan_path
            result["ps-ms_path_new"] = new_ps_ms_path
            result["ps-rgb_path_new"] = new_ps_rgb_path

        except Exception as e:
            print(f"Error processing images for {img_id}: {str(e)}")

    return result


def create_spacenet2_masks(
    df: pd.DataFrame,
    src_root: str,
    output_root: str,
    copy_originals: bool = True,
    num_workers: int = 4,
) -> pd.DataFrame:
    """Create binary semantic and instance segmentation masks for SpaceNet2.

    Args:
        df: DataFrame with SpaceNet2 metadata
        src_root: Root directory of the source dataset
        output_root: Root directory to save the output dataset
        copy_originals: Whether to copy original images to the output directory
        num_workers: Number of parallel processes to use

    Returns:
        Updated DataFrame with added mask paths
    """
    from concurrent.futures import ProcessPoolExecutor

    result_df = df.copy()
    os.makedirs(output_root, exist_ok=True)

    mask_dir_semantic = os.path.join(output_root, "semantic_masks")
    mask_dir_instance = os.path.join(output_root, "instance_masks")
    os.makedirs(mask_dir_semantic, exist_ok=True)
    os.makedirs(mask_dir_instance, exist_ok=True)

    if copy_originals:
        for img_type in ["PAN", "MUL-PanSharpen", "RGB-PanSharpen"]:
            img_dir = os.path.join(output_root, img_type)
            os.makedirs(img_dir, exist_ok=True)

    result_df["semantic_mask_path"] = None
    result_df["instance_mask_path"] = None

    if copy_originals:
        result_df["pan_path_new"] = None
        result_df["ps-ms_path_new"] = None
        result_df["ps-rgb_path_new"] = None

    print(f"Starting parallel processing with {num_workers} workers")

    chunk_size = min(500, len(df))
    chunks = [df.iloc[i : i + chunk_size] for i in range(0, len(df), chunk_size)]

    for chunk_idx, chunk in enumerate(chunks):
        print(f"Processing chunk {chunk_idx + 1}/{len(chunks)} ({len(chunk)} samples)")

        tasks = [
            (idx, row, src_root, output_root, copy_originals)
            for idx, row in chunk.iterrows()
        ]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(process_spacenet2_sample, task) for task in tasks
            ]

            for future in tqdm(
                futures, total=len(futures), desc=f"Processing chunk {chunk_idx + 1}"
            ):
                try:
                    result = future.result()
                    idx = result["idx"]

                    result_df.at[idx, "semantic_mask_path"] = result[
                        "semantic_mask_path"
                    ]
                    result_df.at[idx, "instance_mask_path"] = result[
                        "instance_mask_path"
                    ]

                    if copy_originals and "pan_path_new" in result:
                        result_df.at[idx, "pan_path"] = result["pan_path_new"]
                        result_df.at[idx, "ps-ms_path"] = result["ps-ms_path_new"]
                        result_df.at[idx, "ps-rgb_path"] = result["ps-rgb_path_new"]
                except Exception as e:
                    print(f"Error processing task: {str(e)}")

    result_df.to_parquet(
        os.path.join(output_root, "spacenet2_metadata.parquet"), index=False
    )

    semantic_count = len(result_df[~result_df["semantic_mask_path"].isna()])
    instance_count = len(result_df[~result_df["instance_mask_path"].isna()])
    print(
        f"Generated masks for {len(df)} samples: {semantic_count} semantic, {instance_count} instance"
    )
    if copy_originals:
        print(
            f"Copied and optimized {len(df)} original images with new profile settings"
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


def create_tortilla(root_dir, df, save_dir, tortilla_name):
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
        final_samples,
        os.path.join(save_dir, tortilla_name),
        quiet=True,
        chunk_size="48GB",
    )


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
        n_additional_test_samples: Number of additional test samples to add from the train split
        root_dir: Root directory for the dataset
        save_dir: Directory to save the GeoBench version
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

    result_df = create_spacenet2_masks(
        subset_df,
        src_root=root_dir,
        output_root=save_dir,
        copy_originals=True,
        num_workers=8,
    )
    return result_df


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

    result_path = os.path.join(args.save_dir, "geobench_spacenet2.parquet")
    if os.path.exists(result_path):
        result_df = pd.read_parquet(result_path)
    else:
        full_df = create_city_based_checkerboard_splits(metadata_df)
        result_df = create_geobench_version(
            full_df,
            n_train_samples=4000,
            n_val_samples=-1,
            n_test_samples=-1,
            n_additional_test_samples=0,
            root_dir=args.root,
            save_dir=args.save_dir,
        )

        result_df.to_parquet(result_path, index=False)

    tortilla_name = "geobench_spacenet2.tortilla"

    create_tortilla(
        args.save_dir, result_df, args.save_dir, tortilla_name=tortilla_name
    )

    create_unittest_subset(
        data_dir=args.save_dir,
        tortilla_pattern=f"{tortilla_name.split('.')[0]}.*.part.tortilla",
        test_dir_name="spacenet2",
        n_train_samples=2,
        n_val_samples=1,
        n_test_samples=1,
        n_additional_test_samples=0,
    )


if __name__ == "__main__":
    main()
