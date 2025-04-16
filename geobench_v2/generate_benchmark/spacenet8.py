# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of SpaceNet8 dataset."""

from torchgeo.datasets import SpaceNet8
import geopandas as gpd
import pandas as pd
import os
import argparse
import rasterio
from tqdm import tqdm
import re
import tacotoolbox
import tacoreader
import glob

from concurrent.futures import ProcessPoolExecutor
from rasterio.enums import Compression
from rasterio.features import rasterize
from rasterio.windows import Window


from geobench_v2.generate_benchmark.utils import (
    plot_sample_locations,
    create_unittest_subset,
    create_subset_from_df,
)
from geobench_v2.generate_benchmark.geospatial_split_utils import (
    visualize_checkerboard_pattern,
    split_geospatial_tiles_into_patches,
    visualize_geospatial_split,
    checkerboard_split,
    geographic_buffer_split,
    geographic_distance_split,
    visualize_distance_clusters,
    show_samples_per_valid_ratio,
)
import numpy as np


def process_spacenet8_tile(args):
    """Process a single SpaceNet8 tile into patches.

    Args:
        args: Tuple containing:
            - idx: Row index
            - row: DataFrame row with metadata
            - root_dir: Root directory of the dataset
            - output_dir: Directory to save patches
            - patch_size: Size of patches (height, width)
            - blockxsize: Block width for GeoTIFF
            - blockysize: Block height for GeoTIFF
            - stride: Step size between patches
            - output_format: Output file format
            - patch_id_prefix: Prefix for patch IDs
            - buffer_top: Number of pixels to skip from the top
            - buffer_left: Number of pixels to skip from the left
            - buffer_bottom: Number of pixels to skip from the bottom
            - buffer_right: Number of pixels to skip from the right

    Returns:
        List of patch metadata dictionaries
    """
    (
        idx,
        row,
        root_dir,
        output_dir,
        patch_size,
        blockxsize,
        blockysize,
        stride,
        output_format,
        patch_id_prefix,
        buffer_top,
        buffer_left,
        buffer_bottom,
        buffer_right,
    ) = args

    pre_image_dir = os.path.join(output_dir, "pre-event")
    post_image_dir = os.path.join(output_dir, "post-event")
    mask_dir = os.path.join(output_dir, "mask")

    result_metadata = []

    try:
        # Use the correct path keys from the metadata DataFrame
        pre_img_path = row["pre-path"]
        post_img_path = row["post-path"]
        label_path = row["label-path"]

        tile_basename = os.path.basename(pre_img_path).split(".")[0]

        with rasterio.open(pre_img_path) as src:
            pre_profile = src.profile.copy()
            height, width = src.height, src.width
            transform = src.transform
            crs = src.crs
            pre_data = src.read()

        with rasterio.open(post_img_path) as src:
            post_profile = src.profile.copy()
            post_data = src.read()

        # Load the label/mask as a GeoDataFrame
        gdf = gpd.read_file(label_path)
        mask_data = np.zeros((1, height, width), dtype=np.uint8)

        if len(gdf) > 0 and not gdf.empty:
            if gdf.crs is None:
                gdf.set_crs(crs, inplace=True)
            elif gdf.crs != crs:
                gdf = gdf.to_crs(crs)

            # Create mask from road vectors - distinguish between flooded and non-flooded roads
            road_shapes = []
            for _, feature in gdf.iterrows():
                if feature.geometry is not None and not feature.geometry.is_empty:
                    # Check if the feature has a 'flood' attribute or similar
                    if "flood" in feature and feature["flood"] > 0:
                        road_shapes.append((feature.geometry, 2))  # Flooded roads = 2
                    else:
                        road_shapes.append(
                            (feature.geometry, 1)
                        )  # Non-flooded roads = 1

            if road_shapes:
                mask_data = rasterize(
                    road_shapes,
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    dtype=np.uint8,
                    all_touched=False,
                )
                mask_data = mask_data[np.newaxis, :, :]

        # Apply buffers to get the effective dimensions
        effective_height = height - buffer_top - buffer_bottom
        effective_width = width - buffer_left - buffer_right

        if effective_height <= 0 or effective_width <= 0:
            print(
                f"Error: Image dimensions ({height}x{width}) are smaller than the combined buffers. Skipping."
            )
            return []

        # Calculate number of patches considering buffers and stride
        patches_per_dim_h = max(
            1, (effective_height - patch_size[0] + stride[0]) // stride[0]
        )
        patches_per_dim_w = max(
            1, (effective_width - patch_size[1] + stride[1]) // stride[1]
        )

        for i in range(patches_per_dim_h):
            for j in range(patches_per_dim_w):
                # Calculate row and column start with buffer offsets
                row_start = buffer_top + (i * stride[0])
                col_start = buffer_left + (j * stride[1])

                # Ensure we don't go beyond the effective area (accounting for buffers)
                if row_start + patch_size[0] > height - buffer_bottom:
                    row_start = max(buffer_top, height - buffer_bottom - patch_size[0])
                if col_start + patch_size[1] > width - buffer_right:
                    col_start = max(buffer_left, width - buffer_right - patch_size[1])

                window = Window(col_start, row_start, patch_size[1], patch_size[0])
                patch_transform = rasterio.windows.transform(window, transform)

                mask_patch = mask_data[
                    :,
                    row_start : row_start + patch_size[0],
                    col_start : col_start + patch_size[1],
                ]

                # Calculate road coverage
                road_ratio = np.sum(mask_patch > 0) / mask_patch.size
                flood_ratio = (
                    np.sum(mask_patch == 2) / mask_patch.size if 2 in mask_patch else 0
                )

                patch_id = f"{patch_id_prefix}{tile_basename}_{i:03d}_{j:03d}"

                pre_img_filename = f"{patch_id}_pre_image.{output_format}"
                post_img_filename = f"{patch_id}_post_image.{output_format}"
                mask_filename = f"{patch_id}_mask.{output_format}"

                pre_img_patch_path = os.path.join(pre_image_dir, pre_img_filename)
                post_img_patch_path = os.path.join(post_image_dir, post_img_filename)
                mask_patch_path = os.path.join(mask_dir, mask_filename)

                pre_img_patch = pre_data[
                    :,
                    row_start : row_start + patch_size[0],
                    col_start : col_start + patch_size[1],
                ]

                post_img_patch = post_data[
                    :,
                    row_start : row_start + patch_size[0],
                    col_start : col_start + patch_size[1],
                ]

                profile_template = {
                    "driver": "GTiff",
                    "tiled": True,
                    "blockxsize": blockxsize,
                    "blockysize": blockysize,
                    "interleave": "pixel",
                    "compress": "zstd",
                    "zstd_level": 22,
                    "predictor": 2,
                    "crs": crs,
                    "transform": patch_transform,
                }

                pre_img_profile = profile_template.copy()
                pre_img_profile.update(
                    {
                        "height": patch_size[0],
                        "width": patch_size[1],
                        "count": pre_img_patch.shape[0],
                        "dtype": pre_img_patch.dtype,
                    }
                )

                post_img_profile = profile_template.copy()
                post_img_profile.update(
                    {
                        "height": patch_size[0],
                        "width": patch_size[1],
                        "count": post_img_patch.shape[0],
                        "dtype": post_img_patch.dtype,
                    }
                )

                mask_profile = profile_template.copy()
                mask_profile.update(
                    {
                        "height": patch_size[0],
                        "width": patch_size[1],
                        "count": 1,
                        "dtype": "uint8",
                    }
                )

                with rasterio.open(pre_img_patch_path, "w", **pre_img_profile) as dst:
                    dst.write(pre_img_patch)

                with rasterio.open(post_img_patch_path, "w", **post_img_profile) as dst:
                    dst.write(post_img_patch)

                with rasterio.open(mask_patch_path, "w", **mask_profile) as dst:
                    dst.write(mask_patch)

                patch_bounds = rasterio.windows.bounds(window, transform)
                west, south, east, north = patch_bounds

                center_x = (west + east) / 2
                center_y = (south + north) / 2

                lon, lat = None, None
                if crs.is_projected:
                    try:
                        from pyproj import Transformer

                        transformer = Transformer.from_crs(
                            crs, "EPSG:4326", always_xy=True
                        )
                        lon, lat = transformer.transform(center_x, center_y)
                    except Exception as e:
                        print(f"Error transforming coordinates: {e}")
                        lon, lat = None, None
                else:
                    lon, lat = center_x, center_y

                patch_metadata = {
                    "source_pre_img": os.path.basename(pre_img_path),
                    "source_post_img": os.path.basename(post_img_path),
                    "source_mask": os.path.basename(label_path),
                    "patch_id": patch_id,
                    "lon": lon,
                    "lat": lat,
                    "height_px": patch_size[0],
                    "width_px": patch_size[1],
                    "crs": str(crs),
                    "row": i,
                    "col": j,
                    "row_px": row_start,
                    "col_px": col_start,
                    "road_ratio": float(road_ratio),
                    "flood_ratio": float(flood_ratio),
                    "pre_image_path": os.path.join("pre-event", pre_img_filename),
                    "post_image_path": os.path.join("post-event", post_img_filename),
                    "mask_path": os.path.join("mask", mask_filename),
                    "region": row["region"] if "region" in row else None,
                    "split": row["split"] if "split" in row else "train",
                }

                result_metadata.append(patch_metadata)

        return result_metadata

    except Exception as e:
        print(f"Error processing tile {idx}: {e}")
        import traceback

        traceback.print_exc()
        return []


def split_spacenet8_into_patches(
    metadata_df: pd.DataFrame,
    root_dir: str,
    output_dir: str,
    patch_size: tuple[int, int] = (512, 512),
    block_size: tuple[int, int] = (512, 512),
    stride: tuple[int, int] | None = None,
    output_format: str = "tif",
    patch_id_prefix: str = "p",
    num_workers: int = 4,
    buffer_top: int = 0,
    buffer_left: int = 0,
    buffer_bottom: int = 0,
    buffer_right: int = 0,
) -> pd.DataFrame:
    """Split SpaceNet8 images and road masks into smaller patches.

    Args:
        metadata_df: DataFrame with SpaceNet8 metadata including paths
        root_dir: Root directory of the dataset
        output_dir: Directory to save patches
        patch_size: Size of the patches (height, width)
        block_size: Size of the blocks for optimized GeoTIFF writing
        stride: Step size between patches (height, width)
        output_format: Output file format (e.g., 'tif')
        patch_id_prefix: Prefix for patch IDs
        num_workers: Number of parallel processes to use
        buffer_top: Number of pixels to skip from the top of the image
        buffer_left: Number of pixels to skip from the left of the image
        buffer_bottom: Number of pixels to skip from the bottom of the image
        buffer_right: Number of pixels to skip from the right of the image

    Returns:
        DataFrame containing metadata for all created patches
    """
    from concurrent.futures import ProcessPoolExecutor

    blockxsize, blockysize = block_size
    blockxsize = blockxsize - (blockxsize % 16) if blockxsize % 16 != 0 else blockxsize
    blockysize = blockysize - (blockysize % 16) if blockysize % 16 != 0 else blockysize

    if stride is None:
        stride = patch_size

    os.makedirs(output_dir, exist_ok=True)

    pre_image_dir = os.path.join(output_dir, "pre-event")
    post_image_dir = os.path.join(output_dir, "post-event")
    mask_dir = os.path.join(output_dir, "mask")

    for directory in [pre_image_dir, post_image_dir, mask_dir]:
        os.makedirs(directory, exist_ok=True)

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
                patch_size,
                blockxsize,
                blockysize,
                stride,
                output_format,
                patch_id_prefix,
                buffer_top,
                buffer_left,
                buffer_bottom,
                buffer_right,
            )
            for idx, row in batch.iterrows()
        ]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(
                tqdm(
                    executor.map(process_spacenet8_tile, tasks),
                    total=len(tasks),
                    desc=f"Processing batch {batch_idx + 1}/{len(batches)}",
                )
            )

            for result_list in results:
                all_patch_metadata.extend(result_list)

    patches_df = pd.DataFrame(all_patch_metadata)

    return patches_df


def generate_metadata_df(root_dir) -> pd.DataFrame:
    """Generate metadata DataFrame for SpaceNet8 dataset.

    Args:
        ds: SpaceNet8 dataset.
    """

    paths = [
        os.path.join(root_dir, "Germany_Training_Public_label_image_mapping.csv"),
        os.path.join(
            root_dir, "Louisiana-East_Training_Public_label_image_mapping.csv"
        ),
    ]

    df = pd.concat([pd.read_csv(path) for path in paths])

    metadata: list[dict[str, str]] = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        pre_event_path = os.path.join(
            root_dir, "SN8_floods", "train", "PRE-event", row["pre-event image"]
        )
        post_event_path = os.path.join(
            root_dir, "SN8_floods", "train", "POST-event", row["post-event image 1"]
        )
        label_path = os.path.join(
            root_dir, "SN8_floods", "train", "annotations", row["label"]
        )

        assert os.path.exists(pre_event_path)
        assert os.path.exists(post_event_path)
        assert os.path.exists(label_path)

        with rasterio.open(pre_event_path) as src:
            lng, lat = src.lnglat()
            height_px, width_px = src.height, src.width
        metadata.append(
            {
                "pre-path": pre_event_path,
                "post-path": post_event_path,
                "label-path": label_path,
                "lon": lng,
                "lat": lat,
                "height_px": height_px,
                "width_px": width_px,
            }
        )

    metadata_df = pd.DataFrame(metadata)

    metadata_df["split"] = "train"

    regions = [
        {
            "name": "Louisiana, USA",
            "bounds": {"min_lat": 29, "max_lat": 33, "min_lon": -94, "max_lon": -89},
        },
        {
            "name": "Germany",
            "bounds": {
                "min_lat": 47.5,
                "max_lat": 54.5,
                "min_lon": 6.5,
                "max_lon": 14.5,
            },
        },
    ]

    # match region to each sample
    metadata_df["region"] = "unknown"  # Default value

    for region in regions:
        bounds = region["bounds"]
        metadata_df.loc[
            (metadata_df["lat"] >= bounds["min_lat"])
            & (metadata_df["lat"] <= bounds["max_lat"])
            & (metadata_df["lon"] >= bounds["min_lon"])
            & (metadata_df["lon"] <= bounds["max_lon"]),
            "region",
        ] = region["name"]

    return metadata_df


def create_tortilla(root_dir, df, save_dir, tortilla_name):
    """Create a tortilla version of the dataset."""

    tortilla_dir = os.path.join(save_dir, "tortilla")
    os.makedirs(tortilla_dir, exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating tortilla"):
        modalities = ["pre_image", "post_image", "mask"]
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
                source_mask_file=row["source_mask"],
                source_pre_img_file=row["source_pre_img"],
                source_post_img_file=row["source_post_img"],
                patch_id=row["patch_id"],
                region=row["region"],
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
            source_mask_file=sample_data["source_mask_file"],
            source_pre_img_file=sample_data["source_pre_img_file"],
            source_post_img_file=sample_data["source_post_img_file"],
            patch_id=sample_data["patch_id"],
            region=sample_data["region"],
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
    root_dir: str,
    save_dir: str,
) -> None:
    """Create a GeoBench version of the dataset.
    Args:
        metadata_df: DataFrame with metadata including geolocation for each patch
        n_train_samples: Number of final training samples, -1 means all
        n_val_samples: Number of final validation samples, -1 means all
        n_test_samples: Number of final test samples, -1 means all
        root_dir: Root directory for the dataset
        save_dir: Directory to save the GeoBench version
    """
    random_state = 24

    subset_df = create_subset_from_df(
        metadata_df, n_train_samples, n_val_samples, n_test_samples, random_state
    )

    patch_size = (512, 512)
    stride = (511, 511)

    patches_df = split_spacenet8_into_patches(
        subset_df,
        root_dir=root_dir,
        output_dir=save_dir,
        patch_size=patch_size,
        block_size=(512, 512),
        stride=stride,
        buffer_top=64,
        buffer_bottom=64,
        buffer_left=64,
        buffer_right=64,
        num_workers=8,
    )
    return patches_df


def main():
    """Generate SpaceNet8 Benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for SpaceNet8 dataset"
    )
    parser.add_argument(
        "--save_dir",
        default="geobenchV2/SpaceNet8",
        help="Directory to save the subset",
    )
    args = parser.parse_args()

    metadata_path = os.path.join(args.save_dir, "geobench_metadata.parquet")

    orig_dataset = SpaceNet8(root=args.root, download=False)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if os.path.exists(metadata_path):
        metadata_df = pd.read_parquet(metadata_path)
    else:
        metadata_df = generate_metadata_df(args.root)
        metadata_df.to_parquet(metadata_path)

    df_with_assigned_split = geographic_distance_split(
        metadata_df, n_clusters=8, random_state=42
    )

    visualize_distance_clusters(
        df_with_assigned_split,
        title="Distance Split",
        output_path=os.path.join(args.save_dir, "distance_split.png"),
        buffer_degrees=0.05,
    )

    result_df_path = os.path.join(args.save_dir, "spacenet8_patches.parquet")
    if os.path.exists(result_df_path):
        result_df = pd.read_parquet(result_df_path)
    else:
        result_df = create_geobench_version(
            df_with_assigned_split,
            n_train_samples=4000,
            n_val_samples=-1,
            n_test_samples=-1,
            root_dir=args.root,
            save_dir=args.save_dir,
        )
        result_df.to_parquet(result_df_path)

    tortilla_name = "geobench_spacenet8.tortilla"
    create_tortilla(args.save_dir, result_df, args.save_dir, tortilla_name)

    create_unittest_subset(
        data_dir=args.save_dir,
        tortilla_pattern=tortilla_name,
        test_dir_name="spacenet8",
        n_train_samples=2,
        n_val_samples=1,
        n_test_samples=1,
    )


if __name__ == "__main__":
    main()
