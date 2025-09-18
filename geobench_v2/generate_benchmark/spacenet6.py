# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of SpaceNet6 dataset."""

import argparse
import glob
import os
import re
from concurrent.futures import ProcessPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import tacoreader
import tacotoolbox
from rasterio.features import rasterize
from rasterio.windows import Window
from tqdm import tqdm

from geobench_v2.generate_benchmark.geospatial_split_utils import (
    checkerboard_split,
    visualize_geospatial_split,
)
from geobench_v2.generate_benchmark.utils import (
    create_subset_from_df,
    create_unittest_subset,
    plot_sample_locations,
)


def process_spacenet6_tile(args):
    """Process a single SpaceNet6 tile into patches.

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
    ) = args

    ps_rgbnir_dir = os.path.join(output_dir, "ps-rgbnir")
    sar_intensity_dir = os.path.join(output_dir, "sar_intensity")
    mask_dir = os.path.join(output_dir, "mask")

    result_metadata = []

    try:
        ps_rgbnir_path = os.path.join(root_dir, row["ps_rgbnir_path"])
        sar_intensity_path = os.path.join(root_dir, row["sar_intensity_path"])
        mask_path = os.path.join(root_dir, row["mask_path"])

        tile_basename = os.path.basename(ps_rgbnir_path).split(".")[0]

        with rasterio.open(ps_rgbnir_path) as src:
            height, width = src.height, src.width
            transform = src.transform
            crs = src.crs
            ps_data = src.read()
            ps_nodata = src.nodata

        with rasterio.open(sar_intensity_path) as src:
            sar_data = src.read()

        gdf = gpd.read_file(mask_path)
        building_polygons = []

        if len(gdf) > 0:
            if gdf.crs != crs:
                gdf = gdf.to_crs(crs)

            building_polygons = [
                (geom, 1)
                for geom in gdf.geometry
                if geom is not None and not geom.is_empty
            ]

        building_mask = rasterize(
            building_polygons,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True,
        )

        building_mask = building_mask[np.newaxis, :, :]

        patches_per_dim_h = max(1, (height - patch_size[0] + stride[0]) // stride[0])
        patches_per_dim_w = max(1, (width - patch_size[1] + stride[1]) // stride[1])

        for i in range(patches_per_dim_h):
            for j in range(patches_per_dim_w):
                row_start = i * stride[0]
                col_start = j * stride[1]

                if row_start + patch_size[0] > height:
                    row_start = max(0, height - patch_size[0])
                if col_start + patch_size[1] > width:
                    col_start = max(0, width - patch_size[1])

                window = Window(col_start, row_start, patch_size[1], patch_size[0])
                patch_transform = rasterio.windows.transform(window, transform)

                ps_patch = ps_data[
                    :,
                    row_start : row_start + patch_size[0],
                    col_start : col_start + patch_size[1],
                ]

                sar_patch = sar_data[
                    :,
                    row_start : row_start + patch_size[0],
                    col_start : col_start + patch_size[1],
                ]

                mask_patch = building_mask[
                    :,
                    row_start : row_start + patch_size[0],
                    col_start : col_start + patch_size[1],
                ]

                if ps_nodata is not None:
                    patch_valid_ratio = np.sum(ps_patch != ps_nodata) / ps_patch.size
                else:
                    patch_valid_ratio = 1.0

                building_ratio = np.sum(mask_patch > 0) / mask_patch.size

                patch_id = f"{patch_id_prefix}{tile_basename}_{i:03d}_{j:03d}"

                ps_patch_filename = f"{patch_id}_ps-rgbnir.{output_format}"
                sar_patch_filename = f"{patch_id}_sar-intensity.{output_format}"
                mask_patch_filename = f"{patch_id}_mask.{output_format}"

                ps_patch_path = os.path.join(ps_rgbnir_dir, ps_patch_filename)
                sar_patch_path = os.path.join(sar_intensity_dir, sar_patch_filename)
                mask_patch_path = os.path.join(mask_dir, mask_patch_filename)

                ps_out_profile = {
                    "driver": "GTiff",
                    "height": patch_size[0],
                    "width": patch_size[1],
                    "count": ps_patch.shape[0],
                    "dtype": ps_patch.dtype,
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

                sar_out_profile = {
                    "driver": "GTiff",
                    "height": patch_size[0],
                    "width": patch_size[1],
                    "count": sar_patch.shape[0],
                    "dtype": sar_patch.dtype,
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

                mask_out_profile = {
                    "driver": "GTiff",
                    "height": patch_size[0],
                    "width": patch_size[1],
                    "count": 1,
                    "dtype": "uint8",
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

                with rasterio.open(ps_patch_path, "w", **ps_out_profile) as dst:
                    dst.write(ps_patch)

                with rasterio.open(sar_patch_path, "w", **sar_out_profile) as dst:
                    dst.write(sar_patch)

                with rasterio.open(mask_patch_path, "w", **mask_out_profile) as dst:
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
                    "source_img_file": os.path.basename(ps_rgbnir_path),
                    "source_mask_file": os.path.basename(mask_path),
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
                    "building_ratio": float(building_ratio),
                    "valid_ratio": float(patch_valid_ratio),
                    "is_positive": building_ratio > 0,
                    "ps_rgbnir_path": os.path.join("ps-rgbnir", ps_patch_filename),
                    "sar_intensity_path": os.path.join(
                        "sar_intensity", sar_patch_filename
                    ),
                    "mask_path": os.path.join("mask", mask_patch_filename),
                    "split": row["split"] if "split" in row else "train",
                    "date": row["date"] if "date" in row else None,
                }

                result_metadata.append(patch_metadata)

        return result_metadata

    except Exception as e:
        print(f"Error processing tile {idx}: {e}")
        import traceback

        traceback.print_exc()
        return []


def split_spacenet6_into_patches(
    metadata_df: pd.DataFrame,
    root_dir: str,
    output_dir: str,
    patch_size: tuple[int, int] = (448, 448),
    block_size: tuple[int, int] = (448, 448),
    stride: tuple[int, int] | None = None,
    output_format: str = "tif",
    patch_id_prefix: str = "p",
    num_workers: int = 8,
) -> pd.DataFrame:
    """Split SpaceNet6 images and building annotations into smaller patches.

    Args:
        metadata_df: DataFrame with SpaceNet6 metadata including paths
        root_dir: Root directory of the dataset
        output_dir: Directory to save patches
        patch_size: Size of the patches (height, width)
        block_size: Size of the blocks for optimized GeoTIFF writing
        stride: Step size between patches (height, width)
        output_format: Output file format (e.g., 'tif')
        patch_id_prefix: Prefix for patch IDs
        num_workers: Number of parallel processes to use

    Returns:
        DataFrame containing metadata for all created patches
    """
    blockxsize, blockysize = block_size
    blockxsize = blockxsize - (blockxsize % 16) if blockxsize % 16 != 0 else blockxsize
    blockysize = blockysize - (blockysize % 16) if blockysize % 16 != 0 else blockysize

    if stride is None:
        stride = patch_size

    os.makedirs(output_dir, exist_ok=True)
    ps_rgbnir_dir = os.path.join(output_dir, "ps-rgbnir")
    sar_intensity_dir = os.path.join(output_dir, "sar_intensity")
    mask_dir = os.path.join(output_dir, "mask")

    os.makedirs(ps_rgbnir_dir, exist_ok=True)
    os.makedirs(sar_intensity_dir, exist_ok=True)
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
                patch_size,
                blockxsize,
                blockysize,
                stride,
                output_format,
                patch_id_prefix,
            )
            for idx, row in batch.iterrows()
        ]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(
                tqdm(
                    executor.map(process_spacenet6_tile, tasks),
                    total=len(tasks),
                    desc=f"Processing batch {batch_idx + 1}/{len(batches)}",
                )
            )

            for result_list in results:
                all_patch_metadata.extend(result_list)

    patches_df = pd.DataFrame(all_patch_metadata)

    metadata_path = os.path.join(output_dir, "patch_metadata.parquet")
    patches_df.to_parquet(metadata_path, index=False)

    print(f"Created {len(patches_df)} patches from {len(metadata_df)} source tiles")
    print(f"Patch metadata saved to {metadata_path}")

    pos_patches = patches_df[patches_df["is_positive"]]
    neg_patches = patches_df[~patches_df["is_positive"]]
    pos_pct = len(pos_patches) / len(patches_df) * 100 if len(patches_df) > 0 else 0
    neg_pct = len(neg_patches) / len(patches_df) * 100 if len(patches_df) > 0 else 0
    print(f"Patches with buildings: {len(pos_patches)} ({pos_pct:.1f}%)")
    print(f"Patches without buildings: {len(neg_patches)} ({neg_pct:.1f}%)")

    building_ratio_counts = (
        patches_df["building_ratio"].value_counts(bins=10).sort_index()
    )
    print("\nBuilding ratio distribution:")
    for i, (index, count) in enumerate(building_ratio_counts.items()):
        print(
            f"  {index.left:.3f}-{index.right:.3f}: {count} patches ({100 * count / len(patches_df):.1f}%)"
        )

    return patches_df


def generate_metadata_df(root: str) -> pd.DataFrame:
    """Generate metadata DataFrame for SpaceNet6 dataset."""
    label_paths = glob.glob(
        os.path.join(
            root,
            "SN6_buildings",
            "train",
            "train",
            "AOI_11_Rotterdam",
            "geojson_buildings",
            "*.geojson",
        ),
        recursive=True,
    )

    df = pd.DataFrame(label_paths, columns=["mask_path"])

    df["sar_intensity_path"] = (
        df["mask_path"]
        .str.replace(".geojson", ".tif")
        .str.replace("/geojson_buildings/", "/SAR-Intensity/")
        .str.replace("_Buildings_", "_SAR-Intensity_")
    )

    df["ps_rgbnir_path"] = (
        df["mask_path"]
        .str.replace(".geojson", ".tif")
        .str.replace("/geojson_buildings/", "/PS-RGBNIR/")
        .str.replace("_Buildings_", "_PS-RGBNIR_")
    )

    def extract_lng_lat(path):
        with rasterio.open(path, "r") as src:
            lng, lat = src.lnglat()
            height_px, width_px = src.height, src.width

        filename = os.path.basename(path)
        parts = filename.split("_")
        date_match = re.match(r"(\d{4})(\d{2})(\d{2})(\d{6})", parts[6])
        if date_match:
            year, month, day, _ = date_match.groups()
            date = f"{year}-{month}-{day}"
        else:
            date = None

        return lng, lat, date, height_px, width_px

    df["lon"], df["lat"], df["date"], df["height_px"], df["width_px"] = zip(
        *df["ps_rgbnir_path"].apply(extract_lng_lat)
    )

    df["mask_path"] = df["mask_path"].str.replace(root, "").str.lstrip(os.sep)
    df["sar_intensity_path"] = (
        df["sar_intensity_path"].str.replace(root, "").str.lstrip(os.sep)
    )
    df["ps_rgbnir_path"] = df["ps_rgbnir_path"].str.replace(root, "").str.lstrip(os.sep)

    df["split"] = "train"

    return df


def create_tortilla(root_dir, df, save_dir, tortilla_name):
    """Create a tortilla version of the dataset."""
    tortilla_dir = os.path.join(save_dir, "tortilla")
    os.makedirs(tortilla_dir, exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating tortilla"):
        modalities = ["ps_rgbnir", "sar_intensity", "mask"]
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
                add_test_split=row["is_additional_test"],
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
            )

            modality_samples.append(sample)

        taco_samples = tacotoolbox.tortilla.datamodel.Samples(samples=modality_samples)
        samples_path = os.path.join(tortilla_dir, f"sample_{idx}.tortilla")
        tacotoolbox.tortilla.create(taco_samples, samples_path, quiet=True, nworkers=4)

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
            lon=sample_data["lon"],
            lat=sample_data["lat"],
            source_img_file=sample_data["source_img_file"],
            source_mask_file=sample_data["source_mask_file"],
            patch_id=sample_data["patch_id"],
        )
        samples.append(sample_tortilla)

    final_samples = tacotoolbox.tortilla.datamodel.Samples(samples=samples)
    tacotoolbox.tortilla.create(
        final_samples, os.path.join(save_dir, tortilla_name), quiet=True, nworkers=4
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
        n_additional_test_samples: Number of additional test samples
        root_dir: Root directory for the dataset
        save_dir: Directory to save the GeoBench version
    """
    patch_size = (450, 450)
    stride = (449, 449)

    results_path = os.path.join(save_dir, "patch_metadata.parquet")

    if os.path.exists(results_path):
        patches_df = pd.read_parquet(results_path)
    else:
        patches_df = split_spacenet6_into_patches(
            metadata_df=metadata_df,
            root_dir=root_dir,
            output_dir=save_dir,
            patch_size=patch_size,
            block_size=(448, 448),
            stride=stride,
        )
        patches_df.to_parquet(results_path, index=False)

    patches_df = patches_df[patches_df["valid_ratio"] > 0.4].reset_index(drop=True)

    subset_df = create_subset_from_df(
        patches_df,
        n_train_samples=n_train_samples,
        n_val_samples=n_val_samples,
        n_test_samples=n_test_samples,
        n_additional_test_samples=n_additional_test_samples,
        random_state=24,
    )

    return subset_df


def main():
    """Generate SpaceNet6 Benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for SpaceNet6 dataset"
    )
    parser.add_argument(
        "--save_dir",
        default="geobenchV2/spacenet6",
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
        metadata_df.to_parquet(metadata_path)

    checker_split_df = checkerboard_split(
        metadata_df,
        n_blocks_x=13,
        n_blocks_y=13,
        pattern="balanced",
        random_state=42,
        ratio_tolerance=0.02,
    )

    visualize_geospatial_split(
        checker_split_df,
        title="Checkerboard Split",
        output_path=os.path.join(args.save_dir, "checker_split.png"),
        buffer_degrees=0.05,
    )

    plot_sample_locations(
        checker_split_df,
        os.path.join(args.save_dir, "sample_locations.png"),
        buffer_degrees=1.0,
    )

    results_path = os.path.join(args.save_dir, "geoebench_spacenet6.parquet")
    if os.path.exists(results_path):
        result_df = pd.read_parquet(results_path)
    else:
        result_df = create_geobench_version(
            checker_split_df,
            n_train_samples=4000,
            n_val_samples=990,
            n_test_samples=1890,
            n_additional_test_samples=0,
            root_dir=args.root,
            save_dir=args.save_dir,
        )
        result_df.to_parquet(results_path)

    tortilla_name = "geobench_spacenet6.tortilla"
    create_tortilla(
        args.save_dir, result_df, args.save_dir, tortilla_name=tortilla_name
    )

    create_unittest_subset(
        data_dir=args.save_dir,
        tortilla_pattern=f"{tortilla_name.split('.')[0]}.*.part.tortilla",
        test_dir_name="spacenet6",
        n_train_samples=2,
        n_val_samples=1,
        n_test_samples=1,
        n_additional_test_samples=0,
    )


if __name__ == "__main__":
    main()
