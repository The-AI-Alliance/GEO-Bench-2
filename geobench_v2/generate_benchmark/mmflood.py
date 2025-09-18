# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of MMFlood dataset."""

import argparse
import glob
import os

import pandas as pd
import rasterio
import tacoreader
import tacotoolbox
from tqdm import tqdm

from geobench_v2.generate_benchmark.utils import (
    create_subset_from_df,
    create_unittest_subset,
    plot_sample_locations,
)


def generate_metadata_df(root) -> pd.DataFrame:
    """Generate metadata DataFrame for MMFlood dataset."""
    metadata: list[dict[str, str]] = []

    paths = glob.glob(
        os.path.join(root, "activations", "**", "mask", "*.tif"), recursive=True
    )

    df = pd.DataFrame(paths, columns=["mask_path"])

    meta_data = pd.read_json(os.path.join(root, "activations.json"), orient="index")
    meta_data.reset_index(inplace=True)
    meta_data.rename(columns={"index": "region_id"}, inplace=True)

    df["s1_path"] = df["mask_path"].str.replace("/mask/", "/s1_raw/")
    df["hydro_path"] = df["mask_path"].str.replace("/mask/", "/hydro/")
    df["dem_path"] = df["mask_path"].str.replace("/mask/", "/DEM/")

    # only keep rows where hydro_path exists
    df["hydro_path_exist"] = df["hydro_path"].apply(
        lambda x: True if os.path.exists(x) else False
    )
    df = df[df["hydro_path_exist"]].reset_index(drop=True)
    df.drop(columns=["hydro_path_exist"], inplace=True)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating metadata"):
        mask_path = row["mask_path"]

        with rasterio.open(mask_path) as src:
            lng, lat = src.lnglat()
            height_px, width_px = src.height, src.width

            tags = src.tags()
            event_date = tags["event_date"]

        metadata.append(
            {
                "mask_path": mask_path,
                "longitude": lng,
                "latitude": lat,
                "date": event_date,
                "height_px": height_px,
                "width_px": width_px,
            }
        )

    metadata_df = pd.DataFrame(metadata)

    full_df = pd.merge(df, metadata_df, on="mask_path", how="left")

    # make all paths relative
    for col in ["mask_path", "s1_path", "hydro_path", "dem_path"]:
        full_df[col] = full_df[col].str.replace(root, "")

    full_df["aoi"] = full_df["mask_path"].str.split(os.sep, expand=True)[1]
    full_df["region_id"] = full_df["aoi"].str.split("-", expand=True)[0]

    full_df = pd.merge(
        full_df,
        meta_data[["region_id", "country", "start", "end", "subset"]],
        on="region_id",
        how="left",
    )
    # rename the column correctly
    full_df.rename(columns={"subset": "split"}, inplace=True)
    full_df["split"] = full_df["split"].replace({"val": "validation"})

    return full_df


def create_tortilla(root_dir, df, save_dir, tortilla_name):
    """Create a tortilla version of the dataset."""
    tortilla_dir = os.path.join(save_dir, "tortilla")
    os.makedirs(tortilla_dir, exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating tortilla"):
        modalities = ["s1", "dem", "hydro", "mask"]
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
                    "time_start": row["start"],
                    "time_end": row["end"],
                },
                lon=row["lon"],
                lat=row["lat"],
                source_s1_file=row["source_s1_file"],
                source_mask_file=row["source_mask_file"],
                region_id=row["region_id"],
                aoi=row["aoi"],
                country=row["country"],
            )

            modality_samples.append(sample)

        taco_samples = tacotoolbox.tortilla.datamodel.Samples(samples=modality_samples)
        samples_path = os.path.join(tortilla_dir, f"sample_{idx}.tortilla")
        tacotoolbox.tortilla.create(taco_samples, samples_path, quiet=True, nworkers=4)

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
            source_s1_file=sample_data["source_s1_file"],
            source_mask_file=sample_data["source_mask_file"],
            region_id=sample_data["region_id"],
            aoi=sample_data["aoi"],
            country=sample_data["country"],
        )
        samples.append(sample_tortilla)

    # create final taco file
    final_samples = tacotoolbox.tortilla.datamodel.Samples(samples=samples)
    tacotoolbox.tortilla.create(
        final_samples, os.path.join(save_dir, tortilla_name), quiet=True, nworkers=4
    )


def process_mmflood_patch(args):
    """Process a single MMFlood patch by extracting, optimizing, and writing to disk.

    Args:
        args: Tuple containing:
            - idx: Patch index
            - row_start: Starting row of the patch
            - col_start: Starting column of the patch
            - tile_id: ID of the source tile
            - row_idx: Row index in patch grid
            - col_idx: Column index in patch grid
            - modalities_data: Dict of modalities data arrays
            - modalities_profiles: Dict of modalities profiles
            - patch_size: Size of patch
            - output_dir: Directory to save patches
            - mask_transform: Transform from mask
            - row_metadata: Metadata from parent tile

    Returns:
        Dict with patch metadata or None if processing failed
    """
    try:
        (
            row_start,
            col_start,
            tile_id,
            row_idx,
            col_idx,
            modalities_data,
            modalities_profiles,
            patch_size,
            output_dir,
            mask_transform,
            row_metadata,
        ) = args

        modalities = list(modalities_data.keys())
        patch_id = f"{tile_id}_p{row_idx}_{col_idx}"
        patch_paths = {}

        mask_patch = modalities_data["mask"][
            :, row_start : row_start + patch_size, col_start : col_start + patch_size
        ]
        valid_pixels = (mask_patch != 255).sum()
        valid_ratio = valid_pixels / (patch_size * patch_size * mask_patch.shape[0])
        positive_pixels = (mask_patch == 1).sum()
        positive_ratio = positive_pixels / (
            patch_size * patch_size * mask_patch.shape[0]
        )

        patch_transform = rasterio.transform.from_origin(
            mask_transform.c + col_start * mask_transform.a,
            mask_transform.f + row_start * mask_transform.e,
            mask_transform.a,
            mask_transform.e,
        )

        for modality in modalities:
            modality_patch = modalities_data[modality][
                :,
                row_start : row_start + patch_size,
                col_start : col_start + patch_size,
            ]

            optimized_profile = {
                "driver": "GTiff",
                "height": patch_size,
                "width": patch_size,
                "count": modality_patch.shape[0],
                "dtype": modality_patch.dtype,
                "tiled": True,
                "blockxsize": min(512, patch_size),
                "blockysize": min(512, patch_size),
                "interleave": "pixel",
                "compress": "zstd",
                "zstd_level": 13,
                "predictor": 2,
                "crs": modalities_profiles[modality]["crs"],
                "transform": patch_transform,
            }

            if "nodata" in modalities_profiles[modality]:
                optimized_profile["nodata"] = modalities_profiles[modality]["nodata"]

            os.makedirs(os.path.join(output_dir, modality), exist_ok=True)

            patch_path = os.path.join(output_dir, modality, f"{patch_id}.tif")
            patch_paths[f"{modality}_path"] = os.path.relpath(patch_path, output_dir)

            with rasterio.open(patch_path, "w", **optimized_profile) as dst:
                dst.write(modality_patch)

        patch_bounds = rasterio.transform.array_bounds(
            patch_size, patch_size, patch_transform
        )
        west, south, east, north = patch_bounds
        center_x = (west + east) / 2
        center_y = (north + south) / 2

        patch_metadata = {
            "source_mask_file": row_metadata["mask_path"],
            "source_s1_file": row_metadata["s1_path"],
            "source_hydro_file": row_metadata["hydro_path"],
            "source_dem_file": row_metadata["dem_path"],
            "patch_id": patch_id,
            "mask_path": patch_paths["mask_path"],
            "s1_path": patch_paths["s1_path"],
            "hydro_path": patch_paths["hydro_path"],
            "dem_path": patch_paths["dem_path"],
            "lon": center_x,
            "lat": center_y,
            "height_px": patch_size,
            "width_px": patch_size,
            "row": row_idx,
            "col": col_idx,
            "row_px": int(row_start),
            "col_px": int(col_start),
            "valid_ratio": float(valid_ratio),
            "positive_ratio": float(positive_ratio),
            "date": row_metadata["date"],
            "aoi": row_metadata["aoi"],
            "region_id": row_metadata["region_id"],
            "country": row_metadata["country"],
            "start": row_metadata["start"],
            "end": row_metadata["end"],
            "split": row_metadata["split"],
        }

        return patch_metadata

    except Exception as e:
        print(f"Error processing patch: {e}")
        import traceback

        traceback.print_exc()
        return None


def create_mmflood_patches(
    metadata_df: pd.DataFrame,
    root_dir: str,
    output_dir: str,
    patch_size: int = 512,
    max_overlap_fraction: float = 0.2,
    num_workers: int = 8,
) -> pd.DataFrame:
    """Split MMFlood tiles into patches of specified size.

    Args:
        metadata_df: DataFrame with image/mask paths and metadata
        root_dir: Root directory of MMFlood dataset
        output_dir: Directory to save patches
        patch_size: Size of patches (height=width)
        max_overlap_fraction: Maximum allowed overlap fraction when optimizing coverage
        num_workers: Number of parallel workers

    Returns:
        DataFrame with metadata for all created patches
    """
    from concurrent.futures import ProcessPoolExecutor

    modalities = ["mask", "s1", "hydro", "dem"]

    for mod in modalities:
        os.makedirs(os.path.join(output_dir, mod), exist_ok=True)

    all_tasks = []

    for idx, row in tqdm(
        metadata_df.iterrows(), total=len(metadata_df), desc="Preparing tasks"
    ):
        tile_id = os.path.basename(row["mask_path"]).split(".")[0]
        mask_path = os.path.join(root_dir, row["mask_path"])

        with rasterio.open(mask_path) as src:
            height, width = src.height, src.width
            mask_transform = src.transform

        # Skip tiles smaller than patch size
        if height <= patch_size and width <= patch_size:
            continue

        # Calculate row and column starting positions
        max_overlap_pixels = int(patch_size * max_overlap_fraction)

        if height <= patch_size:
            row_starts = [(height - patch_size) // 2]
        elif height <= patch_size + max_overlap_pixels:
            row_starts = [0]
        else:
            num_rows = max(
                1,
                (height + max_overlap_pixels - 1) // (patch_size - max_overlap_pixels),
            )
            if num_rows == 1:
                row_starts = [(height - patch_size) // 2]
            else:
                row_step = (height - patch_size) / (num_rows - 1) if num_rows > 1 else 0
                row_starts = [int(i * row_step) for i in range(num_rows)]

        if width <= patch_size:
            col_starts = [(width - patch_size) // 2]
        elif width <= patch_size + max_overlap_pixels:
            col_starts = [0]
        else:
            num_cols = max(
                1, (width + max_overlap_pixels - 1) // (patch_size - max_overlap_pixels)
            )
            if num_cols == 1:
                col_starts = [(width - patch_size) // 2]
            else:
                col_step = (width - patch_size) / (num_cols - 1) if num_cols > 1 else 0
                col_starts = [int(i * col_step) for i in range(num_cols)]

        modality_data = {}
        modality_profiles = {}
        for modality in modalities:
            path_key = f"{modality}_path" if modality != "s1" else "s1_path"
            file_path = os.path.join(root_dir, row[path_key])
            with rasterio.open(file_path) as src:
                modality_data[modality] = src.read()
                modality_profiles[modality] = src.profile.copy()

        for i, row_start in enumerate(row_starts):
            for j, col_start in enumerate(col_starts):
                row_start = min(row_start, height - patch_size)
                col_start = min(col_start, width - patch_size)

                task = (
                    row_start,
                    col_start,
                    tile_id,
                    i,
                    j,
                    modality_data,
                    modality_profiles,
                    patch_size,
                    output_dir,
                    mask_transform,
                    row,
                )
                all_tasks.append(task)

    all_patch_metadata = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(
                executor.map(process_mmflood_patch, all_tasks),
                total=len(all_tasks),
                desc="Processing patches",
            )
        )

        for result in results:
            if result is not None:
                all_patch_metadata.append(result)

    patches_df = pd.DataFrame(all_patch_metadata)

    metadata_path = os.path.join(output_dir, "patch_metadata.parquet")
    patches_df.to_parquet(metadata_path, index=False)

    print(f"Created {len(patches_df)} patches from {len(metadata_df)} source tiles")

    for split in patches_df["split"].unique():
        split_count = len(patches_df[patches_df["split"] == split])
        split_pct = split_count / len(patches_df) * 100
        print(f"{split} patches: {split_count} ({split_pct:.1f}%)")

    return patches_df


def create_geobench_version(
    metadata_df: pd.DataFrame,
    n_train_samples: int,
    n_val_samples: int,
    n_test_samples: int,
) -> None:
    """Create a GeoBench version of the dataset.

    Args:
        metadata_df: DataFrame with metadata including geolocation for each patch
        n_train_samples: Number of final training samples, -1 means all
        n_val_samples: Number of final validation samples, -1 means all
        n_test_samples: Number of final test samples, -1 means all
    """
    random_state = 24

    subset_df = create_subset_from_df(
        metadata_df,
        n_train_samples=n_train_samples,
        n_val_samples=n_val_samples,
        n_test_samples=n_test_samples,
        random_state=random_state,
    )

    return subset_df


def main():
    """Generate MMFlood Benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for MMFlood dataset"
    )
    parser.add_argument(
        "--save_dir", default="geobenchV2/MMFlood", help="Directory to save the subset"
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

    plot_sample_locations(
        metadata_df,
        os.path.join(args.save_dir, "sample_locations.png"),
        buffer_degrees=1.0,
        dataset_name="MMFlood",
    )

    path = os.path.join(args.save_dir, "patch_metadata.parquet")

    if os.path.exists(path):
        patch_metadata_df = pd.read_parquet(path)
    else:
        patch_metadata_df = create_mmflood_patches(
            metadata_df,
            args.root,
            os.path.join(args.save_dir, "patches"),
            patch_size=512,
        )
        patch_metadata_df.to_parquet(path)

    tortilla_name = "geobench_mmflood.tortilla"
    create_tortilla(
        os.path.join(args.save_dir, "patches"),
        patch_metadata_df,
        args.save_dir,
        tortilla_name,
    )
    create_unittest_subset(
        data_dir=args.save_dir,
        tortilla_pattern=tortilla_name,
        test_dir_name="mmflood",
        n_train_samples=4,
        n_val_samples=2,
        n_test_samples=2,
    )


if __name__ == "__main__":
    main()
