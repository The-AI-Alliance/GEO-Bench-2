# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of MADOS dataset."""

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import rasterio
import tacoreader
import tacotoolbox
from rasterio.warp import Resampling
from tqdm import tqdm

from geobench_v2.generate_benchmark.utils import (
    create_subset_from_df,
    create_unittest_subset,
)

mados_name_to_band = {
    # 10m resolution bands
    "rhorc_492": "B02",  # Blue
    "rhorc_559": "B03",  # Green
    "rhorc_560": "B03",  # Green
    "rhorc_665": "B04",  # Red
    "rhorc_833": "B08",  # NIR
    # 20m resolution bands
    "rhorc_704": "B05",  # Red Edge 1
    "rhorc_740": "B06",  # Red Edge 2
    "rhorc_739": "B06",  # Red Edge 2
    "rhorc_780": "B07",  # Red Edge 3
    "rhorc_783": "B07",  # Red Edge 3
    "rhorc_864": "B8A",  # Narrow NIR
    "rhorc_865": "B8A",  # Narrow NIR
    "rhorc_1610": "B11",  # SWIR 1
    "rhorc_1614": "B11",  # SWIR 1
    "rhorc_2186": "B12",  # SWIR 2
    "rhorc_2202": "B12",  # SWIR 2
    # 60m resolution bands
    "rhorc_442": "B01",  # Coastal aerosol
    "rhorc_443": "B01",  # Coastal aerosol
    # Special products
    "TUR_Dogliotti": "TUR_Dogliotti",
    "TUR_Dogliotti2015": "TUR_Dogliotti",
    "TUR_Nechad2016_665": "TUR_Nechad",
    # Classification and quality layers
    "cl": "CL",  # Classification layer
    "conf": "CONF",  # Confidence layer
    "rep": "REP",  # Reprojection quality
    "rgb": "RGB",  # RGB visualization
}

bands_to_keep = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B11",
    "B12",
    "CL",
    "RGB",
    "TUR_Dogliotti",
    "TUR_Nechad",
    "CONF",
    "REP",
]


def generate_metadata_df(root: str) -> pd.DataFrame:
    """Generate metadata DataFrame for MADOS dataset with band mapping."""
    paths = glob.glob(os.path.join(root, "MADOS", "Scene_*", "*", "Scene_*_L2*_*.*"))

    splits = {}
    split_files = glob.glob(os.path.join(root, "MADOS", "splits", "*.txt"))
    for split_file in split_files:
        split_name = os.path.basename(split_file).split(".")[0].split("_")[0]
        with open(split_file) as f:
            scene_ids = [line.strip() for line in f.readlines()]
            for scene_id in scene_ids:
                splits[scene_id] = split_name

    metadata_records = []

    for path in tqdm(paths, desc="Processing MADOS files"):
        rel_path = os.path.relpath(path, os.path.join(root, "MADOS"))

        parts = rel_path.split(os.sep)
        scene_dir = parts[0]
        subdirectory = parts[1]

        filename = os.path.basename(path)

        filename_parts = filename.split("_")
        scene_num = filename_parts[1]
        product_type = filename_parts[2]

        if len(filename_parts) >= 5:
            modality = filename_parts[3]
            if len(filename_parts) > 5:
                modality = "_".join(filename_parts[3:-1])
                crop_id = filename_parts[-1].replace(".tif", "").replace(".png", "")
            else:
                crop_id = filename_parts[4].replace(".tif", "").replace(".png", "")
        else:
            modality = filename_parts[3].split(".")[0]
            crop_id = "CROP"
        band_name = mados_name_to_band.get(modality, modality)

        scene_id = f"{scene_dir}_{crop_id}"
        split = splits.get(scene_id, "unknown")

        metadata_records.append(
            {
                "scene": scene_num,
                "scene_dir": scene_dir,
                "resolution": subdirectory,
                "product_type": product_type,
                "modality": modality,
                "band_name": band_name,
                "crop_id": crop_id,
                "path": rel_path,
                "split": split,
            }
        )

    df = pd.DataFrame(metadata_records)

    scene_data = []

    for (scene_dir, crop_id), group in tqdm(
        df.groupby(["scene_dir", "crop_id"]), desc="Organizing samples"
    ):
        record = {
            "scene": group["scene"].iloc[0],
            "scene_dir": scene_dir,
            "crop_id": crop_id,
            "split": group["split"].iloc[0],
        }
        for band in bands_to_keep:
            band_found = False
            for _, row in group.iterrows():
                if mados_name_to_band.get(row["modality"]) == band:
                    record[f"{band}_path"] = row["path"]
                    record[f"{band}_res"] = row["resolution"]
                    band_found = True
                    break

            if not band_found:
                matching_rows = group[group["modality"] == band]
                if not matching_rows.empty:
                    record[f"{band}_path"] = matching_rows.iloc[0]["path"]
                    record[f"{band}_res"] = matching_rows.iloc[0]["resolution"]

        # # For original modality paths, add them directly
        # for modality in df["modality"].unique():
        #     matching_rows = group[group["modality"] == modality]
        #     if not matching_rows.empty:
        #         record[f"{modality}_path"] = matching_rows.iloc[0]["path"]
        #         record[f"{modality}_res"] = matching_rows.iloc[0]["resolution"]

        scene_data.append(record)

    full_df = pd.DataFrame(scene_data)

    full_df["split"] = full_df["split"].replace({"val": "validation"})

    print(f"Found {len(paths)} total files across {len(full_df)} unique samples")
    print(f"Split distribution: {full_df['split'].value_counts().to_dict()}")
    band_path_cols = [name + "_path" for name in bands_to_keep]
    essential_cols = ["scene", "scene_dir", "crop_id", "split"] + band_path_cols

    return full_df[essential_cols]


def create_tortilla(root_dir, df, save_dir, tortilla_name):
    """Create a tortilla version of the MADOS dataset with only Sentinel-2 data and CL mask.

    Args:
        root_dir: Root directory containing MADOS data
        df: DataFrame with metadata
        save_dir: Directory to save tortilla files
        tortilla_name: Name of the final tortilla file
    """
    tortilla_dir = os.path.join(save_dir, "tortilla")
    os.makedirs(tortilla_dir, exist_ok=True)

    for idx, row in tqdm(
        df.iterrows(), total=len(df), desc="Creating tortilla samples"
    ):
        modality_samples = []

        sentinel2_path = os.path.join(save_dir, "converted", row["sentinel2_path"])

        s2_sample = tacotoolbox.tortilla.datamodel.Sample(
            id=f"{row['scene_dir']}_{row['crop_id']}_sentinel2",
            path=sentinel2_path,
            file_format="GTiff",
            data_split=row["split"],
            patch_id=f"{row['scene_dir']}_{row['crop_id']}",
            modality="sentinel2",
        )
        modality_samples.append(s2_sample)

        cl_path = os.path.join(root_dir, "MADOS", row["CL_path"])

        mask_sample = tacotoolbox.tortilla.datamodel.Sample(
            id=f"{row['scene_dir']}_{row['crop_id']}_mask",
            path=cl_path,
            file_format="GTiff" if cl_path.endswith(".tif") else "PNG",
            data_split=row["split"],
            patch_id=f"{row['scene_dir']}_{row['crop_id']}",
            modality="mask",
        )
        modality_samples.append(mask_sample)

        # Skip if we don't have both modalities
        if len(modality_samples) < 2:
            continue

        taco_samples = tacotoolbox.tortilla.datamodel.Samples(samples=modality_samples)
        sample_base_id = f"scene_{row['scene_dir']}_{row['crop_id']}"
        samples_path = os.path.join(tortilla_dir, f"{sample_base_id}.tortilla")
        tacotoolbox.tortilla.create(taco_samples, samples_path, quiet=True, nworkers=4)

    # Find all created tortilla files
    all_tortilla_files = sorted(glob.glob(os.path.join(tortilla_dir, "*.tortilla")))
    print(f"Created {len(all_tortilla_files)} individual tortilla files")

    # final tortilla
    samples = []

    for tortilla_file in tqdm(all_tortilla_files, desc="Building final taco"):
        sample_data = tacoreader.load(tortilla_file).iloc[0]

        sample_tortilla = tacotoolbox.tortilla.datamodel.Sample(
            id=os.path.basename(tortilla_file).split(".")[0],
            path=tortilla_file,
            file_format="TORTILLA",
            data_split=sample_data["tortilla:data_split"],
            patch_id=sample_data["patch_id"],
            modality=sample_data["modality"],
        )
        samples.append(sample_tortilla)

    final_samples = tacotoolbox.tortilla.datamodel.Samples(samples=samples)
    final_taco_path = os.path.join(save_dir, tortilla_name)
    tacotoolbox.tortilla.create(final_samples, final_taco_path, quiet=True, nworkers=4)


def process_mados_sample(args):
    """Process a single MADOS sample by resampling and saving with optimized profile."""
    idx, row, output_dir, root_dir, target_size = args

    try:
        # Output paths for both image and mask
        img_id = f"{row['scene_dir']}_{row['crop_id']}"
        img_output_path = os.path.join(output_dir, "scene", f"{img_id}.tif")
        mask_output_path = os.path.join(
            output_dir, "mask", f"{row['scene_dir']}_{row['crop_id']}.tif"
        )

        os.makedirs(os.path.join(output_dir, "scene"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "mask"), exist_ok=True)

        # Collect all available bands
        bands_data = []
        band_paths = []

        for band in [
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B11",
            "B12",
        ]:
            band_key = f"{band}_path"
            if band_key in row and pd.notna(row[band_key]):
                band_paths.append(os.path.join(root_dir, "MADOS", row[band_key]))

        # Check for mask path
        mask_path = None
        if "CL_path" in row and pd.notna(row["CL_path"]):
            mask_path = os.path.join(root_dir, "MADOS", row["CL_path"])


        for band_path in band_paths:
            with rasterio.open(band_path) as src:
                data = src.read(
                    1,
                    out_shape=(target_size, target_size),
                    resampling=Resampling.bilinear,
                )
                bands_data.append(data)

        stacked_data = np.stack(bands_data, axis=0)

        reference_crs = rasterio.crs.CRS.from_epsg(4326)  # WGS84 as default
        reference_transform = rasterio.transform.from_bounds(
            0, 0, target_size, target_size, target_size, target_size
        )
        reference_width = target_size
        reference_height = target_size

        new_transform = rasterio.transform.from_bounds(
            *rasterio.transform.array_bounds(
                reference_height, reference_width, reference_transform
            ),
            target_size,
            target_size,
        )

        # Create optimized profile
        optimized_profile = {
            "driver": "GTiff",
            "height": target_size,
            "width": target_size,
            "count": len(bands_data),
            "dtype": stacked_data.dtype,
            "tiled": True,
            "blockxsize": 224,
            "blockysize": 224,
            "interleave": "pixel",
            "compress": "zstd",
            "zstd_level": 13,
            "predictor": 2,
            "crs": reference_crs,
            "transform": new_transform,
        }

        # Write stacked bands to a single file
        with rasterio.open(img_output_path, "w", **optimized_profile) as dst:
            dst.write(stacked_data)

        with rasterio.open(mask_path) as src:
            mask_data = src.read(
                out_shape=(src.count, target_size, target_size),
                resampling=Resampling.nearest,
            )

            # Create optimized profile for mask
            mask_profile = {
                "driver": "GTiff",
                "height": target_size,
                "width": target_size,
                "count": mask_data.shape[0],
                "dtype": mask_data.dtype,
                "tiled": True,
                "blockxsize": 224,
                "blockysize": 224,
                "interleave": "pixel",
                "compress": "zstd",
                "zstd_level": 13,
                "predictor": 2,
                "crs": reference_crs,
                "transform": new_transform,
            }

            # Write mask to file
            with rasterio.open(mask_output_path, "w", **mask_profile) as dst:
                dst.write(mask_data)

        return img_output_path, mask_output_path

    except Exception as e:
        print(f"Error processing sample {idx}: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


def convert_mados_dataset(
    metadata_df, output_dir, root_dir, target_size=224, num_workers=8
):
    """Convert MADOS dataset to optimized GeoTIFF format with all bands stacked."""
    os.makedirs(output_dir, exist_ok=True)

    tasks = [
        (idx, row, output_dir, root_dir, target_size)
        for idx, row in metadata_df.iterrows()
    ]

    results = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list_of_results = list(
            tqdm(
                executor.map(process_mados_sample, tasks),
                total=len(tasks),
                desc="Processing MADOS samples",
            )
        )

        for result in list_of_results:
            s2_path = result[0]
            mask_path = result[1]
            results.append(
                {
                    "sentinel2_path": os.path.relpath(s2_path, start=output_dir),
                    "mask_path": os.path.relpath(mask_path, start=output_dir),
                }
            )

    # Update metadata with new paths
    results_df = pd.DataFrame(results)

    results_df["sample_id"] = results_df["sentinel2_path"].apply(
        lambda x: os.path.basename(x).split(".")[0]
    )
    metadata_df["sample_id"] = metadata_df["scene_dir"] + "_" + metadata_df["crop_id"]

    # Merge with original metadata
    optimized_df = metadata_df.copy()
    optimized_df = optimized_df.merge(results_df, on="sample_id", how="left")

    # Save updated metadata
    metadata_path = os.path.join(output_dir, "mados_optimized.parquet")
    optimized_df.to_parquet(metadata_path)

    print(f"Saved {len(results)} optimized MADOS samples")
    print(f"Updated metadata saved to {metadata_path}")

    return optimized_df


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

    converted_df = convert_mados_dataset(
        metadata_df, os.path.join(save_dir, "converted"), root_dir, target_size=224
    )

    subset_df = create_subset_from_df(
        converted_df,
        n_train_samples=n_train_samples,
        n_val_samples=n_val_samples,
        n_test_samples=n_test_samples,
        random_state=random_state,
    )

    return subset_df


def main():
    """Generate MADOS Benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for MADOS dataset"
    )
    parser.add_argument(
        "--save_dir", default="geobenchV2/MADOS", help="Direcdtory to save the subset"
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

    result_path = os.path.join(args.save_dir, "geobench_mados.parquet")

    if os.path.exists(result_path):
        result_df = pd.read_parquet(result_path)
    else:
        result_df = create_geobench_version(
            metadata_df,
            n_train_samples=-1,
            n_val_samples=-1,
            n_test_samples=-1,
            root_dir=args.root,
            save_dir=args.save_dir,
        )
        result_df.to_parquet(result_path)

    tortilla_name = "geobench_mados.tortilla"
    create_tortilla(args.root, result_df, args.save_dir, tortilla_name)

    create_unittest_subset(
        data_dir=args.save_dir,
        tortilla_pattern=tortilla_name,
        test_dir_name="mados",
        n_train_samples=4,
        n_val_samples=2,
        n_test_samples=2,
    )


if __name__ == "__main__":
    main()
