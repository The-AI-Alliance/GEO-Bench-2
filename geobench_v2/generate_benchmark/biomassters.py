# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of BioMassters dataset."""

import argparse
import glob
import os
import shutil
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import rasterio
import tacoreader
import tacotoolbox
from rasterio.errors import NotGeoreferencedWarning
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from geobench_v2.generate_benchmark.utils import create_subset_from_df


def consolidate_bio_meta_df(df: pd.DataFrame) -> pd.DataFrame:
    """Consolidate BioMassters dataset by chip_id to create one row per chip.

    with columns for S1 and S2 paths and months.

    Args:
        df: DataFrame with BioMassters metadata

    Returns:
        Consolidated DataFrame with one row per chip_id
    """
    consolidated_records = []

    unique_chip_ids = df["chip_id"].unique()

    for chip_id in tqdm(unique_chip_ids, desc="Consolidating chip data"):
        chip_data = df[df["chip_id"] == chip_id]

        s1_data = chip_data[chip_data["satellite"] == "S1"]
        s2_data = chip_data[chip_data["satellite"] == "S2"]

        split = chip_data["split"].iloc[0]

        features_dir = "test_features" if split == "test" else "train_features"
        agbm_dir = "test_agbm" if split == "test" else "train_agbm"

        record = {
            "chip_id": chip_id,
            "split": split,
            "S1_paths": [
                os.path.join(features_dir, filename)
                for filename in s1_data["filename"].tolist()
            ],
            "S1_months": s1_data["month"].tolist(),
            "S2_paths": [
                os.path.join(features_dir, filename)
                for filename in s2_data["filename"].tolist()
            ],
            "S2_months": s2_data["month"].tolist(),
            "agbm_path": os.path.join(
                agbm_dir, chip_data["corresponding_agbm"].iloc[0]
            ),
        }

        consolidated_records.append(record)

    consolidated_df = pd.DataFrame(consolidated_records)

    consolidated_df["num_S1_images"] = consolidated_df["S1_paths"].apply(len)
    consolidated_df["num_S2_images"] = consolidated_df["S2_paths"].apply(len)

    print(
        f"Consolidated {len(unique_chip_ids)} chips to {len(consolidated_df)} rows with both S1 and S2 data"
    )
    return consolidated_df


def generate_metadata_df(root: str) -> pd.DataFrame:
    """Generate metadata DataFrame for BioMassters dataset."""
    feature_df = pd.read_csv(os.path.join(root, "biomassters_features_metadata.csv"))
    consolidated_df = consolidate_bio_meta_df(feature_df)

    # create random train/val split from train entries
    train_df = consolidated_df[consolidated_df["split"] == "train"]

    _, val_indices = train_test_split(train_df.index, test_size=0.2, random_state=42)
    consolidated_df.loc[val_indices, "split"] = "validation"

    split_counts = consolidated_df["split"].value_counts()
    print(f"Split distribution: {split_counts.to_dict()}")

    return consolidated_df


def create_tortilla(root_dir, df, save_dir, tortilla_name):
    """Create a tortilla version of the dataset."""
    tortilla_dir = os.path.join(save_dir, "tortilla")
    os.makedirs(tortilla_dir, exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating tortilla"):
        modalities = ["S1", "S2"]
        modality_samples = []

        for modality in modalities:
            # for timesteps
            for path, month in zip(row[f"{modality}_paths"], row[f"{modality}_months"]):
                sample = tacotoolbox.tortilla.datamodel.Sample(
                    id=f"{row['chip_id']}_{modality}_{month}",
                    path=os.path.join(root_dir, path),
                    file_format="GTiff",
                    data_split=row["split"],
                    add_test_split=row["add_test_split"],
                    month=month,
                    source_img_file=path,
                    modality=modality,
                )

                modality_samples.append(sample)

        sample = tacotoolbox.tortilla.datamodel.Sample(
            id=f"{row['chip_id']}_AGBM",
            path=os.path.join(root_dir, row["agbm_path"]),
            file_format="GTiff",
            data_split=row["split"],
            add_test_split=row["add_test_split"],
            modality="AGBM",
            source_img_file=row["agbm_path"],
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
            data_split=sample_data["tortilla:data_split"],
            add_test_split=sample_data["add_test_split"],
        )
        samples.append(sample_tortilla)

    final_samples = tacotoolbox.tortilla.datamodel.Samples(samples=samples)
    tacotoolbox.tortilla.create(
        final_samples,
        os.path.join(save_dir, tortilla_name),
        quiet=True,
        nworkers=4,
        chunk_size="48GB",
    )


def process_biomassters_sample(args):
    """Process a single BioMassters sample by rewriting with optimized profile.

    Args:
        args: Tuple containing:
            - idx: Sample index
            - row: DataFrame row with metadata
            - root_dir: Root directory containing original data
            - output_dir: Directory to save optimized files

    Returns:
        Dict with updated paths or None if processing failed
    """
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
    idx, row, root_dir, output_dir = args

    try:
        chip_id = row["chip_id"]
        split = row["split"]

        features_dir = os.path.join(output_dir, "features")
        agbm_dir = os.path.join(output_dir, "agbm")
        os.makedirs(features_dir, exist_ok=True)
        os.makedirs(agbm_dir, exist_ok=True)

        s1_paths = []
        s1_months = []
        s2_paths = []
        s2_months = []

        for s1_path, s1_month in zip(row["S1_paths"], row["S1_months"]):
            src_path = os.path.join(root_dir, s1_path)
            out_filename = f"{chip_id}_S1_{s1_month}.tif"
            dst_path = os.path.join(features_dir, out_filename)

            with rasterio.open(src_path) as src:
                data = src.read()
                height, width = src.height, src.width

                transform = rasterio.transform.Affine(1.0, 0.0, 0.0, 0.0, -1.0, height)
                crs = rasterio.crs.CRS.from_epsg(4326)

                optimized_profile = {
                    "driver": "GTiff",
                    "height": height,
                    "width": width,
                    "count": data.shape[0],
                    "dtype": data.dtype,
                    "tiled": True,
                    "blockxsize": min(256, data.shape[2]),
                    "blockysize": min(256, data.shape[1]),
                    "interleave": "pixel",
                    "compress": "zstd",
                    "zstd_level": 13,
                    "predictor": 2,
                    "crs": crs,
                    "transform": transform,
                }

                if src.nodata is not None:
                    optimized_profile["nodata"] = src.nodata

                with rasterio.open(dst_path, "w", **optimized_profile) as dst:
                    dst.write(data)

                rel_path = os.path.relpath(dst_path, output_dir)
                s1_paths.append(rel_path)
                s1_months.append(s1_month)

        for s2_path, s2_month in zip(row["S2_paths"], row["S2_months"]):
            src_path = os.path.join(root_dir, s2_path)
            out_filename = f"{chip_id}_S2_{s2_month}.tif"
            dst_path = os.path.join(features_dir, out_filename)

            with rasterio.open(src_path) as src:
                data = src.read()

                transform = rasterio.transform.Affine(1.0, 0.0, 0.0, 0.0, -1.0, height)
                crs = rasterio.crs.CRS.from_epsg(4326)

                optimized_profile = {
                    "driver": "GTiff",
                    "height": data.shape[1],
                    "width": data.shape[2],
                    "count": data.shape[0],
                    "dtype": data.dtype,
                    "tiled": True,
                    "blockxsize": 256,
                    "blockysize": 256,
                    "interleave": "pixel",
                    "compress": "zstd",
                    "zstd_level": 13,
                    "predictor": 2,
                    "crs": crs,
                    "transform": transform,
                }

                if src.nodata is not None:
                    optimized_profile["nodata"] = src.nodata

                with rasterio.open(dst_path, "w", **optimized_profile) as dst:
                    dst.write(data)

                rel_path = os.path.relpath(dst_path, output_dir)
                s2_paths.append(rel_path)
                s2_months.append(s2_month)

        src_path = os.path.join(root_dir, row["agbm_path"])
        out_filename = f"{chip_id}_AGBM.tif"
        dst_path = os.path.join(agbm_dir, out_filename)

        with rasterio.open(src_path) as src:
            data = src.read()

            optimized_profile = {
                "driver": "GTiff",
                "height": data.shape[1],
                "width": data.shape[2],
                "count": data.shape[0],
                "dtype": data.dtype,
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
                "interleave": "pixel",
                "compress": "zstd",
                "zstd_level": 13,
                "predictor": 2,
                "crs": src.crs,
                "transform": src.transform,
            }

            if src.nodata is not None:
                optimized_profile["nodata"] = src.nodata

            with rasterio.open(dst_path, "w", **optimized_profile) as dst:
                dst.write(data)

            rel_agbm_path = os.path.relpath(dst_path, output_dir)

        return {
            "chip_id": chip_id,
            "split": split,
            "S1_paths": s1_paths,
            "S1_months": s1_months,
            "S2_paths": s2_paths,
            "S2_months": s2_months,
            "agbm_path": rel_agbm_path,
            "num_S1_images": len(s1_paths),
            "num_S2_images": len(s2_paths),
            "add_test_split": row["add_test_split"],
        }

    except Exception as e:
        print(f"Error processing sample {idx}: {str(e)}")
        return None


def optimize_biomassters_dataset(metadata_df, root_dir, output_dir, num_workers=1):
    """Rewrite BioMassters dataset with optimized GeoTIFF profile in parallel.

    Args:
        metadata_df: DataFrame with metadata about samples
        root_dir: Root directory containing original data
        output_dir: Directory to save optimized files
        num_workers: Number of parallel workers

    Returns:
        DataFrame with updated metadata
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "features"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "agbm"), exist_ok=True)

    tasks = [(idx, row, root_dir, output_dir) for idx, row in metadata_df.iterrows()]

    processed_samples = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(
                executor.map(process_biomassters_sample, tasks),
                total=len(tasks),
                desc="Processing samples",
            )
        )

        for result in results:
            if result is not None:
                processed_samples.append(result)

    updated_df = pd.DataFrame(processed_samples)

    metadata_path = os.path.join(output_dir, "biomassters_optimized.parquet")
    updated_df.to_parquet(metadata_path)

    print(f"Processed {len(updated_df)} samples successfully")
    print(f"Updated metadata saved to {metadata_path}")

    return updated_df


def create_test_subset(
    root_dir: str,
    df: pd.DataFrame,
    save_dir: str,
    num_train_samples: int = 4,
    num_val_samples: int = 2,
    num_test_samples: int = 2,
    n_additional_test_samples: int = 0,
) -> None:
    """Create a test subset of the BioMassters dataset with downsampled 32x32 images.

    Args:
        root_dir: Root directory containing original BioMassters data
        df: DataFrame with BioMassters metadata
        save_dir: Directory to save the downsampled test subset
        num_train_samples: Number of training samples to include
        num_val_samples: Number of validation samples to include
        num_test_samples: Number of test samples to include
        n_additional_test_samples: Number of additional test samples from train split
    """
    import warnings

    from rasterio.errors import NotGeoreferencedWarning

    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

    test_dir = os.path.join(save_dir, "unittest")
    test_features_dir = os.path.join(test_dir, "test_features")
    test_agbm_dir = os.path.join(test_dir, "test_agbm")
    train_features_dir = os.path.join(test_dir, "train_features")
    train_agbm_dir = os.path.join(test_dir, "train_agbm")

    for directory in [
        test_dir,
        test_features_dir,
        test_agbm_dir,
        train_features_dir,
        train_agbm_dir,
    ]:
        os.makedirs(directory, exist_ok=True)

    # Prepare pools by split
    train_pool = df[df["split"] == "train"]
    val_pool = df[df["split"] == "validation"]
    test_pool = df[df["split"] == "test"]

    total_train_needed = num_train_samples + n_additional_test_samples
    if total_train_needed > len(train_pool):
        raise ValueError(
            f"Not enough training samples: need {total_train_needed} "
            f"({num_train_samples} train + {n_additional_test_samples} additional test) "
            f"but only {len(train_pool)} available."
        )

    # Sample training and additional test disjointly from train
    if n_additional_test_samples > 0:
        selected = train_pool.sample(total_train_needed, random_state=42)
        train_df = selected.iloc[:num_train_samples].copy()
        additional_df = selected.iloc[num_train_samples:].copy()
        additional_df["add_test_split"] = True
    else:
        train_df = train_pool.sample(num_train_samples, random_state=42).copy()
        additional_df = pd.DataFrame(columns=train_pool.columns)

    val_df = val_pool.sample(num_val_samples, random_state=42).copy()
    test_df = test_pool.sample(num_test_samples, random_state=42).copy()

    subset_df = pd.concat([train_df, val_df, test_df, additional_df]).reset_index(
        drop=True
    )
    subset_df["add_test_split"] = subset_df.get("add_test_split", False)

    print(
        f"Created subset with {len(subset_df)} samples: "
        f"{num_train_samples} train, {num_val_samples} validation, "
        f"{num_test_samples} test, {n_additional_test_samples} additional-from-train"
    )

    subset_metadata = []

    for idx, row in tqdm(
        subset_df.iterrows(), total=len(subset_df), desc="Creating downsampled subset"
    ):
        chip_id = row["chip_id"]
        split = row["split"]

        features_dir = test_features_dir if split == "test" else train_features_dir
        agbm_dir = test_agbm_dir if split == "test" else train_agbm_dir

        s1_paths = []
        s1_months = []
        for s1_path, s1_month in zip(row["S1_paths"], row["S1_months"]):
            with rasterio.open(os.path.join(root_dir, s1_path)) as src:
                profile = src.profile.copy()
                data = src.read()

                data_small = np.zeros((data.shape[0], 32, 32), dtype=data.dtype)
                for band_idx in range(data.shape[0]):
                    data_small[band_idx] = resize(
                        data[band_idx], (32, 32), preserve_range=True
                    ).astype(data.dtype)

                profile.update(height=32, width=32)
                filename = os.path.basename(s1_path)
                new_path = os.path.join(features_dir, f"small_{filename}")

                with rasterio.open(new_path, "w", **profile) as dst:
                    dst.write(data_small)

                rel_path = os.path.relpath(new_path, test_dir)
                s1_paths.append(rel_path)
                s1_months.append(s1_month)

        s2_paths = []
        s2_months = []
        for s2_path, s2_month in zip(row["S2_paths"], row["S2_months"]):
            with rasterio.open(os.path.join(root_dir, s2_path)) as src:
                profile = src.profile.copy()
                data = src.read()

                data_small = np.zeros((data.shape[0], 32, 32), dtype=data.dtype)
                for band_idx in range(data.shape[0]):
                    data_small[band_idx] = resize(
                        data[band_idx], (32, 32), preserve_range=True
                    ).astype(data.dtype)

                profile.update(height=32, width=32)
                filename = os.path.basename(s2_path)
                new_path = os.path.join(features_dir, f"small_{filename}")

                with rasterio.open(new_path, "w", **profile) as dst:
                    dst.write(data_small)

                rel_path = os.path.relpath(new_path, test_dir)
                s2_paths.append(rel_path)
                s2_months.append(s2_month)

        with rasterio.open(os.path.join(root_dir, row["agbm_path"])) as src:
            profile = src.profile.copy()
            data = src.read()

            data_small = np.zeros((data.shape[0], 32, 32), dtype=data.dtype)
            for band_idx in range(data.shape[0]):
                data_small[band_idx] = resize(
                    data[band_idx], (32, 32), preserve_range=True
                ).astype(data.dtype)

            profile.update(height=32, width=32)
            filename = os.path.basename(row["agbm_path"])
            new_agbm_path = os.path.join(agbm_dir, f"small_{filename}")

            with rasterio.open(new_agbm_path, "w", **profile) as dst:
                dst.write(data_small)

            rel_agbm_path = os.path.relpath(new_agbm_path, test_dir)

        subset_metadata.append(
            {
                "chip_id": chip_id,
                "split": split,
                "S1_paths": s1_paths,
                "S1_months": s1_months,
                "S2_paths": s2_paths,
                "S2_months": s2_months,
                "agbm_path": rel_agbm_path,
                "num_S1_images": len(s1_paths),
                "num_S2_images": len(s2_paths),
                "add_test_split": row["add_test_split"],
            }
        )

    subset_df = pd.DataFrame(subset_metadata)
    subset_df.to_parquet(os.path.join(test_dir, "subset_metadata.parquet"))

    create_tortilla(
        test_dir, subset_df, os.path.join(save_dir, "unittest"), "biomassters.tortilla"
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(script_dir))
    test_data_dir = os.path.join(repo_root, "tests", "data", "biomassters")
    os.makedirs(test_data_dir, exist_ok=True)

    tortilla_path = os.path.join(save_dir, "unittest", "biomassters.tortilla")

    size_mb = os.path.getsize(tortilla_path) / (1024 * 1024)
    print(f"Tortilla file size: {size_mb:.2f} MB")
    shutil.copy(tortilla_path, os.path.join(test_data_dir, "biomassters.tortilla"))

    print(f"Test subset created successfully at {test_dir}")
    print(
        f"Tortilla file copied to {os.path.join(test_data_dir, 'biomassters.tortilla')}"
    )


def create_geobench_version(
    metadata_df: pd.DataFrame,
    n_train_samples: int,
    n_val_samples: int,
    n_test_samples: int,
    n_additional_test_samples: int,
) -> None:
    """Create a GeoBench version of the dataset.

    Args:
        metadata_df: DataFrame with metadata including geolocation for each patch
        n_train_samples: Number of final training samples, -1 means all
        n_val_samples: Number of final validation samples, -1 means all
        n_test_samples: Number of final test samples, -1 means all
        n_additional_test_samples: Number of additional test samples to include from train set
    """
    subset_df = create_subset_from_df(
        metadata_df,
        n_train_samples=n_train_samples,
        n_val_samples=n_val_samples,
        n_test_samples=n_test_samples,
        n_additional_test_samples=n_additional_test_samples,
        random_state=24,
    )

    return subset_df


def main():
    """Generate BioMassters Benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for BioMassters dataset"
    )
    parser.add_argument(
        "--save_dir",
        default="geobenchV2/BioMassters",
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

    results_path = os.path.join(args.save_dir, "geobench_biomassters.parquet")
    if os.path.exists(results_path):
        results_df = pd.read_parquet(results_path)
    else:
        results_df = create_geobench_version(
            metadata_df,
            n_train_samples=4000,
            n_val_samples=1000,
            n_test_samples=2000,
            n_additional_test_samples=0,
        )
        results_df.to_parquet(results_path)

    optimized_path = os.path.join(args.save_dir, "biomassters_optimized.parquet")
    if os.path.exists(optimized_path):
        optimized_df = pd.read_parquet(optimized_path)
    else:
        optimized_df = optimize_biomassters_dataset(
            results_df, args.root, args.save_dir, num_workers=8
        )
        optimized_df.to_parquet(optimized_path)

    tortilla_name = "geobench_biomassters.tortilla"
    create_tortilla(args.save_dir, optimized_df, args.save_dir, tortilla_name)

    create_test_subset(
        args.save_dir,
        optimized_df,
        save_dir=args.save_dir,
        num_train_samples=4,
        num_val_samples=2,
        num_test_samples=2,
        n_additional_test_samples=0,
    )


if __name__ == "__main__":
    main()
