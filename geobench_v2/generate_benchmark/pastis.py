# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of PASTIS dataset."""

import argparse
import glob
import json
import os

import geopandas as gpd
import pandas as pd
import tacoreader
import tacotoolbox
from torchgeo.datasets import PASTIS
from tqdm import tqdm

import os
import numpy as np
import h5py
from tqdm import tqdm

from geobench_v2.generate_benchmark.utils import (
    create_subset_from_df,
    create_unittest_subset,
    plot_sample_locations,
)


def generate_metadata_df(ds: PASTIS) -> pd.DataFrame:
    """Generate metadata DataFrame for PASTIS Benchmark."""
    geojson_path = f"{ds.root}/PASTIS-R/metadata.geojson"
    print(f"Loading metadata from {geojson_path}")

    # Load the GeoJSON file
    gdf = gpd.read_file(geojson_path)
    print(f"Loaded {len(gdf)} patches")

    fold_to_split = {1: "train", 2: "train", 3: "train", 4: "val", 5: "test"}

    # Map fold to split
    gdf["split"] = gdf["Fold"].map(fold_to_split)

    # Reproject to WGS84 (lat/lon)
    gdf_wgs84 = gdf.to_crs(epsg=4326)

    gdf_projected = gdf_wgs84.to_crs(epsg=3857)
    centroids_projected = gdf_projected.geometry.centroid
    centroids_wgs84 = gpd.GeoSeries(centroids_projected, crs=3857).to_crs(4326)

    # Extract lat/lon from properly calculated centroids
    gdf["longitude"] = centroids_wgs84.x
    gdf["latitude"] = centroids_wgs84.y

    # Now gdf has the lat/lon coordinates you need
    print(
        f"Coordinate range: lon [{gdf['longitude'].min():.6f}, {gdf['longitude'].max():.6f}], "
        f"lat [{gdf['latitude'].min():.6f}, {gdf['latitude'].max():.6f}]"
    )

    columns_to_drop = ["geometry"]
    geometry = gdf["geometry"]

    df = pd.DataFrame(gdf.drop(columns=columns_to_drop))

    df["ID_PATCH"] = df["ID_PATCH"].astype(str)
    files_df = pd.DataFrame(ds.files)
    files_df["ID_PATCH"] = files_df["s2"].apply(
        lambda x: x.split("/")[-1].split("_")[-1].split(".")[0]
    )

    new_df = pd.merge(
        df, files_df, how="left", left_on="ID_PATCH", right_on="ID_PATCH"
    ).reset_index(drop=True)

    # make s2, s1a, s1d, semantic and instance relative paths
    new_df["s2"] = new_df["s2"].apply(lambda x: x.replace(ds.root + "/", ""))
    new_df["s1a"] = new_df["s1a"].apply(lambda x: x.replace(ds.root + "/", ""))
    new_df["s1d"] = new_df["s1d"].apply(lambda x: x.replace(ds.root + "/", ""))
    new_df["semantic"] = new_df["semantic"].apply(
        lambda x: x.replace(ds.root + "/", "")
    )
    new_df["instance"] = new_df["instance"].apply(
        lambda x: x.replace(ds.root + "/", "")
    )

    # rename those columns to append _path
    new_df.rename(
        columns={
            "s2": "s2_path",
            "s1a": "s1a_path",
            "s1d": "s1d_path",
            "semantic": "semantic_path",
            "instance": "instance_path",
        },
        inplace=True,
    )

    new_df["dates-s2"] = new_df["dates-S2"].apply(
        lambda x: list(json.loads(x.replace("'", '"')).values())
    )
    new_df["dates-s1a"] = new_df["dates-S1A"].apply(
        lambda x: list(json.loads(x.replace("'", '"')).values())
    )
    new_df["dates-s1d"] = new_df["dates-S1D"].apply(
        lambda x: list(json.loads(x.replace("'", '"')).values())
    )

    new_df.drop(columns=["dates-S1A", "dates-S1D", "dates-S2"], inplace=True)

    new_df["split"] = new_df["split"].replace("val", "validation")

    # conve
    final_gdf = gpd.GeoDataFrame(new_df, geometry=geometry)

    return final_gdf


def _convert_pastis_row_to_h5(task):
    """Convert a single row of PASTIS DataFrame from .npy to .h5 format.
    
    Args:
        task: Tuple containing:
            - idx: Index of the row in the DataFrame
            - row: Row data as a dictionary
            - root_dir: Root directory of the PASTIS dataset
            - save_dir: Directory to save the converted HDF5 files
            - modalities: List of modalities to convert
            - compression: Compression type for HDF5 files
            - compression_level: Compression level for HDF5 files
            - overwrite: Whether to overwrite existing HDF5 files
    
    Returns:
        Tuple containing:
            - idx: Index of the row in the DataFrame
            - updates: Dictionary with updated paths for each modality
            - If a modality file was successfully converted, the value is the new relative path.
            - If conversion failed, the value is None.
    """
    idx, row, root_dir, save_dir, modalities, compression, compression_level, overwrite = task
    updates = {}
    for modality in modalities:
        col = f"{modality}_path"
        src_rel = row.get(col)
        if not isinstance(src_rel, str) or not src_rel:
            continue

        src_abs = os.path.join(root_dir, src_rel)
        tgt_rel = os.path.splitext(os.path.join("hdf5", src_rel))[0] + ".h5"
        tgt_abs = os.path.join(save_dir, tgt_rel)

        os.makedirs(os.path.dirname(tgt_abs), exist_ok=True)

        if not overwrite and os.path.exists(tgt_abs):
            updates[col] = tgt_rel
            continue

        arr = np.load(src_abs, allow_pickle=False)
        with h5py.File(tgt_abs, "w") as hf:
            dset = hf.create_dataset(
                "data",
                data=arr,
                compression=compression,
                compression_opts=compression_level if compression == "gzip" else None,
                chunks=True,
            )
            dset.attrs["modality"] = modality
            dset.attrs["source_path"] = src_rel
        updates[col] = tgt_rel

    return idx, updates

def convert_pastis_numpy_to_hdf5(
    df: pd.DataFrame,
    root_dir: str,
    save_dir: str,
    modalities: list[str] = ("s2", "s1a", "s1d", "semantic", "instance"),
    compression: str = "gzip",
    compression_level: int = 4,
    overwrite: bool = True,
    num_workers: int = 8,
) -> pd.DataFrame:
    """Convert per-modality .npy arrays to per-modality HDF5 files and update paths (parallel).
    
    Args:
        df: DataFrame with metadata including paths to .npy files.
        root_dir: Root directory of the PASTIS dataset.
        save_dir: Directory to save the converted HDF5 files.
        modalities: List of modalities to convert.
        compression: Compression type for HDF5 files (default: "gzip").
        compression_level: Compression level for HDF5 files (default: 4).
        overwrite: Whether to overwrite existing HDF5 files (default: False).
        num_workers: Number of parallel workers to use (default: 8).
    """
    out_df = df.copy()
    base_out = os.path.join(save_dir, "hdf5")
    os.makedirs(base_out, exist_ok=True)

    tasks = [
        (
            idx,
            {k: v for k, v in row.items()},
            root_dir,
            save_dir,
            modalities,
            compression,
            compression_level,
            overwrite,
        )
        for idx, row in out_df.iterrows()
    ]

    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(_convert_pastis_row_to_h5, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Converting .npy -> .h5"):
            idx, updates = fut.result()
            for col, tgt_rel in updates.items():
                if tgt_rel:
                    out_df.at[idx, col] = tgt_rel

    return out_df


def create_tortilla(root_dir, df, save_dir, tortilla_name) -> None:
    """Create a subset of PASTIS dataset for Tortilla Benchmark."""
    tortilla_dir = os.path.join(save_dir, "tortilla")
    os.makedirs(tortilla_dir, exist_ok=True)

    # for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating tortilla"):
    #     modalities = ["s2", "s1a", "s1d", "semantic", "instance"]
    #     modality_samples = []

    #     for modality in modalities:
    #         path = os.path.join(root_dir, row[modality + "_path"])
            
    #         sample = tacotoolbox.tortilla.datamodel.Sample(
    #             id=modality,
    #             path=path,
    #             file_format="HDF5",
    #             data_split=row["split"],
    #             add_test_split=row["is_additional_test"],
    #             dates=row[f"dates-{modality}"] if f"dates-{modality}" in row else [],
    #             tile=row["TILE"],
    #             lon=row["longitude"],
    #             lat=row["latitude"],
    #             patch_id=row["ID_PATCH"],
    #         )

    #         modality_samples.append(sample)

    #     taco_samples = tacotoolbox.tortilla.datamodel.Samples(samples=modality_samples)
    #     samples_path = os.path.join(tortilla_dir, f"sample_{idx}.tortilla")
    #     tacotoolbox.tortilla.create(taco_samples, samples_path, quiet=True, nworkers=4)

    # # merge tortillas into a single dataset
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
            dates=sample_data["dates"],
            tile=sample_data["tile"],
            lon=sample_data["lon"],
            lat=sample_data["lat"],
            patch_id=sample_data["patch_id"],
        )
        samples.append(sample_tortilla)

    # create final taco file
    final_samples = tacotoolbox.tortilla.datamodel.Samples(samples=samples)
    tacotoolbox.tortilla.create(
        final_samples, os.path.join(save_dir, tortilla_name), quiet=True, nworkers=4
    )


def create_geobench_version(
    metadata_df: pd.DataFrame,
    n_train_samples: int,
    n_val_samples: int,
    n_test_samples: int,
    n_additional_test_samples: int
) -> pd.DataFrame:
    """Create a GeoBench version of the dataset.

    Args:
        metadata_df: DataFrame with metadata including geolocation for each patch
        n_train_samples: Number of final training samples, -1 means all
        n_val_samples: Number of final validation samples, -1 means all
        n_test_samples: Number of final test samples, -1 means all
        n_additional_test_samples: Number of additional test samples from train set
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


def create_unit_test_subset() -> None:
    """Create a subset of PASTIS dataset for GeoBench unit tests."""
    # create random images etc that respect the structure of the dataset in minimal format
    pass


def main():
    """Generate PASTIS Benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for PASTIS dataset"
    )
    parser.add_argument(
        "--save_dir",
        default="geobenchV2/pastis",
        help="Directory to save the subset benchmark data",
    )
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    orig_dataset = PASTIS(root=args.root, download=False)

    metadata_path = os.path.join(args.save_dir, "geobench_metadata.parquet")
    if os.path.exists(metadata_path):
        metadata_df = pd.read_parquet(metadata_path)
    else:
        metadata_df = generate_metadata_df(orig_dataset)
        metadata_df.to_parquet(metadata_path)

    result_path = os.path.join(args.save_dir, "geobench_pastis.parquet")
    if os.path.exists(result_path):
        h5_df = pd.read_parquet(result_path)
    else:
        result_df = create_geobench_version(
            metadata_df, n_train_samples=1200, n_val_samples=482, n_test_samples=496, n_additional_test_samples=255
        )

        h5_df = convert_pastis_numpy_to_hdf5(
            result_df, root_dir=args.root, save_dir=args.save_dir, overwrite=False
        )
        h5_df.to_parquet(result_path)

    tortilla_name = "geobench_pastis.tortilla"
    # create_tortilla(args.root, h5_df, args.save_dir, tortilla_name)

    create_unittest_subset(
        data_dir=args.save_dir,
        tortilla_pattern="geobench_pastis.*.part.tortilla",
        test_dir_name="pastis",
        n_train_samples=4,
        n_val_samples=2,
        n_test_samples=2,
        n_additional_test_samples=1,
    )

    plot_sample_locations(
        metadata_df,
        os.path.join(args.save_dir, "sample_locations.png"),
        buffer_degrees=1.5,
        s=2,
    )


if __name__ == "__main__":
    main()
