# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of PASTIS dataset."""

import argparse
import glob
import io
import json
import os
import re
import shutil
import tempfile

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import tacoreader
import tacotoolbox
from skimage.transform import resize
from torchgeo.datasets import PASTIS
from tqdm import tqdm

from geobench_v2.generate_benchmark.utils import (
    create_subset_from_df,
    plot_sample_locations,
)


def generate_metadata_df(ds: PASTIS) -> pd.DataFrame:
    """Generate metadata DataFrame for PASTIS Benchmark."""
    geojson_path = f"{ds.root}/PASTIS-R/metadata.geojson"
    print(f"Loading metadata from {geojson_path}")

    gdf = gpd.read_file(geojson_path)
    print(f"Loaded {len(gdf)} patches")

    fold_to_split = {1: "train", 2: "train", 3: "train", 4: "val", 5: "test"}

    # Map fold to split
    gdf["split"] = gdf["Fold"].map(fold_to_split)

    gdf_wgs84 = gdf.to_crs(epsg=4326)

    gdf_projected = gdf_wgs84.to_crs(epsg=3857)
    centroids_projected = gdf_projected.geometry.centroid
    centroids_wgs84 = gpd.GeoSeries(centroids_projected, crs=3857).to_crs(4326)

    gdf["longitude"] = centroids_wgs84.x
    gdf["latitude"] = centroids_wgs84.y

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
    (
        idx,
        row,
        root_dir,
        save_dir,
        modalities,
        compression,
        compression_level,
        overwrite,
    ) = task
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
        for fut in tqdm(
            as_completed(futures), total=len(futures), desc="Converting .npy -> .h5"
        ):
            idx, updates = fut.result()
            for col, tgt_rel in updates.items():
                if tgt_rel:
                    out_df.at[idx, col] = tgt_rel

    return out_df


def create_tortilla(root_dir, df, save_dir, tortilla_name) -> None:
    """Create a subset of PASTIS dataset for Tortilla Benchmark."""
    tortilla_dir = os.path.join(save_dir, "tortilla")
    os.makedirs(tortilla_dir, exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating tortilla"):
        modalities = ["s2", "s1a", "s1d", "semantic", "instance"]
        modality_samples = []

        for modality in modalities:
            path = os.path.join(root_dir, row[modality + "_path"])

            sample = tacotoolbox.tortilla.datamodel.Sample(
                id=modality,
                path=path,
                file_format="HDF5",
                data_split=row["split"],
                add_test_split=row["is_additional_test"],
                dates=row[f"dates-{modality}"] if f"dates-{modality}" in row else [],
                tile=row["TILE"],
                lon=row["longitude"],
                lat=row["latitude"],
                patch_id=row["ID_PATCH"],
            )

            modality_samples.append(sample)

        taco_samples = tacotoolbox.tortilla.datamodel.Samples(samples=modality_samples)
        samples_path = os.path.join(tortilla_dir, f"sample_{idx}.tortilla")
        tacotoolbox.tortilla.create(taco_samples, samples_path, quiet=True, nworkers=4)

    all_tortilla_files = glob.glob(os.path.join(tortilla_dir, "*.tortilla"))
    all_tortilla_files = sorted(
        all_tortilla_files,
        key=lambda x: int(re.search(r"sample_(\d+)\.tortilla", x).group(1)),
    )

    samples = []

    for idx, tortilla_file in tqdm(
        enumerate(all_tortilla_files),
        total=len(all_tortilla_files),
        desc="Building taco",
    ):
        taco_df = tacoreader.load(tortilla_file)
        s2_dates = taco_df.loc[taco_df["tortilla:id"] == "s2", "dates"].iloc[0].tolist()

        sample_data = taco_df.iloc[0]

        sample_tortilla = tacotoolbox.tortilla.datamodel.Sample(
            id=os.path.basename(tortilla_file).split(".")[0],
            path=tortilla_file,
            file_format="TORTILLA",
            data_split=sample_data["tortilla:data_split"],
            add_test_split=sample_data["add_test_split"],
            dates=s2_dates,
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
    n_additional_test_samples: int,
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


def create_unit_test_subset(data_dir, test_dir_name) -> None:
    """Create a compact unittest tortilla from the final tortilla by subsampling and shrinking arrays."""
    small_tortilla = create_compact_unittest_tortilla_from_final(
        final_tortilla_path=sorted(
            glob.glob(os.path.join(data_dir, "*.part.tortilla"))
        ),
        save_dir=data_dir,
        n_train_samples=4,
        n_val_samples=2,
        n_test_samples=2,
        n_additional_test_samples=0,
        target_size=32,
        max_timesteps=3,
        random_state=42,
        out_name=f"{test_dir_name}_unittest_small.tortilla",
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(script_dir))
    test_data_dir = os.path.join(repo_root, "tests", "data", test_dir_name)
    os.makedirs(test_data_dir, exist_ok=True)
    dst = os.path.join(test_data_dir, f"{test_dir_name}.tortilla")
    shutil.copyfile(small_tortilla, dst)
    print(f"Unit test tortilla copied to {dst}")
    print(f"Unit test subset saved to {dst}")
    print(f"Filesize: {os.path.getsize(dst) / (1024 * 1024):.2f} MB")


def _determine_layout(arr: np.ndarray):
    """Return axes (t_axis, c_axis, h_axis, w_axis).

    Args:
        arr: array to check
    """
    if arr.ndim == 4:
        # [T, C, H, W]
        return (0, 1, 2, 3)
    if arr.ndim == 3:
        # [C, H, W] (e.g., mask with channel dim)
        return (None, 0, 1, 2)
    if arr.ndim == 2:
        # [H, W]
        return (None, None, 0, 1)
    raise ValueError(
        f"Unsupported array shape for visualization/processing: {arr.shape}"
    )


def _process_modality_arr(
    arr: np.ndarray,
    keep_t: int,
    target_size: int,
    is_label: bool,
    expected_t: int | None,
) -> np.ndarray:
    """Trim time and resize H,W.

    Assumptions:
    - Time-series imagery arrives as [T, C, H, W]
    - Masks arrive as [H, W] or [C, H, W]

    Returns:
        Array with preserved leading dims (T and/or C if present) and resized spatial dims [H, W].
    """
    out_dtype = arr.dtype
    t_axis, c_axis, h_axis, w_axis = _determine_layout(arr)
    axes = list(range(arr.ndim))
    perm = []
    if t_axis is not None:
        perm.append(t_axis)
    if c_axis is not None:
        perm.append(c_axis)
    perm.extend([h_axis, w_axis])
    remaining = [ax for ax in axes if ax not in perm]
    perm.extend(remaining)
    arr_std = np.transpose(arr, perm)

    if t_axis is not None:
        t_len = arr_std.shape[0]
        arr_std = arr_std[: min(keep_t, t_len)]

    if arr_std.ndim == 2:
        arr_std = resize(
            arr_std,
            (target_size, target_size),
            order=0 if is_label else 1,
            preserve_range=True,
            anti_aliasing=(not is_label),
        ).astype(out_dtype, copy=False)
    else:
        lead_shape = arr_std.shape[:-2]
        flat = arr_std.reshape((-1, arr_std.shape[-2], arr_std.shape[-1]))
        order = 0 if is_label else 1
        resized = np.stack(
            [
                resize(
                    x,
                    (target_size, target_size),
                    order=order,
                    preserve_range=True,
                    anti_aliasing=(order != 0),
                ).astype(out_dtype, copy=False)
                for x in flat
            ],
            axis=0,
        )
        arr_std = resized.reshape(tuple(lead_shape) + (target_size, target_size))
    return arr_std


def create_compact_unittest_tortilla_from_final(
    final_tortilla_path: str,
    save_dir: str,
    n_train_samples: int = 4,
    n_val_samples: int = 2,
    n_test_samples: int = 2,
    n_additional_test_samples: int = 1,
    target_size: int = 32,
    max_timesteps: int = 3,
    random_state: int = 42,
    out_name: str = "pastis_unittest_small.tortilla",
) -> str:
    """Create unittest version with fewer time steps and smaller spatial size.

    Args:
        final_tortilla_path: Path to the final PASTIS Benchmark tortilla file
        save_dir: Directory to save the compact unittest tortilla
        n_train_samples: Number of training samples in the unittest subset
        n_val_samples: Number of validation samples in the unittest subset
        n_test_samples: Number of test samples in the unittest subset
        n_additional_test_samples: Number of additional test samples from train set
        target_size: Target spatial size (H, W) for all modalities
        max_timesteps: Maximum number of time steps to keep for time-series modalities
        random_state: Random state for reproducibility
        out_name: Name of the output compact tortilla file

    Returns:
        Path to the created compact unittest tortilla file
    """
    os.makedirs(save_dir, exist_ok=True)

    taco = tacoreader.load(final_tortilla_path)

    subset_top = create_subset_from_df(
        taco,
        n_train_samples=n_train_samples,
        n_val_samples=n_val_samples,
        n_test_samples=n_test_samples,
        n_additional_test_samples=n_additional_test_samples,
        random_state=random_state,
        split_column="tortilla:data_split",
    ).reset_index(drop=True)

    id_to_pos = {taco.iloc[i]["tortilla:id"]: i for i in range(len(taco))}
    small_root = tempfile.mkdtemp(prefix="pastis_unittest_")
    out_h5_root = os.path.join(small_root, "hdf5")
    out_tortilla_dir = os.path.join(small_root, "tortilla")
    os.makedirs(out_h5_root, exist_ok=True)
    os.makedirs(out_tortilla_dir, exist_ok=True)

    def _bytesio_from_internal_subfile(internal_subfile: str) -> io.BytesIO:
        m = re.search(r"/vsisubfile/(\d+)_(\d+),(.+)", internal_subfile)
        if not m:
            raise ValueError(f"Unrecognized internal:subfile: {internal_subfile}")
        offset, length, tortilla_path = int(m.group(1)), int(m.group(2)), m.group(3)
        with open(tortilla_path, "rb") as f:
            f.seek(offset)
            data = f.read(length)
        return io.BytesIO(data)

    written_parts = []
    for _, row in tqdm(
        subset_top.iterrows(), total=len(subset_top), desc="Building compact unittest"
    ):
        sample_id = row["tortilla:id"]
        pos = id_to_pos[sample_id]
        inner = taco.read(pos)

        split = inner["tortilla:data_split"].iloc[0]
        add_test = bool(inner.get("add_test_split", pd.Series([False])).iloc[0])
        patch_id = str(inner.get("patch_id", pd.Series([sample_id])).iloc[0])
        dates = inner["dates"].iloc[0]
        tile = inner.get("tile", pd.Series([""])).iloc[0]
        lon = float(inner.get("lon", pd.Series([np.nan])).iloc[0])
        lat = float(inner.get("lat", pd.Series([np.nan])).iloc[0])

        modality_samples = []
        for _, mrow in inner.iterrows():
            modality = mrow["tortilla:id"]
            internal_subfile = mrow["internal:subfile"]
            dates = mrow.get("dates", [])

            with h5py.File(
                _bytesio_from_internal_subfile(internal_subfile), "r"
            ) as hf_in:
                arr = hf_in["data"][...]

            expected_t = len(dates) if isinstance(dates, (list, tuple)) else None
            is_ts = modality in ("s2", "s1a", "s1d")
            is_label = modality in ("semantic", "instance")

            arr_small = _process_modality_arr(
                arr=arr,
                keep_t=max_timesteps if is_ts else 1_000_000,
                target_size=target_size,
                is_label=is_label,
                expected_t=expected_t,
            )

            if is_ts:
                dates_small = dates[: arr_small.shape[0]]
            else:
                dates_small = []

            h5_name = f"{patch_id}_{modality}.h5"
            h5_rel = os.path.join("hdf5", str(tile), str(patch_id), h5_name)
            h5_abs = os.path.join(small_root, h5_rel)
            os.makedirs(os.path.dirname(h5_abs), exist_ok=True)
            with h5py.File(h5_abs, "w") as hf_out:
                dset = hf_out.create_dataset(
                    "data",
                    data=arr_small,
                    compression="gzip",
                    compression_opts=4,
                    chunks=True,
                )
                dset.attrs["modality"] = modality
                dset.attrs["source_patch_id"] = patch_id

            sample = tacotoolbox.tortilla.datamodel.Sample(
                id=modality,
                path=h5_abs,
                file_format="HDF5",
                data_split=split,
                add_test_split=add_test,
                dates=dates_small,
                tile=tile,
                lon=lon,
                lat=lat,
                patch_id=patch_id,
            )
            modality_samples.append(sample)

        per_sample = tacotoolbox.tortilla.datamodel.Samples(samples=modality_samples)
        part_path = os.path.join(out_tortilla_dir, f"{sample_id}.tortilla")
        tacotoolbox.tortilla.create(per_sample, part_path, quiet=True, nworkers=4)
        written_parts.append(part_path)

    merged_samples = []
    for tf in sorted(written_parts):
        row = tacoreader.load(tf).iloc[0]
        merged_samples.append(
            tacotoolbox.tortilla.datamodel.Sample(
                id=os.path.basename(tf).split(".")[0],
                path=tf,
                file_format="TORTILLA",
                data_split=row["tortilla:data_split"],
                add_test_split=bool(row.get("add_test_split", False)),
                dates=row["dates"],
                tile=row.get("tile"),
                lon=float(row.get("lon", np.nan)),
                lat=float(row.get("lat", np.nan)),
                patch_id=str(row.get("patch_id")),
            )
        )
    final = tacotoolbox.tortilla.datamodel.Samples(samples=merged_samples)
    final_path = os.path.join(save_dir, out_name)
    tacotoolbox.tortilla.create(final, final_path, quiet=True, nworkers=4)
    print(f"Wrote compact unittest tortilla: {final_path}")
    return final_path


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
            metadata_df,
            n_train_samples=1200,
            n_val_samples=482,
            n_test_samples=496,
            n_additional_test_samples=0,
        )

        h5_df = convert_pastis_numpy_to_hdf5(
            result_df, root_dir=args.root, save_dir=args.save_dir, overwrite=False
        )
        h5_df.to_parquet(result_path)

    tortilla_name = "geobench_pastis.tortilla"
    create_tortilla(args.root, h5_df, args.save_dir, tortilla_name)

    create_unit_test_subset(data_dir=args.save_dir, test_dir_name="pastis")

    plot_sample_locations(
        metadata_df,
        os.path.join(args.save_dir, "sample_locations.png"),
        buffer_degrees=1.5,
        s=2,
    )


if __name__ == "__main__":
    main()
