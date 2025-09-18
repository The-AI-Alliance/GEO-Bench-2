# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of SpaceNet8 dataset."""

import argparse
import gzip
import json
import os
import pickle as pkl
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio as rio
import tacoreader
import tacotoolbox
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchgeo.datasets.utils import percentile_normalization
from tqdm import tqdm

from geobench_v2.generate_benchmark.utils import (
    create_subset_from_df,
    create_unittest_subset,
)

# name mappings
modality_mapper = {
    "MK0_DEM": "aux_dem",
    "MK0_SLOPE": "aux_slope",
    "MK0_MLU": "mask_target",
    "MK0_MNA": "mask_invalid_data",
    "SL1_IVV": "pre_event_2_vv",
    "SL1_IVH": "pre_event_2_vh",
    "SL2_IVV": "pre_event_1_vv",
    "SL2_IVH": "pre_event_1_vh",
    "MS1_IVV": "event_vv",
    "MS1_IVH": "event_vh",
}

name_mapper = {
    "aux_dem": "MK0_DEM",
    "aux_slope": "MK0_SLOPE",
    "mask_target": "MK0_MLU",
    "mask_invalid_data": "MK0_MNA",
    "pre_event_2_vv": "SL1_IVV",
    "pre_event_2_vh": "SL1_IVH",
    "pre_event_1_vv": "SL2_IVV",
    "pre_event_1_vh": "SL2_IVH",
    "event_vv": "MS1_IVV",
    "event_vh": "MS1_IVH",
}

modality_order = (
    "pre_event_1_vv",
    "pre_event_1_vh",
    "pre_event_2_vv",
    "pre_event_2_vh",
    "event_vv",
    "event_vh",
    "aux_slope",
    "aux_dem",
    "mask_target",
    "mask_invalid_data",
)

# Split definitions
# https://github.com/Orion-AI-Lab/KuroSiwo/blob/e9ded558cc9a11bdfa2f09727543c715874353b8/utilities/utilities.py#L415
train_acts = [
    130,
    470,
    555,
    118,
    174,
    324,
    421,
    554,
    427,
    518,
    502,
    498,
    497,
    496,
    492,
    147,
    267,
    273,
    275,
    417,
    567,
    1111011,
    1111004,
    1111009,
    1111010,
    1111006,
    1111005,
]
val_acts = [514, 559, 279, 520, 437, 1111003, 1111008]
test_acts = [321, 561, 445, 562, 411, 1111002, 277, 1111007, 205, 1111013]


def create_split_mapper():
    """Create a mapping of event IDs to data splits."""
    split_mapper = {}
    for act in train_acts:
        split_mapper[act] = "train"
    for act in val_acts:
        split_mapper[act] = "validation"
    for act in test_acts:
        split_mapper[act] = "test"
    return split_mapper


def extract_grid_data(path: str):
    """Extract grid data from the Kuro Siwo dataset.

    Args:
        path: Path to the gzipped pickle file containing grid data.
    """
    extracted = []
    with gzip.open(path, "rb") as f:
        data = pkl.load(f)

        for key in data:
            extracted.append(
                {
                    "hex": key,
                    "event_id": data[key]["info"]["actid"],
                    "aoi": "{:02d}".format(data[key]["info"]["aoiid"]),
                    "grid_id": data[key]["info"]["grid_id"],
                }
            )

    return extracted


def visualize_complete_sample(sample, output_path=None):
    """Visualize all data in a TACO sample with proper normalization and titles.

    Args:
        sample: TACO sample dataframe containing all modalities
        output_path: Optional path to save the visualization
    """
    modalities = [
        ("pre_event_1_vv", "Pre-Event 1 VV"),
        ("pre_event_1_vh", "Pre-Event 1 VH"),
        ("pre_event_2_vv", "Pre-Event 2 VV"),
        ("pre_event_2_vh", "Pre-Event 2 VH"),
        ("event_vv", "Event VV"),
        ("event_vh", "Event VH"),
        ("aux_dem", "DEM"),
        ("mask_target", "Target Mask"),
        ("mask_invalid_data", "Invalid Data Mask"),
    ]

    n_rows = 3
    n_cols = 4

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 15))
    axes = axes.flatten()

    event_id = (
        sample.iloc[0]["kurosiwo_actid"]
        if "kurosiwo_actid" in sample.columns
        else "Unknown"
    )
    grid_id = (
        sample.iloc[0]["kurosiwo_grid_id"].split("-")[0]
        if "kurosiwo_grid_id" in sample.columns
        else "Unknown"
    )
    flood_date = (
        sample.iloc[0]["kurosiwo_flood_date"]
        if "kurosiwo_flood_date" in sample.columns
        else "Unknown"
    )

    for i, (modality_id, title) in enumerate(modalities):
        if i >= len(axes):
            break
        modality_row = sample[sample["tortilla:id"] == modality_id]

        if len(modality_row) == 0:
            axes[i].text(
                0.5, 0.5, f"Missing: {title}", ha="center", va="center", fontsize=12
            )
            axes[i].axis("off")
            continue

        file_path = modality_row.iloc[0]["internal:subfile"]

        with rio.open(file_path) as src:
            data = src.read()

            if "vv" in modality_id or "vh" in modality_id:
                data = percentile_normalization(data, 2, 98)
                cmap = "gray"
            elif modality_id == "mask_target":
                cmap = plt.cm.colors.ListedColormap(["black", "blue", "cyan", "yellow"])
                bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
                norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

                class_0_count = np.sum(data == 0)
                class_1_count = np.sum(data == 1)
                class_2_count = np.sum(data == 2)
                class_3_count = np.sum(data == 3)
                total_pixels = data.size
                pct_0 = class_0_count / total_pixels * 100
                pct_1 = class_1_count / total_pixels * 100
                pct_2 = class_2_count / total_pixels * 100
                pct_3 = class_3_count / total_pixels * 100

                title += f"\nNo Water: {pct_0:.1f}%, Permanent: {pct_1:.1f}, Floods: {pct_2:.1f}, Background: {pct_3:.1f}%%"

                im = axes[i].imshow(data[0], cmap=cmap, norm=norm)

                divider = make_axes_locatable(axes[i])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = plt.colorbar(im, cax=cax, ticks=[0, 1, 2, 3])
                cbar.ax.set_yticklabels(
                    ["No-Water", "Permanent", "Floods", "Background"]
                )

                axes[i].set_title(title, fontsize=10)
                axes[i].axis("off")
                continue

            elif modality_id == "mask_invalid_data":
                cmap = "binary"
            else:
                data = percentile_normalization(data, 2, 98)
                cmap = "terrain"

            axes[i].imshow(data[0], cmap=cmap)
            axes[i].set_title(title, fontsize=10)
            axes[i].axis("off")

    for i in range(len(modalities), len(axes)):
        axes[i].axis("off")

    fig.suptitle(
        f"Kuro Siwo Sample - Event ID: {event_id}, Grid: {grid_id}, Date: {flood_date}",
        fontsize=16,
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {output_path}")

    return fig


def process_kurosiwo_sample(task):
    """Process a single KuroSiwo sample with optimized profiles."""
    try:
        sample_id = task["sample_id"]
        output_dir = task["output_dir"]
        input_dir = task["input_dir"]

        output_paths = {
            "pre_event_1": os.path.join(
                output_dir, "pre_event_1", f"{sample_id}_pre_event_1.tif"
            ),
            "pre_event_2": os.path.join(
                output_dir, "pre_event_2", f"{sample_id}_pre_event_2.tif"
            ),
            "post_event": os.path.join(
                output_dir, "post_event", f"{sample_id}_post_event.tif"
            ),
            "dem": os.path.join(output_dir, "dem", f"{sample_id}_dem.tif"),
            "mask": os.path.join(output_dir, "mask", f"{sample_id}_mask.tif"),
            "invalid_data": os.path.join(
                output_dir, "invalid_data", f"{sample_id}_invalid_data.tif"
            ),
        }

        modality_configs = {
            "pre_event_1": {
                "inputs": [task["pre_event_1_vv_path"], task["pre_event_1_vh_path"]],
                "combine": True,
            },
            "pre_event_2": {
                "inputs": [task["pre_event_2_vv_path"], task["pre_event_2_vh_path"]],
                "combine": True,
            },
            "post_event": {
                "inputs": [task["post_event_vv_path"], task["post_event_vh_path"]],
                "combine": True,
            },
            "dem": {"inputs": [task["dem_path"]], "combine": False},
            "mask": {"inputs": [task["mask_path"]], "combine": False, "dtype": "uint8"},
            "invalid_data": {
                "inputs": [task["invalid_data_path"]],
                "combine": False,
                "dtype": "uint8",
            },
        }

        profile_template = None
        sample_crs = None
        sample_transform = None
        sample_height = None
        sample_width = None

        for modality_name, config in modality_configs.items():
            input_paths = [os.path.join(input_dir, path) for path in config["inputs"]]

            data_arrays = []
            for in_path in input_paths:
                with rasterio.open(in_path) as src:
                    if profile_template is None:
                        profile_template = src.profile.copy()
                        sample_crs = src.crs
                        sample_transform = src.transform
                        sample_height = src.height
                        sample_width = src.width

                    data_arrays.append(src.read())

            if config["combine"] and len(data_arrays) > 1:
                combined_data = np.vstack(data_arrays)
            else:
                combined_data = data_arrays[0]

            out_profile = profile_template.copy()
            out_profile.update(
                {
                    "driver": "GTiff",
                    "height": sample_height,
                    "width": sample_width,
                    "count": combined_data.shape[0],
                    "dtype": config.get("dtype", combined_data.dtype),
                    "tiled": True,
                    "blockxsize": sample_height,
                    "blockysize": sample_width,
                    "interleave": "pixel",
                    "compress": "zstd",
                    "zstd_level": 13,
                    "predictor": 2,
                    "crs": sample_crs,
                    "transform": sample_transform,
                }
            )

            out_path = output_paths[modality_name]
            with rasterio.open(
                out_path, "w", **{k: v for k, v in out_profile.items() if v is not None}
            ) as dst:
                dst.write(combined_data)

        metadata = {
            "sample_id": sample_id,
            "data_split": task["data_split"],
            "height": sample_height,
            "width": sample_width,
            "pcovered": task["pcovered"],
            "pwater": task["pwater"],
            "pflood": task["pflood"],
            "event_id": task["event_id"],
            "aoi": task["aoi"],
            "flood_date": task["flood_date"],
            "is_additional_test": task["is_additional_test"],
        }

        for modality, path in output_paths.items():
            metadata[f"{modality}_path"] = path if os.path.exists(path) else None

        metadata.update(
            {
                "original_pre_event_1_vv_path": task["pre_event_1_vv_path"],
                "original_pre_event_1_vh_path": task["pre_event_1_vh_path"],
                "original_pre_event_2_vv_path": task["pre_event_2_vv_path"],
                "original_pre_event_2_vh_path": task["pre_event_2_vh_path"],
                "original_post_event_vv_path": task["post_event_vv_path"],
                "original_post_event_vh_path": task["post_event_vh_path"],
                "original_dem_path": task["dem_path"],
                "original_mask_path": task["mask_path"],
                "original_invalid_data_path": task["invalid_data_path"],
            }
        )

        return metadata

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Error processing sample {task['sample_id']}: {e}")
        return None


def reprocess_kurosiwo_dataset(
    metadata_df: pd.DataFrame, input_dir: str, output_dir: str, num_workers: int = 8
) -> pd.DataFrame:
    """Reprocess KuroSiwo samples with optimized profiles and combined modalities.

    Args:
        metadata_df: DataFrame with original sample metadata
        input_dir: Directory containing original samples
        output_dir: Directory to save optimized samples
        block_size: Size of internal GeoTIFF blocks for optimal I/O
        num_workers: Number of workers for parallel processing

    Returns:
        DataFrame with updated metadata pointing to optimized files
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "pre_event_1"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "pre_event_2"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "post_event"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "dem"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "mask"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "invalid_data"), exist_ok=True)

    tasks = []
    for idx, row in metadata_df.iterrows():
        task = {
            "sample_id": f"sample_{idx}",
            "pre_event_1_vv_path": row["pre_event_1_vv_path"],
            "pre_event_1_vh_path": row["pre_event_1_vh_path"],
            "pre_event_2_vv_path": row["pre_event_2_vv_path"],
            "pre_event_2_vh_path": row["pre_event_2_vh_path"],
            "post_event_vv_path": row["event_vv_path"],
            "post_event_vh_path": row["event_vh_path"],
            "dem_path": row["aux_dem_path"],
            "mask_path": row["mask_path"],
            "invalid_data_path": row["invalid_data_path"],
            "data_split": row["split"],
            "pcovered": row["pcovered"],
            "pwater": row["pwater"],
            "pflood": row["pflood"],
            "event_id": row["event_id"],
            "flood_date": row["flood_date"],
            "aoi": row["aoi"],
            "input_dir": input_dir,
            "output_dir": output_dir,
            "is_additional_test": row["is_additional_test"],
        }
        tasks.append(task)

    process_kurosiwo_sample(tasks[0])

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_kurosiwo_sample, task) for task in tasks]

        for future in tqdm(futures, total=len(futures), desc="Processing samples"):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Error processing sample: {e}")

    updated_metadata = pd.DataFrame(results)

    updated_metadata_path = os.path.join(
        output_dir, "kurosiwo_optimized_metadata.parquet"
    )
    updated_metadata.to_parquet(updated_metadata_path)
    print(f"Saved optimized metadata to {updated_metadata_path}")

    return updated_metadata


def generate_metadata_df(root_dir: str) -> pd.DataFrame:
    """Generate metadata DataFrame from the Kuro Siwo dataset."""
    extracted_data = extract_grid_data(
        os.path.join(root_dir, "KuroSiwo", "KuroV2_grid_dict_test_0_100.gz")
    )
    split_mapper = create_split_mapper()

    df = pd.DataFrame(extracted_data)
    df["split"] = df["event_id"].apply(lambda x: split_mapper[x])

    def extract_paths(row):
        data_dir = os.path.join(
            str(root_dir),
            "KuroSiwo",
            str(row["event_id"]),
            str(row["aoi"]),
            str(row["hex"]),
        )

        info_path = os.path.join(data_dir, "info.json")
        info_json = json.load(open(info_path))
        sample_sources = info_json["datasets"]

        meta_data = {}
        for key, val in sample_sources.items():
            modality_path = os.path.join(data_dir, val["name"] + ".tif")
            modality_type = val["name"].split("_")[0] + "_" + val["name"].split("_")[1]
            meta_data[modality_type] = modality_path

        meta_data["pcovered"] = info_json["pcovered"]
        meta_data["pwater"] = info_json["pwater"]
        meta_data["pflood"] = info_json["pflood"]
        meta_data["flood_date"] = info_json["flood_date"]
        return meta_data

    path_df = pd.DataFrame(df.apply(extract_paths, axis=1).tolist())
    df = pd.concat([df, path_df], axis=1)

    # use modality mapper to rename columns appended with "_path"
    modality_mapper = {
        "MK0_DEM": "aux_dem_path",
        "MK0_SLOPE": "aux_slope_path",
        "MK0_MLU": "mask_path",
        "MK0_MNA": "invalid_data_path",
        "SL1_IVV": "pre_event_2_vv_path",
        "SL1_IVH": "pre_event_2_vh_path",
        "SL2_IVV": "pre_event_1_vv_path",
        "SL2_IVH": "pre_event_1_vh_path",
        "MS1_IVV": "event_vv_path",
        "MS1_IVH": "event_vh_path",
    }
    df.rename(columns=modality_mapper, inplace=True)

    # make paths relative
    for col in df.columns:
        if col.endswith("_path"):
            df[col] = df[col].apply(lambda x: x.replace(str(root_dir), ""))

    return df


def create_tortilla(df: pd.DataFrame, root: str, save_dir: str, tortilla_name: str):
    """Create a Tortilla file from the Kuro Siwo dataset.

    Args:
        df: DataFrame with metadata including geolocation for each patch
        root: Root directory for Kuro Siwo dataset
        save_dir: Directory to save the Tortilla file
        tortilla_name: Name of the Tortilla file to be created
    """
    tortilla_dir = os.path.join(save_dir, "tortilla")
    os.makedirs(tortilla_dir, exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating tortilla"):
        modalities = [
            "pre_event_1",
            "pre_event_2",
            "post_event",
            "dem",
            "mask",
            "invalid_data",
        ]
        modality_samples = []

        for modality in modalities:
            modality_path = os.path.join(root, row[modality + "_path"])

            with rio.open(modality_path) as src:
                profile = src.profile

            date_obj = datetime.strptime(row["flood_date"], "%Y-%m-%d %H:%M:%S")
            sample = tacotoolbox.tortilla.datamodel.Sample(
                id=modality,
                path=modality_path,
                file_format="GTiff",
                data_split=row["data_split"],
                add_test_split=row["is_additional_test"],
                stac_data={
                    "crs": "EPSG:" + str(profile["crs"].to_epsg()),
                    "geotransform": profile["transform"].to_gdal(),
                    "raster_shape": (profile["height"], profile["width"]),
                    "time_start": row["flood_date"].split(" ")[0],
                    "time_end": row["flood_date"].split(" ")[0],
                },
                actid=row["event_id"],
                aoiid=row["aoi"],
                flood_date=date_obj,
                pcovered=row["pcovered"],
                pwater=row["pwater"],
                pflood=row["pflood"],
            )

            modality_samples.append(sample)

        taco_samples = tacotoolbox.tortilla.datamodel.Samples(samples=modality_samples)
        samples_path = os.path.join(tortilla_dir, (f"{idx}.tortilla"))
        tacotoolbox.tortilla.create(taco_samples, samples_path, quiet=True)

    all_tortilla_files = sorted(glob(os.path.join(tortilla_dir, "*.tortilla")))
    samples = []

    for idx, tortilla_file in tqdm(
        enumerate(all_tortilla_files),
        total=len(all_tortilla_files),
        desc="Building taco",
    ):
        sample_data = tacoreader.load(tortilla_file).iloc[0]

        sample_tortilla = tacotoolbox.tortilla.datamodel.Sample(
            id=str(idx),
            path=tortilla_file,
            file_format="TORTILLA",
            stac_data={
                "crs": sample_data["stac:crs"],
                "geotransform": sample_data["stac:geotransform"],
                "raster_shape": sample_data["stac:raster_shape"],
                "time_start": sample_data["stac:time_start"],
                "time_end": sample_data["stac:time_end"],
            },
            data_split=sample_data["tortilla:data_split"],
            add_test_split=sample_data["add_test_split"],
            actid=sample_data["actid"],
            aoiid=sample_data["aoiid"],
            flood_date=sample_data["flood_date"],
            pcovered=sample_data["pcovered"],
            pwater=sample_data["pwater"],
            pflood=sample_data["pflood"],
        )
        samples.append(sample_tortilla)

    samples = tacotoolbox.tortilla.datamodel.Samples(samples=samples)
    tacotoolbox.tortilla.create(
        samples, os.path.join(save_dir, tortilla_name), quiet=True
    )


def create_geobench_version(
    metadata_df: pd.DataFrame,
    n_train_samples: int,
    n_val_samples: int,
    n_test_samples: int,
    n_additional_test_samples: int = 0,
) -> pd.DataFrame:
    """Create a GeoBench version of the dataset.

    Args:
        metadata_df: DataFrame with metadata including geolocation for each patch
        n_train_samples: Number of final training samples, -1 means all
        n_val_samples: Number of final samples, -1 means all
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


def main():
    """Generate KuroSiwo Benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for Kuro Siwo dataset"
    )
    parser.add_argument(
        "--save_dir",
        default="geobenchV2/kuro_siwo",
        help="Directory to save the subset",
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    metadata_path = os.path.join(args.save_dir, "geobench_metadata.parquet")
    if os.path.exists(metadata_path):
        metadata_df = pd.read_parquet(metadata_path)
    else:
        metadata_df = generate_metadata_df(args.root)
        metadata_df.to_parquet(metadata_path)

    result_df_path = os.path.join(args.save_dir, "geobench_kuro_siwo.parquet")

    if os.path.exists(result_df_path):
        result_df = pd.read_parquet(result_df_path)
    else:
        subset_df = create_geobench_version(
            metadata_df,
            n_train_samples=4000,
            n_val_samples=1000,
            n_test_samples=2000,
            n_additional_test_samples=0,
        )

        result_df = reprocess_kurosiwo_dataset(
            subset_df, input_dir=args.root, output_dir=args.save_dir, num_workers=1
        )
        result_df.to_parquet(result_df_path)

    tortilla_name = "geobench_kuro_siwo.tortilla"

    create_tortilla(
        result_df,
        root=args.save_dir,
        save_dir=args.save_dir,
        tortilla_name=tortilla_name,
    )

    create_unittest_subset(
        data_dir=args.save_dir,
        tortilla_pattern=tortilla_name,
        test_dir_name="kuro_siwo",
        n_train_samples=4,
        n_val_samples=2,
        n_test_samples=2,
        n_additional_test_samples=0,
    )


if __name__ == "__main__":
    main()
