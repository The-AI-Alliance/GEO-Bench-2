# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of MADOS dataset."""

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
        with open(split_file, "r") as f:
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
        ext = os.path.splitext(filename)[1]

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


def create_tortilla(root_dir, df, save_dir):
    """Create a tortilla version of the MADOS dataset with all available modalities.

    Args:
        root_dir: Root directory containing MADOS data
        df: DataFrame with metadata
        save_dir: Directory to save tortilla files
    """
    tortilla_dir = os.path.join(save_dir, "tortilla")
    os.makedirs(tortilla_dir, exist_ok=True)

    for idx, row in tqdm(
        df.iterrows(), total=len(df), desc="Creating tortilla samples"
    ):
        modality_samples = []

        # Process each available modality
        for modality in bands_to_keep:
            path_col = f"{modality}_path"
            path = os.path.join(root_dir, "MADOS", row[path_col])

            sample = tacotoolbox.tortilla.datamodel.Sample(
                id=f"{row['scene_dir']}_{row['crop_id']}_{modality}",
                path=path,
                file_format="GTiff" if path.endswith(".tif") else "PNG",
                data_split=row["split"],
                patch_id=f"{row['scene_dir']}_{row['crop_id']}",
                modality=modality,
            )

            modality_samples.append(sample)

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
    final_taco_path = os.path.join(save_dir, "MADOS.tortilla")
    tacotoolbox.tortilla.create(final_samples, final_taco_path, quiet=True, nworkers=4)


def main():
    """Generate MADOS Benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for MADOS dataset"
    )
    parser.add_argument(
        "--save_dir", default="geobenchV2/MADOS", help="Directory to save the subset"
    )
    args = parser.parse_args()

    metadata_path = os.path.join(args.save_dir, "geobench_mados.parquet")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if os.path.exists(metadata_path):
        metadata_df = pd.read_parquet(metadata_path)
    else:
        metadata_df = generate_metadata_df(args.root)
        metadata_df.to_parquet(metadata_path)

    create_tortilla(args.root, metadata_df, args.save_dir)

    # taco = tacoreader.load(os.path.join(args.save_dir, "MADOS.tortilla"))


if __name__ == "__main__":
    main()
