# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of SpaceNet8 dataset."""

import argparse


import geopandas as gpd
import tacotoolbox
import rasterio as rio
import pathlib
import tacoreader
import pandas as pd
from glob import glob
import json
import os
import gzip
import pickle as pkl
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from torchgeo.datasets.utils import percentile_normalization
from mpl_toolkits.axes_grid1 import make_axes_locatable

from geobench_v2.generate_benchmark.utils import (
    plot_sample_locations,
    create_subset_from_tortilla,
)


# Constants
OUTPUT_FOLDER = pathlib.Path(
    "/mnt/rg_climate_benchmark/data/datasets_segmentation/kuro_siwo/taco_kuro_siwo"
)
MAIN_FOLDER = pathlib.Path("KuroSiwo")
GRID_DATA_PATH = "/mnt/rg_climate_benchmark/data/datasets_segmentation/kuro_siwo/KuroSiwo/KuroV2_grid_dict_test_0_100.gz"
FINAL_TACO_PATH = "kurosiwo.tortilla"

# Configuration mappings
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
    split_mapper = {}
    for act in train_acts:
        split_mapper[act] = "train"
    for act in val_acts:
        split_mapper[act] = "validation"
    for act in test_acts:
        split_mapper[act] = "test"
    return split_mapper


def extract_grid_data(path):
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


def process_grid_samples(df, root):
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing"):
        data_dir = os.path.join(
            str(root), str(row["event_id"]), str(row["aoi"]), str(row["hex"])
        )

        if not os.path.exists(data_dir):
            print(f"Warning: {data_dir} does not exist, skipping")
            continue

        # Read info.json
        info_path = os.path.join(data_dir, "info.json")
        info_json = json.load(open(info_path))

        grid_id = info_json["grid_id"]
        sample_sources = info_json["datasets"]
        event_id = info_json["actid"]
        aoi_id = info_json["aoiid"]

        modality_samples = []

        # Process each dataset modality
        for key, val in sample_sources.items():
            modality_path = os.path.join(data_dir, val["name"] + ".tif")
            modality_type = val["name"].split("_")[0] + "_" + val["name"].split("_")[1]

            with rio.open(modality_path) as src:
                profile = src.profile

            sample = tacotoolbox.tortilla.datamodel.Sample(
                id=modality_mapper[modality_type],
                path=modality_path,
                file_format="GTiff",
                data_split=row["split"],
                stac_data={
                    "crs": "EPSG:" + str(profile["crs"].to_epsg()),
                    "geotransform": profile["transform"].to_gdal(),
                    "raster_shape": (profile["height"], profile["width"]),
                    "time_start": val["source_date"],
                    "time_end": val["source_date"],
                },
                kurosiwo_actid=event_id,
                kurosiwo_aoiid=aoi_id,
                kurosiwo_grid_id=grid_id,
                kurosiwo_flood_date=info_json["flood_date"],
                kurosiwo_pcovered=info_json["pcovered"],
                kurosiwo_pwater=info_json["pwater"],
                kurosiwo_pflood=info_json["pflood"],
            )
            modality_samples.append(sample)

        # Sort samples according to modality order
        sorted_samples = []
        for modality in modality_order:
            for sample in modality_samples:
                if sample.id == modality:
                    sorted_samples.append(sample)
                    break

        # Create and save TACO samples
        taco_samples = tacotoolbox.tortilla.datamodel.Samples(samples=sorted_samples)
        samples_path = OUTPUT_FOLDER / (f"{event_id}_{aoi_id}_{grid_id}.tortilla")
        tacotoolbox.tortilla.create(taco_samples, samples_path, quiet=True)


def merge_tortilla_files():
    all_tortilla_files = list(OUTPUT_FOLDER.glob("*.tortilla"))
    samples = []

    for idx, tortilla_file in tqdm(
        enumerate(all_tortilla_files),
        total=len(all_tortilla_files),
        desc="Building taco",
    ):
        # Read the event data
        sample_data = tacoreader.load(tortilla_file.as_posix())
        sample_data_flood = sample_data[sample_data["tortilla:id"] == "event_vh"].iloc[
            0
        ]
        pre_event_1 = sample_data[sample_data["tortilla:id"] == "pre_event_1_vh"].iloc[
            0
        ]
        pre_event_2 = sample_data[sample_data["tortilla:id"] == "pre_event_2_vh"].iloc[
            0
        ]

        # Get time delay between the pre-events and the event
        pre_event_1_date = pd.to_datetime(pre_event_1["stac:time_start"], unit="s")
        pre_event_2_date = pd.to_datetime(pre_event_2["stac:time_start"], unit="s")
        event_date = pd.to_datetime(sample_data_flood["stac:time_start"], unit="s")

        delay1 = (event_date - pre_event_1_date).days
        delay2 = (event_date - pre_event_2_date).days

        # Create sample
        sample_tortilla = tacotoolbox.tortilla.datamodel.Sample(
            id=tortilla_file.stem,
            path=tortilla_file,
            file_format="TORTILLA",
            stac_data={
                "crs": sample_data_flood["stac:crs"],
                "geotransform": sample_data_flood["stac:geotransform"],
                "raster_shape": sample_data_flood["stac:raster_shape"],
                "time_start": sample_data_flood["stac:time_start"],
                "time_end": sample_data_flood["stac:time_end"],
            },
            data_split=sample_data_flood["tortilla:data_split"],
            kurosiwo_delay1=delay1,
            kurosiwo_delay2=delay2,
        )
        samples.append(sample_tortilla)

    # Create the final TACO file
    samples = tacotoolbox.tortilla.datamodel.Samples(samples=samples)
    tacotoolbox.tortilla.create(samples, FINAL_TACO_PATH, quiet=True)


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

    # Process and plot each modality
    for i, (modality_id, title) in enumerate(modalities):
        if i >= len(axes):
            break

        # Find the corresponding row in the sample dataframe
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

                # Compute some class statistics and add to plot
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

    metadata_path = os.path.join(args.save_dir, "geobench_kuro_siwo.parquet")

    metadata_df = pd.read_parquet(metadata_path)

    plot_sample_locations(
        metadata_df,
        os.path.join(args.save_dir, "sample_locations.png"),
        split_column="tortilla:data_split",
        dataset_name="Kuro Siwo",
    )

    # # Create output directory
    # OUTPUT_FOLDER.mkdir(exist_ok=True)

    # # Setup data and splits
    # split_mapper = create_split_mapper()
    # extracted_data = extract_grid_data(GRID_DATA_PATH)

    # # Create dataframe and assign splits
    # df = pd.DataFrame(extracted_data)
    # df["split"] = df["event_id"].apply(lambda x: split_mapper[x])
    # df.to_csv("full_df.csv", index=False)

    # process_grid_samples(df, args.root)
    # merge_tortilla_files()

    taco_glob = sorted(glob(os.path.join(args.save_dir, "kurosiwo.*.part.tortilla")))
    taco_ben = tacoreader.load(taco_glob)

    # create unit test subset
    unit_test_taco = create_subset_from_tortilla(
        taco_ben, n_train_samples=4, n_val_samples=2, n_test_samples=2
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(script_dir))
    test_data_dir = os.path.join(repo_root, "tests", "data", "kuro_siwo")
    os.makedirs(test_data_dir, exist_ok=True)
    tacoreader.compile(
        dataframe=unit_test_taco,
        output=os.path.join(test_data_dir, "kuro_siwo.tortilla"),
    )


if __name__ == "__main__":
    main()
