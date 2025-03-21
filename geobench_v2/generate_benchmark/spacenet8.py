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
from geobench_v2.generate_benchmark.utils import plot_sample_locations
import tacotoolbox
import tacoreader
import glob

from geobench_v2.generate_benchmark.utils import (
    plot_sample_locations,
)

from geobench_v2.generate_benchmark.geospatial_split_utils import (
    visualize_checkerboard_pattern,
    split_geospatial_tiles_into_patches,
    visualize_geospatial_split,
    checkerboard_split,
    geographic_buffer_split,
    geographic_distance_split,
    visualize_distance_clusters,
    show_samples_per_valid_ratio
)


def create_geobench_ds(
    metadata_df: pd.DataFrame, save_dir: str
) -> None:
    """Create a subset of SpaceNet6 dataset."""
    os.makedirs(save_dir, exist_ok=True)



    modal_path_dict = {}
    modal_path_dict["PRE-event"] = metadata_df["pre-path"].tolist()
    modal_path_dict["POST-event"] = metadata_df["post-path"].tolist()
    modal_path_dict["mask"] = metadata_df["label-path"].tolist()

    patch_size = (512, 512)
    stride = (511, 511)

    patches_df = split_geospatial_tiles_into_patches(
        modal_path_dict=modal_path_dict,
        output_dir=save_dir,
        patch_size=patch_size,
        stride=stride,
        buffer_top=64,
        buffer_bottom=64,
        buffer_left=64,
        buffer_right=64,
    )
    return patches_df


def generate_metadata_df(root_dir) -> pd.DataFrame:
    """Generate metadata DataFrame for SpaceNet8 dataset.

    Args:
        ds: SpaceNet8 dataset.
    """
    # metadata_file = os.path.join(ds.root, ds.dataset_id, "train", "train", "AOI_11_Rotterdam", "SummaryData", "SN6_Train_AOI_11_Rotterdam_Buildings.csv")

    paths = [
        "/mnt/rg_climate_benchmark/data/datasets_segmentation/SpaceNet8/Germany_Training_Public_label_image_mapping.csv",
        "/mnt/rg_climate_benchmark/data/datasets_segmentation/SpaceNet8/Louisiana-East_Training_Public_label_image_mapping.csv"
    ]

    df = pd.concat([pd.read_csv(path) for path in paths])

    metadata: list[dict[str, str]] = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        

        pre_event_path = os.path.join(root_dir, "SN8_floods",  "train", "PRE-event", row["pre-event image"])
        post_event_path = os.path.join(root_dir, "SN8_floods",  "train", "POST-event", row["post-event image 1"])
        label_path = os.path.join(root_dir, "SN8_floods",  "train", "annotations", row["label"])

        assert os.path.exists(pre_event_path)
        assert os.path.exists(post_event_path)
        assert os.path.exists(label_path)
        

        with rasterio.open(pre_event_path) as src:
            lng, lat = src.lnglat()
            height_px, width_px = src.height, src.width
        metadata.append(
            {"pre-path": pre_event_path, "post-path": post_event_path, "label-path": label_path, "longitude": lng, "latitude": lat, "height_px": height_px, "width_px": width_px}
        )


    metadata_df = pd.DataFrame(metadata)

    metadata_df["split"] = "train"

    regions = [
        {
            'name': 'Louisiana, USA',
            'bounds': {'min_lat': 29, 'max_lat': 33, 'min_lon': -94, 'max_lon': -89}
        },
        {
            'name': 'Germany',
            'bounds': {'min_lat': 47.5, 'max_lat': 54.5, 'min_lon': 6.5, 'max_lon': 14.5}
        }
    ]

    # match region to each sample
    metadata_df['region'] = 'unknown'  # Default value
    
    for region in regions:
        bounds = region["bounds"]
        metadata_df.loc[
            (metadata_df["latitude"] >= bounds["min_lat"])
            & (metadata_df["latitude"] <= bounds["max_lat"])
            & (metadata_df["longitude"] >= bounds["min_lon"])
            & (metadata_df["longitude"] <= bounds["max_lon"]),
            "region",
        ] = region["name"]


    return metadata_df


def create_tortilla(root_dir, df, save_dir):
    """Create a tortilla version of the dataset."""

    tortilla_dir = os.path.join(save_dir, "tortilla")
    os.makedirs(tortilla_dir, exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating tortilla"):

        modalities = ["PRE-event", "POST-event", "mask"]
        modality_samples = []

        for modality in modalities:
            path = os.path.join(root_dir, row[modality+"_path"])
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
                spacenet8_source_img_file=row["source_img_file"],
                spacenet8_source_mask_file=row["source_mask_file"],
                spacenet8_patch_id=row["patch_id"],
                spacenet8_region=row["region"],
            )

            modality_samples.append(sample)

        taco_samples = tacotoolbox.tortilla.datamodel.Samples(samples=modality_samples)
        samples_path = os.path.join(tortilla_dir, f"sample_{idx}.tortilla")
        tacotoolbox.tortilla.create(taco_samples, samples_path, quiet=True)

    # merge tortillas into a single dataset
    all_tortilla_files = sorted(glob.glob(os.path.join(tortilla_dir, "*.tortilla")))

    samples = []

    for idx, tortilla_file in tqdm(enumerate(all_tortilla_files), total=len(all_tortilla_files), desc="Building taco"):
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
            spacenet8_source_img_file=sample_data["spacenet8_source_img_file"],
            spacenet8_source_mask_file=sample_data["spacenet8_source_mask_file"],
            spacenet8_patch_id=sample_data["spacenet8_patch_id"],
            spacenet8_region=sample_data["spacenet8_region"],
        )
        samples.append(sample_tortilla)

    # create final taco file
    final_samples = tacotoolbox.tortilla.datamodel.Samples(samples=samples)
    tacotoolbox.tortilla.create(final_samples, os.path.join(save_dir, "SpaceNet8.tortilla"), quiet=True)
        
        


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

    metadata_df = generate_metadata_df(args.root)
    metadata_df.to_parquet(metadata_path)

    # create_geobench_ds(metadata_df, save_dir=args.save_dir)
    path = "/mnt/rg_climate_benchmark/data/geobenchV2/spacenet8/patch_metadata.parquet"

    df = pd.read_parquet(path)

    regions = [
        {
            'name': 'Louisiana, USA',
            'bounds': {'min_lat': 29, 'max_lat': 33, 'min_lon': -94, 'max_lon': -89}
        },
        {
            'name': 'Germany',
            'bounds': {'min_lat': 47.5, 'max_lat': 54.5, 'min_lon': 6.5, 'max_lon': 14.5}
        }
    ]

    # match region to each sample
    df['region'] = 'unknown'  # Default value
    
    for region in regions:
        bounds = region["bounds"]
        df.loc[
            (df["lat"] >= bounds["min_lat"])
            & (df["lat"] <= bounds["max_lat"])
            & (df["lon"] >= bounds["min_lon"])
            & (df["lon"] <= bounds["max_lon"]),
            "region",
        ] = region["name"]

    # show_samples_per_valid_ratio(df, os.path.join(args.save_dir, "samples_per_valid_ratio.png"), dataset_name="SpaceNet8")

    df_with_assigned_split = geographic_distance_split(
        df,
        n_clusters=8,
        random_state=42
    )

    visualize_distance_clusters(
        df_with_assigned_split,
        title='Distance Split',
        output_path=os.path.join(args.save_dir, 'distance_split.png'),
        buffer_degrees=0.05
    )

    create_tortilla(args.save_dir, df_with_assigned_split, args.save_dir)



    # create taco version of the dataset

    # checker_split_df = checkerboard_split(
    #     df,
    #     n_blocks_x=10,
    #     n_blocks_y=10,
    #     pattern="other",
    #     random_state=42,
    # )

    # visualize_geospatial_split(
    #     checker_split_df,
    #     title='Checkerboard Split',
    #     output_path=os.path.join(args.save_dir, 'checker_split.png'),
    #     buffer_degrees=0.05
    # )


    # plot_sample_locations(
    #     distance_df,
    #     output_path=os.path.join(args.save_dir, "sample_locations.png"),
    #     buffer_degrees=0.5,
    # )


if __name__ == "__main__":
    main()