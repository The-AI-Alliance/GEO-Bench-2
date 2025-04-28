# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of PASTIS dataset."""

import geopandas as gpd
from torchgeo.datasets import PASTIS
import pandas as pd
import os
import argparse
import json
import tacotoolbox
import tacoreader
import glob
from tqdm import tqdm

from geobench_v2.generate_benchmark.utils import (
    plot_sample_locations,
    create_subset_from_df,
    create_unittest_subset,
)


def create_subset(ds: PASTIS, metadata_df: pd.DataFrame, save_dir: str) -> None:
    """Create a subset of PASTIS dataset.

    Args:
        ds: PASTIS dataset.
        metadata_df: Metadata DataFrame.
        save_dir: Directory to save the subset.
    """
    # based on the metadata_df create a subset of the dataset and copy it
    # with the same structure to the save_dir
    pass


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

    # conve
    final_gdf = gpd.GeoDataFrame(new_df, geometry=geometry)

    return final_gdf


def create_tortilla(root_dir, df, save_dir, tortilla_name) -> None:
    """Create a subset of PASTIS dataset for Tortilla Benchmark."""

    tortilla_dir = os.path.join(save_dir, "tortilla")
    os.makedirs(tortilla_dir, exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating tortilla"):
        modalities = ["s2", "s1a", "s1d", "semantic", "instance"]
        modality_samples = []

        for modality in modalities:
            path = os.path.join(root_dir, row[modality + "_path"])
            # start date

            sample = tacotoolbox.tortilla.datamodel.Sample(
                id=modality,
                path=path,
                file_format="NUMPY",
                data_split=row["split"],
                stac_data={
                    "crs": "EPSG:" + str(profile["crs"].to_epsg()),
                    "geotransform": profile["transform"].to_gdal(),
                    "raster_shape": (profile["height"], profile["width"]),
                    "time_start": row[f"dates-{modality}"][0],
                },
                dates=row[f"dates-{modality}"],
                tile=row["tile"],
                lon=row["lon"],
                lat=row["lat"],
                patch_id=row["ID_PATCH"],
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
            dates=sample_data["dates"],
            tile=sample_data["tile"],
            lon=sample_data["lon"],
            lat=sample_data["lat"],
            fold=sample_data["fold"],
            patch_id=sample_data["patch_id"],
            n_parcel=sample_data["n_parcel"],
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
) -> pd.DataFrame:
    """Create a GeoBench version of the dataset.
    Args:
        metadata_df: DataFrame with metadata including geolocation for each patch
        n_train_samples: Number of final training samples, -1 means all
        n_val_samples: Number of final validation samples, -1 means all
        n_test_samples: Number of final test samples, -1 means all
        root_dir: Root directory for FLAIR2 dataset
        save_dir: Directory to save the subset benchmark data
        block_size: Size of blocks for optimized GeoTIFF writing
        num_workers: Number of parallel workers
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
        result_df = pd.read_parquet(result_path)
    else:
        result_df = create_geobench_version(
            metadata_df, n_train_samples=4000, n_val_samples=1000, n_test_samples=2000
        )
        result_df.to_parquet(result_path)

    tortilla_name = "geobench_pastis.tortilla"
    create_tortilla(args.root, result_df, args.save_dir, tortilla_name)

    create_unittest_subset(
        data_dir=args.save_dir,
        tortilla_pattern=tortilla_name,
        test_dir_name="pastis",
        n_train_samples=4,
        n_val_samples=2,
        n_test_samples=2,
    )

    plot_sample_locations(
        metadata_df,
        os.path.join(args.save_dir, "sample_locations.png"),
        buffer_degrees=1.5,
        s=2,
    )


if __name__ == "__main__":
    main()
