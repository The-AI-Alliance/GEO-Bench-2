# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate Benchmark version of PASTIS dataset."""

import geopandas as gpd
from torchgeo.datasets import PASTIS
import pandas as pd
import os
import argparse
from geobench_v2.generate_benchmark.utils import plot_sample_locations


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
    df = pd.DataFrame(gdf.drop(columns=columns_to_drop))
    return df


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

    plot_sample_locations(
        metadata_df,
        os.path.join(args.save_dir, "sample_locations.png"),
        buffer_degrees=1.5,
        s=2,
    )


if __name__ == "__main__":
    main()
