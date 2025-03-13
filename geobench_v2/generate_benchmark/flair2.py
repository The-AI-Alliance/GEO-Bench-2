# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate GeoBenchV2 version of FLAIR2 dataset."""

from torchvision.datasets.utils import download_url
import pandas as pd
import os
import argparse
import pyproj

from geobench_v2.generate_benchmark.utils import plot_sample_locations

from geobench_v2.datasets.flair2 import GeoBenchFLAIR2


def generate_metadata_df(orig_dataset=None, save_dir=None):
    """"""
    metadata_link = "https://huggingface.co/datasets/IGNF/FLAIR/resolve/main/aux-data/flair_aerial_metadata.json"
    download_url(metadata_link, save_dir)
    metadata_df = pd.read_json(metadata_link, orient="index")

    # Create coordinate transformer from Lambert-93 (EPSG:2154) to WGS84 (EPSG:4326)
    transformer = pyproj.Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)

    # Convert patch centroids to lat/lon
    lon_lat_coords = [
        transformer.transform(row.patch_centroid_x, row.patch_centroid_y)
        for _, row in metadata_df.iterrows()
    ]
    metadata_df["longitude"] = [coord[0] for coord in lon_lat_coords]
    metadata_df["latitude"] = [coord[1] for coord in lon_lat_coords]

    print("Coordinate conversion example:")
    for i, (_, row) in enumerate(metadata_df.head().iterrows()):
        print(
            f"  {row.name}: Lambert-93 ({row.patch_centroid_x}, {row.patch_centroid_y}) -> "
            f"WGS84 ({row.longitude:.6f}, {row.latitude:.6f})"
        )
        if i >= 4:
            break

    # Add spatial region info
    def assign_region(lon, lat):
        # Define simple regions within France
        if lat > 48.5:
            return "Northern France"
        elif lat < 44.5:
            return "Southern France"
        elif lon < 2.0:
            return "Western France"
        else:
            return "Eastern France"

    metadata_df["region"] = metadata_df.apply(
        lambda row: assign_region(row.longitude, row.latitude), axis=1
    )

    # Assign a standard split (we can customize this based on domains or regions)
    # Using domain to ensure spatial coherence in splits
    all_domains = metadata_df["domain"].unique()
    n_domains = len(all_domains)

    # Assign approximately 70/15/15 split based on domains
    train_domains = all_domains[: int(0.7 * n_domains)]
    val_domains = all_domains[int(0.7 * n_domains) : int(0.85 * n_domains)]
    test_domains = all_domains[int(0.85 * n_domains) :]

    def assign_split(domain):
        if domain in train_domains:
            return "train"
        elif domain in val_domains:
            return "val"
        else:
            return "test"

    metadata_df["split"] = metadata_df["domain"].apply(assign_split)

    # Add summary statistics
    print(f"\nTotal patches: {len(metadata_df)}")
    print(f"Split distribution:")
    split_counts = metadata_df["split"].value_counts()
    for split, count in split_counts.items():
        print(f"  {split}: {count} ({100 * count / len(metadata_df):.1f}%)")

    return metadata_df


def main():
    """Generate FLAIR2 Benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for FLAIR2 dataset"
    )
    parser.add_argument(
        "--save_dir",
        default="geobenchV2/flair2",
        help="Directory to save the subset benchmark data",
    )
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    metadata_path = os.path.join(args.save_dir, "geobench_metadata.parquet")
    if os.path.exists(metadata_path):
        metadata_df = pd.read_parquet(metadata_path)
    else:
        metadata_df = generate_metadata_df(save_dir=args.save_dir)
        metadata_df.to_parquet(metadata_path)

    plot_sample_locations(
        metadata_df,
        output_path=os.path.join(args.save_dir, "sample_locations.png"),
        buffer_degrees=1.5,
        sample_fraction=0.1,
        split_column="split",
        alpha=0.5,
    )

    print("\nFLAIR2 benchmark metadata generation complete.")


if __name__ == "__main__":
    main()
