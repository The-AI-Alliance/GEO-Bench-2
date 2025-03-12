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

# TODO add automatic download of dataset to have a starting point for benchmark generation

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

    # from https://huggingface.co/datasets/IGNF/FLAIR#data-splits
    train_ids = (
        "D006", "D007", "D008", "D009", "D013", "D016", "D017", "D021", "D023", 
        "D030", "D032", "D033", "D034", "D035", "D038", "D041", "D044", "D046", 
        "D049", "D051", "D052", "D055", "D060", "D063", "D070", "D072", "D074", 
        "D078", "D080", "D081", "D086", "D091"
    )
    val_ids = (
        "D004", "D014", "D029", "D031", "D058", "D066", "D067", "D077"
    )
    test_ids = (
        "D015", "D022", "D026", "D036", "D061", "D064", "D068", "D069", "D071", "D084"
    )
    # find match in the domain column which has values of id_year
    metadata_df["split"] = (
        metadata_df["domain"]
        .apply(lambda x: x.split("_")[0])
        .replace({train_id: "train" for train_id in train_ids})
        .replace({val_id: "val" for val_id in val_ids})
        .replace({test_id: "test" for test_id in test_ids})
    )

    metadata_df = metadata_df.reset_index().rename(columns={"index": "image_id"})

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
        split_column="split",
        s=2.0,
    )

    print("\nFLAIR2 benchmark metadata generation complete.")


if __name__ == "__main__":
    main()
