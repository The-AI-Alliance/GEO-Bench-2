# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate GeoBenchV2 version of RESISC45 dataset."""

from torchgeo.datasets import RESISC45

def create_subset(
    ds: RESISC45, df: pd.DataFrame, save_dir: str, random_state: int = 42
) -> None:
    pass

def main():
    """Generate RESISC45 Benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for RESISC45 dataset"
    )
    parser.add_argument(
        "--save_dir", default="geobenchV2/resisc45", help="Directory to save the subset benchmark data"
    )
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # collect dataframe based on which to do the split and subsetting and copying
    
    # There is no geospatial metadata for RESISC45 dataset


if __name__ == "__main__":
    main()