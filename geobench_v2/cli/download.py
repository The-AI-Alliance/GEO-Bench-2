# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""CLI entry point for downloading GeoBench-2 datasets."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Run the download script.

    This command-line tool downloads GeoBench-2 datasets from Hugging Face.

    Usage:
        geobench-download --root /path/to/data               # Download all datasets (requires --root)
        geobench-download --root /path/to/data all           # Explicit all
        geobench-download --root /path/to/data spacenet7 caffe  # Download specific datasets

    Note: --root is required to force callers to choose a download location.
    """
    parser = argparse.ArgumentParser(
        prog="geobench-download",
        description="Download GeoBench-2 datasets from Hugging Face (requires --root).",
    )
    parser.add_argument(
        "--root",
        "-r",
        required=True,
        help="Root directory for downloads (required).",
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Dataset names to download (default: all). Use 'all' to explicitly download all datasets.",
    )
    args = parser.parse_args()

    # Find the download script in the same directory as this file
    script_path = Path(__file__).parent / "download_geobenchV2.sh"

    if not script_path.exists():
        print(f"Error: Download script not found at {script_path}", file=sys.stderr)
        print("Please ensure the package is installed correctly.", file=sys.stderr)
        sys.exit(1)

    script_path.chmod(0o755)

    cmd_args = []
    if not args.datasets:
        cmd_args = ["all"]
    else:
        cmd_args = list(args.datasets)

    # Prepare environment with the required DOWNLOAD_ROOT (force user-specified root)
    env = os.environ.copy()
    env["DOWNLOAD_ROOT"] = str(args.root)

    # Invoke the bash script with the provided datasets and forced root
    result = subprocess.run([str(script_path)] + cmd_args, check=False, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()