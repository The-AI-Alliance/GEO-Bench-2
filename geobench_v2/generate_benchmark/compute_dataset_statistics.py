# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Script to compute Dataset Statistics for GeoBenchV2."""

import argparse
import json
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from geobench_v2.generate_benchmark.utils_dataset_statistics import (
    ClassificationStatistics,
    ObjectDetectionStatistics,
    PxRegressionStatistics,
    SegmentationStatistics,
)


def _determine_task_type(stats_computer):
    """Determine the task type from target statistics."""
    if isinstance(stats_computer, PxRegressionStatistics):
        return "px_regression"
    elif isinstance(stats_computer, SegmentationStatistics):
        return "segmentation"
    elif isinstance(stats_computer, ClassificationStatistics):
        return "classification"
    elif isinstance(stats_computer, ObjectDetectionStatistics):
        return "object_detection"


def _create_histogram_plot(stats, keys, group_name, vis_dir, dataset_name):
    """Create histogram plot for a group of keys."""
    all_bands = []
    all_histograms = []
    all_bin_centers = []

    for key in keys:
        if "histograms" not in stats[key] or "histogram_bins" not in stats[key]:
            continue

        histograms = np.array(stats[key]["histograms"])
        bin_edges = np.array(stats[key]["histogram_bins"])
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        band_names = stats[key].get(
            "band_names", [f"Band {i}" for i in range(histograms.shape[0])]
        )

        all_bands.extend([f"{key}_{band}" for band in band_names])
        all_histograms.append(histograms)
        all_bin_centers.append(bin_centers)

    if not all_histograms:
        return

    n_keys = len(keys)
    fig, axes = plt.subplots(n_keys, 1, figsize=(10, 4 * n_keys), squeeze=False)

    for i, key in enumerate(keys):
        if i >= len(all_histograms):
            continue

        ax = axes[i, 0]
        histograms = all_histograms[i]
        bin_centers = all_bin_centers[i]

        for j in range(histograms.shape[0]):
            ax.plot(bin_centers, histograms[j], label=f"Band {j}")

        ax.set_title(f"{key}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(
        os.path.join(vis_dir, f"{dataset_name}_{group_name}_histograms.png"), dpi=300
    )
    plt.close(fig)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types and sets."""

    def default(self, obj):
        """Convert NumPy types and sets to native Python types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)


def process_dataset(
    dataset_config: dict[str, Any], save_dir: str, device: str, normalization_mode: str
) -> None:
    """Process a single dataset and compute its statistics.

    Args:
        dataset_config: Configuration for the dataset from YAML
        save_dir: Directory to save results
        device: Device to use for computation
        normalization_mode: Normalization mode to use
    """
    dataset_name = dataset_config.get("name", "unknown_dataset")
    print(f"\nProcessing dataset: {dataset_name}")

    dataset_dir = os.path.join(save_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    stats_computer_config = dataset_config["stats_computer"]
    stats_computer_config["device"] = device
    stats_computer_config["save_dir"] = dataset_dir

    stats_computer_config["datamodule"]["data_normalizer"] = {
        "_target_": "torch.nn.Identity"
    }

    stats_computer = instantiate(
        stats_computer_config, normalization_mode=normalization_mode
    )

    print(f"Computing statistics for {dataset_name}...")
    stats = stats_computer.compute_statistics()

    save_statistics(stats, dataset_dir, dataset_name)

    print(f"Statistics for {dataset_name} saved to {save_dir}")

    print(f"Completed processing for {dataset_name}")


def save_statistics(stats: tuple, save_dir: str, dataset_name: str) -> None:
    """Save statistics and create visualizations.

    Args:
        stats: Tuple of (input_stats, target_stats)
        save_dir: Directory to save results
        dataset_name: Name of the dataset
    """
    input_stats, target_stats = stats

    dataset_stats = {"input_stats": input_stats, "target_stats": target_stats}
    dataset_stats_path = os.path.join(save_dir, f"{dataset_name}_stats.json")

    with open(dataset_stats_path, "w") as f:
        json.dump(dataset_stats, f, cls=NumpyEncoder, indent=4)

    vis_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)


def main():
    """Main function to compute dataset statistics."""
    parser = argparse.ArgumentParser(
        description="Compute dataset statistics for GeoBenchV2."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset_statistics.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="dataset_statistics",
        help="Directory to save statistics",
    )
    parser.add_argument(
        "--normalization_mode",
        type=str,
        default="none",
        choices=["none", "clip_only", "clip_rescale", "satmae"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for computation",
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    config = OmegaConf.load(args.config)

    for dataset_config in config["datamodules"]:
        process_dataset(
            dataset_config, args.save_dir, args.device, args.normalization_mode
        )

    print(f"\nAll dataset statistics computed and saved to {args.save_dir}")


if __name__ == "__main__":
    main()
