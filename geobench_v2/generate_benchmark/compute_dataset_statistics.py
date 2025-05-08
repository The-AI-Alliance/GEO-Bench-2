# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Script to compute Dataset Statistics for GeoBenchV2."""

import argparse
import json
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from geobench_v2.generate_benchmark.utils_dataset_statistics import (
    ClassificationStatistics,
    ObjectDetectionStatistics,
    PxRegressionStatistics,
    SegmentationStatistics,
)

sns.set(style="whitegrid")


def create_visualizations(input_stats, target_stats, vis_dir, dataset_name, task_type):
    """Create visualizations for dataset statistics."""
    os.makedirs(vis_dir, exist_ok=True)
    modalities = {}
    for key, stats in input_stats.items():
        if isinstance(stats, dict) and "modality" in stats:
            modality = stats["modality"]
            if modality not in modalities:
                modalities[modality] = []
            modalities[modality].append(key)

    if not modalities:
        modalities = {"image": list(input_stats.keys())}
    _create_input_visualizations(input_stats, modalities, vis_dir, dataset_name)

    if task_type == "px_regression":
        _create_regression_visualizations(target_stats, vis_dir, dataset_name)
    elif task_type == "classification":
        _create_classification_visualizations(target_stats, vis_dir, dataset_name)
    elif task_type == "segmentation":
        _create_segmentation_visualizations(target_stats, vis_dir, dataset_name)
    elif task_type == "object_detection":
        _create_object_detection_visualizations(target_stats, vis_dir, dataset_name)


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


def _create_input_visualizations(input_stats, modalities, vis_dir, dataset_name):
    """Create visualizations for input statistics."""
    for modality, keys in modalities.items():
        valid_keys = [
            k
            for k in keys
            if isinstance(input_stats[k], dict) and "mean" in input_stats[k]
        ]
        if not valid_keys:
            continue
        _create_histogram_plot(input_stats, valid_keys, modality, vis_dir, dataset_name)


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


def _create_regression_visualizations(target_stats, vis_dir, dataset_name):
    """Create visualizations for regression target statistics."""
    _create_histogram_plot(
        {"target": target_stats}, ["target"], "target", vis_dir, dataset_name
    )

    if all(k in target_stats for k in ["mean", "min", "max"]):
        plt.figure(figsize=(8, 6))

        stat_names = ["min", "mean", "max"]
        values = [np.array(target_stats[s]).mean() for s in stat_names]

        x_positions = np.arange(len(stat_names))
        plt.bar(x_positions, values)
        plt.title(f"{dataset_name} - Target Statistics")
        plt.ylabel("Value")
        plt.xticks(x_positions, stat_names)

        for i, v in enumerate(values):
            plt.text(i, v, f"{v:.4f}", ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"{dataset_name}_target_stats.png"), dpi=300)
        plt.close()


def _create_classification_visualizations(target_stats, vis_dir, dataset_name):
    """Create visualizations for classification target statistics."""
    if "class_frequencies" in target_stats:
        freqs = np.array(target_stats["class_frequencies"])
        class_ids = np.arange(len(freqs))
        class_names = target_stats.get(
            "class_names", [f"Class {i}" for i in range(len(freqs))]
        )

        plt.figure(figsize=(max(10, len(freqs) * 0.5), 6))
        plt.bar(class_ids, freqs)
        plt.title(f"{dataset_name} - Class Distribution")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.xticks(class_ids, class_names, rotation=45, ha="right")

        for i, v in enumerate(freqs):
            plt.text(i, v, str(int(v)), ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"{dataset_name}_class_dist.png"), dpi=300)
        plt.close()

        plt.figure(figsize=(10, 10))
        plt.pie(freqs, labels=class_names, autopct="%1.1f%%")
        plt.title(f"{dataset_name} - Class Distribution")
        plt.savefig(os.path.join(vis_dir, f"{dataset_name}_class_pie.png"), dpi=300)
        plt.close()


def _create_segmentation_visualizations(target_stats, vis_dir, dataset_name):
    """Create visualizations for segmentation target statistics."""
    if "pixel_distribution" in target_stats:
        dist = np.array(target_stats["pixel_distribution"])
        class_ids = np.arange(len(dist))
        class_names = target_stats.get(
            "class_names", [f"Class {i}" for i in range(len(dist))]
        )

        plt.figure(figsize=(max(10, len(dist) * 0.5), 6))
        plt.bar(class_ids, dist)
        plt.title(f"{dataset_name} - Pixel Distribution")
        plt.xlabel("Class")
        plt.ylabel("Pixel Count")
        plt.xticks(class_ids, class_names, rotation=45, ha="right")

        total_pixels = np.sum(dist)
        for i, v in enumerate(dist):
            percentage = 100 * v / total_pixels
            plt.text(i, v, f"{int(v)}\n({percentage:.1f}%)", ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"{dataset_name}_pixel_dist.png"), dpi=300)
        plt.close()

        if "class_presence_ratio" in target_stats:
            presence = np.array(target_stats["class_presence_ratio"])

            plt.figure(figsize=(max(10, len(presence) * 0.5), 6))
            plt.bar(class_ids, presence)
            plt.title(f"{dataset_name} - Class Presence in Images")
            plt.xlabel("Class")
            plt.ylabel("Presence Ratio")
            plt.xticks(class_ids, class_names, rotation=45, ha="right")
            plt.ylim(0, 1.05)

            for i, v in enumerate(presence):
                plt.text(i, v, f"{v * 100:.1f}%", ha="center", va="bottom")

            plt.tight_layout()
            plt.savefig(
                os.path.join(vis_dir, f"{dataset_name}_class_presence.png"), dpi=300
            )
            plt.close()


def _create_object_detection_visualizations(target_stats, vis_dir, dataset_name):
    """Create visualizations for object detection target statistics."""
    pass


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types and sets."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, set):
            return list(obj)  # Convert sets to lists for JSON serialization
        return super(NumpyEncoder, self).default(obj)


def process_dataset(dataset_config: dict[str, Any], save_dir: str, device: str) -> None:
    """Process a single dataset and compute its statistics.

    Args:
        dataset_config: Configuration for the dataset from YAML
        save_dir: Directory to save results
        device: Device to use for computation
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

    stats_computer = instantiate(stats_computer_config)

    print(f"Computing statistics for {dataset_name}...")
    stats = stats_computer.compute_statistics()

    save_statistics(stats, dataset_dir, dataset_name)

    input_stats, target_stats = stats
    vis_dir = os.path.join(dataset_dir, "visualizations")
    create_visualizations(
        input_stats,
        target_stats,
        vis_dir,
        dataset_name,
        _determine_task_type(stats_computer),
    )

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
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for computation",
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    config = OmegaConf.load(args.config)

    for dataset_config in config["datamodules"]:
        process_dataset(dataset_config, args.save_dir, args.device)

    print(f"\nAll dataset statistics computed and saved to {args.save_dir}")


if __name__ == "__main__":
    main()
