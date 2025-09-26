# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Visualization utilities for GeoBench datasets."""

import json
import matplotlib.pyplot as plt
import numpy as np

import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from typing import Dict, List, Tuple, Union, Optional, Type
import torch.nn as nn


def plot_channel_histograms(stats_json_path: str) -> plt.Figure:
    """
    Plots channel-wise histograms for each modality from a dataset statistics JSON file.

    Args:
        stats_json_path: Path to the JSON file containing dataset statistics.
                         Expected format includes an 'input_stats' key, which is a
                         dictionary mapping modality keys (e.g., 'image_s1', 'image_s2')
                         to statistics including 'band_names', 'histograms', and
                         'histogram_bins'.
    """
    with open(stats_json_path, "r") as f:
        stats = json.load(f)

    input_stats = stats["input_stats"]

    for modality_key, modality_stats in input_stats.items():
        band_names = modality_stats["band_names"]
        # Ensure histograms is a list of lists, even if only one band exists
        histograms = modality_stats["histograms"]
        if not isinstance(histograms[0], list):
            histograms = [histograms]  # Wrap if it's a single flat list for one band

        bins = modality_stats["histogram_bins"]

        if not band_names or not histograms or not bins:
            print(
                f"Warning: Missing band_names, histograms, or bins for modality {modality_key}. Skipping."
            )
            continue

        if len(band_names) != len(histograms):
            print(
                f"Warning: Mismatch between number of band names ({len(band_names)}) and histograms ({len(histograms)}) for {modality_key}. Skipping."
            )
            continue
        fig, ax = plt.subplots(figsize=(12, 6))

        bin_edges = np.array(bins)

        for i, band_name in enumerate(band_names):
            counts = np.array(histograms[i])

            # Determine x-values (pixel values) for plotting
            if len(bin_edges) == len(counts) + 1:
                # Standard case: bins are edges, plot at bin centers
                x_values = (bin_edges[:-1] + bin_edges[1:]) / 2
            elif len(bin_edges) == len(counts):
                # Assume bins are centers or left edges
                x_values = bin_edges
                print(
                    f"Warning: Assuming histogram_bins for {band_name} in {modality_key} represent bin centers/starts."
                )
            else:
                print(
                    f"Warning: Unexpected relationship between bin count ({len(bin_edges)}) and histogram count ({len(counts)}) for band {band_name} in {modality_key}. Skipping band."
                )
                continue

            # Plot as a line plot
            ax.plot(x_values, counts, label=band_name, alpha=0.8)

        ax.set_title(f"Channel Histograms for Modality: {modality_key}")
        ax.set_xlabel("Pixel Value (Bin Edge)")
        ax.set_ylabel("Frequency (Count)")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
        fig.tight_layout()

    return fig


def create_normalization_stats(dataset_stats: dict) -> dict:
    """Extract normalization stats from dataset stats format to normalizer format."""
    norm_stats = {"means": {}, "stds": {}, "clip_min": {}, "clip_max": {}}

    for modality_key, modality_stats in dataset_stats["input_stats"].items():
        band_names = modality_stats["band_names"]
        means = modality_stats["mean"]
        stds = modality_stats["std"]

        # Handle case where not all bands have stats
        if len(band_names) > len(means):
            print(
                f"Warning: Number of band names ({len(band_names)}) > number of mean values ({len(means)}) for {modality_key}"
            )
            band_names = band_names[: len(means)]

        for i, band in enumerate(band_names):
            if i < len(means) and i < len(stds):
                norm_stats["means"][band] = means[i]
                norm_stats["stds"][band] = stds[i]

                # Add optional clipping values (2% to 98% percentiles if available)
                if "pct_02" in modality_stats and "pct_98" in modality_stats:
                    if (
                        modality_stats["pct_02"] is not None
                        and modality_stats["pct_98"] is not None
                    ):
                        if i < len(modality_stats["pct_02"]) and i < len(
                            modality_stats["pct_98"]
                        ):
                            norm_stats["clip_min"][band] = modality_stats["pct_02"][i]
                            norm_stats["clip_max"][band] = modality_stats["pct_98"][i]

    return norm_stats


def compute_batch_histograms(
    batch: Dict[str, Tensor],
    n_bins: int = 100,
    hist_range: Optional[Tuple[float, float]] = None,
) -> Dict[str, Dict[str, Union[List, np.ndarray]]]:
    """
    Compute channel-wise histograms for image modalities in a batch.

    Args:
        batch: Dictionary with keys like 'image_s1', 'image_s2' containing tensors [B, C, H, W]
        n_bins: Number of bins for histogram
        hist_range: Optional range for all histograms (min, max)

    Returns:
        Dictionary with statistics for each modality
    """
    batch_stats = {}

    for key, tensor in batch.items():
        if key.startswith("image_") and isinstance(tensor, Tensor) and tensor.ndim == 4:
            modality = key

            # Extract number of channels
            num_channels = tensor.shape[1]
            histograms = []
            bin_edges = None

            # Compute per-channel histograms
            for c in range(num_channels):
                channel_data = tensor[:, c, :, :].detach().cpu().numpy().flatten()
                counts, edges = np.histogram(
                    channel_data, bins=n_bins, range=hist_range
                )
                histograms.append(counts.tolist())

                # Store bin edges (same for all channels if range is fixed)
                if bin_edges is None:
                    bin_edges = edges.tolist()

            # Store stats for this modality
            batch_stats[modality] = {
                "histograms": histograms,
                "histogram_bins": bin_edges,
                "min": [float(tensor[:, c, :, :].min()) for c in range(num_channels)],
                "max": [float(tensor[:, c, :, :].max()) for c in range(num_channels)],
                "mean": [float(tensor[:, c, :, :].mean()) for c in range(num_channels)],
                "std": [float(tensor[:, c, :, :].std()) for c in range(num_channels)],
            }

    return batch_stats


def plot_batch_histograms(
    batch_stats: Dict[str, Dict[str, Union[List, np.ndarray]]],
    band_names: Optional[Dict[str, List[str]]] = None,
    figsize: Tuple[int, int] = (12, 5),
    title_suffix: str = "",
) -> List[plt.Figure]:
    """
    Plot channel-wise histograms for image modalities.

    Args:
        batch_stats: Dictionary with statistics for each modality
        band_names: Optional dictionary mapping modality keys to lists of band names
        figsize: Figure size
        title_suffix: Suffix to add to plot titles (e.g., "Raw" or "Normalized")

    Returns:
        List of matplotlib figures
    """
    figs = []

    for modality, stats in batch_stats.items():
        fig, ax = plt.subplots(figsize=figsize)
        figs.append(fig)

        # Get histogram data
        histograms = stats["histograms"]
        bin_edges = stats["histogram_bins"]

        if not histograms or not bin_edges:
            print(
                f"Warning: Missing histograms or bin_edges for modality {modality}. Skipping."
            )
            continue

        # Get labels for each channel
        if band_names and modality in band_names:
            labels = band_names[modality]
            # Ensure length matches number of histograms
            if len(labels) != len(histograms):
                print(
                    f"Warning: Number of band names ({len(labels)}) != number of histograms ({len(histograms)}) for {modality}"
                )
                labels = [f"Channel {i}" for i in range(len(histograms))]
        else:
            labels = [f"Channel {i}" for i in range(len(histograms))]

        # Plot histograms as line plots
        for i, (hist, label) in enumerate(zip(histograms, labels)):
            # Convert bin edges to bin centers for plotting
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            ax.plot(bin_centers, hist, label=label, alpha=0.7)

        # Customize plot
        title = f"Histogram for {modality}{title_suffix}"
        ax.set_title(title)
        ax.set_xlabel("Pixel Value")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)

        # Add stats to plot as text
        stats_text = []
        for i, label in enumerate(labels):
            if i < len(stats["mean"]) and i < len(stats["std"]):
                stats_text.append(
                    f"{label}: μ={stats['mean'][i]:.2f}, σ={stats['std'][i]:.2f}"
                )

        ax.text(
            0.02,
            0.98,
            "\n".join(stats_text),
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

        plt.tight_layout()

    return figs


def get_normalized_batch(
    datamodule, normalizer: nn.Module, split: str = "train", batch_index: int = 0
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    """
    Get a batch of data and its normalized version.

    Args:
        datamodule: Lightning DataModule instance
        normalizer: Normalizer module
        split: Data split ('train', 'val', 'test')
        batch_index: Index of batch to retrieve

    Returns:
        Tuple of (original_batch, normalized_batch)
    """
    # Store original normalizer
    original_normalizer = None

    # Find the dataset attribute based on split
    if split == "train":
        dataset = datamodule.train_dataset
    elif split == "val":
        dataset = datamodule.val_dataset
    elif split == "test":
        dataset = datamodule.test_dataset
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")

    # Store original normalizer and set to None temporarily
    if hasattr(dataset, "data_normalizer"):
        original_normalizer = dataset.data_normalizer
        dataset.data_normalizer = nn.Identity()  # Temporarily disable normalization

    # Get dataloader for specified split
    if split == "train":
        dataloader = datamodule.train_dataloader()
    elif split == "val":
        dataloader = datamodule.val_dataloader()
    else:
        dataloader = datamodule.test_dataloader()

    # Get batch
    for i, batch in enumerate(dataloader):
        if i == batch_index:
            # Apply custom normalizer
            normalized_batch = normalizer(batch)

            # Restore original normalizer
            if original_normalizer is not None:
                dataset.data_normalizer = original_normalizer

            return batch, normalized_batch

    # Restore original normalizer if we didn't find the batch
    if original_normalizer is not None:
        dataset.data_normalizer = original_normalizer

    raise IndexError(f"Batch index {batch_index} out of range for {split} split")


def visualize_normalization_effects(
    datamodule,
    stats_json_path: str,
    normalizer_classes: List[Type[nn.Module]],
    split: str = "train",
    batch_index: int = 0,
    n_bins: int = 100,
):
    """
    Visualize the effects of different normalization schemes on a batch.

    Args:
        datamodule: Lightning DataModule instance
        stats_json_path: Path to JSON file with dataset statistics
        normalizer_classes: List of normalizer classes to compare
        split: Data split to use ('train', 'val', 'test')
        batch_index: Index of batch to retrieve
        n_bins: Number of bins for histograms
    """
    # Get band order from datamodule
    band_order = datamodule.band_order

    # Extract band names for plotting
    band_names = {}
    for modality, bands in band_order.items():
        # Filter out fill values (non-string elements)
        band_names[f"image_{modality}"] = [b for b in bands if isinstance(b, str)]

    # Get raw batch and compute histograms
    raw_batch, _ = get_normalized_batch(datamodule, nn.Identity(), split, batch_index)
    raw_stats = compute_batch_histograms(raw_batch, n_bins=n_bins)

    print("Raw batch statistics:")
    for modality, stats in raw_stats.items():
        print(f"  {modality}:")
        print(f"    Mean: {stats['mean']}")
        print(f"    Std:  {stats['std']}")
        print(f"    Min:  {stats['min']}")
        print(f"    Max:  {stats['max']}")

    # Plot raw histograms
    plot_batch_histograms(raw_stats, band_names, title_suffix=" (Raw)")

    # For each normalizer, apply and visualize
    for normalizer_class in normalizer_classes:
        normalizer_name = normalizer_class.__name__
        print(f"\nApplying {normalizer_name}...")

        # Create normalizer
        normalizer = create_normalizer(stats_json_path, normalizer_class, band_order)

        # Get normalized batch
        _, normalized_batch = get_normalized_batch(
            datamodule, normalizer, split, batch_index
        )

        # Compute and plot histograms for normalized batch
        norm_stats = compute_batch_histograms(normalized_batch, n_bins=n_bins)

        print(f"{normalizer_name} statistics:")
        for modality, stats in norm_stats.items():
            print(f"  {modality}:")
            print(f"    Mean: {stats['mean']}")
            print(f"    Std:  {stats['std']}")
            print(f"    Min:  {stats['min']}")
            print(f"    Max:  {stats['max']}")

        plot_batch_histograms(
            norm_stats, band_names, title_suffix=f" ({normalizer_name})"
        )

    plt.show()
