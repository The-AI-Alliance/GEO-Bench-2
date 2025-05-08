# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Visualization utilities for GeoBench datasets."""

import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch.nn as nn
from torch import Tensor


def plot_channel_histograms(stats_json_path: str) -> plt.Figure:
    """Plots channel-wise histograms for each modality from a dataset statistics JSON file.

    Args:
        stats_json_path: Path to the JSON file containing dataset statistics.
                         Expected format includes an 'input_stats' key, which is a
                         dictionary mapping modality keys (e.g., 'image_s1', 'image_s2')
                         to statistics including 'band_names', 'histograms', and
                         'histogram_bins'.
    """
    with open(stats_json_path) as f:
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
    batch: dict[str, Tensor],
    n_bins: int = 100,
    hist_range: tuple[float, float] | None = None,
) -> dict[str, dict[str, list | np.ndarray]]:
    """Compute channel-wise histograms for image modalities in a batch.

    Args:
        batch: Dictionary with keys like 'image_s1', 'image_s2' containing tensors [B, C, H, W]
        n_bins: Number of bins for histogram
        hist_range: Optional range for all histograms (min, max)

    Returns:
        Dictionary with statistics for each modality
    """
    batch_stats = {}

    for key, tensor in batch.items():
        if key.startswith("image") and isinstance(tensor, Tensor) and tensor.ndim == 4:
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
    batch_stats: dict[str, dict[str, list | np.ndarray]],
    band_order: dict[str, list[str | float]] | list[str | float] | None = None,
    figsize: tuple[int, int] = (12, 5),
    title_suffix: str = "",
) -> list[plt.Figure]:
    """Plot channel-wise histograms for image modalities.

    Args:
        batch_stats: Dictionary with statistics for each modality
        band_order: Either a dictionary mapping modality keys to lists of band names/scaling factors,
                    or a single list of band names/scaling factors for a single modality
        figsize: Figure size
        title_suffix: Suffix to add to plot titles (e.g., "Raw" or "Normalized")

    Returns:
        List of matplotlib figures
    """
    figs = []

    # Convert band_order to dictionary format if it's a list
    if isinstance(band_order, list):
        # If band_order is a list, we assume there's only one modality
        # Check if there's just one key in batch_stats that starts with "image"
        # or exactly one key named "image"
        image_keys = [key for key in batch_stats.keys() if key.startswith("image")]

        if len(image_keys) == 1:
            # Use the detected image key
            band_order = {image_keys[0]: band_order}
        elif "image" in batch_stats:
            # Use the "image" key directly
            band_order = {"image": band_order}
        else:
            # Default to creating entries for all modalities with the same band_order
            band_order = {modality: band_order for modality in batch_stats.keys()}
            print(
                f"Warning: Applying the same band names to all modalities: {list(batch_stats.keys())}"
            )

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

        # Convert bin_edges to numpy array if it's not already
        bin_edges = np.array(bin_edges)

        # Get labels for each channel
        if band_order and modality in band_order:
            # Get all band names including numeric scaling factors
            labels = [str(item) for item in band_order[modality]]

            # Ensure length matches number of histograms
            if len(labels) != len(histograms):
                # Keep only the number of labels that match the histograms
                if len(labels) > len(histograms):
                    labels = labels[: len(histograms)]
                else:
                    # Append generic labels if we have more histograms than labels
                    labels.extend(
                        [
                            f"Channel {i + len(labels)}"
                            for i in range(len(histograms) - len(labels))
                        ]
                    )
        else:
            labels = [f"Channel {i}" for i in range(len(histograms))]

        # Plot histograms as line plots
        for i, (hist, label) in enumerate(zip(histograms, labels)):
            # Convert histogram to numpy array if it's not already
            hist = np.array(hist)
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
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Get a batch of data and its normalized version.

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


def visualize_segmentation_target_statistics(
    stats_json_path: str, dataset_name: str = None, figsize: tuple[int, int] = (26, 10)
) -> plt.Figure:
    """Visualizes target statistics from earth observation datasets with three informative subplots.

    Args:
        stats_json_path: Path to dataset statistics JSON file.
        dataset_name: Optional name for the dataset. If None, derived from filename.
        figsize: Figure size as (width, height) tuple.

    Returns:
        Matplotlib figure with subplots showing class distribution, presence, and co-occurrence
    """
    with open(stats_json_path) as f:
        stats = json.load(f)

    target_stats = stats.get("target_stats", {})

    pixel_distribution = target_stats.get("pixel_distribution", [])
    class_presence_ratio = target_stats.get("class_presence_ratio", [])
    pixel_counts = target_stats.get("pixel_counts", [])
    num_classes = target_stats.get("num_classes", len(pixel_distribution))
    total_images = target_stats.get("total_images", 0)

    class_names = target_stats.get(
        "class_names", [f"Class {i}" for i in range(num_classes)]
    )

    has_cooccurrence = "class_cooccurrence_ratio" in target_stats
    cooccurrence_ratio = target_stats.get("class_cooccurrence_ratio", None)

    if dataset_name is None:
        dataset_name = os.path.basename(stats_json_path).split("_stats")[0]

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2])

    ax1 = fig.add_subplot(gs[0, 0])

    bars = ax1.bar(np.arange(num_classes), class_presence_ratio, color="skyblue")

    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.annotate(
            f"{height:.2f}\n({int(target_stats.get('class_presence_counts', [])[i]) if i < len(target_stats.get('class_presence_counts', [])) else 'N/A'})",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax1.set_xlabel("Class Label/Name (in enumerated order)", fontsize=16)
    ax1.set_ylabel("Presence Ratio (fraction of images)", fontsize=16)
    ax1.set_title("Class Presence Distribution", fontsize=16)
    ax1.set_xticks(np.arange(num_classes))
    ax1.set_xticklabels(class_names, fontsize=13, rotation=45, ha="right")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)
    ax1.tick_params(axis="both", which="major", labelsize=13)

    # center plot about overall class distribution
    ax2 = fig.add_subplot(gs[0, 1])

    sorted_indices = np.argsort(pixel_distribution)[::-1]
    sorted_distribution = np.array(pixel_distribution)[sorted_indices]
    sorted_class_names = [class_names[i] for i in sorted_indices]

    sorted_percentages = sorted_distribution * 100

    bars = ax2.bar(np.arange(num_classes), sorted_percentages, color="skyblue")

    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height > 0.1:
            ax2.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                rotation=90,
            )

    ax2.set_xlabel("Class (sorted by frequency)", fontsize=16)
    ax2.set_ylabel("Pixel Distribution (%)", fontsize=16)
    ax2.set_title("Class Distribution Analysis", fontsize=16)
    ax2.set_xticks(np.arange(num_classes))
    ax2.set_xticklabels(sorted_class_names, rotation=45, fontsize=13, ha="right")
    ax2.grid(axis="y", linestyle="--", alpha=0.7)
    ax2.tick_params(axis="both", which="major", labelsize=13)

    ax3 = fig.add_subplot(gs[0, 2])

    if has_cooccurrence:
        mask = np.zeros_like(cooccurrence_ratio, dtype=bool)
        np.fill_diagonal(mask, True)

        cmap = sns.color_palette("Blues", as_cmap=True)

        sns.heatmap(
            cooccurrence_ratio,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            mask=mask,
            vmin=0,
            vmax=min(1.0, np.max(cooccurrence_ratio) * 1.2),
            linewidths=0.5,
            ax=ax3,
            cbar_kws={"label": "Co-occurrence Probability"},
        )

        ax3.set_title(
            "Class Co-occurrence Analysis\n(How often classes appear together)",
            fontsize=16,
        )
        ax3.set_xlabel("Class Label/Name", fontsize=12)
        ax3.set_ylabel("Class Label/Name", fontsize=12)

        ax3.set_xticks(np.arange(num_classes) + 0.5)
        ax3.set_yticks(np.arange(num_classes) + 0.5)
        ax3.set_xticklabels(class_names, rotation=45, ha="right", fontsize=13)
        ax3.set_yticklabels(class_names, rotation=0, fontsize=13)
    else:
        ax3.text(
            0.5,
            0.5,
            "Co-occurrence data not available",
            ha="center",
            va="center",
            fontsize=14,
        )
        ax3.axis("off")

    fig.suptitle(
        f"Target Statistics for {dataset_name.upper()} Dataset\n(Total Images: {total_images}, Classes: {num_classes})",
        fontsize=18,
        y=0.98,
    )

    plt.tight_layout()
    fig.subplots_adjust(top=0.90)

    return fig
