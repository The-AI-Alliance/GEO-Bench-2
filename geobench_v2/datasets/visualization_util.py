# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Visualization utilities for GeoBench datasets."""

import json
from typing import Any

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tacoreader
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch, Rectangle
from shapely.geometry import Point
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
        histograms = modality_stats["histograms"]
        if not isinstance(histograms[0], list):
            histograms = [histograms]

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

            if len(bin_edges) == len(counts) + 1:
                x_values = (bin_edges[:-1] + bin_edges[1:]) / 2
            elif len(bin_edges) == len(counts):
                x_values = bin_edges
                print(
                    f"Warning: Assuming histogram_bins for {band_name} in {modality_key} represent bin centers/starts."
                )
            else:
                print(
                    f"Warning: Unexpected relationship between bin count ({len(bin_edges)}) and histogram count ({len(counts)}) for band {band_name} in {modality_key}. Skipping band."
                )
                continue

            ax.plot(x_values, counts, label=band_name, alpha=0.8)

        ax.set_title(f"Channel Histograms for Modality: {modality_key}")
        ax.set_xlabel("Pixel Value (Bin Edge)")
        ax.set_ylabel("Frequency (Count)")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
        fig.tight_layout()

    return fig


def compute_batch_histograms(
    batch: dict[str, Tensor],
    n_bins: int = 100,
    hist_range: tuple[float, float] | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute channel-wise histograms for image modalities in a batch.

    Args:
        batch: Dictionary with keys like 'image_s1', 'image_s2' containing tensors [B, C, H, W], or [B, T, C, H, W]
        n_bins: Number of bins for histogram
        hist_range: Optional range for all histograms (min, max)

    Returns:
        Dictionary with statistics for each modality
    """
    batch_stats = {}

    for key, tensor in batch.items():
        if key.startswith("image") and isinstance(tensor, Tensor) and tensor.ndim >= 4:
            modality = key

            num_channels = tensor.shape[1]
            histograms = []
            bin_edges = None

            for c in range(num_channels):
                channel_data = tensor[:, c, :, :].detach().cpu().numpy().flatten()
                counts, edges = np.histogram(
                    channel_data, bins=n_bins, range=hist_range
                )
                histograms.append(counts.tolist())
                if bin_edges is None:
                    bin_edges = edges.tolist()

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

    if isinstance(band_order, list):
        image_keys = [key for key in batch_stats.keys() if key.startswith("image")]

        if len(image_keys) == 1:
            band_order = {image_keys[0]: band_order}
        elif "image" in batch_stats:
            band_order = {"image": band_order}
        else:
            band_order = {modality: band_order for modality in batch_stats.keys()}
            print(
                f"Warning: Applying the same band names to all modalities: {list(batch_stats.keys())}"
            )

    for modality, stats in batch_stats.items():
        fig, ax = plt.subplots(figsize=figsize)
        figs.append(fig)

        histograms = stats["histograms"]
        bin_edges = stats["histogram_bins"]

        if not histograms or not bin_edges:
            print(
                f"Warning: Missing histograms or bin_edges for modality {modality}. Skipping."
            )
            continue

        bin_edges = np.array(bin_edges)

        if band_order and modality in band_order:
            labels = [str(item) for item in band_order[modality]]

            if len(labels) != len(histograms):
                if len(labels) > len(histograms):
                    labels = labels[: len(histograms)]
                else:
                    labels.extend(
                        [
                            f"Channel {i + len(labels)}"
                            for i in range(len(histograms) - len(labels))
                        ]
                    )
        else:
            labels = [f"Channel {i}" for i in range(len(histograms))]

        for i, (hist, label) in enumerate(zip(histograms, labels)):
            hist = np.array(hist)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            ax.plot(bin_centers, hist, label=label, alpha=0.7)

        title = f"Histogram for {modality}{title_suffix}"
        ax.set_title(title)
        ax.set_xlabel("Pixel Value")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)

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


def visualize_segmentation_target_statistics(
    stats_json_path: str, dataset_name: str, figsize: tuple[int, int] = (26, 10)
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
    num_classes = target_stats.get("num_classes", len(pixel_distribution))
    total_images = target_stats.get("total_images", 0)

    class_names = target_stats.get(
        "class_names", [f"Class {i}" for i in range(num_classes)]
    )

    has_cooccurrence = "class_cooccurrence_ratio" in target_stats
    cooccurrence_ratio = target_stats.get("class_cooccurrence_ratio", None)

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


def compare_normalization_methods(
    batch, normalizer_modules, datamodule, figsize=(20, 8)
) -> tuple[plt.Figure, list[dict[str, Tensor]]]:
    """Create a visualization showing before/after distributions for multiple normalization methods.

    The visualization is organized with:
    - Rows: Different modalities (e.g., S1, S2) or single modality ("image")
    - Columns: Raw data and different normalization methods

    Args:
        batch: Dictionary containing the batch data
        normalizer_modules: List of normalizer modules to compare
        datamodule: Datamodule to extract band information
        figsize: Base figure size (width, height), will be adjusted based on number of normalizers

    Returns:
        matplotlib figure with the visualizations, and list of normalized batches
    """
    n_normalizers = len(normalizer_modules)
    normalized_batches = [normalizer(batch.copy()) for normalizer in normalizer_modules]
    modalities = [key for key in batch.keys() if key.startswith("image")]
    n_modalities = len(modalities)

    adjusted_width = figsize[0] * (n_normalizers + 1) / 3
    adjusted_height = figsize[1] * n_modalities
    fig = plt.figure(figsize=(adjusted_width, adjusted_height))

    gs = GridSpec(
        n_modalities * 2,
        n_normalizers + 1,
        height_ratios=[4, 1] * n_modalities,
        width_ratios=[1] * (n_normalizers + 1),
        figure=fig,
    )

    column_titles = ["Raw Data"] + [
        f"{norm.__class__.__name__}" for norm in normalizer_modules
    ]

    for row_idx, modality in enumerate(modalities):
        if modality == "image":
            modality_prefix = "main"
            plot_title = "Image"
        else:
            modality_prefix = modality.replace("image_", "")
            plot_title = f"{modality_prefix.upper()} Modality"

        band_names = []
        modality_config = datamodule.dataset_band_config

        if hasattr(datamodule, "dataset_band_config") and hasattr(
            datamodule.dataset_band_config, "modalities"
        ):
            if modality_prefix in datamodule.dataset_band_config.modalities:
                modality_config = datamodule.dataset_band_config.modalities[
                    modality_prefix
                ]
            elif (
                modality == "image"
                and len(datamodule.dataset_band_config.modalities) == 1
            ):
                config_key = next(iter(datamodule.dataset_band_config.modalities))
                modality_config = datamodule.dataset_band_config.modalities[config_key]

        band_names = modality_config.plot_bands
        if modality == "image":
            plot_title = "Single Modality"
        else:
            plot_title = f"{modality_prefix.upper()}"

        band_indices = []
        for band in band_names:
            try:
                if isinstance(datamodule.band_order, dict):
                    if modality_prefix in datamodule.band_order:
                        band_indices.append(
                            datamodule.band_order[modality_prefix].index(band)
                        )
                    elif modality == "image" and len(datamodule.band_order) == 1:
                        config_key = next(iter(datamodule.band_order))
                        band_indices.append(
                            datamodule.band_order[config_key].index(band)
                        )
                else:
                    band_indices.append(datamodule.band_order.index(band))
            except (ValueError, KeyError):
                print(
                    f"Warning: Band {band} not found in band_order for modality {modality}"
                )

        if not band_indices and modality in batch:
            band_indices = list(range(batch[modality].shape[1]))
            band_names = [f"Band {i}" for i in band_indices]

        all_data: dict[str, dict[str, Any]] = {"raw": {}}
        normalizer_display_names = {}

        for i, normalizer in enumerate(normalizer_modules):
            base_name = normalizer.__class__.__name__
            if hasattr(normalizer, "processing_mode"):
                base_name = f"{base_name} ({normalizer.processing_mode})"

            norm_key = base_name
            suffix = 1
            while norm_key in all_data:
                norm_key = f"{base_name}_{suffix}"
                suffix += 1

            all_data[norm_key] = {}
            normalizer_display_names[i] = norm_key

        for i, band_idx in enumerate(band_indices):
            if band_idx < batch[modality].shape[1]:
                band_label = (
                    band_names[i] if i < len(band_names) else f"Band {band_idx}"
                )
                all_data["raw"][band_label] = (
                    batch[modality][:, band_idx].flatten().numpy()
                )

                for j, norm_batch in enumerate(normalized_batches):
                    norm_key = normalizer_display_names[j]
                    all_data[norm_key][band_label] = (
                        norm_batch[modality][:, band_idx].flatten().numpy()
                    )

        for col_idx in range(n_normalizers + 1):
            data_key = "raw" if col_idx == 0 else normalizer_display_names[col_idx - 1]

            main_ax = fig.add_subplot(gs[row_idx * 2, col_idx])
            stats_ax = fig.add_subplot(gs[row_idx * 2 + 1, col_idx])
            stats_ax.axis("off")

            bins = 100
            for band in all_data[data_key].keys():
                hist, bin_edges = np.histogram(all_data[data_key][band], bins=bins)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                main_ax.plot(bin_centers, hist, label=band, linewidth=2.5)

            if row_idx == 0:
                display_title = column_titles[col_idx]
                main_ax.set_title(display_title, fontsize=16, pad=20)

            main_ax.set_title(f"{plot_title}", fontsize=14, pad=5)
            main_ax.set_xlabel("Value")
            main_ax.set_ylabel("Frequency")
            main_ax.legend(loc="upper right")

            main_ax.grid(True, alpha=0.3)
            main_ax.spines["top"].set_visible(False)
            main_ax.spines["right"].set_visible(False)

            stats_text = f"Statistics for {plot_title} ({data_key}):\n"
            for band in all_data[data_key].keys():
                data = all_data[data_key][band]
                mean_val = np.mean(data)
                std_val = np.std(data)
                min_val = np.min(data)
                max_val = np.max(data)
                stats_text += f"{band}: Mean={mean_val:.3f}, Std={std_val:.3f}, Range=[{min_val:.3f}, {max_val:.3f}]\n"

            stats_ax.text(
                0.5,
                0.5,
                stats_text,
                ha="center",
                va="center",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )

    normalizer_labels = []
    for norm in normalizer_modules:
        if hasattr(norm, "processing_mode"):
            normalizer_labels.append(
                f"{norm.__class__.__name__} ({norm.processing_mode})"
            )
        else:
            normalizer_labels.append(norm.__class__.__name__)

    normalizer_names = ", ".join(normalizer_labels)
    plt.suptitle(
        f"Comparison of Normalization Methods: {normalizer_names}", fontsize=18, y=0.995
    )
    plt.tight_layout()
    fig.subplots_adjust(top=0.95 if n_modalities == 1 else 0.97)

    return fig, normalized_batches


DATASET_COLORS = {
    "biomassters": "#E69F00",
    "benv2": "#0072B2",
    "burn_scars": "#CC79A7",
    "caffe": "#009E73",
    "cloudsen12": "#D55E00",
    "dynamic_earthnet": "#F0E442",
    "everwatch": "#56B4E9",
    "flair2": "#9400D3",
    "fotw": "#A52A2A",
    "kuro_siwo": "#FFC0CB",
    "pastis": "#2F4F4F",
    "spacenet2": "#40E0D0",
    "spacenet7": "#808000",
    "treesatai": "#FF4500",
    "wind_turbine": "#4682B4",
    "substation": "#8A2BE2",
    "nzcattle": "#DAA520",
    "forestnet": "#228B22",
}


def plot_global_sample_distribution(
    taco_paths: dict[str, list[str]], labels=None, output_path="global_distribution.png"
) -> None:
    """Plots the distribution of samples across the globe, with a Europe zoom above the global map and legend below.

    Args:
        taco_paths: dict[str, list[str]] mapping dataset labels to lists of TACO file
        labels: Optional list of dataset labels to include. If None, all keys from taco_paths are used.
        output_path: Path to save the output figure
    """
    fig = plt.figure(figsize=(20, 14))

    gs = GridSpec(2, 1, height_ratios=[1.8, 2.2], hspace=0.04)

    lon_min, lon_max = -25, 45
    lat_min, lat_max = 34, 72

    # Europe zoom (top)
    ax_europe = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax_europe.coastlines()
    ax_europe.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # Global map (bottom)
    ax_global = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    ax_global.coastlines()
    ax_global.set_global()

    legend_handles = []
    for label, path_list in taco_paths.items():
        df = tacoreader.load(path_list)
        df = df.sample(n=500)
        if "lon" not in df.columns:
            continue

        color = DATASET_COLORS[label]

        # Global scatter
        ax_global.scatter(
            df["lon"],
            df["lat"],
            s=2,
            alpha=0.7,
            color=color,
            label=label,
            transform=ccrs.PlateCarree(),
        )
        # Europe scatter (only points in Europe extent)
        europe_mask = (
            (df["lon"] >= lon_min)
            & (df["lon"] <= lon_max)
            & (df["lat"] >= lat_min)
            & (df["lat"] <= lat_max)
        )
        ax_europe.scatter(
            df.loc[europe_mask, "lon"],
            df.loc[europe_mask, "lat"],
            s=2,
            alpha=0.5,
            color=color,
            label=label,
            transform=ccrs.PlateCarree(),
        )

        handle = mlines.Line2D(
            [],
            [],
            color=color,
            marker="o",
            linestyle="None",
            markersize=14,
            alpha=1.0,
            label=label,
        )
        legend_handles.append(handle)

    # Rectangle for Europe zoom on global map
    width = lon_max - lon_min
    height = lat_max - lat_min
    europe_rect = Rectangle(
        (lon_min, lat_min),
        width,
        height,
        linewidth=2.5,
        edgecolor="black",
        facecolor="none",
        linestyle="dotted",
        zorder=10,
    )
    europe_rect.set_transform(ccrs.PlateCarree())
    ax_global.add_patch(europe_rect)

    # Connecting lines
    rect_top_left = (lon_min, lat_max)
    rect_top_right = (lon_max, lat_max)
    europe_axes_bottom_left = (0.0, 0.0)
    europe_axes_bottom_right = (1.0, 0.0)

    for rect_xy, europe_xy in [
        (rect_top_left, europe_axes_bottom_left),
        (rect_top_right, europe_axes_bottom_right),
    ]:
        con = ConnectionPatch(
            xyA=rect_xy,
            xyB=europe_xy,
            coordsA="data",
            coordsB="axes fraction",
            axesA=ax_global,
            axesB=ax_europe,
            linestyle="dotted",
            linewidth=2.0,
            color="black",
            alpha=0.7,
            zorder=100,
        )
        fig.add_artist(con)

    fig.tight_layout(rect=[0, 0.1, 1, 0.96])

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
        fontsize=24,
        title="Datasets",
        title_fontsize=22,
        handletextpad=0.1,
        columnspacing=0.2,
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"Saved global distribution map as {output_path}")


def extract_continent_names(df):
    """Extract continent names for each point in the DataFrame based on lat/lon.

    Args:
        df: DataFrame with 'lat' and 'lon' columns.

    Returns:
        GeoDataFrame with an additional 'continent_name' column.
    """
    geometry = [Point(xy) for xy in zip(df["lon"], df["lat"])]
    gdf_points = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    url = (
        "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    )
    world = gpd.read_file(url)
    continents = world.dissolve(by="CONTINENT")[["geometry"]]
    gdf_points = gpd.sjoin(gdf_points, continents, how="left", predicate="within")
    gdf_points = gdf_points.rename(columns={"CONTINENT": "continent_name"})
    return gdf_points


def plot_continent_bar(
    taco_paths: dict[str, list[str]], output_path="continent_bar.png"
):
    """Plots the aggregate percentage distribution of samples across continents for all datasets combined.

    Args:
        taco_paths: dict[str, list[str]] (dict): dictionary of paths to tacos to visualize
        output_path (str): Path to save the output plot.
    """
    all_samples = []

    for paths in taco_paths.values():
        df = tacoreader.load(paths)
        if "lon" not in df.columns or "lat" not in df.columns:
            continue
        gdf_points = extract_continent_names(df)
        gdf_points = gdf_points.dropna(subset=["continent_name"])
        all_samples.append(gdf_points["continent_name"])

    all_continents_series = pd.concat(all_samples)
    continent_counts = all_continents_series.value_counts()
    total_samples = continent_counts.sum()
    continent_percentages = continent_counts / total_samples * 100

    continent_percentages = continent_percentages.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        continent_percentages.index, continent_percentages.values, color="skyblue"
    )
    ax.set_xticklabels(continent_percentages.index, rotation=45, fontsize=14)
    ax.set_ylabel("Pct of Total Samples (%)", fontsize=14)
    ax.set_title("Aggregate Sample Percentage by Continent", fontsize=18)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Saved aggregate continent bar chart as {output_path}")
