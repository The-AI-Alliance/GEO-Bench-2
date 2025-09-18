# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Utility functions for geospatial dataset splitting and visualization."""

import os
import re

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from rasterio.enums import Compression
from rasterio.features import rasterize
from rasterio.windows import Window
from skimage.transform import resize
from sklearn.cluster import DBSCAN
from tqdm import tqdm


def dataframe_to_geodataframe(
    df: pd.DataFrame, lon_col: str = "lon", lat_col: str = "lat", crs: str = "EPSG:4326"
) -> gpd.GeoDataFrame:
    """Convert a standard pandas DataFrame with lat/lon columns to GeoDataFrame."""
    import shapely.geometry as sg

    df_copy = df.copy()
    geometries = [sg.Point(x, y) for x, y in zip(df_copy[lon_col], df_copy[lat_col])]
    gdf = gpd.GeoDataFrame(df_copy, geometry=geometries, crs=crs)

    return gdf


def geographic_distance_split(
    df: pd.DataFrame,
    lon_col: str = "lon",
    lat_col: str = "lat",
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    n_clusters: int | None = None,
    crs: str = "EPSG:4326",
    random_state: int = 42,
) -> pd.DataFrame:
    """Split dataset based on geographic distance between samples.

    Uses K-means clustering to group nearby points and assigns entire
    clusters to train/validation/test splits to maintain spatial coherence.

    Args:
        df: DataFrame containing latitude and longitude columns
        lon_col: Name of the longitude column
        lat_col: Name of the latitude column
        test_ratio: Proportion of data for test set
        val_ratio: Proportion of data for validation set
        n_clusters: Number of clusters to create (defaults to 10% of data points)
        crs: Coordinate reference system of the input coordinates
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with additional columns: 'split' and 'distance_cluster'
    """
    gdf = dataframe_to_geodataframe(df, lon_col, lat_col, crs)

    coords = np.array([(geom.x, geom.y) for geom in gdf.geometry])

    coords_norm = (coords - coords.mean(axis=0)) / coords.std(axis=0)

    if n_clusters is None:
        n_clusters = max(3, int(len(gdf) * 0.1))  # At least 3 clusters, or 10% of data

    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(coords_norm)

    gdf["distance_cluster"] = clusters

    n_total = len(gdf)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)

    unique_clusters = np.unique(clusters)
    np.random.seed(random_state)
    np.random.shuffle(unique_clusters)

    test_clusters = []
    val_clusters = []
    test_count = 0
    val_count = 0

    for c in unique_clusters:
        cluster_size = (clusters == c).sum()
        if test_count < n_test:
            test_clusters.append(c)
            test_count += cluster_size
        elif val_count < n_val:
            val_clusters.append(c)
            val_count += cluster_size

    gdf["split"] = "train"
    gdf.loc[gdf["distance_cluster"].isin(test_clusters), "split"] = "test"
    gdf.loc[gdf["distance_cluster"].isin(val_clusters), "split"] = "validation"

    result_df = df.copy()
    result_df["split"] = gdf["split"]
    result_df["distance_cluster"] = gdf["distance_cluster"]

    return result_df


def geographic_buffer_split(
    df: pd.DataFrame,
    lon_col: str = "lon",
    lat_col: str = "lat",
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    buffer_distance: float = 0.1,
    min_samples: int = 5,
    crs: str = "EPSG:4326",
    random_state: int = 42,
) -> pd.DataFrame:
    """Split data with buffer zones between train/val/test to prevent data leakage."""
    gdf = dataframe_to_geodataframe(df, lon_col, lat_col, crs)

    coords = np.array([(geom.x, geom.y) for geom in gdf.geometry])

    dbscan = DBSCAN(eps=buffer_distance, min_samples=min_samples)
    clusters = dbscan.fit_predict(coords)

    if (clusters == -1).any():
        max_cluster = max(0, clusters.max())
        outlier_indices = np.where(clusters == -1)[0]
        clusters[outlier_indices] = np.arange(
            max_cluster + 1, max_cluster + 1 + len(outlier_indices)
        )

    gdf["spatial_cluster"] = clusters

    n_total = len(gdf)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)

    unique_clusters = np.unique(clusters)
    np.random.seed(random_state)
    np.random.shuffle(unique_clusters)

    test_clusters = []
    val_clusters = []
    test_count = 0
    val_count = 0

    for c in unique_clusters:
        cluster_size = (clusters == c).sum()
        if test_count < n_test:
            test_clusters.append(c)
            test_count += cluster_size
        elif val_count < n_val:
            val_clusters.append(c)
            val_count += cluster_size

    gdf["split"] = "train"
    gdf.loc[gdf["spatial_cluster"].isin(test_clusters), "split"] = "test"
    gdf.loc[gdf["spatial_cluster"].isin(val_clusters), "split"] = "validation"

    result_df = df.copy()
    result_df["split"] = gdf["split"]
    result_df["spatial_cluster"] = gdf["spatial_cluster"]

    return result_df


def checkerboard_split(
    df: pd.DataFrame,
    lon_col: str = "lon",
    lat_col: str = "lat",
    n_blocks_x: int = 5,
    n_blocks_y: int = 5,
    pattern: str = "checkerboard",
    test_blocks_ratio: float = 0.2,
    val_blocks_ratio: float = 0.1,
    target_test_ratio: float = 0.2,
    target_val_ratio: float = 0.1,
    max_iterations: int = 50,
    ratio_tolerance: float = 0.02,
    crs: str = "EPSG:4326",
    random_state: int = 42,
) -> pd.DataFrame:
    """Split dataset into a spatial pattern to ensure coherent geographic distribution.

    Args:
        df: DataFrame containing latitude and longitude columns
        lon_col: Name of the longitude column
        lat_col: Name of the latitude column
        n_blocks_x: Number of blocks along the x-axis
        n_blocks_y: Number of blocks along the y-axis
        pattern: Pattern type - one of "checkerboard", "random", "balanced"
        test_blocks_ratio: Proportion of blocks for test set (when using "random")
        val_blocks_ratio: Proportion of blocks for validation set (when using "random")
        target_test_ratio: Target ratio of samples for test set (when using "balanced")
        target_val_ratio: Target ratio of samples for validation set (when using "balanced")
        max_iterations: Maximum number of iterations for ratio optimization
        ratio_tolerance: Acceptable deviation from target ratios
        crs: Coordinate reference system of the input coordinates
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with additional columns: 'split', 'block_x', 'block_y', and 'block_id'
    """
    gdf = dataframe_to_geodataframe(df, lon_col, lat_col, crs)

    minx, miny, maxx, maxy = gdf.total_bounds

    buffer_x = (maxx - minx) * 0.001
    buffer_y = (maxy - miny) * 0.001
    minx -= buffer_x
    maxx += buffer_x
    miny -= buffer_y
    maxy += buffer_y

    x_step = (maxx - minx) / n_blocks_x
    y_step = (maxy - miny) / n_blocks_y

    gdf["block_x"] = (
        ((gdf.geometry.x - minx) / x_step).astype(int).clip(0, n_blocks_x - 1)
    )
    gdf["block_y"] = (
        ((gdf.geometry.y - miny) / y_step).astype(int).clip(0, n_blocks_y - 1)
    )
    gdf["block_id"] = gdf["block_y"] * n_blocks_x + gdf["block_x"]

    block_counts = gdf["block_id"].value_counts().to_dict()
    n_blocks = n_blocks_x * n_blocks_y
    blocks = np.arange(n_blocks)

    np.random.seed(random_state)

    if pattern == "checkerboard":
        block_matrix = np.zeros((n_blocks_y, n_blocks_x), dtype=int)

        np.random.seed(random_state)
        start_pattern = np.random.randint(0, 3)

        for i in range(n_blocks_y):
            for j in range(n_blocks_x):
                block_matrix[i, j] = (i + j + start_pattern) % 3

        block_assignments = block_matrix.flatten()

        test_value = 0
        val_value = 1

        block_splits = {}
        for block_id in range(len(block_assignments)):
            if block_assignments[block_id] == test_value:
                block_splits[block_id] = "test"
            elif block_assignments[block_id] == val_value:
                block_splits[block_id] = "validation"
            else:
                block_splits[block_id] = "train"

    elif pattern == "balanced":
        np.random.shuffle(blocks)

        n_total_samples = len(gdf)
        n_test_samples_target = int(n_total_samples * target_test_ratio)
        n_val_samples_target = int(n_total_samples * target_val_ratio)

        block_splits = {}
        for block_id in range(n_blocks):
            block_splits[block_id] = "train"

        assigned_blocks = []
        test_blocks = []
        val_blocks = []
        test_samples = 0
        val_samples = 0
        remaining_blocks = blocks.copy()

        def get_initial_assignments():
            nonlocal \
                test_samples, \
                val_samples, \
                test_blocks, \
                val_blocks, \
                assigned_blocks, \
                remaining_blocks
            remaining_blocks = list(set(blocks) - set(assigned_blocks))
            np.random.shuffle(remaining_blocks)

            test_samples = 0
            for block_id in remaining_blocks:
                if test_samples < n_test_samples_target:
                    block_count = block_counts.get(block_id, 0)
                    test_samples += block_count
                    test_blocks.append(block_id)
                    assigned_blocks.append(block_id)
                    block_splits[block_id] = "test"
                else:
                    break

            remaining_blocks = list(set(blocks) - set(assigned_blocks))
            np.random.shuffle(remaining_blocks)

            val_samples = 0
            for block_id in remaining_blocks:
                if val_samples < n_val_samples_target:
                    block_count = block_counts.get(block_id, 0)
                    val_samples += block_count
                    val_blocks.append(block_id)
                    assigned_blocks.append(block_id)
                    block_splits[block_id] = "validation"
                else:
                    break

            remaining_blocks = list(set(blocks) - set(assigned_blocks))
            for block_id in remaining_blocks:
                block_splits[block_id] = "train"

        get_initial_assignments()

        test_ratio = test_samples / n_total_samples
        val_ratio = val_samples / n_total_samples

        iteration = 0
        while (
            abs(test_ratio - target_test_ratio) > ratio_tolerance
            or abs(val_ratio - target_val_ratio) > ratio_tolerance
        ) and iteration < max_iterations:
            # If test set is too large, try moving smallest test block to train
            if test_ratio > target_test_ratio + ratio_tolerance and test_blocks:
                # Find test block with smallest sample count
                test_block_counts = [
                    (block, block_counts.get(block, 0)) for block in test_blocks
                ]
                sorted_blocks = sorted(test_block_counts, key=lambda x: x[1])

                if sorted_blocks:
                    block_to_move = sorted_blocks[0][0]
                    test_samples -= block_counts.get(block_to_move, 0)
                    test_blocks.remove(block_to_move)
                    assigned_blocks.remove(block_to_move)
                    block_splits[block_to_move] = "train"

            # If test set is too small, try moving smallest train block to test
            elif test_ratio < target_test_ratio - ratio_tolerance:
                train_blocks = [b for b in blocks if block_splits.get(b) == "train"]
                if train_blocks:
                    train_block_counts = [
                        (block, block_counts.get(block, 0)) for block in train_blocks
                    ]
                    sorted_blocks = sorted(train_block_counts, key=lambda x: x[1])

                    if sorted_blocks:
                        block_to_move = sorted_blocks[0][0]
                        test_samples += block_counts.get(block_to_move, 0)
                        test_blocks.append(block_to_move)
                        assigned_blocks.append(block_to_move)
                        block_splits[block_to_move] = "test"

            # If validation set is too large, try moving smallest val block to train
            if val_ratio > target_val_ratio + ratio_tolerance and val_blocks:
                # Find validation block with smallest sample count
                val_block_counts = [
                    (block, block_counts.get(block, 0)) for block in val_blocks
                ]
                sorted_blocks = sorted(val_block_counts, key=lambda x: x[1])

                if sorted_blocks:
                    block_to_move = sorted_blocks[0][0]
                    val_samples -= block_counts.get(block_to_move, 0)
                    val_blocks.remove(block_to_move)
                    assigned_blocks.remove(block_to_move)
                    block_splits[block_to_move] = "train"

            # If validation set is too small, try moving smallest train block to validation
            elif val_ratio < target_val_ratio - ratio_tolerance:
                train_blocks = [b for b in blocks if block_splits.get(b) == "train"]
                if train_blocks:
                    train_block_counts = [
                        (block, block_counts.get(block, 0)) for block in train_blocks
                    ]
                    sorted_blocks = sorted(train_block_counts, key=lambda x: x[1])

                    if sorted_blocks:
                        block_to_move = sorted_blocks[0][0]
                        val_samples += block_counts.get(block_to_move, 0)
                        val_blocks.append(block_to_move)
                        assigned_blocks.append(block_to_move)
                        block_splits[block_to_move] = "validation"

            test_ratio = test_samples / n_total_samples
            val_ratio = val_samples / n_total_samples

            iteration += 1

            print(iteration)

    elif pattern == "random":
        blocks = np.arange(n_blocks_x * n_blocks_y)
        np.random.seed(random_state)
        np.random.shuffle(blocks)

        n_blocks = len(blocks)
        n_test_blocks = max(1, int(n_blocks * test_blocks_ratio))
        n_val_blocks = max(1, int(n_blocks * val_blocks_ratio))

        test_blocks = blocks[:n_test_blocks]
        val_blocks = blocks[n_test_blocks : n_test_blocks + n_val_blocks]

        block_splits = {}
        for block_id in range(n_blocks):
            if block_id in test_blocks:
                block_splits[block_id] = "test"
            elif block_id in val_blocks:
                block_splits[block_id] = "validation"
            else:
                block_splits[block_id] = "train"

    else:
        raise ValueError(
            f"Unknown pattern '{pattern}'. Use 'checkerboard', 'random', or 'balanced'."
        )

    gdf["split"] = gdf["block_id"].map(block_splits).fillna("train")

    result_df = df.copy()
    result_df["split"] = gdf["split"]
    result_df["block_x"] = gdf["block_x"]
    result_df["block_y"] = gdf["block_y"]
    result_df["block_id"] = gdf["block_id"]

    return result_df


def visualize_geospatial_split(
    df: pd.DataFrame,
    split_col: str = "split",
    lon_col: str = "lon",
    lat_col: str = "lat",
    cluster_col: str | None = None,
    title: str = "Geospatial Data Split",
    marker_size: int = 20,
    alpha: float = 0.7,
    figsize: tuple[int, int] = (15, 10),
    output_path: str | None = None,
    buffer_degrees: float = 1.0,
) -> None:
    """Visualize the spatial distribution of data splits using Cartopy features."""
    split_markers = {"train": "o", "validation": "^", "test": "x"}

    unfilled_markers = ["x", "+", "|", "_"]

    split_colors = {"train": "#1f77b4", "validation": "#ff7f0e", "test": "#2ca02c"}

    min_lon = df[lon_col].min() - buffer_degrees
    max_lon = df[lon_col].max() + buffer_degrees
    min_lat = df[lat_col].min() - buffer_degrees
    max_lat = df[lat_col].max() + buffer_degrees

    min_lon = max(-180, min_lon)
    max_lon = min(180, max_lon)
    min_lat = max(-90, min_lat)
    max_lat = min(90, max_lat)

    plt.figure(figsize=figsize)
    projection = ccrs.PlateCarree()

    ax = plt.axes(projection=projection)
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    # map features
    scale = "110m"
    ax.add_feature(cfeature.LAND.with_scale(scale), facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN.with_scale(scale), facecolor="lightblue")
    ax.add_feature(cfeature.COASTLINE.with_scale(scale), linewidth=0.5)
    ax.add_feature(cfeature.BORDERS.with_scale(scale), linewidth=0.3)

    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False

    splits = df[split_col].unique()
    if cluster_col and cluster_col in df.columns:
        unique_clusters = df[cluster_col].unique()

        cluster_cmap = cm.get_cmap("tab20", len(unique_clusters))
        cluster_colors = {
            cluster: cluster_cmap(i) for i, cluster in enumerate(unique_clusters)
        }

        for split in splits:
            split_data = df[df[split_col] == split]
            marker = split_markers.get(split, "o")

            marker_adj_size = marker_size * 1.5 if marker == "x" else marker_size

            for cluster in unique_clusters:
                cluster_data = split_data[split_data[cluster_col] == cluster]
                if len(cluster_data) > 0:
                    if marker in unfilled_markers:
                        ax.scatter(
                            cluster_data[lon_col],
                            cluster_data[lat_col],
                            transform=ccrs.PlateCarree(),
                            c=[split_colors.get(split, "gray")] * len(cluster_data),
                            marker=marker,
                            s=marker_adj_size,
                            alpha=alpha,
                            label=f"{split}_{cluster}"
                            if cluster == unique_clusters[0]
                            else None,
                        )
                    else:
                        ax.scatter(
                            cluster_data[lon_col],
                            cluster_data[lat_col],
                            transform=ccrs.PlateCarree(),
                            c=[cluster_colors[cluster]] * len(cluster_data),
                            marker=marker,
                            edgecolor=split_colors.get(split, "gray"),
                            s=marker_adj_size,
                            alpha=alpha,
                            label=f"{split}_{cluster}"
                            if cluster == unique_clusters[0]
                            else None,
                        )
    else:
        for split in splits:
            split_data = df[df[split_col] == split]
            marker = split_markers.get(split, "o")

            marker_adj_size = marker_size * 1.5 if marker == "x" else marker_size

            if marker in unfilled_markers:
                ax.scatter(
                    split_data[lon_col],
                    split_data[lat_col],
                    transform=ccrs.PlateCarree(),
                    c=split_colors.get(split, "gray"),
                    marker=marker,
                    s=marker_adj_size,
                    alpha=alpha,
                    label=f"{split} (n={len(split_data)})",
                )
            else:
                ax.scatter(
                    split_data[lon_col],
                    split_data[lat_col],
                    transform=ccrs.PlateCarree(),
                    c=split_colors.get(split, "gray"),
                    marker=marker,
                    edgecolor="black",
                    s=marker_adj_size,
                    alpha=alpha,
                    label=f"{split} (n={len(split_data)})",
                )

    legend_elements = []
    for split in splits:
        marker = split_markers.get(split, "o")
        split_count = len(df[df[split_col] == split])

        if marker in unfilled_markers:
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    color=split_colors.get(split, "gray"),
                    linestyle="None",
                    markersize=8,
                    label=f"{split} (n={split_count})",
                )
            )
        else:
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    color="w",
                    markerfacecolor=split_colors.get(split, "gray"),
                    markeredgecolor="black",
                    markersize=8,
                    label=f"{split} (n={split_count})",
                )
            )

    ax.legend(handles=legend_elements, loc="best", title="Data Splits")

    ax.set_title(title, fontsize=14)

    split_stats = df[split_col].value_counts()
    stats_text = "Split Distribution:\n"
    for split, count in split_stats.items():
        pct = 100 * count / len(df)
        stats_text += f"{split}: {count} ({pct:.1f}%)\n"

    plt.figtext(
        0.01,
        0.01,
        stats_text,
        bbox=dict(facecolor="white", alpha=0.8),
        verticalalignment="bottom",
    )

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()

    plt.close()


def visualize_distance_clusters(
    df: pd.DataFrame,
    lon_col: str = "lon",
    lat_col: str = "lat",
    cluster_col: str = "distance_cluster",
    split_col: str = "split",
    title: str = "Distance-Based Clustering Split",
    marker_size: int = 40,
    alpha: float = 0.8,
    figsize: tuple[int, int] = (15, 10),
    output_path: str | None = None,
    show_cluster_centers: bool = True,
    buffer_degrees: float = 1.0,
) -> None:
    """Visualize the distance-based clustering and splits using Cartopy features."""
    min_lon = df[lon_col].min() - buffer_degrees
    max_lon = df[lon_col].max() + buffer_degrees
    min_lat = df[lat_col].min() - buffer_degrees
    max_lat = df[lat_col].max() + buffer_degrees

    min_lon = max(-180, min_lon)
    max_lon = min(180, max_lon)
    min_lat = max(-90, min_lat)
    max_lat = min(90, max_lat)

    plt.figure(figsize=figsize)

    projection = ccrs.PlateCarree()

    ax = plt.axes(projection=projection)
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    scale = "110m"
    ax.add_feature(cfeature.LAND.with_scale(scale), facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN.with_scale(scale), facecolor="lightblue")
    ax.add_feature(cfeature.COASTLINE.with_scale(scale), linewidth=0.5)
    ax.add_feature(cfeature.BORDERS.with_scale(scale), linewidth=0.3)

    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False

    unique_clusters = sorted(df[cluster_col].unique())
    splits = df[split_col].unique()

    from matplotlib.colors import to_rgba

    cluster_cmap = plt.cm.get_cmap("tab20", len(unique_clusters))
    cluster_colors = {c: cluster_cmap(i % 20) for i, c in enumerate(unique_clusters)}

    split_colors = {"train": "#1f77b4", "validation": "#ff7f0e", "test": "#2ca02c"}

    split_markers = {"train": "o", "validation": "^", "test": "x"}

    unfilled_markers = ["x", "+", "|", "_"]

    for cluster in unique_clusters:
        cluster_data = df[df[cluster_col] == cluster]

        for split in splits:
            split_cluster_data = cluster_data[cluster_data[split_col] == split]
            if len(split_cluster_data) > 0:
                base_color = cluster_colors[cluster]
                marker = split_markers[split]

                if split == "train":
                    color = split_colors[split]
                    face_color = to_rgba(base_color, 0.7)
                    zorder = 1
                elif split == "validation":
                    color = split_colors[split]
                    face_color = to_rgba(base_color, 0.8)
                    zorder = 2
                else:
                    color = split_colors[split]
                    face_color = to_rgba(base_color, 0.9)
                    zorder = 3

                marker_adj_size = marker_size * 1.5 if marker == "x" else marker_size

                if marker in unfilled_markers:
                    ax.scatter(
                        split_cluster_data[lon_col],
                        split_cluster_data[lat_col],
                        transform=ccrs.PlateCarree(),
                        c=[color] * len(split_cluster_data),
                        marker=marker,
                        linewidth=1.5,
                        s=marker_adj_size,
                        alpha=alpha,
                        label=f"Cluster {cluster} ({split})"
                        if split == splits[0]
                        else None,
                        zorder=zorder,
                    )
                else:
                    ax.scatter(
                        split_cluster_data[lon_col],
                        split_cluster_data[lat_col],
                        transform=ccrs.PlateCarree(),
                        c=[face_color] * len(split_cluster_data),
                        marker=marker,
                        edgecolor=color,
                        linewidth=1.5,
                        s=marker_adj_size,
                        alpha=alpha,
                        label=f"Cluster {cluster} ({split})"
                        if split == splits[0]
                        else None,
                        zorder=zorder,
                    )

    if show_cluster_centers:
        for cluster in unique_clusters:
            cluster_data = df[df[cluster_col] == cluster]
            if len(cluster_data) > 0:
                center_x = cluster_data[lon_col].mean()
                center_y = cluster_data[lat_col].mean()

                ax.scatter(
                    center_x,
                    center_y,
                    transform=ccrs.PlateCarree(),
                    c="black",
                    marker="X",
                    s=marker_size * 2,
                    alpha=1.0,
                    edgecolor="white",
                    linewidth=1.5,
                    zorder=100,
                )

                ax.text(
                    center_x,
                    center_y,
                    f"{cluster}",
                    transform=ccrs.PlateCarree(),
                    ha="center",
                    va="center",
                    fontweight="bold",
                    fontsize=10,
                    color="white",
                    bbox=dict(facecolor="black", alpha=0.7, boxstyle="round,pad=0.3"),
                    zorder=101,
                )

    ax.set_title(title, fontsize=14)

    legend_elements = []
    for split, color in split_colors.items():
        marker = split_markers[split]
        if marker in unfilled_markers:
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    color=color,
                    linestyle="None",
                    markersize=8,
                    label=split,
                )
            )
        else:
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    color="w",
                    markerfacecolor=color,
                    markeredgecolor="black",
                    markersize=8,
                    label=split,
                )
            )

    split_legend = ax.legend(
        handles=legend_elements, loc="upper right", title="Split", framealpha=0.9
    )
    ax.add_artist(split_legend)

    if len(unique_clusters) <= 10:
        cluster_legend_elements = []
        for i, cluster in enumerate(unique_clusters[:10]):
            cluster_legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=cluster_colors[cluster],
                    markersize=8,
                    label=f"Cluster {cluster}",
                )
            )

        ax.legend(
            handles=cluster_legend_elements,
            loc="lower right",
            title="Clusters",
            framealpha=0.9,
        )

    split_stats = df[split_col].value_counts()
    stats_text = "Split Distribution:\n"
    for split, count in split_stats.items():
        pct = 100 * count / len(df)
        stats_text += f"{split}: {count} ({pct:.1f}%)\n"

    plt.figtext(
        0.01,
        0.01,
        stats_text,
        bbox=dict(facecolor="white", alpha=0.8),
        verticalalignment="bottom",
    )

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()

    plt.close()


def visualize_checkerboard_pattern(
    df: pd.DataFrame,
    n_blocks_x: int = 5,
    n_blocks_y: int = 5,
    split_col: str = "split",
    block_x_col: str = "block_x",
    block_y_col: str = "block_y",
    title: str = "Checkerboard Split Pattern",
    figsize: tuple[int, int] = (10, 8),
    output_path: str | None = None,
) -> None:
    """Visualize the checkerboard pattern used for splitting."""
    grid = np.zeros((n_blocks_y, n_blocks_x), dtype=object)

    split_map = {"train": 0, "validation": 1, "test": 2}

    split_markers = {"train": "o", "validation": "^", "test": "x"}

    for _, row in df.iterrows():
        x = int(row[block_x_col])
        y = int(row[block_y_col])
        if 0 <= x < n_blocks_x and 0 <= y < n_blocks_y:
            grid[y, x] = row[split_col]

    block_counts = {}
    for _, row in df.iterrows():
        x = int(row[block_x_col])
        y = int(row[block_y_col])
        block_id = (x, y)
        if block_id not in block_counts:
            block_counts[block_id] = 0
        block_counts[block_id] += 1

    grid_numerical = np.zeros((n_blocks_y, n_blocks_x))
    for i in range(n_blocks_y):
        for j in range(n_blocks_x):
            if grid[i, j] in split_map:
                grid_numerical[i, j] = split_map[grid[i, j]]
            else:
                grid_numerical[i, j] = -1

    fig, ax = plt.subplots(figsize=figsize)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    cmap = ListedColormap(colors)

    ax.imshow(grid_numerical, cmap=cmap, interpolation="nearest")

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="#d62728",
            markersize=10,
            label="No data",
        ),
        Line2D(
            [0],
            [0],
            marker=split_markers["train"],
            color="w",
            markerfacecolor="#1f77b4",
            markersize=10,
            label="Train",
        ),
        Line2D(
            [0],
            [0],
            marker=split_markers["validation"],
            color="w",
            markerfacecolor="#ff7f0e",
            markersize=10,
            label="Validation",
        ),
        Line2D(
            [0],
            [0],
            marker=split_markers["test"],
            color="w",
            markerfacecolor="#2ca02c",
            markersize=10,
            label="Test",
        ),
    ]

    ax.legend(handles=legend_elements, loc="upper right", title="Splits")

    for i in range(n_blocks_y):
        for j in range(n_blocks_x):
            count = block_counts.get((j, i), 0)
            split = grid[i, j]
            if split is not None:
                text_color = "black" if split == "train" else "white"
                ax.text(
                    j,
                    i,
                    f"{count}\n({split})",
                    ha="center",
                    va="center",
                    color=text_color,
                )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Block X")
    ax.set_ylabel("Block Y")

    ax.set_xticks(np.arange(-0.5, n_blocks_x, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_blocks_y, 1), minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    split_stats = df[split_col].value_counts()
    stats_text = "Split Distribution:\n"
    for split, count in split_stats.items():
        pct = 100 * count / len(df)
        marker = split_markers.get(split, "o")
        stats_text += f"{split} ({marker}): {count} ({pct:.1f}%)\n"

    plt.figtext(
        0.01,
        0.01,
        stats_text,
        bbox=dict(facecolor="white", alpha=0.8),
        verticalalignment="bottom",
    )

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()


def analyze_split_results(
    df: pd.DataFrame,
    split_col: str = "split",
    additional_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Analyze the distribution of data across splits."""
    split_counts = df[split_col].value_counts().reset_index()
    split_counts.columns = ["Split", "Count"]
    split_counts["Percentage"] = 100 * split_counts["Count"] / len(df)

    print("=== Split Distribution ===")
    for _, row in split_counts.iterrows():
        print(f"{row['Split']}: {row['Count']} samples ({row['Percentage']:.1f}%)")

    if additional_columns:
        print("\n=== Distribution of Key Variables ===")
        for col in additional_columns:
            if col in df.columns:
                print(f"\n{col} statistics:")
                try:
                    if np.issubdtype(df[col].dtype, np.number):
                        stats = df.groupby(split_col)[col].agg(
                            ["mean", "std", "min", "max"]
                        )
                        print(stats)
                    else:
                        for split in df[split_col].unique():
                            split_data = df[df[split_col] == split]
                            top_values = split_data[col].value_counts().head(3)
                            print(f"\n{split} top {col} values:")
                            for val, count in top_values.items():
                                print(
                                    f"  {val}: {count} ({100 * count / len(split_data):.1f}%)"
                                )
                except Exception as e:
                    print(f"Could not analyze column {col}: {e}")

    return split_counts


def split_geospatial_tiles_into_patches(
    modal_path_dict: dict[str, list[str]],
    output_dir: str,
    patch_size: tuple[int, int] = (512, 512),
    block_size: tuple[int, int] = (512, 512),
    stride: tuple[int, int] | None = None,
    output_format: str = "tif",
    patch_id_prefix: str = "p",
    buffer_top: int = 0,
    buffer_left: int = 0,
    buffer_bottom: int = 0,
    buffer_right: int = 0,
) -> pd.DataFrame:
    """Split large geospatial image and mask pairs into smaller patches across multiple modalities.

    Args:
        modal_path_dict: Dictionary mapping modality names to lists of image paths
        output_dir: Directory to save patches and metadata
        patch_size: Size of the patches (height, width)
        block_size: Size of the blocks (height, width) to write tiff profile with
        stride: Step size between patches (height, width)
        output_format: Output file format (e.g., 'tif')
        patch_id_prefix: Prefix for patch IDs
        buffer_top: Number of pixels to skip from the top of the image
        buffer_left: Number of pixels to skip from the left of the image
        buffer_bottom: Number of pixels to skip from the bottom of the image
        buffer_right: Number of pixels to skip from the right of the image

    Returns:
        DataFrame containing metadata for all created patches
    """
    blockxsize, blockysize = block_size

    modalities = list(modal_path_dict.keys())
    mask_modality = "mask"

    first_modality = [m for m in modalities if m != mask_modality][0]
    image_paths = modal_path_dict[first_modality]
    mask_paths = modal_path_dict[mask_modality]

    assert len(image_paths) == len(mask_paths), (
        f"Number of images ({len(image_paths)}) does not match number of masks ({len(mask_paths)})"
    )

    for modality in modalities:
        if modality != mask_modality:
            assert len(modal_path_dict[modality]) == len(image_paths), (
                f"Modality {modality} has {len(modal_path_dict[modality])} images, expected {len(image_paths)}"
            )

    if stride is None:
        stride = patch_size

    os.makedirs(output_dir, exist_ok=True)

    for modality in modalities:
        modality_dir = os.path.join(output_dir, modality)
        os.makedirs(modality_dir, exist_ok=True)

    all_patch_metadata = []

    for idx in tqdm(
        range(len(image_paths)), desc="Splitting tiles", total=len(image_paths)
    ):
        modal_img_paths = {
            modality: modal_path_dict[modality][idx] for modality in modalities
        }

        img_path = modal_img_paths[first_modality]
        mask_path = modal_img_paths[mask_modality]

        img_filename = os.path.basename(img_path)
        img_basename = os.path.splitext(img_filename)[0]
        mask_filename = os.path.basename(mask_path)

        with rasterio.open(img_path) as img_src:
            src_height = img_src.height
            src_width = img_src.width
            src_crs = img_src.crs
            src_transform = img_src.transform
            src_nodata = img_src.nodata

            effective_height = src_height - buffer_top - buffer_bottom
            effective_width = src_width - buffer_left - buffer_right

            if effective_height <= 0 or effective_width <= 0:
                print(
                    f"Error: Image dimensions ({src_height}x{src_width}) are smaller than the combined buffers. Skipping."
                )
                continue

            try:
                mask_full = np.zeros((1, src_height, src_width), dtype=np.uint8)

                gdf = gpd.read_file(mask_path)

                if len(gdf) > 0 and not gdf.empty:
                    if gdf.crs is None:
                        gdf.set_crs(src_crs, inplace=True)
                    elif gdf.crs != src_crs:
                        gdf = gdf.to_crs(src_crs)

                    # Extract building and street features with flood status
                    street_not_flooded = gdf[
                        (gdf["flooded"] != "yes")
                        & (gdf["highway"].notna())
                        & (gdf["building"].isna())
                    ]

                    street_flooded = gdf[
                        (gdf["flooded"] == "yes")
                        & (gdf["highway"].notna())
                        & (gdf["building"].isna())
                    ]

                    building_not_flooded = gdf[
                        (gdf["flooded"] != "yes") & (gdf["building"].notna())
                    ]

                    building_flooded = gdf[
                        (gdf["flooded"] == "yes") & (gdf["building"].notna())
                    ]

                    # Create shapes with appropriate class values
                    building_not_flooded_shapes = [
                        (geom, 1)
                        for geom in building_not_flooded.geometry
                        if geom is not None and not geom.is_empty
                    ]

                    building_flooded_shapes = [
                        (geom, 2)
                        for geom in building_flooded.geometry
                        if geom is not None and not geom.is_empty
                    ]

                    street_not_flooded_shapes = [
                        (geom, 3)
                        for geom in street_not_flooded.geometry
                        if geom is not None and not geom.is_empty
                    ]

                    street_flooded_shapes = [
                        (geom, 4)
                        for geom in street_flooded.geometry
                        if geom is not None and not geom.is_empty
                    ]

                    all_shapes = (
                        street_not_flooded_shapes
                        + building_not_flooded_shapes
                        + street_flooded_shapes
                        + building_flooded_shapes
                    )

                    if all_shapes:
                        mask_full = rasterize(
                            all_shapes,
                            out_shape=(src_height, src_width),
                            transform=src_transform,
                            fill=0,
                            dtype=np.uint8,
                            all_touched=True,
                            merge_alg=rasterio.enums.MergeAlg.replace,
                        )
                        mask_full = mask_full[np.newaxis, :, :]

            except Exception as e:
                print(f"Warning: Error processing mask {mask_path}: {e}")
                mask_full = np.zeros((1, src_height, src_width), dtype=np.uint8)

            patches_per_dim_h = max(
                1, (effective_height - patch_size[0] + stride[0]) // stride[0]
            )
            patches_per_dim_w = max(
                1, (effective_width - patch_size[1] + stride[1]) // stride[1]
            )

            patches_created = 0

            modality_patches = {modality: [] for modality in modalities}
            modality_tiles = {}

            for modality in modalities:
                if modality != "mask":
                    try:
                        with rasterio.open(modal_img_paths[modality]) as modal_src:
                            modal_data = modal_src.read()
                            modal_height, modal_width = (
                                modal_data.shape[1],
                                modal_data.shape[2],
                            )

                            if modal_height != src_height or modal_width != src_width:
                                resized_bands = []
                                for band_idx in range(modal_data.shape[0]):
                                    resized_band = resize(
                                        modal_data[band_idx],
                                        (src_height, src_width),
                                        order=1,
                                        preserve_range=True,
                                    ).astype(modal_data.dtype)
                                    resized_bands.append(resized_band)

                                modal_data = np.stack(resized_bands, axis=0)

                            modality_tiles[modality] = modal_data

                    except Exception as e:
                        print(f"Error opening or resizing {modality} source: {e}")
                        modality_tiles[modality] = None

            modality_tiles["mask"] = mask_full

            for i in range(patches_per_dim_h):
                for j in range(patches_per_dim_w):
                    row_start = buffer_top + i * stride[0]
                    col_start = buffer_left + j * stride[1]
                    max_row = src_height - buffer_bottom - patch_size[0]
                    max_col = src_width - buffer_right - patch_size[1]

                    if row_start > max_row or col_start > max_col:
                        continue

                    if row_start + patch_size[0] > src_height - buffer_bottom:
                        row_start = max(
                            buffer_top, src_height - buffer_bottom - patch_size[0]
                        )
                    if col_start + patch_size[1] > src_width - buffer_right:
                        col_start = max(
                            buffer_left, src_width - buffer_right - patch_size[1]
                        )

                    window = Window(col_start, row_start, patch_size[1], patch_size[0])

                    try:
                        img_data = img_src.read(window=window)

                        if src_nodata is not None:
                            valid_ratio = np.sum(img_data != src_nodata) / img_data.size
                        else:
                            valid_ratio = 1.0

                        mask_data = mask_full[
                            :,
                            row_start : row_start + patch_size[0],
                            col_start : col_start + patch_size[1],
                        ]

                        positive_ratio = np.sum(mask_data > 0) / mask_data.size

                    except Exception as e:
                        print(f"Error reading patch at ({row_start}, {col_start}): {e}")
                        continue

                    patch_id = f"{patch_id_prefix}{i:03d}_{j:03d}"

                    patch_transform = rasterio.windows.transform(window, src_transform)

                    modal_patch_paths = {}

                    for modality in modalities:
                        modality_path = modal_img_paths[modality]
                        modality_dir = os.path.join(output_dir, modality)

                        if modality == mask_modality:
                            patch_filename = (
                                f"{img_basename}_mask_{patch_id}.{output_format}"
                            )
                        else:
                            patch_filename = (
                                f"{img_basename}_{modality}_{patch_id}.{output_format}"
                            )

                        patch_path = os.path.join(modality_dir, patch_filename)
                        modal_patch_paths[modality] = patch_path

                        try:
                            if modality == mask_modality:
                                patch_data = mask_data

                                patch_meta = {
                                    "driver": "GTiff",
                                    "compress": Compression.lzw,
                                    "interleave": "pixel",
                                    "tiled": True,
                                    "blockxsize": blockxsize,
                                    "blockysize": blockysize,
                                    "predictor": 2,
                                    "zlevel": 9,
                                    "count": patch_data.shape[0],
                                    "dtype": patch_data.dtype,
                                    "crs": src_crs,
                                    "transform": patch_transform,
                                    "width": patch_size[1],
                                    "height": patch_size[0],
                                }
                            else:
                                if modality_tiles[modality] is not None:
                                    patch_data = modality_tiles[modality][
                                        :,
                                        row_start : row_start + patch_size[0],
                                        col_start : col_start + patch_size[1],
                                    ]

                                    patch_meta = {
                                        "driver": "GTiff",
                                        "compress": Compression.lzw,
                                        "interleave": "pixel",
                                        "tiled": True,
                                        "blockxsize": blockxsize,
                                        "blockysize": blockysize,
                                        "predictor": 2,
                                        "zlevel": 9,
                                        "count": patch_data.shape[0],
                                        "dtype": patch_data.dtype,
                                        "crs": src_crs,
                                        "transform": patch_transform,
                                        "width": patch_size[1],
                                        "height": patch_size[0],
                                    }
                                else:
                                    with rasterio.open(modality_path) as src:
                                        patch_data = src.read(window=window)

                                        patch_meta = {
                                            "driver": "GTiff",
                                            "compress": Compression.lzw,
                                            "interleave": "pixel",
                                            "tiled": True,
                                            "blockxsize": blockxsize,
                                            "blockysize": blockysize,
                                            "predictor": 2,
                                            "zlevel": 9,
                                            "count": patch_data.shape[0],
                                            "dtype": patch_data.dtype,
                                            "crs": src_crs,
                                            "transform": patch_transform,
                                            "width": patch_size[1],
                                            "height": patch_size[0],
                                        }

                            with rasterio.open(patch_path, "w", **patch_meta) as dst:
                                dst.write(patch_data)

                            modality_patches[modality].append(
                                (patch_path, patch_id, i, j)
                            )

                        except Exception as e:
                            print(
                                f"Error processing {modality} patch at ({i}, {j}): {e}"
                            )
                            continue

                    patch_bounds = rasterio.windows.bounds(window, src_transform)
                    west, south, east, north = patch_bounds

                    center_x = (west + east) / 2
                    center_y = (south + north) / 2

                    lon, lat = None, None
                    if src_crs and not src_crs.is_geographic:
                        try:
                            from pyproj import Transformer

                            transformer = Transformer.from_crs(
                                src_crs, "EPSG:4326", always_xy=True
                            )
                            lon, lat = transformer.transform(center_x, center_y)
                        except Exception:
                            pass
                    else:
                        lon, lat = center_x, center_y

                    patch_metadata = {
                        "source_img_file": img_filename,
                        "source_mask_file": mask_filename,
                        "patch_id": patch_id,
                        "lon": lon,
                        "lat": lat,
                        "west": west,
                        "south": south,
                        "east": east,
                        "north": north,
                        "height_px": patch_size[0],
                        "width_px": patch_size[1],
                        "crs": str(src_crs),
                        "row": i,
                        "col": j,
                        "row_px": row_start,
                        "col_px": col_start,
                        "valid_ratio": float(valid_ratio),
                        "positive_ratio": float(positive_ratio),
                        "is_positive": positive_ratio > 0,
                    }

                    for modality, path in modal_patch_paths.items():
                        patch_metadata[f"{modality}_path"] = os.path.relpath(
                            path, start=output_dir
                        )

                    if hasattr(img_src, "tags") and img_src.tags().get(
                        "TIFFTAG_DATETIME"
                    ):
                        patch_metadata["date"] = img_src.tags().get("TIFFTAG_DATETIME")
                    else:
                        date_match = re.search(
                            r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", img_filename
                        )
                        if date_match:
                            year, month, day = date_match.groups()
                            patch_metadata["date"] = f"{year}-{month}-{day}"

                    all_patch_metadata.append(patch_metadata)
                    patches_created += 1

    patches_df = pd.DataFrame(all_patch_metadata)

    if len(patches_df) > 0:
        metadata_path = os.path.join(output_dir, "patch_metadata.parquet")
        patches_df.to_parquet(metadata_path, index=False)

        print(
            f"Created {len(patches_df)} patches from {len(image_paths)} source images"
        )
        print(f"Patch metadata saved to {metadata_path}")

        if "positive_ratio" in patches_df.columns:
            pos_patches = patches_df[patches_df["is_positive"]]
            neg_patches = patches_df[~patches_df["is_positive"]]
            pos_pct = (
                len(pos_patches) / len(patches_df) * 100 if len(patches_df) > 0 else 0
            )
            neg_pct = (
                len(neg_patches) / len(patches_df) * 100 if len(patches_df) > 0 else 0
            )
            print(f"Positive patches: {len(pos_patches)} ({pos_pct:.1f}%)")
            print(f"Negative patches: {len(neg_patches)} ({neg_pct:.1f}%)")
    else:
        print("No patches were created. Check filtering criteria and input data.")

    return patches_df


def show_samples_per_valid_ratio(
    df: pd.DataFrame, output_path: str = None, dataset_name: str = "Dataset"
):
    """Show the number of samples (rows) that would remain in dataframe after filtering by valid_ratio."""
    valid_ratios = np.arange(0, 1.0, 0.05)

    samples_per_valid_ratio = []

    for valid_ratio in valid_ratios:
        samples_per_valid_ratio.append(len(df[df["valid_ratio"] >= valid_ratio]))

    fig, ax = plt.subplots()
    ax.plot(valid_ratios, samples_per_valid_ratio, marker="o")
    ax.set_xlabel("Minimum Valid Data Ratio")
    ax.set_ylabel("Samples Remaining")
    ax.set_title(
        f"Samples in {dataset_name} Remaining After Filtering by Valid Data Ratio"
    )

    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

    plt.close()


def create_geospatial_temporal_split(
    metadata_df: pd.DataFrame,
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2,
    geo_exclusive_ratio=0.3,
    temporal_ratio=0.7,
    random_seed=42,
):
    """Create a train/validation/test split with both geospatial and temporal constraints.

    This function creates a split that combines:
    1. Geographic exclusivity - some areas appear only in one split
    2. Temporal progression - other areas have their time series split chronologically

    Args:
        metadata_df: DataFrame containing DynamicEarthNet metadata
        train_ratio: Proportion of data for training set (default: 0.7)
        val_ratio: Proportion of data for validation set (default: 0.1)
        test_ratio: Proportion of data for test set (default: 0.2)
        geo_exclusive_ratio: Ratio of areas to assign exclusively to one split (default: 0.3)
        temporal_ratio: Ratio of areas to split temporally (default: 0.7)
        random_seed: Random seed for reproducibility (default: 42)

    Returns:
        DataFrame with an additional 'split' column indicating the assigned split
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, (
        "Split ratios must sum to 1.0"
    )
    assert abs(geo_exclusive_ratio + temporal_ratio - 1.0) < 1e-10, (
        "Geographic and temporal ratios must sum to 1.0"
    )

    df = metadata_df.copy()
    np.random.seed(random_seed)

    print(
        f"Creating enhanced geospatial-temporal split with {train_ratio:.1%}/{val_ratio:.1%}/{test_ratio:.1%} ratios"
    )
    print(
        f"Using {geo_exclusive_ratio:.1%} areas for geographic exclusivity and {temporal_ratio:.1%} for temporal splits"
    )

    unique_areas = df["area_id"].unique()
    np.random.shuffle(unique_areas)

    print(f"Found {len(unique_areas)} unique geographic areas")

    n_geo_exclusive_areas = int(len(unique_areas) * geo_exclusive_ratio)

    geo_train_areas = int(
        n_geo_exclusive_areas * (train_ratio / (train_ratio + val_ratio + test_ratio))
    )
    geo_val_areas = int(
        n_geo_exclusive_areas * (val_ratio / (train_ratio + val_ratio + test_ratio))
    )

    geo_exclusive_areas = unique_areas[:n_geo_exclusive_areas]
    temporal_areas = unique_areas[n_geo_exclusive_areas:]

    train_geo_areas = geo_exclusive_areas[:geo_train_areas]
    val_geo_areas = geo_exclusive_areas[
        geo_train_areas : geo_train_areas + geo_val_areas
    ]
    test_geo_areas = geo_exclusive_areas[geo_train_areas + geo_val_areas :]

    print(
        f"Geographic exclusivity: {len(train_geo_areas)} train, {len(val_geo_areas)} val, {len(test_geo_areas)} test areas"
    )
    print(f"Temporal split: {len(temporal_areas)} areas")

    train_indices = []
    val_indices = []
    test_indices = []

    area_stats = []

    for area_id in tqdm(
        geo_exclusive_areas, desc="Processing geographically exclusive areas"
    ):
        area_df = df[df["area_id"] == area_id].copy()

        if area_id in train_geo_areas:
            split = "train"
            train_indices.extend(area_df.index.tolist())
        elif area_id in val_geo_areas:
            split = "validation"
            val_indices.extend(area_df.index.tolist())
        else:
            split = "test"
            test_indices.extend(area_df.index.tolist())

        area_stats.append(
            {
                "area_id": area_id,
                "split_type": "geographic",
                "split": split,
                "total_periods": len(area_df["year_month"].unique()),
                "train_periods": len(area_df["year_month"].unique())
                if split == "train"
                else 0,
                "val_periods": len(area_df["year_month"].unique())
                if split == "validation"
                else 0,
                "test_periods": len(area_df["year_month"].unique())
                if split == "test"
                else 0,
                "train_samples": len(area_df) if split == "train" else 0,
                "val_samples": len(area_df) if split == "validation" else 0,
                "test_samples": len(area_df) if split == "test" else 0,
                "total_samples": len(area_df),
            }
        )

    for area_id in tqdm(temporal_areas, desc="Processing temporal split areas"):
        area_df = df[df["area_id"] == area_id].copy()

        time_periods = sorted(area_df["year_month"].unique())
        total_periods = len(time_periods)

        train_end = int(np.floor(total_periods * train_ratio))
        val_end = train_end + int(np.floor(total_periods * val_ratio))
        train_periods = time_periods[:train_end]
        val_periods = time_periods[train_end:val_end]
        test_periods = time_periods[val_end:]

        train_mask = area_df["year_month"].isin(train_periods)
        val_mask = area_df["year_month"].isin(val_periods)
        test_mask = area_df["year_month"].isin(test_periods)

        train_indices.extend(area_df.index[train_mask].tolist())
        val_indices.extend(area_df.index[val_mask].tolist())
        test_indices.extend(area_df.index[test_mask].tolist())

        area_stats.append(
            {
                "area_id": area_id,
                "split_type": "temporal",
                "split": "mixed",
                "total_periods": total_periods,
                "train_periods": len(train_periods),
                "val_periods": len(val_periods),
                "test_periods": len(test_periods),
                "train_samples": train_mask.sum(),
                "val_samples": val_mask.sum(),
                "test_samples": test_mask.sum(),
                "total_samples": len(area_df),
            }
        )

    df["split"] = "unknown"
    df.loc[train_indices, "split"] = "train"
    df.loc[val_indices, "split"] = "validation"
    df.loc[test_indices, "split"] = "test"

    total_samples = len(df)
    train_count = len(train_indices)
    val_count = len(val_indices)
    test_count = len(test_indices)

    print("\nOverall Split Statistics:")
    print(f"  Train: {train_count} samples ({train_count / total_samples:.1%})")
    print(f"  Validation: {val_count} samples ({val_count / total_samples:.1%})")
    print(f"  Test: {test_count} samples ({test_count / total_samples:.1%})")

    stats_df = pd.DataFrame(area_stats)
    geo_stats = stats_df[stats_df["split_type"] == "geographic"]
    print("\nGeographically Exclusive Areas:")
    print(f"  Train: {len(geo_stats[geo_stats['split'] == 'train'])} areas")
    print(f"  Validation: {len(geo_stats[geo_stats['split'] == 'validation'])} areas")
    print(f"  Test: {len(geo_stats[geo_stats['split'] == 'test'])} areas")

    temp_stats = stats_df[stats_df["split_type"] == "temporal"]
    train_periods_total = temp_stats["train_periods"].sum()
    val_periods_total = temp_stats["val_periods"].sum()
    test_periods_total = temp_stats["test_periods"].sum()
    total_periods_all = temp_stats["total_periods"].sum()

    print("\nTemporal Distribution (for temporally split areas):")
    print(
        f"  Train: {train_periods_total} periods ({train_periods_total / total_periods_all:.1%})"
    )
    print(
        f"  Validation: {val_periods_total} periods ({val_periods_total / total_periods_all:.1%})"
    )
    print(
        f"  Test: {test_periods_total} periods ({test_periods_total / total_periods_all:.1%})"
    )

    train_areas = df[df["split"] == "train"]["area_id"].nunique()
    val_areas = df[df["split"] == "validation"]["area_id"].nunique()
    test_areas = df[df["split"] == "test"]["area_id"].nunique()

    print("\nGeographic Coverage:")
    print(
        f"  Train: {train_areas}/{len(unique_areas)} areas ({train_areas / len(unique_areas):.1%})"
    )
    print(
        f"  Validation: {val_areas}/{len(unique_areas)} areas ({val_areas / len(unique_areas):.1%})"
    )
    print(
        f"  Test: {test_areas}/{len(unique_areas)} areas ({test_areas / len(unique_areas):.1%})"
    )

    unknown_count = (df["split"] == "unknown").sum()
    if unknown_count > 0:
        print(
            f"\nWARNING: {unknown_count} samples ({unknown_count / total_samples:.1%}) were not assigned to any split!"
        )

    print("\nSplit Type by Area:")
    for split_type in ["geographic", "temporal"]:
        if split_type == "geographic":
            print("\nGeographically Exclusive Areas:")
            for split in ["train", "validation", "test"]:
                areas = stats_df[
                    (stats_df["split_type"] == "geographic")
                    & (stats_df["split"] == split)
                ]["area_id"].tolist()
                print(f"  {split}: {areas}")
        else:
            print("\nTemporally Split Areas:")
            areas = stats_df[stats_df["split_type"] == "temporal"]["area_id"].tolist()
            print(f"  {areas}")

    return df
