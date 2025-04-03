# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Utility functions for geospatial dataset splitting and visualization."""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from matplotlib.colors import ListedColormap
from typing import Optional, Tuple, Union, List

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from pathlib import Path
from tqdm import tqdm

import os
from tqdm import tqdm
import rasterio
import re
from rasterio.windows import Window

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec


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
    n_clusters: Optional[int] = None,
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
    target_test_ratio: float = 0.2,  # Target sample ratio for test set
    target_val_ratio: float = 0.1,  # Target sample ratio for validation set
    max_iterations: int = 50,  # Maximum iterations for ratio optimization
    ratio_tolerance: float = 0.02,  # Acceptable deviation from target ratios
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

    # Count points per block for balancing
    block_counts = gdf["block_id"].value_counts().to_dict()
    n_blocks = n_blocks_x * n_blocks_y
    blocks = np.arange(n_blocks)

    # Set random seed for reproducibility
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
        train_value = 2

        block_splits = {}
        for block_id in range(len(block_assignments)):
            if block_assignments[block_id] == test_value:
                block_splits[block_id] = "test"
            elif block_assignments[block_id] == val_value:
                block_splits[block_id] = "validation"
            else:
                block_splits[block_id] = "train"

    elif pattern == "balanced":
        # Initialize with random assignment
        np.random.shuffle(blocks)

        # Start with initial ratio-based assignment
        n_total_samples = len(gdf)
        n_test_samples_target = int(n_total_samples * target_test_ratio)
        n_val_samples_target = int(n_total_samples * target_val_ratio)

        # Initialize block splits
        block_splits = {}
        for block_id in range(n_blocks):
            block_splits[block_id] = "train"  # Default all to train

        # Calculate initial block assignments based on ratios
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
            # First assign test blocks
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

            # Then assign validation blocks
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

            # Rest are train
            remaining_blocks = list(set(blocks) - set(assigned_blocks))
            for block_id in remaining_blocks:
                block_splits[block_id] = "train"

        get_initial_assignments()

        # Calculate actual ratios achieved
        test_ratio = test_samples / n_total_samples
        val_ratio = val_samples / n_total_samples

        # Iteratively improve to get closer to target ratios
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

            # Recalculate ratios
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
    cluster_col: Optional[str] = None,
    title: str = "Geospatial Data Split",
    marker_size: int = 20,
    alpha: float = 0.7,
    figsize: Tuple[int, int] = (15, 10),
    output_path: Optional[str] = None,
    buffer_degrees: float = 1.0,
) -> None:
    """Visualize the spatial distribution of data splits using Cartopy features."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.lines import Line2D

    split_markers = {"train": "o", "validation": "^", "test": "x"}

    unfilled_markers = ["x", "+", "|", "_"]

    split_colors = {
        "train": "#1f77b4",  # Blue
        "validation": "#ff7f0e",  # Orange
        "test": "#2ca02c",  # Green
    }

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

        import matplotlib.cm as cm

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
    figsize: Tuple[int, int] = (15, 10),
    output_path: Optional[str] = None,
    show_cluster_centers: bool = True,
    buffer_degrees: float = 1.0,
) -> None:
    """Visualize the distance-based clustering and splits using Cartopy features."""
    if cluster_col not in df.columns or split_col not in df.columns:
        raise ValueError(
            f"DataFrame must contain {cluster_col} and {split_col} columns"
        )

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.lines import Line2D

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

    import matplotlib.cm as cm
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
    figsize: Tuple[int, int] = (10, 8),
    output_path: Optional[str] = None,
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

    colors = [
        "#d62728",
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
    ]  # No data (red), Train (blue), Val (orange), Test (green)
    cmap = ListedColormap(colors)

    im = ax.imshow(grid_numerical, cmap=cmap, interpolation="nearest")

    from matplotlib.lines import Line2D

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
    additional_columns: Optional[List[str]] = None,
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
    import numpy as np
    from skimage.transform import resize

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
            img_meta = img_src.meta.copy()
            src_height = img_src.height
            src_width = img_src.width
            src_crs = img_src.crs
            src_transform = img_src.transform
            src_nodata = img_src.nodata
            primary_shape = (src_height, src_width)

            effective_height = src_height - buffer_top - buffer_bottom
            effective_width = src_width - buffer_left - buffer_right

            if effective_height <= 0 or effective_width <= 0:
                print(
                    f"Error: Image dimensions ({src_height}x{src_width}) are smaller than the combined buffers. Skipping."
                )
                continue

            try:
                mask_full = np.zeros((1, src_height, src_width), dtype=np.uint8)

                import geopandas as gpd
                from rasterio.features import rasterize

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

            total_patches = patches_per_dim_h * patches_per_dim_w
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
                                    "height": patch_size[0],
                                    "width": patch_size[1],
                                    "count": 1,
                                    "dtype": np.uint8,
                                    "crs": src_crs,
                                    "transform": patch_transform,
                                }
                            else:
                                if modality_tiles[modality] is not None:
                                    patch_data = modality_tiles[modality][
                                        :,
                                        row_start : row_start + patch_size[0],
                                        col_start : col_start + patch_size[1],
                                    ]

                                    with rasterio.open(
                                        modal_img_paths[modality]
                                    ) as src:
                                        patch_meta = src.meta.copy()
                                        patch_meta.update(
                                            {
                                                "height": patch_size[0],
                                                "width": patch_size[1],
                                                "transform": patch_transform,
                                                "count": patch_data.shape[0],
                                            }
                                        )
                                else:
                                    with rasterio.open(modality_path) as src:
                                        patch_data = src.read(window=window)
                                        patch_meta = src.meta.copy()
                                        patch_meta.update(
                                            {
                                                "height": patch_size[0],
                                                "width": patch_size[1],
                                                "transform": patch_transform,
                                            }
                                        )

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

            # if patches_created > 0:
            #     visualize_dir = os.path.join(output_dir, "visualizations")
            #     os.makedirs(visualize_dir, exist_ok=True)
            #     vis_output_path = os.path.join(
            #         visualize_dir, f"{img_basename}_patches.png"
            #     )

            #     visualize_current_patches(
            #         modality_tiles=modality_tiles,
            #         modality_patches=modality_patches,
            #         output_path=vis_output_path,
            #         buffer_top=buffer_top,
            #         buffer_left=buffer_left,
            #         buffer_bottom=buffer_bottom,
            #         buffer_right=buffer_right
            #     )

            #     import pdb; pdb.set_trace()
            # print(
            #     f"Created {patches_created}/{total_patches} patches for {img_filename}"
            # )

    patches_df = pd.DataFrame(all_patch_metadata)

    if len(patches_df) > 0:
        metadata_path = os.path.join(output_dir, "patch_metadata.parquet")
        patches_df.to_parquet(metadata_path, index=False)

        print(
            f"Created {len(patches_df)} patches from {len(image_paths)} source images"
        )
        print(f"Patch metadata saved to {metadata_path}")

        if "positive_ratio" in patches_df.columns:
            pos_patches = patches_df[patches_df["is_positive"] == True]
            neg_patches = patches_df[patches_df["is_positive"] == False]
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


def visualize_current_patches(
    modality_tiles,
    modality_patches,
    output_path=None,
    buffer_top=0,
    buffer_left=0,
    buffer_bottom=0,
    buffer_right=0,
):
    """Visualize the original images and their patches with one modality per row.

    Args:
        modality_tiles: Dictionary of full-sized tiles for each modality
        modality_patches: Dictionary of patches for each modality
        output_path: Path to save the visualization (optional)
        buffer_top: Top buffer offset (pixels to skip from top edge)
        buffer_left: Left buffer offset (pixels to skip from left edge)
        buffer_bottom: Bottom buffer offset (pixels to skip from bottom edge)
        buffer_right: Right buffer offset (pixels to skip from right edge)
    """
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap

    modalities = list(modality_patches.keys())
    n_rows = len(modalities)
    n_cols = 5

    fig = plt.figure(figsize=(22, 4 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, wspace=0.05, hspace=0.2)

    colors = ["r", "g", "b", "y"]
    drawn_rectangles = {}

    # Custom colormap for mask visualization - black, blue, red for background, non-flooded, flooded
    mask_cmap = ListedColormap(["black", "blue", "red"])
    first_col_axes = {}

    for row_idx, modality in enumerate(modalities):
        patches = modality_patches[modality]
        tile = modality_tiles.get(modality)

        if tile is not None:
            if isinstance(tile, np.ndarray):
                if modality == "mask":
                    orig_data = tile[0] if tile.shape[0] == 1 else tile
                    cmap = mask_cmap
                elif tile.ndim == 3 and tile.shape[0] <= 3:
                    if tile.shape[0] >= 3:
                        orig_data = np.stack([tile[i] for i in range(3)], axis=2)
                        cmap = None
                    else:
                        orig_data = tile[0]
                        cmap = None
                elif tile.ndim == 2:
                    orig_data = tile
                    cmap = None
                else:
                    orig_data = tile
                    cmap = None
            elif hasattr(tile, "read"):
                original_img = tile.read()
                if modality == "mask":
                    orig_data = original_img[0]
                    cmap = mask_cmap
                elif original_img.shape[0] >= 3:
                    orig_data = np.stack([original_img[i] for i in range(3)], axis=2)
                    cmap = None
                else:
                    orig_data = original_img[0]
                    cmap = None
            else:
                if len(patches) > 0:
                    with rasterio.open(patches[0][0]) as src:
                        orig_data = np.zeros((src.height * 2, src.width * 2))
                        cmap = mask_cmap if modality == "mask" else "gray"
                else:
                    orig_data = np.zeros((100, 100))
                    cmap = mask_cmap if modality == "mask" else "gray"
        else:
            if len(patches) > 0:
                with rasterio.open(patches[0][0]) as src:
                    orig_data = np.zeros((src.height * 2, src.width * 2))
                    cmap = mask_cmap if modality == "mask" else "gray"
            else:
                orig_data = np.zeros((100, 100))
                cmap = mask_cmap if modality == "mask" else "gray"

        if modality != "mask" and orig_data.dtype != np.uint8:
            orig_data = np.clip(
                orig_data / np.percentile(orig_data, 99)
                if np.percentile(orig_data, 99) > 0
                else 1,
                0,
                1,
            )
        ax_orig = fig.add_subplot(gs[row_idx, 0])
        img = ax_orig.imshow(orig_data, cmap=cmap)

        if modality == "mask":
            if isinstance(orig_data, np.ndarray):
                if orig_data.ndim == 3 and orig_data.shape[0] == 1:
                    mask_data = orig_data[0]
                elif orig_data.ndim == 2:
                    mask_data = orig_data
                else:
                    mask_data = orig_data

                background_count = np.sum(mask_data == 0)
                non_flooded_building = np.sum(mask_data == 1)
                flooded_building = np.sum(mask_data == 2)
                non_flooded_street = np.sum(mask_data == 3)
                flooded_street = np.sum(mask_data == 4)
                total_pixels = mask_data.size

                legend_patches = [
                    mpatches.Patch(
                        color="black",
                        label=f"Background: {background_count} px ({100 * background_count / total_pixels:.1f}%)",
                    ),
                    mpatches.Patch(
                        color="blue",
                        label=f"Non-flooded Street: {non_flooded_street} px ({100 * non_flooded_street / total_pixels:.1f}%)",
                    ),
                    mpatches.Patch(
                        color="red",
                        label=f"Flooded Street: {flooded_street} px ({100 * flooded_street / total_pixels:.1f}%)",
                    ),
                    mpatches.Patch(
                        color="green",
                        label=f"Non-flooded Building: {non_flooded_building} px ({100 * non_flooded_building / total_pixels:.1f}%)",
                    ),
                    mpatches.Patch(
                        color="yellow",
                        label=f"Flooded Building: {flooded_building} px ({100 * flooded_building / total_pixels:.1f}%)",
                    ),
                ]

                mask_legend = ax_orig.legend(
                    handles=legend_patches,
                    loc="lower right",
                    fontsize=8,
                    framealpha=0.7,
                )
                ax_orig.add_artist(mask_legend)

        if buffer_top > 0 or buffer_left > 0 or buffer_bottom > 0 or buffer_right > 0:
            img_height, img_width = (
                orig_data.shape[:2]
                if len(orig_data.shape) >= 2
                else (orig_data.shape[0], orig_data.shape[0])
            )

            if buffer_top > 0:
                ax_orig.add_patch(
                    Rectangle(
                        (0, 0),
                        img_width,
                        buffer_top,
                        facecolor="gray",
                        alpha=0.3,
                        edgecolor=None,
                    )
                )
            if buffer_left > 0:
                ax_orig.add_patch(
                    Rectangle(
                        (0, 0),
                        buffer_left,
                        img_height,
                        facecolor="gray",
                        alpha=0.3,
                        edgecolor=None,
                    )
                )
            if buffer_bottom > 0:
                ax_orig.add_patch(
                    Rectangle(
                        (0, img_height - buffer_bottom),
                        img_width,
                        buffer_bottom,
                        facecolor="gray",
                        alpha=0.3,
                        edgecolor=None,
                    )
                )
            if buffer_right > 0:
                ax_orig.add_patch(
                    Rectangle(
                        (img_width - buffer_right, 0),
                        buffer_right,
                        img_height,
                        facecolor="gray",
                        alpha=0.3,
                        edgecolor=None,
                    )
                )
            ax_orig.set_title(
                f"Original {modality}\nBuffer: T{buffer_top}, L{buffer_left}, B{buffer_bottom}, R{buffer_right}"
            )
        else:
            ax_orig.set_title(f"Original {modality}")

        ax_orig.axis("off")

        first_col_axes[modality] = {
            "ax": ax_orig,
            "data": orig_data,
            "height": orig_data.shape[0]
            if hasattr(orig_data, "shape") and len(orig_data.shape) >= 2
            else 0,
            "width": orig_data.shape[1]
            if hasattr(orig_data, "shape") and len(orig_data.shape) >= 2
            else 0,
        }

        for i, (patch_path, patch_id, row, col) in enumerate(patches[:4]):
            if i >= 4:
                break

            with rasterio.open(patch_path) as patch_src:
                patch_img = patch_src.read()

                if modality == "mask":
                    patch_vis = patch_img[0]
                    patch_cmap = mask_cmap
                elif patch_img.shape[0] >= 3:
                    patch_vis = np.stack([patch_img[i] for i in range(3)], axis=2)
                    patch_cmap = None
                else:
                    patch_vis = patch_img[0]
                    patch_cmap = None

                if patch_vis.size > 0 and modality != "mask":
                    patch_vis = np.clip(
                        patch_vis / np.percentile(patch_vis, 99)
                        if np.percentile(patch_vis, 99) > 0
                        else 1,
                        0,
                        1,
                    )

                ax = fig.add_subplot(gs[row_idx, i + 1])
                ax.imshow(patch_vis, cmap=patch_cmap)

                if modality == "mask":
                    background_count = np.sum(patch_vis == 0)
                    non_flooded_street = np.sum(patch_vis == 1)
                    flooded_street = np.sum(patch_vis == 2)
                    non_flooded_building = np.sum(patch_vis == 3)
                    flooded_building = np.sum(patch_vis == 4)
                    total_pixels = patch_vis.size

                    ratio_text = f"BG: {100 * background_count / total_pixels:.1f}%\n"
                    ratio_text += (
                        f"NF Street: {100 * non_flooded_street / total_pixels:.1f}%\n"
                    )
                    ratio_text += (
                        f"F Street: {100 * flooded_street / total_pixels:.1f}%\n"
                    )
                    ratio_text += f"NF Building: {100 * non_flooded_building / total_pixels:.1f}%\n"
                    ratio_text += (
                        f"F Building: {100 * flooded_building / total_pixels:.1f}%"
                    )

                    ax.text(
                        0.98,
                        0.02,
                        ratio_text,
                        transform=ax.transAxes,
                        ha="right",
                        va="bottom",
                        fontsize=8,
                        color="white",
                        bbox=dict(
                            facecolor="black", alpha=0.7, boxstyle="round,pad=0.3"
                        ),
                    )

                ax.set_title(f"{modality} ({row},{col})", color=colors[i % len(colors)])
                for spine in ax.spines.values():
                    spine.set_color(colors[i % len(colors)])
                    spine.set_linewidth(3)
                ax.axis("off")

    for row_idx, modality in enumerate(modalities):
        patches = modality_patches[modality]

        ax_orig = first_col_axes[modality]["ax"]

        for i, (patch_path, patch_id, row, col) in enumerate(patches[:4]):
            if i >= 4:
                break

            with rasterio.open(patch_path) as patch_src:
                rect_key = f"{modality}_{row}_{col}"

                if rect_key not in drawn_rectangles:
                    x = buffer_left + col * patch_src.width
                    y = buffer_top + row * patch_src.height
                    width = patch_src.width
                    height = patch_src.height

                    rect = Rectangle(
                        (x, y),
                        width,
                        height,
                        linewidth=2,
                        edgecolor=colors[i % len(colors)],
                        facecolor="none",
                        alpha=0.8,
                    )
                    ax_orig.add_patch(rect)

                    ax_orig.text(
                        x + width // 2,
                        y + height // 2,
                        f"{row},{col}",
                        color="white",
                        ha="center",
                        va="center",
                        fontsize=10,
                        fontweight="bold",
                        bbox=dict(
                            facecolor="black", alpha=0.5, pad=0.5, boxstyle="round"
                        ),
                    )
                    drawn_rectangles[rect_key] = True

    if "mask" in modality_tiles:
        mask_data = modality_tiles["mask"]
        if isinstance(mask_data, np.ndarray):
            if mask_data.ndim == 3 and mask_data.shape[0] == 1:
                mask_flat = mask_data[0].flatten()
            elif mask_data.ndim == 2:
                mask_flat = mask_data.flatten()
            else:
                mask_flat = mask_data.flatten()

            background_count = np.sum(patch_vis == 0)
            non_flooded_street = np.sum(patch_vis == 1)
            flooded_street = np.sum(patch_vis == 2)
            non_flooded_building = np.sum(patch_vis == 3)
            flooded_building = np.sum(patch_vis == 4)
            total_pixels = patch_vis.size
            total = mask_flat.size

            class_info = (
                f"Overall Class Distribution:\n"
                f"BG: {100 * background_count / total:.1f}%\n"
                f"NF Street: {100 * non_flooded_street / total:.1f}%\n"
                f"F Street: {100 * flooded_street / total:.1f}%\n"
                f"NF Building: {100 * non_flooded_building / total:.1f}%\n"
                f"F Building: {100 * flooded_building / total:.1f}%"
            )

            fig.text(
                0.01,
                0.01,
                class_info,
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.8, boxstyle="round"),
            )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
    else:
        plt.show()

    plt.close()


def show_samples_per_valid_ratio(
    df: pd.DataFrame, output_path: str = None, dataset_name: str = "Dataset"
):
    """Show the number of samples (rows) that would remain in dataframe after filtering by valid_ratio."""
    import matplotlib.pyplot as plt

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


def create_mmflood_patches(
    metadata_df: pd.DataFrame,
    root_dir: str,
    output_dir: str,
    patch_size: int = 512,
    max_overlap_fraction: float = 0.2,
) -> pd.DataFrame:
    """Split MMFlood tiles into patches of specified size.

    Args:
        metadata_df: DataFrame with image/mask paths and metadata
        root_dir: Root directory of MMFlood dataset
        output_dir: Directory to save patches
        patch_size: Size of patches (height=width)
        max_overlap_fraction: Maximum allowed overlap fraction when optimizing coverage

    Returns:
        DataFrame with metadata for all created patches
    """

    modalities = ["mask", "s1", "hydro", "dem"]

    for mod in modalities:
        os.makedirs(os.path.join(output_dir, mod), exist_ok=True)

    all_patch_metadata = []

    for idx, row in tqdm(
        metadata_df.iterrows(), total=len(metadata_df), desc="Creating patches"
    ):
        tile_id = os.path.basename(row["mask_path"]).split(".")[0]

        mask_path = os.path.join(root_dir, row["mask_path"])
        with rasterio.open(mask_path) as src:
            height, width = src.height, src.width
            mask_profile = src.profile.copy()
            mask_transform = src.transform

        # for tiles that are close to desired patch size multiple, crop with overlap
        # for tiles that are close to desired patch size, center crop
        max_overlap_pixels = int(patch_size * max_overlap_fraction)

        if height <= patch_size and width <= patch_size:
            continue

        if height <= patch_size:
            row_starts = [(height - patch_size) // 2]
            row_step = 0
        elif height <= patch_size + max_overlap_pixels:
            row_starts = [0]
            row_step = 0
        else:
            num_rows = max(
                1,
                (height + max_overlap_pixels - 1) // (patch_size - max_overlap_pixels),
            )
            if num_rows == 1:
                row_starts = [(height - patch_size) // 2]
                row_step = 0
            else:
                row_step = (height - patch_size) / (num_rows - 1) if num_rows > 1 else 0
                row_starts = [int(i * row_step) for i in range(num_rows)]

        if width <= patch_size:
            col_starts = [(width - patch_size) // 2]
            col_step = 0
        elif width <= patch_size + max_overlap_pixels:
            col_starts = [0]
            col_step = 0
        else:
            num_cols = max(
                1, (width + max_overlap_pixels - 1) // (patch_size - max_overlap_pixels)
            )
            if num_cols == 1:
                col_starts = [(width - patch_size) // 2]
                col_step = 0
            else:
                col_step = (width - patch_size) / (num_cols - 1) if num_cols > 1 else 0
                col_starts = [int(i * col_step) for i in range(num_cols)]

        modality_data = {}
        modality_profiles = {}

        for modality in modalities:
            path_key = f"{modality}_path" if modality != "s1" else "s1_path"
            file_path = os.path.join(root_dir, row[path_key])

            with rasterio.open(file_path) as src:
                modality_data[modality] = src.read()
                modality_profiles[modality] = src.profile.copy()

        for i, row_start in enumerate(row_starts):
            for j, col_start in enumerate(col_starts):
                row_start = min(row_start, height - patch_size)
                col_start = min(col_start, width - patch_size)

                mask_patch = modality_data["mask"][
                    :,
                    row_start : row_start + patch_size,
                    col_start : col_start + patch_size,
                ]

                valid_pixels = (mask_patch != 255).sum()
                valid_ratio = valid_pixels / (
                    patch_size * patch_size * mask_patch.shape[0]
                )

                positive_pixels = (mask_patch == 1).sum()
                positive_ratio = positive_pixels / (
                    patch_size * patch_size * mask_patch.shape[0]
                )

                patch_id = f"{tile_id}_p{i}_{j}"

                patch_paths = {}

                for modality in modalities:
                    modality_patch = modality_data[modality][
                        :,
                        row_start : row_start + patch_size,
                        col_start : col_start + patch_size,
                    ]

                    profile = modality_profiles[modality].copy()
                    patch_transform = rasterio.transform.from_origin(
                        mask_transform.c + col_start * mask_transform.a,
                        mask_transform.f + row_start * mask_transform.e,
                        mask_transform.a,
                        mask_transform.e,
                    )

                    profile.update(
                        {
                            "height": patch_size,
                            "width": patch_size,
                            "transform": patch_transform,
                        }
                    )

                    patch_path = os.path.join(output_dir, modality, f"{patch_id}.tif")
                    patch_paths[f"{modality}_path"] = os.path.relpath(
                        patch_path, output_dir
                    )

                    with rasterio.open(patch_path, "w", **profile) as dst:
                        dst.write(modality_patch)

                patch_bounds = rasterio.transform.array_bounds(
                    patch_size, patch_size, patch_transform
                )
                west, south, east, north = patch_bounds
                center_x = (west + east) / 2
                center_y = (north + south) / 2

                patch_metadata = {
                    "source_mask_file": row["mask_path"],
                    "source_s1_file": row["s1_path"],
                    "source_hydro_file": row["hydro_path"],
                    "source_dem_file": row["dem_path"],
                    "patch_id": patch_id,
                    "mask_path": patch_paths["mask_path"],
                    "s1_path": patch_paths["s1_path"],
                    "hydro_path": patch_paths["hydro_path"],
                    "dem_path": patch_paths["dem_path"],
                    "lon": center_x,
                    "lat": center_y,
                    "west": west,
                    "south": south,
                    "east": east,
                    "north": north,
                    "height_px": patch_size,
                    "width_px": patch_size,
                    "row": i,
                    "col": j,
                    "row_px": int(row_start),
                    "col_px": int(col_start),
                    "valid_ratio": float(valid_ratio),
                    "positive_ratio": float(positive_ratio),
                    "date": row["date"],
                    "aoi": row["aoi"],
                    "region_id": row["region_id"],
                    "country": row["country"],
                    "start": row["start"],
                    "end": row["end"],
                    "split": row["split"],
                }

                all_patch_metadata.append(patch_metadata)

    patches_df = pd.DataFrame(all_patch_metadata)

    metadata_path = os.path.join(output_dir, "patch_metadata.parquet")
    patches_df.to_parquet(metadata_path, index=False)

    print(f"Created {len(patches_df)} patches from {len(metadata_df)} source tiles")

    for split in patches_df["split"].unique():
        split_count = len(patches_df[patches_df["split"] == split])
        split_pct = split_count / len(patches_df) * 100
        print(f"{split} patches: {split_count} ({split_pct:.1f}%)")

    return patches_df
