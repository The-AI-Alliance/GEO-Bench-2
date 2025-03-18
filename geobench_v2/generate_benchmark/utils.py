# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Utility functions for benchmark generation."""

from matplotlib.lines import Line2D
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from typing import Tuple, Optional, Dict, List
import os
import rasterio
import numpy as np
import re
from rasterio.windows import Window
from tqdm import tqdm


def plot_sample_locations(
    metadata_df: pd.DataFrame,
    output_path: str = None,
    buffer_degrees: float = 5.0,
    split_column: str = "split",
    sample_fraction: float = 1.0,  # Reduced default to 10%
    alpha: float = 0.5,
    s: float = 0.5,
) -> None:
    """Plot the geolocation of samples on a map, differentiating by dataset splits.

    Args:
        metadata_df: DataFrame with metadata including lat and lon columns
        output_path: Path to save the figure. If None, the figure is displayed but not saved.
        buffer_degrees: Buffer around the data extent in degrees
        split_column: Column name that indicates the dataset split
        sample_fraction: Fraction of samples to plot for better performance (0.0-1.0)
        alpha: Transparency of plotted points
        s: Size of plotted points
    """
    # Sample data if fraction is less than 1.0
    if sample_fraction < 1.0:
        sample_size = int(len(metadata_df) * sample_fraction)
        # Use stratified sampling to maintain split proportions
        if split_column in metadata_df.columns:
            sampled_dfs = []
            for split in metadata_df[split_column].unique():
                split_df = metadata_df[metadata_df[split_column] == split]
                split_sample_size = max(
                    1, int(sample_size * len(split_df) / len(metadata_df))
                )
                sampled_dfs.append(
                    split_df.sample(
                        n=min(len(split_df), split_sample_size), random_state=42
                    )
                )
            metadata_df = pd.concat(sampled_dfs)
        else:
            metadata_df = metadata_df.sample(sample_size, random_state=42)
        print(f"Sampled {len(metadata_df)} points for plotting")

    # Ensure column names are standardized
    if "latitude" in metadata_df.columns:
        metadata_df = metadata_df.rename(
            columns={"latitude": "lat", "longitude": "lon"}
        )

    # Determine the geographic extent of the data with buffer
    min_lon = metadata_df["lon"].min() - buffer_degrees
    max_lon = metadata_df["lon"].max() + buffer_degrees
    min_lat = metadata_df["lat"].min() - buffer_degrees
    max_lat = metadata_df["lat"].max() + buffer_degrees

    # Ensure the extent is valid
    min_lon = max(-180, min_lon)
    max_lon = min(180, max_lon)
    min_lat = max(-90, min_lat)
    max_lat = min(90, max_lat)

    print(
        f"Map extent: Longitude [{min_lon:.2f}° to {max_lon:.2f}°], "
        f"Latitude [{min_lat:.2f}° to {max_lat:.2f}°]"
    )

    # Create figure with a suitable projection for this extent
    plt.figure(figsize=(10, 8))  # Smaller figure for faster rendering

    # For France-specific data, use an optimized projection
    if 41 < min_lat < 51.5 and -5 < min_lon < 10:
        # France-specific projection
        projection = ccrs.LambertConformal(
            central_longitude=3.0, central_latitude=46.5, standard_parallels=(44, 49)
        )
    else:
        # Choose appropriate projection for other regions
        central_lon = (min_lon + max_lon) / 2
        central_lat = (min_lat + max_lat) / 2
        projection = ccrs.LambertConformal(
            central_longitude=central_lon, central_latitude=central_lat
        )

    ax = plt.axes(projection=projection)

    # Set the map extent
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    # Use lower resolution features for speed
    scale = "50m"  # Very low resolution for speed

    # Add ONLY essential map features - removing detailed features that slow down rendering
    ax.add_feature(cfeature.LAND.with_scale(scale), facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN.with_scale(scale), facecolor="lightblue")
    ax.add_feature(cfeature.COASTLINE.with_scale(scale), linewidth=0.5)
    ax.add_feature(cfeature.BORDERS.with_scale(scale), linewidth=0.3)

    # REMOVED: Rivers and Lakes features which cause the slowdown
    # Also removed the conditional for detailed features

    # Get unique splits
    # if split_column in metadata_df.columns:
    #     splits = metadata_df[split_column].unique()
    # else:
    splits = ["train", "val", "test"]  # Default splits if column doesn't exist

    print(f"Found {len(splits)} dataset splits: {', '.join(map(str, splits))}")

    # Define colors for different splits
    split_colors = {
        "train": "blue",
        "val": "green",
        "test": "red",
        "validation": "green",
        "testing": "red",
    }

    # Create a legend handle list
    legend_elements = []

    # Create a scatter plot for each split
    for split in splits:
        split_data = metadata_df[
            metadata_df[split_column] == split
        ]  # Use all data if split column doesn't exist

        if len(split_data) > 0:
            # Get color (default to blue if not in split_colors)
            color = split_colors.get(split, "blue")

            # Plot the points
            ax.scatter(
                split_data["lon"],
                split_data["lat"],
                transform=ccrs.PlateCarree(),
                c=color,
                s=s,
                alpha=alpha,
                label=split,
            )

            # Add to legend
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markersize=8,
                    label=f"{split} (n={len(split_data)})",
                )
            )

    ax.legend(handles=legend_elements, loc="lower right", title="Dataset Splits")

    # Set appropriate title based on output path
    if "pastis" in output_path.lower():
        title = "Geographic Distribution of PASTIS Samples"
    elif "flair" in output_path.lower():
        title = "Geographic Distribution of FLAIR2 Samples"
    else:
        title = "Geographic Distribution of Dataset Samples"

    # Use simplified gridlines (no labels)
    gl = ax.gridlines(
        draw_labels=False, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
    )

    # Set title
    plt.title(title, fontsize=14)

    # Save the figure if output_path is provided
    if output_path:
        try:
            plt.savefig(
                output_path, dpi=150, bbox_inches="tight"
            )  # Reduced DPI for faster saving
            print(f"Map saved to {output_path}")
        except Exception as e:
            print(f"Error saving figure: {e}")
    else:
        plt.show()

    # Explicitly close the figure to free memory
    plt.close()


def visualize_current_patches(modality_tiles, modality_patches, output_path=None):
    """Visualize the original images and their patches with one modality per row."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import matplotlib.gridspec as gridspec

    modalities = list(modality_patches.keys())
    n_rows = len(modalities)
    n_cols = 5

    fig = plt.figure(figsize=(20, 4 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, wspace=0.05, hspace=0.2)

    colors = ["r", "g", "b", "y"]
    drawn_rectangles = {}

    for row_idx, modality in enumerate(modalities):
        patches = modality_patches[modality]
        tile = modality_tiles.get(modality)

        if tile is not None:
            if isinstance(tile, np.ndarray):
                if tile.ndim == 3 and tile.shape[0] <= 3:
                    if tile.shape[0] >= 3:
                        orig_data = np.stack([tile[i] for i in range(3)], axis=2)
                        cmap = None
                    else:
                        orig_data = tile[0]
                        cmap = None if modality != "mask" else "gray"
                elif tile.ndim == 2:
                    orig_data = tile
                    cmap = None if modality != "mask" else "gray"
                else:
                    orig_data = tile
                    cmap = None
            elif hasattr(tile, "read"):
                original_img = tile.read()
                if original_img.shape[0] >= 3 and modality != "mask":
                    orig_data = np.stack([original_img[i] for i in range(3)], axis=2)
                    cmap = None
                else:
                    orig_data = original_img[0]
                    cmap = None if modality != "mask" else "gray"
            else:
                if len(patches) > 0:
                    with rasterio.open(patches[0][0]) as src:
                        orig_data = np.zeros((src.height * 2, src.width * 2))
                        cmap = "gray"
                else:
                    orig_data = np.zeros((100, 100))
                    cmap = "gray"
        else:
            if len(patches) > 0:
                with rasterio.open(patches[0][0]) as src:
                    orig_data = np.zeros((src.height * 2, src.width * 2))
                    cmap = "gray"
            else:
                orig_data = np.zeros((100, 100))
                cmap = "gray"

        if orig_data.dtype != np.uint8:
            orig_data = np.clip(orig_data / np.percentile(orig_data, 99), 0, 1)

        ax_orig = fig.add_subplot(gs[row_idx, 0])
        ax_orig.imshow(orig_data, cmap=cmap)
        ax_orig.set_title(f"Original {modality}")
        ax_orig.axis("off")

        for i, (patch_path, patch_id, row, col) in enumerate(patches[:4]):
            if i >= 4:
                break

            with rasterio.open(patch_path) as patch_src:
                patch_img = patch_src.read()

                if patch_img.shape[0] >= 3 and modality != "mask":
                    patch_vis = np.stack([patch_img[i] for i in range(3)], axis=2)
                    patch_cmap = None
                else:
                    patch_vis = patch_img[0]
                    patch_cmap = "gray" if modality == "mask" else None

                if patch_vis.size > 0:
                    patch_vis = np.clip(patch_vis / np.percentile(patch_vis, 99), 0, 1)

                ax = fig.add_subplot(gs[row_idx, i + 1])
                ax.imshow(patch_vis, cmap=patch_cmap)
                ax.set_title(f"{modality} ({row},{col})", color=colors[i % len(colors)])
                for spine in ax.spines.values():
                    spine.set_color(colors[i % len(colors)])
                    spine.set_linewidth(3)
                ax.axis("off")

                if row_idx == 0:
                    rect_key = f"{row}_{col}"
                    if rect_key not in drawn_rectangles:
                        x = col * patch_src.width
                        y = row * patch_src.height
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
                            fontsize=12,
                        )
                        drawn_rectangles[rect_key] = True

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
    else:
        plt.show()

    plt.close()


def split_geospatial_tiles_into_patches(
    modal_path_dict: Dict[str, List[str]],
    output_dir: str,
    patch_size: Tuple[int, int] = (512, 512),
    stride: Optional[Tuple[int, int]] = None,
    min_valid_data_ratio: float = 0.7,
    min_positive_pixels_ratio: float = 0.01,
    output_format: str = "tif",
    patch_id_prefix: str = "p",
) -> pd.DataFrame:
    """Split large geospatial image and mask pairs into smaller patches across multiple modalities."""
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

                    shapes = [
                        (geom, 1)
                        for geom in gdf.geometry
                        if geom is not None and not geom.is_empty
                    ]

                    if shapes:
                        mask_full = rasterize(
                            shapes,
                            out_shape=(src_height, src_width),
                            transform=src_transform,
                            fill=0,
                            dtype=np.uint8,
                        )
                        mask_full = mask_full[np.newaxis, :, :]
            except Exception as e:
                print(f"Warning: Error processing mask {mask_path}: {e}")
                mask_full = np.zeros((1, src_height, src_width), dtype=np.uint8)

            patches_per_dim_h = src_height // patch_size[0]
            patches_per_dim_w = src_width // patch_size[1]

            total_patches = patches_per_dim_h * patches_per_dim_w
            patches_created = 0

            modality_patches = {modality: [] for modality in modalities}
            modality_tiles = {}

            for modality in modalities:
                if modality != "mask":
                    try:
                        with rasterio.open(modal_img_paths[modality]) as modal_src:
                            modality_tiles[modality] = modal_src.read().transpose(
                                1, 2, 0
                            )
                    except Exception as e:
                        print(f"Error opening {modality} source: {e}")
                        modality_tiles[modality] = None
            modality_tiles["mask"] = mask_full

            for i in range(patches_per_dim_h):
                for j in range(patches_per_dim_w):
                    row_start = i * stride[0]
                    col_start = j * stride[1]

                    if (
                        row_start + patch_size[0] > src_height
                        or col_start + patch_size[1] > src_width
                    ):
                        if row_start + patch_size[0] > src_height:
                            row_start = max(0, src_height - patch_size[0])
                        if col_start + patch_size[1] > src_width:
                            col_start = max(0, src_width - patch_size[1])

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
            #     )

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


def show_samples_per_valid_ratio(
    df: pd.DataFrame, output_path: str = None, dataset_name: str = "Dataset"
):
    """Show the number of samples (rows) that would remain in dataframe after filtering by valid_ratio."""
    import matplotlib.pyplot as plt

    # valid_ratios = df["valid_ratio"].unique()
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
