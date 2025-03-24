# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate GeoBenchV2 version of Fields of the World dataset."""

import argparse
import os
import json
import shutil
import pandas as pd
from torchgeo.datasets import FieldsOfTheWorld
import shapely.wkb
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from geobench_v2.generate_benchmark.utils import (
    plot_sample_locations,
    validate_metadata_with_geo,
)
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D
import numpy as np


TOTAL_N = 20000

CC_BY_COUNTRIES = (
    'austria',
    'brazil',
    'corsica',
    'denmark',
    'estonia',
    'finland',
    'france',
    'india',
    'kenya',
    'luxembourg',
    'netherlands',
    'rwanda',
    'slovakia',
    'spain',
    'vietnam',
)

def create_subset(
    ds: FieldsOfTheWorld, df: pd.DataFrame, save_dir: str, random_state: int = 42
) -> None:
    """Create a subset of Fields of the World dataset. Creates a stratified
    subset that maintains the train/val/test and country distribtion for a new TOTAL_N.

    Args:
        ds: Fields of the World dataset.
        df: Metadata DataFrame.
        save_dir: Directory to save the subset.
    """
    # based on the metadata_df create a subset of the dataset and copy it
    # with the same structure to the save_dir
    total_orig = len(df)

    def sample_group(group):
        """Determine how many samples to draw from this group."""
        fraction = len(group) / total_orig
        n_samples = int(round(fraction * TOTAL_N))
        n_samples = min(n_samples, len(group))
        if n_samples == 0 and len(group) > 0:
            n_samples = 1
        return group.sample(n=n_samples, random_state=random_state)

    # Group by 'split' and 'country', then sample from each group.
    subset = df.groupby(["split", "country"], group_keys=False).apply(sample_group)

    # In case due to rounding the total number of samples is not exactly TOTAL_N,
    # one might need to adjust. Here we reset index and trim or pad with extra samples:
    subset = subset.reset_index(drop=True)
    current_n = len(subset)
    if current_n > TOTAL_N:
        # Remove extra rows randomly
        subset = subset.sample(n=TOTAL_N, random_state=random_state).reset_index(
            drop=True
        )
    elif current_n < TOTAL_N:
        # Optionally, add extra rows by oversampling from some groups (if desired)
        extra_needed = TOTAL_N - current_n
        # Here, we simply sample extra rows from the original dataset.
        extra_samples = df.sample(n=extra_needed, random_state=random_state)
        subset = (
            pd.concat([subset, extra_samples])
            .sample(frac=1, random_state=random_state)
            .reset_index(drop=True)
        )

    return subset


def copy_subset_files(ds, subset: pd.DataFrame, save_dir: str) -> None:
    """
    Copy files from the original dataset (ds.root) to the new directory (save_dir)
    following the same directory structure and naming convention.

    For each sample (identified by its country and aoi_id) the following files are
    copied:
      - Sentinel-2 images in two windows:
          {ds.root}/{country}/s2_images/window_a/{aoi_id}.tif
          {ds.root}/{country}/s2_images/window_b/{aoi_id}.tif
      - Label masks in three categories:
          {ds.root}/{country}/label_masks/instance/{aoi_id}.tif
          {ds.root}/{country}/label_masks/semantic_2class/{aoi_id}.tif
          {ds.root}/{country}/label_masks/semantic_3class/{aoi_id}.tif

    Args:
        ds: The original dataset object. It is expected to have an attribute `root`
            that points to the base directory of the dataset.
        subset: A pandas DataFrame with at least the columns "country" and "aoi_id".
        save_dir: Destination directory in which the subset should be created.
    """
    for _, row in tqdm(
        subset.iterrows(), total=len(subset), desc="Copying sample files"
    ):
        country = row["country"]
        aoi_id = row["aoi_id"]

        # Copy Sentinel-2 images: window_a and window_b
        for window in ["window_a", "window_b"]:
            src_path = os.path.join(
                ds.root, country, "s2_images", window, f"{aoi_id}.tif"
            )
            if not os.path.exists(src_path):
                continue
            dst_path = os.path.join(
                save_dir, country, "s2_images", window, f"{aoi_id}.tif"
            )
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)

        # Copy label masks: instance, semantic_2class, semantic_3class
        for subdir in ["instance", "semantic_2class", "semantic_3class"]:
            src_path = os.path.join(
                ds.root, country, "label_masks", subdir, f"{aoi_id}.tif"
            )
            if not os.path.exists(src_path):
                continue
            dst_path = os.path.join(
                save_dir, country, "label_masks", subdir, f"{aoi_id}.tif"
            )
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)


def generate_metadata_df(ds: FieldsOfTheWorld) -> pd.DataFrame:
    """Generate metadata DataFrame for Fields of the World Benchmark.

    Args:
        ds: Fields of the World dataset.

    Returns:
        Metadata DataFrame.
    """
    overall_df = pd.DataFrame()
    selected_countries = ds.countries

    for country in tqdm(selected_countries, desc="Collecting metadata"):
        country_df = pd.read_parquet(f"{ds.root}/{country}/chips_{country}.parquet")

        with open(f"{ds.root}/{country}/data_config_{country}.json", "r") as f:
            data_config = json.load(f)

        country_df["year_of_collection"] = data_config["year_of_collection"]
        country_df["geometry_obj"] = country_df["geometry"].apply(
            lambda x: shapely.wkb.loads(x)
        )
        country_df["lon"] = country_df["geometry_obj"].apply(lambda g: g.centroid.x)
        country_df["lat"] = country_df["geometry_obj"].apply(lambda g: g.centroid.y)

        country_df.drop(columns=["geometry", "geometry_obj"], inplace=True)

        country_df["country"] = country

        overall_df = pd.concat([overall_df, country_df], ignore_index=True)

    overall_df["aoi_id"] = overall_df["aoi_id"].astype(str)

    # country of india has some samples that are 'none' for the split, so drop them
    overall_df = overall_df[overall_df["split"] != "none"]
    return overall_df


def create_unit_test_subset() -> None:
    """Create a subset of Fields of the World dataset for GeoBench unit tests."""

    # create random images etc that respect the structure of the dataset in minimal format
    pass


def generate_benchmark(
    ds: FieldsOfTheWorld, metadata_df: pd.DataFrame, save_dir: str
) -> None:
    """Generate Fields of the World Benchmark.

    Args:
        ds: Fields of the World dataset.
        save_dir: Directory to save the subset benchmark data.
    """
    subset = create_subset(ds, metadata_df, save_dir)
    copy_subset_files(ds, subset, save_dir)


def plot_country_distribution(
    metadata_df: pd.DataFrame,
    output_path: str = None,
    title: str = "Geographic Distribution of Dataset Samples",
    highlight_countries: bool = True,
    show_country_labels: bool = True,
    min_samples_for_label: int = 100,
    figsize: tuple = (14, 10),
) -> None:
    """Plot the geolocation of samples on a world map with country-level highlighting."""
    # Count samples per country and split
    country_counts = (
        metadata_df.groupby("country")["aoi_id"].count().sort_values(ascending=False)
    )
    split_counts = (
        metadata_df.groupby(["country", "split"])["aoi_id"]
        .count()
        .unstack(fill_value=0)
    )

    # Get global stats
    total_samples = len(metadata_df)
    n_countries = len(country_counts)

    print(f"Dataset contains {total_samples:,} samples across {n_countries} countries")
    print(f"Top 5 countries by sample count:")
    for country, count in country_counts.head(5).items():
        percentage = 100 * count / total_samples
        print(f"  {country}: {count:,} samples ({percentage:.1f}%)")

    # Create figure with Robinson projection for global data
    plt.figure(figsize=figsize)
    projection = ccrs.Robinson()
    ax = plt.axes(projection=projection)

    # Set global extent for world map
    ax.set_global()

    # Add map features with improved styling
    ax.add_feature(cfeature.LAND, facecolor="#EFEFEF")
    ax.add_feature(cfeature.OCEAN, facecolor="#D8E9F5")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#888888")
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle="-", edgecolor="#888888")

    # Define color palette for splits
    split_colors = {
        "train": "#1f77b4",  # Blue
        "val": "#ff7f0e",  # Orange
        "test": "#2ca02c",  # Green
    }

    # Define country highlight colors if needed
    if highlight_countries:
        # Use tab20 colormap for up to 20 countries
        country_colormap = plt.cm.get_cmap("tab20", n_countries)
        country_colors = {
            country: country_colormap(i)
            for i, country in enumerate(country_counts.index)
        }

    # Get country shapes for highlighting
    if highlight_countries:
        # Create a list of countries with samples for highlighting
        countries_with_samples = set(metadata_df["country"].unique())

        # Use shapereader to directly access country geometries
        import cartopy.io.shapereader as shpreader

        # Get shapefile from cartopy's data store
        shapename = "admin_0_countries"
        countries_shp = shpreader.natural_earth(
            resolution="50m", category="cultural", name=shapename
        )

        # Read the shapefile
        reader = shpreader.Reader(countries_shp)

        # Match countries in dataset with geometries in shapefile
        for country_record in reader.records():
            country_name = country_record.attributes["NAME"].lower()

            # Try to match with dataset countries
            for ds_country in countries_with_samples:
                if (
                    ds_country.lower() in country_name
                    or country_name in ds_country.lower()
                ):
                    # Add the geometry for this country with its color
                    ax.add_geometries(
                        [country_record.geometry],
                        ccrs.PlateCarree(),
                        facecolor=country_colors.get(ds_country, "#CCCCCC"),
                        alpha=0.3,
                        edgecolor="#444444",
                        linewidth=0.5,
                    )
                    break

    # Plot points by country and split
    for country in country_counts.index:
        country_data = metadata_df[metadata_df["country"] == country]

        # Calculate point size inversely proportional to number of points (with limits)
        n_points = len(country_data)
        point_size = max(0.5, min(3.0, 50.0 / np.sqrt(n_points)))

        # Plot each split separately
        for split in ["train", "val", "test"]:
            split_data = country_data[country_data["split"] == split]
            if len(split_data) > 0:
                ax.scatter(
                    split_data["lon"],
                    split_data["lat"],
                    transform=ccrs.PlateCarree(),
                    c=split_colors[split],
                    s=point_size,
                    alpha=0.7,
                    label=f"{split} ({len(split_data)})"
                    if country == country_counts.index[0]
                    else "",
                    zorder=3,
                )

    # Add country labels with much better positioning
    if show_country_labels:
        # Prepare data for label positioning
        countries_to_label = []
        for country, count in country_counts.items():
            if count >= min_samples_for_label:
                # Calculate centroid of samples for this country
                country_data = metadata_df[metadata_df["country"] == country]
                center_lon = country_data["lon"].mean()
                center_lat = country_data["lat"].mean()

                # Store for later use
                countries_to_label.append(
                    {
                        "name": country,
                        "lon": center_lon,
                        "lat": center_lat,
                        "count": count,
                        "importance": count,  # Use count as initial importance
                    }
                )

        # Define regions with their center points
        regions = {
            "europe": {"center": (15, 50), "countries": []},
            "africa": {"center": (20, 0), "countries": []},
            "asia": {"center": (100, 30), "countries": []},
            "north_america": {"center": (-100, 40), "countries": []},
            "south_america": {"center": (-60, -20), "countries": []},
            "oceania": {"center": (135, -25), "countries": []},
            "other": {"center": None, "countries": []},
        }

        # Assign countries to regions
        for country in countries_to_label:
            lon, lat = country["lon"], country["lat"]

            # Determine region based on coordinates
            if -20 <= lon <= 40 and 35 <= lat <= 75:
                region = "europe"
            elif -20 <= lon <= 55 and -40 <= lat <= 35:
                region = "africa"
            elif 55 <= lon <= 150 and -10 <= lat <= 75:
                region = "asia"
            elif -170 <= lon <= -50 and 25 <= lat <= 75:
                region = "north_america"
            elif -80 <= lon <= -30 and -60 <= lat <= 15:
                region = "south_america"
            elif 100 <= lon <= 180 and -50 <= lat <= -10:
                region = "oceania"
            else:
                region = "other"

            regions[region]["countries"].append(country)

        # Function to position labels in a curved grid pattern
        def position_labels_in_grid(
            countries, center, min_radius=15, grid_width=5, vertical_spacing=5
        ):
            if not countries:
                return

            # Sort by importance (sample count)
            countries.sort(key=lambda x: x["importance"], reverse=True)

            # Calculate positions in a curved grid
            positions = []
            rows = (len(countries) + grid_width - 1) // grid_width

            for i, country in enumerate(countries):
                row = i // grid_width
                col = i % grid_width

                # Position along a curved grid (arc)
                angle_range = 120  # degrees
                angle_offset = -60  # center around due north
                angle = (
                    angle_offset
                    + (col / (grid_width - 1 if grid_width > 1 else 1)) * angle_range
                )

                # Convert to radians
                angle_rad = np.radians(angle)

                # Radius increases with row number
                radius = min_radius + row * vertical_spacing

                # Convert to cartesian coordinates
                offset_x = center[0] + radius * np.sin(angle_rad)
                offset_y = center[1] + radius * np.cos(angle_rad)

                # Store position with the country
                country["label_x"] = offset_x
                country["label_y"] = offset_y

        # Position labels for each region
        for region_name, region_data in regions.items():
            if region_data["countries"]:
                # Skip positioning for "other" region
                if region_name == "other":
                    for country in region_data["countries"]:
                        country["label_x"] = country["lon"]
                        country["label_y"] = country["lat"]
                    continue

                # For regions with multiple countries, position in grid
                center = region_data["center"]
                countries = region_data["countries"]

                # Adjust parameters for different regions
                if region_name == "europe":
                    # European countries need more spread
                    position_labels_in_grid(
                        countries,
                        center,
                        min_radius=20,  # Larger starting radius
                        grid_width=4,  # Fewer columns for more spread
                        vertical_spacing=8,  # More vertical space
                    )
                elif region_name == "asia" and len(countries) > 5:
                    # Asia might need custom layout too
                    position_labels_in_grid(
                        countries,
                        center,
                        min_radius=15,
                        grid_width=4,
                        vertical_spacing=8,
                    )
                else:
                    # Default positioning
                    position_labels_in_grid(
                        countries,
                        center,
                        min_radius=10,
                        grid_width=5,
                        vertical_spacing=6,
                    )

        # Draw all labels and connecting lines
        for region_name, region_data in regions.items():
            for country in region_data["countries"]:
                # Draw connecting line
                if region_name != "other":  # Don't draw lines for "other" region
                    plt.plot(
                        [country["lon"], country["label_x"]],
                        [country["lat"], country["label_y"]],
                        transform=ccrs.PlateCarree(),
                        color="gray",
                        linewidth=0.5,
                        alpha=0.3,
                        zorder=3,
                    )

                # Add text label
                ax.text(
                    country["label_x"],
                    country["label_y"],
                    country["name"].capitalize(),
                    transform=ccrs.PlateCarree(),
                    fontsize=8,
                    ha="center",
                    va="center",
                    bbox=dict(
                        facecolor="white",
                        alpha=0.7,
                        boxstyle="round,pad=0.2",
                        edgecolor="none",
                    ),
                    zorder=4,
                )

    # Create legend for dataset splits
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markersize=8,
            label=f"{split} ({metadata_df[metadata_df['split'] == split].shape[0]:,})",
        )
        for split, color in split_colors.items()
        if split in metadata_df["split"].unique()
    ]

    ax.legend(
        handles=legend_elements,
        loc="lower left",
        title="Dataset Splits",
        framealpha=0.9,
    )

    # Add gridlines (lighter for better readability)
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="gray", alpha=0.3, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False

    # Add a title with dataset statistics
    plt.title(
        f"{title}\n{total_samples:,} samples across {n_countries} countries",
        fontsize=14,
    )

    # Add a small inset with numerical summary
    summary_text = f"Total: {total_samples:,} samples\n"
    for split in ["train", "val", "test"]:
        if split in metadata_df["split"].unique():
            count = metadata_df[metadata_df["split"] == split].shape[0]
            percentage = 100 * count / total_samples
            summary_text += f"{split.capitalize()}: {count:,} ({percentage:.1f}%)\n"

    plt.figtext(
        0.02,
        0.02,
        summary_text,
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.5"),
    )

    # Save or display the figure
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Map saved to {output_path}")

        # Also save split counts by country as CSV
        csv_path = output_path.replace(".png", "_country_stats.csv")
        split_counts.to_csv(csv_path)
        print(f"Country statistics saved to {csv_path}")
    else:
        plt.tight_layout()
        plt.show()



def main():
    """Generate Fields of the World Benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for Fields of the World dataset"
    )
    parser.add_argument(
        "--save_dir",
        default="geobenchV2/fotw",
        help="Directory to save the subset benchmark data",
    )
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    orig_dataset = FieldsOfTheWorld(root=args.root, download=False, countries=CC_BY_COUNTRIES)
    
    metadata_path = os.path.join(args.save_dir, "metadata.parquet")
    # if os.path.exists(metadata_path):
    #     metadata_df = pd.read_parquet(metadata_path)
    # else:
    metadata_df = generate_metadata_df(orig_dataset)
    metadata_df.to_parquet(metadata_path)

    # assert that only CC_BY_COUNTRIES are present
    assert set(metadata_df["country"].unique()) == set(CC_BY_COUNTRIES)

    validate_metadata_with_geo(metadata_df)

    plot_country_distribution(
        metadata_df,
        output_path=os.path.join(args.save_dir, "country_distribution.png"),
        title="Fields of the World Dataset - Geographic Distribution",
    )

    # generate_benchmark(orig_dataset, metadata_df, args.save_dir)


if __name__ == "__main__":
    main()
