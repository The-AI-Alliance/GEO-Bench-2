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
    create_subset_from_tortilla,
)
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D
import numpy as np

import rasterio
import tacotoolbox
import tacoreader
import glob
import cartopy.io.shapereader as shpreader


TOTAL_N = 20000

CC_BY_COUNTRIES = (
    "austria",
    "brazil",
    "corsica",
    "denmark",
    "estonia",
    "finland",
    "france",
    "india",
    "kenya",
    "luxembourg",
    "netherlands",
    "rwanda",
    "slovakia",
    "spain",
    "vietnam",
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

    Includes relative filepaths to window_a, window_b, instance mask, and semantic_3class mask.

    Args:
        ds: Fields of the World dataset.

    Returns:
        Metadata DataFrame with file paths.
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

        # relative filepaths for tortialla creation
        country_df["win_a_path"] = country_df["aoi_id"].apply(
            lambda aoi: os.path.join(country, "s2_images", "window_a", f"{aoi}.tif")
        )
        country_df["win_b_path"] = country_df["aoi_id"].apply(
            lambda aoi: os.path.join(country, "s2_images", "window_b", f"{aoi}.tif")
        )
        country_df["instance_mask_path"] = country_df["aoi_id"].apply(
            lambda aoi: os.path.join(country, "label_masks", "instance", f"{aoi}.tif")
        )
        country_df["semantic_2class_mask_path"] = country_df["aoi_id"].apply(
            lambda aoi: os.path.join(
                country, "label_masks", "semantic_2class", f"{aoi}.tif"
            )
        )
        country_df["semantic_3class_mask_path"] = country_df["aoi_id"].apply(
            lambda aoi: os.path.join(
                country, "label_masks", "semantic_3class", f"{aoi}.tif"
            )
        )

        # sanity check to see if the files exist
        country_df["win_a_exists"] = country_df["win_a_path"].apply(
            lambda path: os.path.exists(os.path.join(ds.root, path))
        )
        country_df["win_b_exists"] = country_df["win_b_path"].apply(
            lambda path: os.path.exists(os.path.join(ds.root, path))
        )
        country_df["instance_mask_exists"] = country_df["instance_mask_path"].apply(
            lambda path: os.path.exists(os.path.join(ds.root, path))
        )
        country_df["semantic_3class_mask_exists"] = country_df[
            "semantic_3class_mask_path"
        ].apply(lambda path: os.path.exists(os.path.join(ds.root, path)))

        # Drop intermediate geometry objects
        country_df.drop(columns=["geometry", "geometry_obj"], inplace=True)

        country_df["country"] = country

        overall_df = pd.concat([overall_df, country_df], ignore_index=True)

    overall_df["aoi_id"] = overall_df["aoi_id"].astype(str)

    # Drop samples with 'none' split or missing essential files
    overall_df = overall_df[
        (overall_df["split"] != "none")
        & (overall_df["win_a_exists"])
        & (overall_df["win_b_exists"])
    ]

    # Drop the existence check columns after filtering
    overall_df = overall_df.drop(
        columns=[
            "win_a_exists",
            "win_b_exists",
            "instance_mask_exists",
            "semantic_3class_mask_exists",
        ]
    ).reset_index(drop=True)

    return overall_df


def create_unit_test_subset() -> None:
    """Create a subset of Fields of the World dataset for GeoBench unit tests."""

    # create random images etc that respect the structure of the dataset in minimal format
    pass


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
    country_counts = (
        metadata_df.groupby("country")["aoi_id"].count().sort_values(ascending=False)
    )
    split_counts = (
        metadata_df.groupby(["country", "split"])["aoi_id"]
        .count()
        .unstack(fill_value=0)
    )

    total_samples = len(metadata_df)
    n_countries = len(country_counts)

    print(f"Dataset contains {total_samples:,} samples across {n_countries} countries")
    print(f"Top 5 countries by sample count:")
    for country, count in country_counts.head(5).items():
        percentage = 100 * count / total_samples
        print(f"  {country}: {count:,} samples ({percentage:.1f}%)")

    plt.figure(figsize=figsize)
    projection = ccrs.Robinson()
    ax = plt.axes(projection=projection)

    ax.set_global()

    ax.add_feature(cfeature.LAND, facecolor="#EFEFEF")
    ax.add_feature(cfeature.OCEAN, facecolor="#D8E9F5")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#888888")
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle="-", edgecolor="#888888")

    # Define color palette for splits
    split_colors = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}

    if highlight_countries:
        country_colormap = plt.cm.get_cmap("tab20", n_countries)
        country_colors = {
            country: country_colormap(i)
            for i, country in enumerate(country_counts.index)
        }

    if highlight_countries:
        countries_with_samples = set(metadata_df["country"].unique())

        shapename = "admin_0_countries"
        countries_shp = shpreader.natural_earth(
            resolution="50m", category="cultural", name=shapename
        )
        reader = shpreader.Reader(countries_shp)

        for country_record in reader.records():
            country_name = country_record.attributes["NAME"].lower()

            for ds_country in countries_with_samples:
                if (
                    ds_country.lower() in country_name
                    or country_name in ds_country.lower()
                ):
                    ax.add_geometries(
                        [country_record.geometry],
                        ccrs.PlateCarree(),
                        facecolor=country_colors.get(ds_country, "#CCCCCC"),
                        alpha=0.3,
                        edgecolor="#444444",
                        linewidth=0.5,
                    )
                    break

    for country in country_counts.index:
        country_data = metadata_df[metadata_df["country"] == country]

        n_points = len(country_data)
        point_size = max(0.5, min(3.0, 50.0 / np.sqrt(n_points)))

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

    if show_country_labels:
        countries_to_label = []
        for country, count in country_counts.items():
            if count >= min_samples_for_label:
                country_data = metadata_df[metadata_df["country"] == country]
                center_lon = country_data["lon"].mean()
                center_lat = country_data["lat"].mean()

                countries_to_label.append(
                    {
                        "name": country,
                        "lon": center_lon,
                        "lat": center_lat,
                        "count": count,
                        "importance": count,
                    }
                )

        regions = {
            "europe": {"center": (15, 50), "countries": []},
            "africa": {"center": (20, 0), "countries": []},
            "asia": {"center": (100, 30), "countries": []},
            "north_america": {"center": (-100, 40), "countries": []},
            "south_america": {"center": (-60, -20), "countries": []},
            "oceania": {"center": (135, -25), "countries": []},
            "other": {"center": None, "countries": []},
        }

        for country in countries_to_label:
            lon, lat = country["lon"], country["lat"]

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

        def position_labels_in_grid(
            countries, center, min_radius=15, grid_width=5, vertical_spacing=5
        ):
            if not countries:
                return

            countries.sort(key=lambda x: x["importance"], reverse=True)

            positions = []
            rows = (len(countries) + grid_width - 1) // grid_width

            for i, country in enumerate(countries):
                row = i // grid_width
                col = i % grid_width

                angle_range = 120
                angle_offset = -60
                angle = (
                    angle_offset
                    + (col / (grid_width - 1 if grid_width > 1 else 1)) * angle_range
                )
                angle_rad = np.radians(angle)
                radius = min_radius + row * vertical_spacing

                offset_x = center[0] + radius * np.sin(angle_rad)
                offset_y = center[1] + radius * np.cos(angle_rad)
                country["label_x"] = offset_x
                country["label_y"] = offset_y

        for region_name, region_data in regions.items():
            if region_data["countries"]:
                if region_name == "other":
                    for country in region_data["countries"]:
                        country["label_x"] = country["lon"]
                        country["label_y"] = country["lat"]
                    continue

                center = region_data["center"]
                countries = region_data["countries"]

                if region_name == "europe":
                    position_labels_in_grid(
                        countries,
                        center,
                        min_radius=20,
                        grid_width=4,
                        vertical_spacing=8,
                    )
                elif region_name == "asia" and len(countries) > 5:
                    position_labels_in_grid(
                        countries,
                        center,
                        min_radius=15,
                        grid_width=4,
                        vertical_spacing=8,
                    )
                else:
                    position_labels_in_grid(
                        countries,
                        center,
                        min_radius=10,
                        grid_width=5,
                        vertical_spacing=6,
                    )

        for region_name, region_data in regions.items():
            for country in region_data["countries"]:
                if region_name != "other":
                    plt.plot(
                        [country["lon"], country["label_x"]],
                        [country["lat"], country["label_y"]],
                        transform=ccrs.PlateCarree(),
                        color="gray",
                        linewidth=0.5,
                        alpha=0.3,
                        zorder=3,
                    )

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

    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="gray", alpha=0.3, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False

    plt.title(
        f"{title}\n{total_samples:,} samples across {n_countries} countries",
        fontsize=14,
    )

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

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Map saved to {output_path}")

        csv_path = output_path.replace(".png", "_country_stats.csv")
        split_counts.to_csv(csv_path)
        print(f"Country statistics saved to {csv_path}")
    else:
        plt.tight_layout()
        plt.show()


def create_tortilla(root_dir, df, save_dir):
    """Create a tortilla version of the dataset."""

    df["split"] = df["split"].replace("val", "validation")
    tortilla_dir = os.path.join(save_dir, "tortilla")
    os.makedirs(tortilla_dir, exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating tortilla"):
        modalities = ["win_a", "win_b", "instance_mask", "semantic_3class_mask"]
        modality_samples = []

        for modality in modalities:
            path = os.path.join(root_dir, row[modality + "_path"])
            with rasterio.open(path) as src:
                profile = src.profile

            sample = tacotoolbox.tortilla.datamodel.Sample(
                id=modality,
                path=path,
                file_format="GTiff",
                data_split=row["split"],
                stac_data={
                    "crs": "EPSG:" + str(profile["crs"].to_epsg()),
                    "geotransform": profile["transform"].to_gdal(),
                    "raster_shape": (profile["height"], profile["width"]),
                    "time_start": row["year_of_collection"],
                },
                lon=row["lon"],
                lat=row["lat"],
                aoi_id=row["aoi_id"],
                country=row["country"],
            )

            modality_samples.append(sample)

        taco_samples = tacotoolbox.tortilla.datamodel.Samples(samples=modality_samples)
        samples_path = os.path.join(tortilla_dir, f"sample_{idx}.tortilla")
        tacotoolbox.tortilla.create(taco_samples, samples_path, quiet=True, nworkers=4)

    # merge tortillas into a single dataset
    all_tortilla_files = sorted(glob.glob(os.path.join(tortilla_dir, "*.tortilla")))

    samples = []

    for idx, tortilla_file in tqdm(
        enumerate(all_tortilla_files),
        total=len(all_tortilla_files),
        desc="Building taco",
    ):
        sample_data = tacoreader.load(tortilla_file).iloc[0]

        sample_tortilla = tacotoolbox.tortilla.datamodel.Sample(
            id=os.path.basename(tortilla_file).split(".")[0],
            path=tortilla_file,
            file_format="TORTILLA",
            stac_data={
                "crs": sample_data["stac:crs"],
                "geotransform": sample_data["stac:geotransform"],
                "raster_shape": sample_data["stac:raster_shape"],
                "time_start": sample_data["stac:time_start"],
            },
            data_split=sample_data["tortilla:data_split"],
            lon=sample_data["lon"],
            lat=sample_data["lat"],
            aoi_id=sample_data["aoi_id"],
            country=sample_data["country"],
        )
        samples.append(sample_tortilla)

    # create final taco file
    final_samples = tacotoolbox.tortilla.datamodel.Samples(samples=samples)
    tacotoolbox.tortilla.create(
        final_samples,
        os.path.join(save_dir, "FullFOTW.tortilla"),
        quiet=True,
        nworkers=4,
    )


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

    orig_dataset = FieldsOfTheWorld(
        root=args.root, download=False, countries=CC_BY_COUNTRIES
    )

    metadata_path = os.path.join(args.save_dir, "metadata.parquet")
    if os.path.exists(metadata_path):
        metadata_df = pd.read_parquet(metadata_path)
    else:
        metadata_df = generate_metadata_df(orig_dataset)
        metadata_df.to_parquet(metadata_path)

    # assert that only CC_BY_COUNTRIES are present
    assert set(metadata_df["country"].unique()) == set(CC_BY_COUNTRIES)

    # validate_metadata_with_geo(metadata_df)

    # create_tortilla(args.root, metadata_df, args.save_dir)

    plot_country_distribution(
        metadata_df,
        output_path=os.path.join(args.save_dir, "country_distribution.png"),
        title="Fields of the World Dataset - Geographic Distribution",
    )

    # create unit test subset
    taco_glob = sorted(
        glob.glob(os.path.join(args.save_dir, "FullFOTW.*.part.tortilla"))
    )
    taco_ben = tacoreader.load(taco_glob)

    unit_test_taco = create_subset_from_tortilla(
        taco_ben, n_train_samples=4, n_val_samples=2, n_test_samples=2
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(script_dir))
    test_data_dir = os.path.join(repo_root, "tests", "data", "fotw")
    os.makedirs(test_data_dir, exist_ok=True)
    tacoreader.compile(
        dataframe=unit_test_taco, output=os.path.join(test_data_dir, "fotw.tortilla")
    )


if __name__ == "__main__":
    main()
