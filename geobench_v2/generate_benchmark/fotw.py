# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate GeoBenchV2 version of Fields of the World dataset."""

import argparse
import concurrent
import glob
import json
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import shapely.wkb
import tacoreader
import tacotoolbox
from matplotlib.lines import Line2D
from torchgeo.datasets import FieldsOfTheWorld
from tqdm import tqdm

from geobench_v2.generate_benchmark.utils import (
    create_subset_from_df,
    create_unittest_subset,
)

CC_BY_COUNTRIES = (
    "austria",
    "brazil",
    "corsica",
    "denmark",
    "estonia",
    "finland",
    "france",
    "india",
    "luxembourg",
    "netherlands",
    "rwanda",
    "slovakia",
    "spain",
    "vietnam",
)


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

        with open(f"{ds.root}/{country}/data_config_{country}.json") as f:
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

    overall_df = overall_df.drop(
        columns=[
            "win_a_exists",
            "win_b_exists",
            "instance_mask_exists",
            "semantic_3class_mask_exists",
        ]
    ).reset_index(drop=True)

    overall_df["split"] = overall_df["split"].replace("val", "validation")

    return overall_df


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
    print("Top 5 countries by sample count:")
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
    split_colors = {"train": "#1f77b4", "validation": "#ff7f0e", "test": "#2ca02c"}

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

        for split in ["train", "validation", "test"]:
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
    for split in ["train", "validation", "test"]:
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


def process_fotw_sample(args):
    """Process a single Fields of the World sample by rewriting with optimized profile."""
    idx, row, root_dir, save_dir = args

    try:
        country = row["country"]
        aoi_id = row["aoi_id"]

        for subdir in ["win_a", "win_b", "instance_mask", "semantic_3class_mask"]:
            os.makedirs(os.path.join(save_dir, subdir), exist_ok=True)

        result = {}
        modalities = {
            "win_a": row["win_a_path"],
            "win_b": row["win_b_path"],
            "instance_mask": row["instance_mask_path"],
            "semantic_3class_mask": row["semantic_3class_mask_path"],
        }

        for modality, rel_path in modalities.items():
            src_path = os.path.join(root_dir, rel_path)
            if not os.path.exists(src_path):
                continue

            dst_filename = f"{country}_{aoi_id}_{modality}.tif"
            dst_path = os.path.join(save_dir, modality, dst_filename)

            with rasterio.open(src_path) as src:
                data = src.read()

                if modality == "instance_mask" and data.dtype not in [
                    np.uint8,
                    np.int8,
                ]:
                    data = data.astype(np.uint8)

                optimized_profile = {
                    "driver": "GTiff",
                    "height": data.shape[1],
                    "width": data.shape[2],
                    "count": data.shape[0],
                    "dtype": data.dtype,
                    "tiled": True,
                    "blockxsize": 256,
                    "blockysize": 256,
                    "interleave": "pixel",
                    "compress": "zstd",
                    "zstd_level": 13,
                    "predictor": 2,
                    "crs": src.crs,
                    "transform": src.transform,
                }

                with rasterio.open(dst_path, "w", **optimized_profile) as dst:
                    dst.write(data)

            result[f"{modality}_path"] = os.path.relpath(dst_path, start=save_dir)

        result["country"] = country
        result["aoi_id"] = row["aoi_id"]
        result["split"] = row["split"]
        result["year_of_collection"] = row["year_of_collection"]
        result["lon"] = row["lon"]
        result["lat"] = row["lat"]

        return result

    except Exception as e:
        print(
            f"Error processing sample {idx} (country: {row.get('country', '')}, aoi_id: {row.get('aoi_id', '')}): {e}"
        )
        import traceback

        traceback.print_exc()
        return None


def optimize_fotw_dataset(metadata_df, root_dir, save_dir, num_workers=1):
    """Store FOTW dataset with optimized GeoTIFF profiles."""
    os.makedirs(save_dir, exist_ok=True)

    tasks = [(idx, row, root_dir, save_dir) for idx, row in metadata_df.iterrows()]
    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        list_of_results = list(
            tqdm(
                executor.map(process_fotw_sample, tasks),
                total=len(tasks),
                desc="Optimizing FOTW samples",
            )
        )

        for result in list_of_results:
            if result is not None:
                results.append(result)

    optimized_df = pd.DataFrame(results)

    return optimized_df


def create_tortilla(root_dir, df, save_dir, tortilla_name):
    """Create a tortilla version of the dataset."""
    tortilla_dir = os.path.join(save_dir, "tortilla")
    os.makedirs(tortilla_dir, exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating tortilla"):
        modalities = ["win_a", "win_b", "instance_mask", "semantic_3class_mask"]
        modality_samples = []

        for modality in modalities:
            path = os.path.join(
                root_dir,
                modality,
                row["country"]
                + "_"
                + os.path.basename(row[modality + "_path"]).replace(".tif", "")
                + "_"
                + modality
                + ".tif",
            )

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
                add_test_split=row["is_additional_test"],
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
            add_test_split=sample_data["add_test_split"],
            lon=sample_data["lon"],
            lat=sample_data["lat"],
            aoi_id=sample_data["aoi_id"],
            country=sample_data["country"],
        )
        samples.append(sample_tortilla)

    # create final taco file
    final_samples = tacotoolbox.tortilla.datamodel.Samples(samples=samples)
    tacotoolbox.tortilla.create(
        final_samples, os.path.join(save_dir, tortilla_name), quiet=True, nworkers=4
    )


def create_geobench_version(
    metadata_df: pd.DataFrame,
    n_train_samples: int,
    n_val_samples: int,
    n_test_samples: int,
    n_additional_test_samples: int = 0,
) -> None:
    """Create a GeoBench version of the dataset.

    Args:
        metadata_df: DataFrame with metadata including geolocation for each patch
        n_train_samples: Number of final training samples, -1 means all
        n_val_samples: Number of final validation samples, -1 means all
        n_test_samples: Number of final test samples, -1 means all
        n_additional_test_samples: Number of additional test samples from train split
    """
    random_state = 24

    subset_df = create_subset_from_df(
        metadata_df,
        n_train_samples=n_train_samples,
        n_val_samples=n_val_samples,
        n_test_samples=n_test_samples,
        n_additional_test_samples=n_additional_test_samples,
        random_state=random_state,
    )

    return subset_df


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

    assert set(metadata_df["country"].unique()) == set(CC_BY_COUNTRIES)

    plot_country_distribution(
        metadata_df,
        output_path=os.path.join(args.save_dir, "country_distribution.png"),
        title="Fields of the World Dataset - Geographic Distribution",
    )

    results_df_path = os.path.join(args.save_dir, "geobench_fotw.parquet")
    if os.path.exists(results_df_path):
        results_df = pd.read_parquet(results_df_path)
    else:
        results_df = create_geobench_version(
            metadata_df,
            n_train_samples=4000,
            n_val_samples=1000,
            n_test_samples=2000,
            n_additional_test_samples=0,
        )
        results_df.to_parquet(results_df_path)

    optimized_path = os.path.join(args.save_dir, "optimized.parquet")
    if os.path.exists(optimized_path):
        optimized_df = pd.read_parquet(optimized_path)
    else:
        optimized_df = optimize_fotw_dataset(
            results_df,
            root_dir=args.root,
            save_dir=os.path.join(args.save_dir, "optimized"),
            num_workers=8,
        )
        optimized_df.to_parquet(optimized_path)

    tortilla_name = "geobench_fotw.tortilla"
    create_tortilla(
        root_dir=os.path.join(args.save_dir, "optimized"),
        df=results_df,
        save_dir=args.save_dir,
        tortilla_name=tortilla_name,
    )

    create_unittest_subset(
        data_dir=args.save_dir,
        tortilla_pattern=tortilla_name,
        test_dir_name="fotw",
        n_train_samples=4,
        n_val_samples=2,
        n_test_samples=2,
        n_additional_test_samples=0,
    )


if __name__ == "__main__":
    main()
