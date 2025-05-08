# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Generate FLOGA benchmark."""

"""
This script creates an analysis-ready dataset from the downloaded FLOGA imagery.
"""

import argparse
from itertools import product
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class font_colors:
    PURPLE = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    ENDC = "\033[0m"


SEN2_BORDER = 65535
NODATA_VALUES = {"SEN2": 0, "MOD": -28672}


def get_padding_offset(img_size, out_size):
    img_size_x = img_size[0]
    img_size_y = img_size[1]

    output_size_x = out_size[0]
    output_size_y = out_size[1]

    if img_size_x >= output_size_x:
        pad_x = int(output_size_x - img_size_x % output_size_x)
    else:
        pad_x = output_size_x - img_size_x

    if img_size_y >= output_size_y:
        pad_y = int(output_size_y - img_size_y % output_size_y)
    else:
        pad_y = output_size_y - img_size_y

    if not pad_x == output_size_x:
        pad_top = int(pad_x // 2)
        pad_bot = int(pad_x // 2)

        if not pad_x % 2 == 0:
            pad_top += 1
    else:
        pad_top = 0
        pad_bot = 0

    if not pad_y == output_size_y:
        pad_left = int(pad_y // 2)
        pad_right = int(pad_y // 2)

        if not pad_y % 2 == 0:
            pad_left += 1
    else:
        pad_left = 0
        pad_right = 0

    return pad_top, pad_bot, pad_left, pad_right


def pad_image(img, out_size, pad_top, pad_bot, pad_left, pad_right):
    if img.ndim == 2:
        img = img[None, :, :]

    padding = ((pad_top, pad_bot), (pad_left, pad_right), (0, 0))[: img.ndim]

    try:
        img = np.pad(img, padding, mode="reflect")
    except Exception as e:
        print(f"Padding error: {e}")
        img = np.pad(img, padding, mode="constant")

    return img


def export_patches(floga_path, out_path, out_size, sea_ratio):
    cloudy_patches = []
    other_burnt_areas_patches = []
    sea_patches = []
    recorded = []

    hdf_files = list(floga_path.glob("*.h5"))

    with tqdm(initial=0, total=len(hdf_files)) as pbar:
        for hdf_file_i, hdf_file in enumerate(hdf_files):
            pbar.set_description(f"({hdf_file_i + 1}/{len(hdf_files)}) {hdf_file.name}")

            hdf = h5py.File(hdf_file, "r")
            year, _, sen_gsd, _, _ = hdf_file.stem.split("_")[2:]

            image_names = [
                "clc_100_mask",
                "label",
                "mod_500_cloud_post",
                "mod_500_cloud_pre",
                "mod_500_post",
                "mod_500_pre",
                "sea_mask",
                f"sen2_{sen_gsd}_post",
                f"sen2_{sen_gsd}_pre",
            ]

            if sen_gsd in ["20", "60"]:
                image_names += [
                    f"sen2_{sen_gsd}_cloud_post",
                    f"sen2_{sen_gsd}_cloud_pre",
                ]

            out_path_hdf = out_path / year
            out_path_hdf.mkdir(parents=True, exist_ok=True)

            for event_id, event_imgs in hdf[year].items():
                img = event_imgs["label"][:].squeeze()

                padding_offsets = get_padding_offset(img.shape, out_size)

                if out_size is not None:
                    x_idx = list(
                        range(
                            0,
                            img.shape[0] + padding_offsets[0] + padding_offsets[1],
                            out_size[0],
                        )
                    )
                    y_idx = list(
                        range(
                            0,
                            img.shape[1] + padding_offsets[2] + padding_offsets[3],
                            out_size[1],
                        )
                    )
                else:
                    x_idx = [0]
                    y_idx = [0]
                    out_size = [img.shape[0], img.shape[1]]

                crop_indices = [x_idx, y_idx]

                patch_data_dict = {}

                for img_name in image_names:
                    img = event_imgs[img_name][:]

                    if img.ndim == 2:
                        img = img[None, :, :]

                    if ("label" not in img_name) and ("mask" not in img_name):
                        if "sen2" in img_name:
                            img[img == NODATA_VALUES["SEN2"]] = 0
                        elif "mod" in img_name:
                            img[img == NODATA_VALUES["MOD"]] = 0
                    elif img_name == "sea_mask":
                        img[(img == 2) | (img == 4)] = 0
                        img[img != 0] = 1

                    img = pad_image(img, out_size, *padding_offsets)

                    patch_offset = 0
                    for x, y in product(*crop_indices):
                        patch_key = f"sample{(patch_offset):08d}_{event_id}_{year}"

                        if patch_key not in patch_data_dict:
                            patch_data_dict[patch_key] = {}

                        patch = img[:, x : x + out_size[0], y : y + out_size[1]]
                        patch_data_dict[patch_key][img_name] = patch
                        patch_offset += 1

                is_positive = bool(np.max(patch_data_dict[patch_key]["label"]))

                if is_positive:
                    positive_flag = 1
                else:
                    positive_flag = 0

                for patch_key, patch_data in patch_data_dict.items():
                    out_file = out_path_hdf / f"{patch_key}.h5"

                    with h5py.File(out_file, "w") as h5f:
                        for k, v in patch_data.items():
                            h5f.create_dataset(k, data=v, compression="gzip")
                        h5f.attrs["positive_flag"] = positive_flag

            hdf.close()
            pbar.update(1)


def export_csv_with_patch_paths(
    events_list,
    out_path,
    mode,
    random_seed,
    split_mode,
    ratio,
    train_years,
    test_years,
    suffix=None,
    sampling=None,
):
    if suffix is None:
        suffix = ""
    else:
        suffix = f"_{suffix}"

    patch_keys = []
    hdf5_paths = []
    positive_flags = []

    for event in events_list:
        event_id, year = event.split("_")

        event_patches = list((out_path / f"{year}").glob(f"*_{event_id}_{year}.h5"))

        for patch_file in event_patches:
            patch_key = patch_file.stem

            with h5py.File(patch_file, "r") as h5f:
                is_positive = bool(h5f.attrs.get("positive_flag", False))

            patch_keys.append(patch_key)
            hdf5_paths.append(str(patch_file))
            positive_flags.append(is_positive)

    df = pd.DataFrame(
        {
            "patch_key": patch_keys,
            "hdf5_path": hdf5_paths,
            "positive_flag": positive_flags,
        }
    )

    if sampling is not None:
        df = sample_patches_df(sampling, df, random_seed)
        sampling_str = f"r{sampling}"
    else:
        sampling_str = ""

    ratio_str = "-".join([str(i) for i in ratio])

    if split_mode == "year":
        output_file = (
            out_path
            / f"{''.join(train_years)}_{''.join(test_years)}_{ratio_str}_{sampling_str}{suffix}_{mode}.parquet"
        )
        df.to_parquet(output_file, index=False)
    else:
        output_file = (
            out_path / f"allEvents_{ratio_str}_{sampling_str}{suffix}_{mode}.parquet"
        )
        df.to_parquet(output_file, index=False)

    print(f"Saved {len(df)} samples to {output_file}")


def sample_patches_df(neg_ratio, df, random_seed):
    positives = df[df["positive_flag"] == True]
    negatives = df[df["positive_flag"] == False]

    num_negatives = len(positives) * neg_ratio

    if len(positives) >= len(negatives):
        return df

    rng = np.random.default_rng(random_seed)
    if num_negatives < len(negatives):
        negative_indices = negatives.index.to_numpy()
        selected_indices = rng.choice(
            negative_indices, size=int(num_negatives), replace=False
        )

        sampled_negatives = df.loc[selected_indices]
        return pd.concat([positives, sampled_negatives])
    else:
        return df


def export_unified_parquet(
    events_lists,
    out_path,
    random_seed,
    split_mode,
    ratio,
    train_years,
    test_years,
    suffix=None,
    sampling=None,
):
    if suffix is None:
        suffix = ""
    else:
        suffix = f"_{suffix}"
    patch_keys = []
    filepaths = []
    positive_flags = []
    splits = []

    split_names = ["train", "val", "test"]

    for split_idx, events_list in enumerate(events_lists):
        split_name = split_names[split_idx]
        print(f"Processing {len(events_list)} events for {split_name} split...")

        for event in events_list:
            event_id, year = event.split("_")

            event_patches = list((out_path / f"{year}").glob(f"*_{event_id}_{year}.h5"))

            for patch_file in event_patches:
                patch_key = patch_file.stem

                with h5py.File(patch_file, "r") as h5f:
                    is_positive = bool(h5f.attrs.get("positive_flag", False))

                patch_keys.append(patch_key)
                filepaths.append(str(patch_file).replace(f"{out_path}/", ""))
                positive_flags.append(is_positive)
                splits.append(split_name)

    df = pd.DataFrame(
        {
            "patch_key": patch_keys,
            "year": [int(k.split("_")[-1]) for k in patch_keys],
            "filepath": filepaths,
            "positive_flag": positive_flags,
            "split": splits,
        }
    )

    if sampling is not None:
        sampled_dfs = []
        for split_name in split_names:
            split_df = df[df["split"] == split_name]
            sampled_split_df = sample_patches_df(sampling, split_df, random_seed)
            sampled_dfs.append(sampled_split_df)

        df = pd.concat(sampled_dfs)
        sampling_str = f"r{sampling}"
    else:
        sampling_str = ""

    ratio_str = "-".join([str(i) for i in ratio])

    if split_mode == "year":
        output_file = (
            out_path
            / f"{''.join(train_years)}_{''.join(test_years)}_{ratio_str}_{sampling_str}{suffix}_all.parquet"
        )
    else:
        output_file = (
            out_path / f"allEvents_{ratio_str}_{sampling_str}{suffix}_all.parquet"
        )

    df.to_parquet(output_file, index=False)


def sample_patches(neg_ratio, patches, random_seed):
    rng = np.random.default_rng(random_seed)
    positives = [p for p in patches if p[1]]
    negatives = [p for p in patches if not p[1]]

    num_negatives = len(positives) * neg_ratio

    if len(positives) >= len(negatives):
        return patches

    if num_negatives < len(negatives):
        selected_negatives = rng.choice(
            negatives, size=int(num_negatives), replace=False
        )
        return np.concatenate([positives, selected_negatives])
    else:
        return patches


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create FLOGA dataset.")
    parser.add_argument(
        "--floga_path",
        type=str,
        required=True,
        help="Path to the downloaded FLOGA imagery.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Path to save the processed dataset.",
    )
    parser.add_argument(
        "--out_size",
        type=int,
        nargs=2,
        required=True,
        help="Size of the output patches (height width).",
    )
    parser.add_argument(
        "--sea_ratio", type=float, default=0.1, help="Ratio of sea patches to include."
    )
    parser.add_argument(
        "--split_mode",
        type=str,
        default="random",
        choices=["random", "year"],
        help="Split mode: random or by year.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        nargs=3,
        default=[0.7, 0.15, 0.15],
        help="Train/val/test split ratio.",
    )
    parser.add_argument(
        "--train_years",
        type=int,
        nargs="+",
        default=[2018, 2019, 2020],
        help="Years to use for training.",
    )
    parser.add_argument(
        "--test_years",
        type=int,
        nargs="+",
        default=[2021],
        help="Years to use for testing.",
    )
    parser.add_argument(
        "--random_seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--sample", type=int, default=None, help="Negative sampling ratio (1:X)."
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="Suffix to add to the output file name.",
    )
    args = parser.parse_args()

    floga_path = Path(args.floga_path)
    out_path = Path(args.out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    export_patches(floga_path, out_path, args.out_size, args.sea_ratio)

    events = [
        f.stem.split("_")[1] + "_" + f.stem.split("_")[2]
        for f in floga_path.glob("*.h5")
    ]
    train_events, test_events = train_test_split(
        events, test_size=sum(args.ratio[1:]), random_state=args.random_seed
    )
    val_events, test_events = train_test_split(
        test_events,
        test_size=args.ratio[2] / sum(args.ratio[1:]),
        random_state=args.random_seed,
    )

    train_events_list = train_events
    val_events_list = val_events
    test_events_list = test_events

    export_unified_parquet(
        [train_events_list, val_events_list, test_events_list],
        out_path,
        args.random_seed,
        args.split_mode,
        args.ratio,
        args.train_years,
        args.test_years,
        suffix=args.suffix,
        sampling=args.sample,
    )
