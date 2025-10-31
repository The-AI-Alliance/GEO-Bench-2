# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""SpaceNet6 dataset."""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

import numpy as np
import rasterio
import torch
import torch.nn as nn
from shapely import wkt
from torch import Tensor

from .base import GeoBenchBaseDataset
from .normalization import ZScoreNormalizer
from .sensor_util import DatasetBandRegistry


class GeoBenchSpaceNet6(GeoBenchBaseDataset):
    """GeoBench version of SpaceNet6 dataset."""

    url = "https://hf.co/datasets/aialliance/spacenet6/resolve/main/{}"

    paths = [
        "geobench_spacenet6.0000.part.tortilla",
        "geobench_spacenet6.0001.part.tortilla",
    ]

    sha256str = ["", "", ""]

    dataset_band_config = DatasetBandRegistry.SPACENET6

    band_default_order = {
        "rgbn": ("red", "green", "blue", "nir"),
        "sar": ("hh", "hv", "vv", "vh"),
    }

    normalization_stats: dict[str, dict[str, float]] = {
        "means": {
            "red": 101.56404876708984,
            "green": 140.59695434570312,
            "blue": 146.70387268066406,
            "nir": 340.8776550292969,
            "hh": 24.750904083251953,
            "hv": 31.68429183959961,
            "vv": 29.68717384338379,
            "vh": 22.68701171875,
        },
        "stds": {
            "red": 109.73048400878906,
            "green": 124.5447998046875,
            "blue": 149.98680114746094,
            "nir": 297.4772033691406,
            "hh": 12.217103004455566,
            "hv": 14.078553199768066,
            "vv": 13.503046035766602,
            "vh": 11.729385375976562,
        },
    }

    classes = ("background", "no-building", "building")

    num_classes = len(classes)

    valid_metadata = ("lat", "lon")

    def __init__(
        self,
        root: Path,
        split: Literal["train", "val", "validation", "test"],
        rename_modalities: dict | None = None,
        band_order: Sequence[str] = band_default_order,
        data_normalizer: type[nn.Module] = ZScoreNormalizer,
        transforms: nn.Module | None = None,
        return_stacked_image: bool = False,
        metadata: Sequence[str] | None = None,
        download: bool = False,
    ) -> None:
        """Initialize SpaceNet6 dataset.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'validation', 'test'
            band_order: The order of bands to return, defaults to ['red', 'green', 'blue', 'nir'], if one would
                specify ['red', 'green', 'blue', 'nir', 'nir'], the dataset would return images with 5 channels
                in that order. This is useful for models that expect a certain band order, or
                test the impact of band order on model performance.
            data_normalizer: The data normalizer to apply to the data, defaults to :class:`data_util.ZScoreNormalizer`,
                which applies z-score normalization to each band.
            transforms: image transformations to apply to the data, defaults to None
            metadata: metadata names to be returned as part of the sample in the
            return_stacked_image: if true, returns a single image tensor with all modalities stacked in band_order
            rename_modalities: dictionary with information to rename modalities in output e.g. {image: {sar:  S1RTC, rgbn: S2L2A}}
            transforms: image transformations to apply to the data, defaults to None
            download: Whether to download the dataset
        """
        split_norm: Literal["train", "validation", "test"]
        if split == "val":
            split_norm = "validation"
        else:
            split_norm = cast(Literal["train", "validation", "test"], split)
        super().__init__(
            root=root,
            split=split_norm,
            band_order=band_order,
            data_normalizer=data_normalizer,
            transforms=transforms,
            metadata=metadata,
            download=download,
        )
        if return_stacked_image:
            assert rename_modalities is None, (
                "Cannot return a stacked image if modalities are renamed"
            )
        self.return_stacked_image = return_stacked_image
        self.rename_modalities = rename_modalities

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        sample: dict[str, Tensor] = {}

        sample_row = self.data_df.read(index)

        ps_rgbn_path = sample_row.read(0)
        sar_intensity_path = sample_row.read(1)
        mask_path = sample_row.read(2)

        img_dict: dict[str, Tensor] = {}
        if "rgbn" in self.band_order:
            with rasterio.open(ps_rgbn_path) as src:
                rgbn_img = src.read()
                # if all values across channels are 0, get mask
                masked_no_data = np.all(rgbn_img == 0, axis=0)

            rgbn_img = torch.from_numpy(rgbn_img).float()
            img_dict["rgbn"] = rgbn_img
        else:
            rgbn_img = None
            masked_no_data = None

        if "sar" in self.band_order:
            with rasterio.open(sar_intensity_path) as src:
                sar_img = src.read()
            sar_img = torch.from_numpy(sar_img).float()
            img_dict["sar"] = sar_img
        else:
            sar_img = None

        img_dict = self.rearrange_bands(img_dict, self.band_order)
        image_dict = self.data_normalizer(img_dict)

        with rasterio.open(mask_path) as src:
            mask = src.read()

        # We add 1 to the mask to map the current {background, building} labels to have a
        # true background class.
        mask = torch.from_numpy(mask).long().squeeze(0) + 1

        if masked_no_data is not None:
            # if all values across channels are 0, set mask to 0
            mask[masked_no_data] = 0

        sample.update(image_dict)
        sample["mask"] = mask

        if self.return_stacked_image:
            sample = {
                "image": torch.cat(
                    [sample[f"image_{key}"] for key in self.band_order.keys()], 0
                ),
                "mask": sample["mask"],
            }

        if self.transforms is not None:
            sample = self.transforms(sample)

        if self.rename_modalities is not None:
            for key, value in self.rename_modalities.items():
                if isinstance(value, dict):
                    sample[key] = {}
                    for old_sub_key in value:
                        if old_sub_key in self.band_order:
                            new_sub_key = value[old_sub_key]
                            sample[key][new_sub_key] = sample[f"image_{old_sub_key}"]
                            del sample[f"image_{old_sub_key}"]
                else:
                    if key in self.band_order:
                        new_sub_key = value
                        sample[new_sub_key] = sample[f"image_{key}"]
                        del sample[f"image_{key}"]
                    else:
                        raise ValueError(
                            "rename_modalities must include names that exist in the dataset"
                        )

        point = wkt.loads(sample_row.iloc[0]["stac:centroid"])
        lon, lat = point.x, point.y

        if "lon" in self.metadata:
            sample["lon"] = torch.tensor(lon)
        if "lat" in self.metadata:
            sample["lat"] = torch.tensor(lat)

        return sample
