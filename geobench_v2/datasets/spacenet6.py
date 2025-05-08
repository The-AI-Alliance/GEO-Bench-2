# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""SpaceNet6 dataset."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
from shapely import wkt
from torch import Tensor

from .base import GeoBenchBaseDataset
from .data_util import ClipZScoreNormalizer
from .sensor_util import DatasetBandRegistry


class GeoBenchSpaceNet6(GeoBenchBaseDataset):
    """SpaceNet6 dataset with enhanced functionality.

    Allows:
    - Variable Band Selection
    - Return band wavelengths

    Classes are:

    0. Background
    1. No Building
    2. Building
    """

    url = "https://hf.co/datasets/aialliance/spacenet6/resolve/main/{}"
    # paths = [
    #     "SpaceNet6.0000.part.tortilla",
    #     "SpaceNet6.0001.part.tortilla",
    #     "SpaceNet6.0002.part.tortilla",
    # ]
    paths = [
        "geobench_spacenet6.0000.part.tortilla",
        "geobench_spacenet6.0001.part.tortilla",
    ]

    sha256str = ["", "", ""]

    dataset_band_config = DatasetBandRegistry.SPACENET6

    band_default_order = {
        "rgbn": ("r", "g", "b", "nir"),
        "sar": ("hh", "hv", "vv", "vh"),
    }

    normalization_stats = {
        "means": {
            "r": 101.56404876708984,
            "g": 140.59695434570312,
            "b": 146.70387268066406,
            "nir": 340.8776550292969,
            "hh": 24.750904083251953,
            "hv": 31.68429183959961,
            "vv": 29.68717384338379,
            "vh": 22.68701171875,
        },
        "stds": {
            "r": 109.73048400878906,
            "g": 124.5447998046875,
            "b": 149.98680114746094,
            "nir": 297.4772033691406,
            "hh": 12.217103004455566,
            "hv": 14.078553199768066,
            "vv": 13.503046035766602,
            "vh": 11.729385375976562
        },
    }

    classes = ("background", "no-building", "building")

    num_classes = len(classes)

    valid_metadata = ("lat", "lon")

    def __init__(
        self,
        root: Path,
        split: str,
        band_order: Sequence[str] = band_default_order,
        data_normalizer: type[nn.Module] = ClipZScoreNormalizer,
        transforms: nn.Module | None = None,
        return_stacked_image: bool = False,
        metadata: Sequence[str] | None = None,
        download: bool = False,
    ) -> None:
        """Initialize SpaceNet6 dataset.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'val', 'test'
            band_order: The order of bands to return, defaults to ['red', 'green', 'blue', 'nir'], if one would
                specify ['red', 'green', 'blue', 'nir', 'nir'], the dataset would return images with 5 channels
                in that order. This is useful for models that expect a certain band order, or
                test the impact of band order on model performance.
            data_normalizer: The data normalizer to apply to the data, defaults to :class:`data_util.ClipZScoreNormalizer`,
                which applies z-score normalization to each band.
            transforms:
            metadata: metadata names to be returned as part of the sample in the
            return_stacked_image: if true, returns a single image tensor with all modalities stacked in band_order
        """
        super().__init__(
            root=root,
            split=split,
            band_order=band_order,
            data_normalizer=data_normalizer,
            transforms=transforms,
            metadata=metadata,
            download=download,
        )

        self.return_stacked_image = return_stacked_image

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

        point = wkt.loads(sample_row.iloc[0]["stac:centroid"])
        lon, lat = point.x, point.y

        if "lon" in self.metadata:
            sample["lon"] = torch.tensor(lon)
        if "lat" in self.metadata:
            sample["lat"] = torch.tensor(lat)

        return sample
