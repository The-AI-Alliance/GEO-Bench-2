# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""SpaceNet6 dataset."""

from torch import Tensor
from torchgeo.datasets import SpaceNet6
from pathlib import Path
from typing import Sequence, Type
import torch.nn as nn

from .sensor_util import DatasetBandRegistry
from .base import GeoBenchBaseDataset
from .data_util import MultiModalNormalizer
import torch.nn as nn
import rasterio
import numpy as np
import torch


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

    paths = [
        "SpaceNet6.0000.part.tortilla",
        "SpaceNet6.0001.part.tortilla",
        "SpaceNet6.0002.part.tortilla",
    ]

    classes = ("background", "no_building", "building")

    num_classes = len(classes)

    def __init__(
        self,
        root: Path,
        split: str,
        band_order: Sequence[str] = band_default_order,
        data_normalizer: Type[nn.Module] = MultiModalNormalizer,
        transforms: nn.Module | None = None,
        return_stacked_image: bool = True,
        **kwargs,
    ) -> None:
        """Initialize SpaceNet6 dataset.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'val', 'test'
            band_order: The order of bands to return, defaults to ['red', 'green', 'blue', 'nir'], if one would
                specify ['red', 'green', 'blue', 'nir', 'nir'], the dataset would return images with 5 channels
                in that order. This is useful for models that expect a certain band order, or
                test the impact of band order on model performance.
            data_normalizer: The data normalizer to apply to the data, defaults to :class:`data_util.MultiModalNormalizer`,
                which applies z-score normalization to each band.
            transforms:
            return_stacked_image: if true, returns a single image tensor with all modalities stacked in band_order
            **kwargs: Additional keyword arguments passed to ``torchgeo.datasets.SpaceNet6``
        """
        super().__init__(
            root=root,
            split=split,
            band_order=band_order,
            data_normalizer=data_normalizer,
            transforms=transforms,
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

            rgbn_img = torch.from_numpy(rgbn_img).float()
            img_dict["rgbn"] = rgbn_img
        else:
            rgbn_img = None

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

        # We add 1 to the mask to map the current {background, building} labels to
        # the values {1, 2}. This is necessary because we add 0 padding to the
        # mask that we want to ignore in the loss function.
        mask = torch.from_numpy(mask).long() + 1

        sample.update(image_dict)
        sample["mask"] = mask

        if self.transforms is not None:
            sample = self.transforms(sample)

        if self.return_stacked_image:
            stacked_image = []
            for mod in self.band_order:
                if mod == "rgbn":
                    stacked_image.append(sample["image_rgbn"])
                if mod == "sar":
                    stacked_image.append(sample["image_sar"])
            output = {}
            output["image"] = torch.cat(stacked_image, 0)
            output["mask"] = sample["mask"]
        else:
            output = sample 
        return output