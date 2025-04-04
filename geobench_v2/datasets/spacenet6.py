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
            "r": 0.0,
            "g": 0.0,
            "b": 0.0,
            "nir": 0.0,
            "hh": 0.0,
            "hv": 0.0,
            "vv": 0.0,
            "vh": 0.0,
        },
        "stds": {
            "r": 1000.0,
            "g": 1000.0,
            "b": 1000.0,
            "nir": 1000.0,
            "hh": 100.0,
            "hv": 100.0,
            "vv": 100.0,
            "vh": 100.0,
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
            **kwargs: Additional keyword arguments passed to ``torchgeo.datasets.SpaceNet6``
        """
        super().__init__(
            root=root,
            split=split,
            band_order=band_order,
            data_normalizer=data_normalizer,
            transforms=transforms,
        )

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


        stacked_image = []
        for mod in self.band_order:
            if mod == "rgbn":
                stacked_image.append(sample["image_rgbn"])
            if mod == "sar":
                stacked_image.append(sample["image_sar"])
        output = {}
        output["image"] = torch.cat(stacked_image, 0)
        output["mask"] = sample["mask"]

        return output