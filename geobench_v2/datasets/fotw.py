# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Fields of the World Dataset."""

from torch import Tensor
from torchgeo.datasets import SpaceNet6
from pathlib import Path
from typing import Sequence, Type, Literal
import torch.nn as nn
from shapely import wkt

from .sensor_util import DatasetBandRegistry
from .base import GeoBenchBaseDataset
from .data_util import MultiModalNormalizer
import torch.nn as nn
import rasterio
import numpy as np
import torch


class GeoBenchFieldsOfTheWorld(GeoBenchBaseDataset):
    """Fields of the World Dataset with enhanced functionality.

    Allows:
    - Variable Band Selection
    - Return band wavelengths
    """

    dataset_band_config = DatasetBandRegistry.FOTW

    # keys should be specified according to the sensor default values
    # defined in sensor_util.py
    band_default_order = ("r", "g", "b", "nir")

    # Define normalization stats using canonical names
    normalization_stats = {
        "means": {"r": 0.0, "g": 0.0, "b": 0.0, "nir": 0.0},
        "stds": {"r": 3000.0, "g": 3000.0, "b": 3000.0, "nir": 3000.0},
    }
    paths = [
        "FullFOTW.0000.part.tortilla",
        "FullFOTW.0001.part.tortilla",
        "FullFOTW.0002.part.tortilla",
        "FullFOTW.0003.part.tortilla",
    ]

    classes = ("background", "field", "field-boundary")
    num_classes = len(classes)

    valid_metadata = ("lat", "lon")

    # TODO maybe add country argument?
    def __init__(
        self,
        root: Path,
        split: str,
        band_order: Sequence[str | float] = dataset_band_config.default_order,
        data_normalizer: Type[nn.Module] = MultiModalNormalizer,
        label_type: Literal["instance_seg", "semantic_seg"] = "semantic_seg",
        transforms: nn.Module | None = None,
        metadata: Sequence[str] | None = None,
    ) -> None:
        """Initialize Fields of the World Dataset.

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
            metadata: metadata names to be returned under specified keys as part of the sample in the
                __getitem__ method. If None, no metadata is returned.
        """
        super().__init__(
            root=root,
            split=split,
            band_order=band_order,
            data_normalizer=data_normalizer,
            transforms=transforms,
            metadata=metadata,
        )

        self.label_type = label_type

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Return the image and mask at the given index.

        Args:
            idx: index of the image and mask to return

        Returns:
            dict: a dict containing the image and mask
        """
        sample: dict[str, Tensor] = {}

        sample: dict[str, Tensor] = {}

        sample_row = self.data_df.read(idx)

        win_a_path = sample_row.read(0)
        win_b_path = sample_row.read(1)

        mask_path = (
            sample_row.read(2)
            if self.label_type == "instance_seg"
            else sample_row.read(3)
        )

        with (
            rasterio.open(win_a_path) as win_a_src,
            rasterio.open(win_b_path) as win_b_src,
            rasterio.open(mask_path) as mask_src,
        ):
            win_a = win_a_src.read()
            win_b = win_b_src.read()
            mask = mask_src.read(1)

        win_a = torch.from_numpy(win_a).float()
        win_b = torch.from_numpy(win_b).float()
        mask = torch.from_numpy(mask).long()

        # TODO how to handle window a and b?
        win_a = self.rearrange_bands(win_a, self.band_order)
        win_a = self.data_normalizer(win_a)

        sample.update(win_a)

        sample["mask"] = mask

        point = wkt.loads(sample_row.iloc[0]["stac:centroid"])
        lon, lat = point.x, point.y

        if "lon" in self.metadata:
            sample["lon"] = torch.tensor(lon)
        if "lat" in self.metadata:
            sample["lat"] = torch.tensor(lat)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
