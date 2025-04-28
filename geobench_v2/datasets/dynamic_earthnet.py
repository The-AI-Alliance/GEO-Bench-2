# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""DynamicEarthNet Dataset."""

from torch import Tensor
from pathlib import Path
import numpy as np
from typing import Any, Sequence, Union, Type, Literal
import torch
import os
import json
import pandas as pd
import torch.nn as nn
import rasterio
from shapely import wkt

from .base import GeoBenchBaseDataset

from .sensor_util import DatasetBandRegistry
from .data_util import DataUtilsMixin, MultiModalNormalizer


class GeoBenchDynamicEarthNet(GeoBenchBaseDataset):
    """DynamicEarthNet dataset."""

    dataset_band_config = DatasetBandRegistry.DYNAMICEARTHNET

    band_default_order = {
        "planet": ("b", "g", "r", "nir"),
        "s2": (
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B10",
            "B11",
            "B12",
        ),
    }

    # Appendix C normalization stats for planet
    # mean = [1042.59, 915.62, 671.26, 2605.21] and
    # std = [957.96, 715.55, 596.94, 1059.90],
    # https://github.com/aysim/dynnet/blob/1e7d90294b54f52744ae2b35db10b4d0a48d093d/data/utae_dynamicen.py#L13
    # TODO check
    normalization_stats = {
        "means": {
            "r": 0.0,
            "g": 0.0,
            "b": 0.0,
            "nir": 0.0,
            "B01": 0.0,
            "B02": 0.0,
            "B03": 0.0,
            "B04": 0.0,
            "B05": 0.0,
            "B06": 0.0,
            "B07": 0.0,
            "B08": 0.0,
            "B8A": 0.0,
            "B10": 0.0,
            "B11": 0.0,
            "B12": 0.0,
        },
        "stds": {
            "r": 3000.0,
            "g": 3000.0,
            "b": 3000.0,
            "nir": 3000.0,
            "B01": 3000.0,
            "B02": 3000.0,
            "B03": 3000.0,
            "B04": 3000.0,
            "B05": 3000.0,
            "B06": 3000.0,
            "B07": 3000.0,
            "B08": 3000.0,
            "B8A": 3000.0,
            "B10": 3000.0,
            "B11": 3000.0,
            "B12": 3000.0,
        },
    }

    # paths = [
    #     "FullDynamicEarthNet.0000.part.tortilla",
    #     "FullDynamicEarthNet.0001.part.tortilla",
    #     "FullDynamicEarthNet.0002.part.tortilla",
    #     "FullDynamicEarthNet.0003.part.tortilla",
    #     "FullDynamicEarthNet.0004.part.tortilla",
    #     "FullDynamicEarthNet.0005.part.tortilla",
    #     "FullDynamicEarthNet.0006.part.tortilla",
    #     "FullDynamicEarthNet.0007.part.tortilla",
    #     "FullDynamicEarthNet.0008.part.tortilla",
    #     "FullDynamicEarthNet.0009.part.tortilla",
    #     "FullDynamicEarthNet.0010.part.tortilla",
    #     "FullDynamicEarthNet.0011.part.tortilla",
    #     "FullDynamicEarthNet.0012.part.tortilla",
    #     "FullDynamicEarthNet.0013.part.tortilla",
    #     "FullDynamicEarthNet.0014.part.tortilla",
    #     "FullDynamicEarthNet.0015.part.tortilla",
    #     "FullDynamicEarthNet.0016.part.tortilla",
    #     "FullDynamicEarthNet.0017.part.tortilla",
    #     "FullDynamicEarthNet.0018.part.tortilla",
    # ]

    paths = [
        "geobench_dynamic_earthnet.0000.part.tortilla",
        "geobench_dynamic_earthnet.0001.part.tortilla",
        "geobench_dynamic_earthnet.0002.part.tortilla",
    ]

    # temporal setting described in A.3 of the paper
    # weekly 1st, 5th, 10th, 15th, 20th and 25th time steps
    # daily returns all available days between 28 and 30 days
    # single returns the 30th day

    # new order of classes
    classes = [
        "Impervious surfaces",
        "Agriculture",
        "Forest & other vegetation",
        "Wetlands",
        "Soil",
        "Water",
        "Snow & ice",
    ]

    num_classes = len(classes)

    valid_metadata = ("lat", "lon", "time")

    def __init__(
        self,
        root: Path,
        split: str,
        band_order: dict[str, Sequence[float | str]] = {
            "plane": ["r", "g", "b", "nir"]
        },
        data_normalizer: Type[nn.Module] = MultiModalNormalizer,
        transforms: nn.Module | None = None,
        metadata: Sequence[str] | None = None,
        temporal_setting: Literal["single", "daily", "weekly"] = "single",
    ) -> None:
        """Initialize the dataset.

        Args:
            root: Root directory where the dataset can be found
            split: The dataset split, supports 'train', 'val', 'test'
            band_order: Band order for the dataset
            data_normalizer: Data normalizer
            transforms: A composition of transformations to apply to the data
            metadata: metadata names to be returned as part of the sample in the
                __getitem__ method. If None, no metadata is returned.
            temporal_setting: The temporal setting to use, either 'single', 'daily' or 'weekly'
        """
        super().__init__(
            root=root,
            split=split,
            band_order=band_order,
            data_normalizer=data_normalizer,
            transforms=transforms,
            metadata=metadata,
        )

        self.temporal_setting = temporal_setting

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            idx: index to return

        Returns:
            data and label at that index
        """
        sample: dict[str, Tensor] = {}
        sample_row = self.data_df.read(idx)

        if self.temporal_setting == "single":
            indices = [0]

        elif self.temporal_setting == "daily":
            indices = sample_row[sample_row["modality"] == "planet"].index
        elif self.temporal_setting == "weekly":
            # ['01', '05', '10', '15', '20', '25']
            # https://github.com/aysim/dynnet/blob/1e7d90294b54f52744ae2b35db10b4d0a48d093d/data/utae_dynamicen.py#L87
            indices = [0, 4, 9, 14, 19, 24]

        img_dict: dict[str, Tensor] = {}
        planet_imgs: list[Tensor] = []
        for i in indices:
            with rasterio.open(sample_row.read(i)) as src:
                img = src.read()
                planet_imgs.append(torch.from_numpy(img))

        # [T, C, H, W]
        planet_imgs = torch.stack(planet_imgs, dim=0).float()

        img_dict["planet"] = planet_imgs
        # [C, T, H, W]

        if self.temporal_setting == "single":
            planet_imgs = planet_imgs.squeeze(0)

        if "s2" in self.band_order:
            sentinel_2_row = sample_row[sample_row["tortilla:id"] == "s2"]
            if sentinel_2_row.empty:
                img = torch.zeros(
                    (12, planet_imgs.shape[-2], planet_imgs.shape[-1]),
                    dtype=torch.float32,
                )
            else:
                with rasterio.open(sample_row.read(-2)) as src:
                    img = src.read()
                    img = torch.from_numpy(img).float()
            img_dict["s2"] = img

        img_dict = self.rearrange_bands(img_dict, self.band_order)
        img_dict = self.data_normalizer(img_dict)

        sample.update(img_dict)

        with rasterio.open(sample_row.read(-1)) as src:
            mask = src.read()

        sample["mask"] = torch.from_numpy(mask).long()

        point = wkt.loads(sample_row.iloc[0]["stac:centroid"])
        lon, lat = point.x, point.y

        if "lon" in self.metadata:
            sample["lon"] = torch.tensor(lon)
        if "lat" in self.metadata:
            sample["lat"] = torch.tensor(lat)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
