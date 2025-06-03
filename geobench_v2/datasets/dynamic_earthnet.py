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

    url = "https://hf.co/datasets/aialliance/dynamic_earthnet/resolve/main/{}"

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

    sha256str = ["", "", ""]

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
            "b": 641.124267578125,
            "g": 881.2556762695312,
            "r": 1011.3512573242188,
            "nir": 2609.922607421875,
            "B01": 1091.76220703125,
            "B02": 1318.852783203125,
            "B03": 1380.147216796875,
            "B04": 2678.525146484375,
            "B05": 1730.9559326171875,
            "B06": 2373.4130859375,
            "B07": 2630.05322265625,
            "B08": 2782.686767578125,
            "B8A": 2307.15869140625,
            "B10": 1719.888671875,
            "B11": 1003.9291381835938,
            "B12": 3031.021728515625,
        },
        "stds": {
            "b": 523.4900512695312,
            "g": 647.6270141601562,
            "r": 888.1035766601562,
            "nir": 992.0601806640625,
            "B01": 1414.6219482421875,
            "B02": 1343.7620849609375,
            "B03": 1427.9449462890625,
            "B04": 1376.4869384765625,
            "B05": 1429.6456298828125,
            "B06": 1333.841064453125,
            "B07": 1370.47802734375,
            "B08": 1386.9127197265625,
            "B8A": 1394.8505859375,
            "B10": 1304.7115478515625,
            "B11": 1475.8455810546875,
            "B12": 2124.4130859375,
        },
    }

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
        download: bool = False,
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
            download=download,
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
