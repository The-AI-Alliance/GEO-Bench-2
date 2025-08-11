# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""DynamicEarthNet Dataset."""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import rasterio
import torch
import torch.nn as nn
from shapely import wkt
from torch import Tensor

from .base import GeoBenchBaseDataset
from .normalization import ZScoreNormalizer
from .sensor_util import DatasetBandRegistry


class GeoBenchDynamicEarthNet(GeoBenchBaseDataset):
    """DynamicEarthNet dataset."""

    url = "https://hf.co/datasets/aialliance/dynamic_earthnet/resolve/main/{}"

    paths = [
        "geobench_dynamic_earthnet.0000.part.tortilla",
        "geobench_dynamic_earthnet.0001.part.tortilla",
        "geobench_dynamic_earthnet.0002.part.tortilla",
    ]

    sha256str = [
        "11204f926bfd89feeaa4c045a79a640137a21f7c038f71665583e7a35d6ab73f", 
        "b3444cab2ac66b2cb225bb1fc4c71c298b6e31a1a1c5eaea72df7d820bb640b7", 
        "3e11fdb9726c4448c07ca6b33f94dfb2c07deb3caed4721c47fae46018b3a011"
    ]

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
            "planet": ["r", "g", "b", "nir"]
        },
        data_normalizer: type[nn.Module] = ZScoreNormalizer,
        transforms: nn.Module | None = None,
        metadata: Sequence[str] | None = None,
        temporal_setting: Literal["single", "daily", "weekly"] = "single",
        return_stacked_image: bool = False,
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
            download: Whether to download the dataset 
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
        assert temporal_setting in ["single", "daily", "weekly"], (
            "temporal_setting must be one of the following: single, daily, or weekly"
        )
        self.temporal_setting = temporal_setting
        self.return_stacked_image = return_stacked_image

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

        if "planet" in self.band_order:
            #convert from T, C, H, W -> C, T, H, W
            sample["image_planet"] = sample["image_planet"].permute(1, 0, 2, 3)

        if self.return_stacked_image:
            if ("s2" in self.band_order):
                if (self.temporal_setting == "single"):
                    sample["image_planet"] = sample["image_planet"].squeeze(1)
                else:
                    raise ValueError("To stack Sentinel 2 (s2) with Planet, please use temporal_setting = single")
            
            sample = { 
                "image": torch.cat(
                    [sample[f"image_{key}"] for key in self.band_order.keys()], 0
                ),
                "mask": sample["mask"],
            }
            sample["mask"] = torch.squeeze(sample["mask"])
            

        if "lon" in self.metadata:
            sample["lon"] = torch.tensor(lon)
        if "lat" in self.metadata:
            sample["lat"] = torch.tensor(lat)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
