# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""MMFlood dataset."""

from torch import Tensor
from torchgeo.datasets import MMFlood
from pathlib import Path
from typing import Sequence, Type
import torch.nn as nn
from shapely import wkt

from .sensor_util import DatasetBandRegistry
from .base import GeoBenchBaseDataset
from .data_util import MultiModalNormalizer
import torch.nn as nn
import rasterio
import numpy as np
import torch


class GeoBenchMMFlood(GeoBenchBaseDataset):
    """MMFlood dataset with enhanced functionality.

    Allows:
    - Variable Band Selection
    - Return band wavelengths

    Classes are:

    0. Background
    1. No Building
    2. Building
    """

    dataset_band_config = DatasetBandRegistry.MMFLOOD

    band_default_order = {"s1": ("vv", "vh"), "dem": ("dem",), "hydro": ("hydro",)}

    normalization_stats = {
        "means": {"vv": 0.0, "vh": 0.0, "dem": 0.0, "hydro": 0.0},
        "stds": {"vv": 1.0, "vh": 1.0, "hydro": 1.0, "dem": 100.0},
    }

    paths = ["MMFlood.tortilla"]

    classes = ("no-data", "no-flood", "flood")

    num_classes = len(classes)

    def __init__(
        self,
        root: Path,
        split: str,
        band_order: dict[str, Sequence[str]] = band_default_order,
        data_normalizer: Type[nn.Module] = MultiModalNormalizer,
        transforms: nn.Module | None = None,
    ) -> None:
        """Initialize MMFlood dataset.

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

        s1_path = sample_row.read(0)
        dem_path = sample_row.read(1)
        hydro_path = sample_row.read(2)
        mask_path = sample_row.read(3)

        img_dict: dict[str, Tensor] = {}
        if "s1" in self.band_order:
            with rasterio.open(s1_path) as src:
                s1_img = src.read()

            nan_mask = torch.from_numpy(np.isnan(s1_img)[0])
            s1_img = torch.from_numpy(s1_img).float()
            s1_img[..., nan_mask] = 0.0
            img_dict["s1"] = s1_img

        if "dem" in self.band_order:
            with rasterio.open(dem_path) as src:
                dem_img = src.read()
            dem_img = torch.from_numpy(dem_img).float()
            img_dict["dem"] = dem_img

        if "hydro" in self.band_order:
            with rasterio.open(hydro_path) as src:
                hydro_img = src.read()
            hydro_img = torch.from_numpy(hydro_img).float()
            img_dict["hydro"] = hydro_img

        img_dict = self.rearrange_bands(img_dict, self.band_order)
        image_dict = self.data_normalizer(img_dict)

        # across all items in the image_dict replace nan_mask with 0 again
        for key in img_dict.keys():
            img_dict[key][..., nan_mask] = 0.0

        sample.update(image_dict)

        with rasterio.open(mask_path) as src:
            mask = src.read()

        # add 1 to the mask to map classes to 1, and 2 and have 0 as no-data class
        mask = torch.from_numpy(mask).long() + 1

        mask[..., nan_mask] = 0

        sample["mask"] = mask

        point = wkt.loads(sample_row.iloc[0]["stac:centroid"])
        lon, lat = point.x, point.y
        sample["lon"], sample["lat"] = torch.tensor(lon), torch.tensor(lat)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
