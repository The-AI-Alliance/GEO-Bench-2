# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""SpaceNet7 dataset."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
from shapely import wkt
from torch import Tensor

from .base import GeoBenchBaseDataset
from .normalization import ZScoreNormalizer
from .sensor_util import DatasetBandRegistry


class GeoBenchSpaceNet7(GeoBenchBaseDataset):
    """SpaceNet7 dataset with enhanced functionality.

    Allows:
    - Variable Band Selection
    - Return band wavelengths
    """

    url = "https://hf.co/datasets/aialliance/spacenet7/resolve/main/{}"

    # paths = ["SpaceNet7.tortilla"]
    paths = ["geobench_spacenet7.tortilla"]

    sha256str = ["dc2364926ce2b247d183f77fadf778f3a679d6be6ef891ffed92cff230722ee4"]

    dataset_band_config = DatasetBandRegistry.SPACENET7

    normalization_stats = {
        "means": {"red": 0.0, "green": 0.0, "blue": 0.0, "nir": 0.0},
        "stds": {"red": 255.0, "green": 255.0, "blue": 255.0, "nir": 255.0},
    }

    band_default_order = ("red", "green", "blue")

    classes = ("background", "no-building", "building")

    num_classes = len(classes)

    valid_metadata = ("lat", "lon")

    def __init__(
        self,
        root: Path,
        split: str,
        band_order: list[str] = band_default_order,
        data_normalizer: type[nn.Module] = ZScoreNormalizer,
        transforms: nn.Module = None,
        metadata: Sequence[str] | None = None,
        download: bool = False,
    ) -> None:
        """Initialize SpaceNet7 dataset.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'val', 'test'
            band_order: The order of bands to return, defaults to ['red', 'green', 'blue'], if one would
                specify ['red', 'green', 'blue', 'blue', 'blue'], the dataset would return images with 5 channels
                in that order. This is useful for models that expect a certain band order, or
                test the impact of band order on model performance.
            data_normalizer: The data normalizer to apply to the data, defaults to :class:`data_util.ZScoreNormalizer`,
            transforms: The transforms to apply to the data, defaults to None
            metadata: metadata names to be returned as part of the sample in the
                __getitem__ method. If None, no metadata is returned.
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
        # TODO how to setup for time-series prediction

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        sample: dict[str, Tensor] = {}

        sample_row = self.data_df.read(index)

        image_path = sample_row.read(0)
        mask_path = sample_row.read(1)

        with rasterio.open(image_path) as img_src, rasterio.open(mask_path) as mask_src:
            image: np.ndarray = img_src.read(out_dtype="float32")
            mask: np.ndarray = mask_src.read()

        image = torch.from_numpy(image).float()

        # add 1 to mask to have a true background class
        mask = torch.from_numpy(mask).long().squeeze(0) + 1

        image = self.rearrange_bands(image, self.band_order)
        image = self.data_normalizer(image)

        sample.update(image)

        sample["mask"] = mask

        if self.transforms is not None:
            sample = self.transforms(sample)

        point = wkt.loads(sample_row.iloc[0]["stac:centroid"])
        lon, lat = point.x, point.y

        if "lon" in self.metadata:
            sample["lon"] = torch.tensor(lon)
        if "lat" in self.metadata:
            sample["lat"] = torch.tensor(lat)

        return sample
