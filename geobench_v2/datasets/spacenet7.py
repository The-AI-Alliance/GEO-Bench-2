# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""SpaceNet7 dataset."""

from torch import Tensor
from torchgeo.datasets import SpaceNet7
from pathlib import Path
from typing import Type

from .sensor_util import DatasetBandRegistry
from .data_util import MultiModalNormalizer
from .base import GeoBenchBaseDataset
import torch.nn as nn
import rasterio
import numpy as np
import torch


class GeoBenchSpaceNet7(GeoBenchBaseDataset):
    """SpaceNet7 dataset with enhanced functionality.

    Allows:
    - Variable Band Selection
    - Return band wavelengths
    """

    dataset_band_config = DatasetBandRegistry.SPACENET7

    normalization_stats = {
        "means": {"r": 0.0, "g": 0.0, "b": 0.0, "nir": 0.0},
        "stds": {"r": 255.0, "g": 255.0, "b": 255.0, "nir": 255.0},
    }

    band_default_order = ("red", "green", "blue")

    paths = ["SpaceNet7.tortilla"]

    classes = ("background", "building")

    num_classes = len(classes)

    def __init__(
        self,
        root: Path,
        split: str,
        band_order: list[str] = band_default_order,
        data_normalizer: Type[nn.Module] = MultiModalNormalizer,
        transforms: nn.Module = None,
        **kwargs,
    ) -> None:
        """Initialize SpaceNet7 dataset.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'val', 'test'
            band_order: The order of bands to return, defaults to ['red', 'green', 'blue'], if one would
                specify ['red', 'green', 'blue', 'blue', 'blue'], the dataset would return images with 5 channels
                in that order. This is useful for models that expect a certain band order, or
                test the impact of band order on model performance.
            **kwargs: Additional keyword arguments passed to ``SpaceNet7``
        """
        super().__init__(
            root=root,
            split=split,
            band_order=band_order,
            data_normalizer=data_normalizer,
            transforms=transforms,
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
        mask = torch.from_numpy(mask).long()

        image = self.rearrange_bands(image, self.band_order)
        image = self.data_normalizer(image)

        sample.update(image)

        sample["mask"] = mask

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
