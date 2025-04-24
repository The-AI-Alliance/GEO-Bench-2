# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""QFabric dataset."""

from torch import Tensor
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


class GeoBenchQFabric(GeoBenchBaseDataset):
    """QFabric dataset with enhanced functionality.

    Allows:
    - Variable Band Selection
    - Return band wavelengths

    Classes are:

    0. Background
    1. No Building
    2. Building
    """

    dataset_band_config = DatasetBandRegistry.QFABRIC

    band_default_order = ("red", "green", "blue")

    normalization_stats = {
        "means": {"r": 0.0, "g": 0.0, "b": 0.0},
        "stds": {"r": 255.0, "g": 255.0, "b": 255.0},
    }

    paths = ["geobench_qfabric.tortilla"]

    classes = ("no-data", "no-flood", "flood")

    num_classes = len(classes)

    def __init__(
        self,
        root: Path,
        split: str,
        band_order: Sequence[str] = band_default_order,
        data_normalizer: Type[nn.Module] = MultiModalNormalizer,
        transforms: nn.Module | None = None,
        time_steps: Sequence[int] = [0, 1, 2, 3, 4],
    ) -> None:
        """Initialize QFabric dataset.

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
            time_steps: QFabric contains 5 time steps, this allows to select which time steps to use. Specified time steps
                will be returned in that order

        Raises:
            AssertionError: If time steps are not in the range [0, 4], or invalid
        """
        super().__init__(
            root=root,
            split=split,
            band_order=band_order,
            data_normalizer=data_normalizer,
            transforms=transforms,
        )
        assert len(time_steps) <= 5, "QFabric only contains 5 time steps"
        assert all(isinstance(ts, int) and 0 <= ts < 5 for ts in time_steps), (
            "Time steps must be integers between 0 and 4"
        )
        assert len(time_steps) == len(set(time_steps)), "Time steps must be unique"
        self.time_steps = time_steps

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        sample: dict[str, Tensor] = {}

        sample_row = self.data_df.read(index)

        images = []
        for i in self.time_steps:
            img_path = sample_row.read(i)
            with rasterio.open(img_path) as src:
                img = src.read()
            images.append(torch.from_numpy(img).float())
        image = torch.stack(images, dim=0)

        image_dict = self.rearrange_bands(image, self.band_order)
        image_dict = self.data_normalizer(image_dict)
        sample.update(image_dict)

        status_masks = []
        for i in self.time_steps:
            status_mask_path = sample_row.read(i + 5)
            with rasterio.open(status_mask_path) as src:
                status_mask = src.read(1)
            status_masks.append(torch.from_numpy(status_mask).long())
        status_mask = torch.stack(status_masks, dim=0)

        sample["mask_status"] = status_mask

        change_mask_path = sample_row.read(-1)
        with rasterio.open(change_mask_path) as src:
            change_mask = src.read(1)
        change_mask = torch.from_numpy(change_mask).long()

        sample["mask_change"] = change_mask

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
