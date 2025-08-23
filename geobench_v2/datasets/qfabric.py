# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""QFabric dataset."""

from collections.abc import Sequence
from pathlib import Path

import rasterio
import torch
import torch.nn as nn
from torch import Tensor

from .base import GeoBenchBaseDataset
from .normalization import ZScoreNormalizer
from .sensor_util import DatasetBandRegistry


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

    url = "https://hf.co/datasets/aialliance/qfabric/resolve/main/{}"

    paths = ["geobench_qfabric.tortilla"]

    sha256str = [""]

    dataset_band_config = DatasetBandRegistry.QFABRIC

    band_default_order = ("red", "green", "blue")

    normalization_stats = {
        "means": {"red": 0.0, "green": 0.0, "blue": 0.0},
        "stds": {"red": 255.0, "green": 255.0, "blue": 255.0},
    }

    # add an extra background class here
    classes = (
        "residential",
        "commercial",
        "industrial",
        "road",
        "demolition",
        "mega projects",
    )

    status_classes = (
        "no change",
        "prior construction",
        "greenland",
        "land cleared",
        "excavation",
        "materials dumped",
        "construction started",
        "construction midway",
        "construction done",
        "operational",
    )
    num_classes = len(classes)

    def __init__(
        self,
        root: Path,
        split: str,
        band_order: Sequence[str] = band_default_order,
        data_normalizer: type[nn.Module] = ZScoreNormalizer,
        transforms: nn.Module | None = None,
        time_steps: Sequence[int] = [0, 1, 2, 3, 4],
        metadata: Sequence[str] | None = None,
        download: bool = False,
    ) -> None:
        """Initialize QFabric dataset.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'val', 'test'
            band_order: The order of bands to return, defaults to ['red', 'green', 'blue', 'nir'], if one would
                specify ['red', 'green', 'blue', 'nir', 'nir'], the dataset would return images with 5 channels
                in that order. This is useful for models that expect a certain band order, or
                test the impact of band order on model performance.
            data_normalizer: The data normalizer to apply to the data, defaults to :class:`data_util.ZScoreNormalizer`,
                which applies z-score normalization to each band.
            transforms: The transforms to apply to the data, defaults to None
            time_steps: QFabric contains 5 time steps, this allows to select which time steps to use. Specified time steps
                will be returned in that order
            metadata: metadata names to be returned as part of the sample in the
                __getitem__ method. If None, no metadata is returned.
            download: Whether to download the dataset

        Raises:
            AssertionError: If time steps are not in the range [0, 4], or invalid
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
