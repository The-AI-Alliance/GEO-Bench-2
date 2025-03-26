# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""SpaceNet6 dataset."""

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
from PIL import Image


class GeoBenchBioMassters(GeoBenchBaseDataset):
    """BioMassters dataset.

    There are always 12 S1 time steps available but the number of S2 time steps can vary.
    """

    dataset_band_config = DatasetBandRegistry.BIOMASSTERS

    normalization_stats = {
        "means": {
            "VV_asc": 0.0,
            "VH_asc": 0.0,
            "VV_desc": 0.0,
            "VH_desc": 0.0,
            "B02": 0.0,
            "B03": 0.0,
            "B04": 0.0,
            "B05": 0.0,
            "B06": 0.0,
            "B07": 0.0,
            "B08": 0.0,
            "B8A": 0.0,
            "B11": 0.0,
            "B12": 0.0,
        },
        "stds": {
            "VV_asc": 1.0,
            "VH_asc": 1.0,
            "VV_desc": 1.0,
            "VH_desc": 1.0,
            "B02": 3000.0,
            "B03": 3000.0,
            "B04": 3000.0,
            "B05": 3000.0,
            "B06": 3000.0,
            "B07": 3000.0,
            "B08": 3000.0,
            "B8A": 3000.0,
            "B11": 3000.0,
            "B12": 3000.0,
        },
    }

    paths = ["BioMassters.tortilla"]

    def __init__(
        self,
        root: Path,
        split: str,
        band_order: dict[str, Sequence[str]] = {
            "s1": ["VV_asc", "VH_asx"],
            "s2": ["B04", "B03", "B02", "B08"],
        },
        data_normalizer: Type[nn.Module] = MultiModalNormalizer,
        transforms: nn.Module | None = None,
        num_time_steps: int = 1,
        **kwargs,
    ) -> None:
        """Initialize BioMassters dataset.

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
            num_time_steps: Number of last time steps to include in the dataset, maximum is 12, for S2
                missing time steps are filled with zeros.
            **kwargs: Additional keyword arguments passed to ``torchgeo.datasets.BioMassters``

        Raises:
            AssertionError: If the number of time steps is greater than 12
        """
        super().__init__(
            root=root,
            split=split,
            band_order=band_order,
            data_normalizer=data_normalizer,
            transforms=transforms,
        )
        assert num_time_steps <= 12, (
            "Number of time steps must be less than or equal to 12"
        )
        self.num_time_steps = num_time_steps

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index

        If num_time_steps is 1, the dataset will return image samples with shape [C, H, W],
        if num_time_steps is greater than 1, the dataset will return image samples with shape [T, C, H, W],
        where T is the number of time steps.
        """
        sample: dict[str, Tensor] = {}
        sample_row = self.metadata_df.iloc[index]

        if "s1" in self.band_order:
            pass

        if "s2" in self.band_order:
            pass

        agb_path = sample_row.read(-1)

        agb = Image.open(label_path)
        agb = torch.from_numpy(np.array(agb))
        sample["mask"] = agb

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
