# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Resisc45 Dataset."""

import torch
from torch import Tensor
from torchgeo.datasets import RESISC45
from pathlib import Path
from typing import Sequence, Type
import torch.nn as nn

from .sensor_util import DatasetBandRegistry
from .data_util import DataUtilsMixin, MultiModalNormalizer


class GeoBenchRESISC45(RESISC45, DataUtilsMixin):
    """Resisc45 Dataset with enhanced functionality.

    Allows:
    - Variable Band Selection
    - Return band wavelengths
    """

    dataset_band_config = DatasetBandRegistry.RESISC45

    band_orig_order = {"r": 0, "g": 1, "b": 2}
    band_default_order = ("r", "g", "b")

    normalization_stats = {
        "means": {"r": 0.0, "g": 0.0, "b": 0.0},
        "stds": {"r": 255.0, "g": 255.0, "b": 255.0},
    }

    def __init__(
        self,
        root: Path,
        split: str,
        band_order: Sequence["str"] = band_default_order,
        data_normalizer: Type[nn.Module] = MultiModalNormalizer,
        **kwargs,
    ):
        """Initialize Resisc45 Dataset.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'val', 'test'
            band_order: The order of bands to return, defaults to ['r', 'g', 'b'], if one would
                specify ['g', 'r', 'b', 'b], the dataset would return the green band first, then the red band,
                and then the blue band twice. This is useful for models that expect a certain band order, or
                test the impact of band order on model performance.
            data_normalizer: The data normalizer to apply to the data, defaults to :class:`data_util.MultiModalNormalizer`,
                which applies z-score normalization to each band.
            **kwargs: Additional keyword arguments passed to ``torchgeo.datasts.RESISC45``
        """
        if split == "validation":
            split = "val"
        super().__init__(root=root, split=split, **kwargs)

        self.band_order = self.resolve_band_order(band_order)

        self.data_normalizer = data_normalizer(
            self.normalization_stats, self.band_order
        )

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        sample: dict[str, Tensor] = {}
        image, label = self._load_image(index)

        image_dict = self.rearrange_bands(image, self.band_order)

        image_dict = self.data_normalizer(image_dict)

        sample.update(image_dict)
        sample["label"] = label

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
