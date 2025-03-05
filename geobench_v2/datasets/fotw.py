# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Fields of the World Dataset."""

import torch
from torch import Tensor
from torchgeo.datasets import FieldsOfTheWorld
from pathlib import Path


from typing import List, Union, Optional
from .sensor_util import BandRegistry, SatelliteType
from .data_util import DataUtilsMixin


class GeoBenchFieldsOfTheWorld(FieldsOfTheWorld, DataUtilsMixin):
    """Fields of the World Dataset with enhanced functionality.

    Allows:
    - Variable Band Selection
    - Return band wavelengths
    """

    # keys should be specified according to the sensor default values
    # defined in sensor_util.py
    band_default_order = ("r", "g", "b", "n")

    # Define normalization stats using canonical names
    normalization_stats = {
        "means": {"r": 0.0, "g": 0.0, "b": 0.0, "n": 0.0},
        "stds": {"r": 3000.0, "g": 3000.0, "b": 3000.0, "n": 3000.0},
    }

    def __init__(
        self,
        root: Path,
        split: str,
        band_order: list[str] = ["red", "green", "blue", "nir"],
        **kwargs,
    ) -> None:
        """Initialize Fields of the World Dataset.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'val', 'test'
            band_order: The order of bands to return, defaults to ['red', 'green', 'blue', 'nir'], if one would
                specify ['red', 'green', 'blue', 'nir', 'nir'], the dataset would return images with 5 channels
                in that order. This is useful for models that expect a certain band order, or
                test the impact of band order on model performance.
            **kwargs: Additional keyword arguments passed to ``FieldsOfTheWorld``
        """
        super().__init__(root=root, split=split, **kwargs)
        # TODO allow input of blank channels
        # assert all(band in self.band_default_order.keys() for band in band_order), (
        #     f"Invalid bands in {band_order}. Must be among {list(self.band_default_order.keys())}"
        # )

        self.band_order = []
        for band in band_order:
            if not isinstance(band, (int, float)):
                self.band_order.append(
                    BandRegistry.resolve_band(band, SatelliteType.RGBN)
                )
            else:
                self.band_order.append(band)

        self.set_normalization_stats(self.band_order)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Return the image and mask at the given index.

        Args:
            idx: index of the image and mask to return

        Returns:
            dict: a dict containing the image and mask
        """
        win_a_fn = self.files[idx]["win_a"]
        win_b_fn = self.files[idx]["win_b"]
        mask_fn = self.files[idx]["mask"]

        win_a = self._load_image(win_a_fn)
        win_b = self._load_image(win_b_fn)

        # adapt img according to band_order
        # win_a = torch.stack(
        #     [win_a[self.band_default_order[band]] for band in self.band_order]
        # )
        # win_b = torch.stack(
        #     [win_b[self.band_default_order[band]] for band in self.band_order]
        # )
        mask = self._load_target(mask_fn)

        win_a = self.rearrange_bands(win_a, self.band_default_order, self.band_order)

        win_a = self.normalizer(win_a)

        win_b = self.rearrange_bands(win_b, self.band_default_order, self.band_order)

        win_b = self.normalizer(win_b)

        # concat or return two separate images?
        image = torch.cat((win_a, win_b), dim=0)
        sample = {"image": image, "mask": mask}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
