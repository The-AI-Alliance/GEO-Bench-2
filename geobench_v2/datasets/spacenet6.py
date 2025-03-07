# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""SpaceNet6 dataset."""

from torch import Tensor
from torchgeo.datasets import SpaceNet6
from pathlib import Path
from typing import Sequence

from .sensor_util import DatasetBandRegistry
from .data_util import DataUtilsMixin


class GeoBenchSpaceNet6(SpaceNet6, DataUtilsMixin):
    """SpaceNet6 dataset with enhanced functionality.

    Allows:
    - Variable Band Selection
    - Return band wavelengths
    """

    dataset_band_config = DatasetBandRegistry.SPACENET6

    band_default_order = {"red": 0, "green": 1, "blue": 2, "nir": 3}

    band_default_order = ("r", "g", "b", "n")

    normalization_stats = {
        "means": {"r": 0.0, "g": 0.0, "b": 0.0, "n": 0.0},
        "stds": {"r": 3000.0, "g": 3000.0, "b": 3000.0, "n": 3000.0},
    }

    def __init__(
        self,
        root: Path,
        split: str,
        band_order: Sequence[str] = band_default_order,
        **kwargs,
    ) -> None:
        """Initialize SpaceNet6 dataset.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'val', 'test'
            band_order: The order of bands to return, defaults to ['red', 'green', 'blue', 'nir'], if one would
                specify ['red', 'green', 'blue', 'nir', 'nir'], the dataset would return images with 5 channels
                in that order. This is useful for models that expect a certain band order, or
                test the impact of band order on model performance.
            **kwargs: Additional keyword arguments passed to ``SpaceNet6``
        """
        super().__init__(root=root, split=split, **kwargs)

        self.band_order = self.resolve_band_order(band_order)

        self.set_normalization_module(self.band_order)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        image_path = self.images[index]
        img, tfm, raster_crs = self._load_image(image_path)
        h, w = img.shape[1:]

        img = self.rearrange_bands(img, self.band_default_order, self.band_order)

        img = self.normalizer(img)

        sample = {"image": img}

        if self.split == "train":
            mask_path = self.masks[index]
            mask = self._load_mask(mask_path, tfm, raster_crs, (h, w))
            sample["mask"] = mask

        # We add 1 to the mask to map the current {background, building} labels to
        # the values {1, 2}. This is necessary because we add 0 padding to the
        # mask that we want to ignore in the loss function.
        if "mask" in sample:
            sample["mask"] += 1

        return sample
