# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""SpaceNet8 dataset."""

from torch import Tensor
from torchgeo.datasets import SpaceNet8
from pathlib import Path

from .sensor_util import DatasetBandRegistry
from .data_util import DataUtilsMixin, MultiModalNormalizer


class GeoBenchSpaceNet8(SpaceNet8, DataUtilsMixin):
    """SpaceNet8 dataset with enhanced functionality.

    Allows:
    - Variable Band Selection
    - Return band wavelengths

    3 classes: background, building or road (not flooded), building or road (flooded)
    """

    dataset_band_config = DatasetBandRegistry.SPACENET8

    normalization_stats = {
        "means": {"red": 0.0, "green": 0.0, "blue": 0.0, "nir": 0.0},
        "stds": {"red": 3000.0, "green": 3000.0, "blue": 3000.0, "nir": 3000.0},
    }

    def __init__(
        self,
        root: Path,
        split: str,
        band_order: list[str] = ["red", "green", "blue", "nir"],
        **kwargs,
    ) -> None:
        """Initialize SpaceNet8 dataset.

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

        self.normalizer = MultiModalNormalizer(
            self.normalization_stats, self.band_order
        )

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


        sample = {"image": img}

        mask_path = self.masks[index]
        mask = self._load_mask(mask_path, tfm, raster_crs, (h, w))
        sample["mask"] = mask

        # We add 1 to the mask to map the current {background, building} labels to
        # the values {1, 2}. This is necessary because we add 0 padding to the
        # mask that we want to ignore in the loss function.
        if "mask" in sample:
            sample["mask"] += 1

        return sample
