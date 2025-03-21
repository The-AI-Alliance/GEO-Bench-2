# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""SpaceNet8 dataset."""

from torch import Tensor
from torchgeo.datasets import SpaceNet8
from pathlib import Path

from .sensor_util import DatasetBandRegistry
from .base import GeoBenchBaseDataset
import torch.nn as nn
import rasterio
import numpy as np
import torch




class GeoBenchSpaceNet8(GeoBenchBaseDataset):
    """SpaceNet8 dataset with enhanced functionality.

    Allows:
    - Variable Band Selection
    - Return band wavelengths

    3 classes: background, building or road (not flooded), building or road (flooded)
    """

    dataset_band_config = DatasetBandRegistry.SPACENET8

    normalization_stats = {
        "means": {"r": 0.0, "g": 0.0, "b": 0.0, "n": 0.0},
        "stds": {"r": 255.0, "g": 255.0, "b": 255.0, "n": 255.0},
    }

    band_default_order = ("red", "green", "blue", "nir")

    paths = [
        "SpaceNet8.tortilla"
    ]

    def __init__(
        self,
        root: Path,
        split: str,
        band_order: list[str] = band_default_order,
        transforms: nn.Module = None,
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
        super().__init__(root=root, split=split, band_order=band_order, transforms=transforms)
        

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        # image_path = self.images[index]
        sample: dict[str, Tensor] = {}
        sample_row = self.data_df.read(index)
        pre_event_path = sample_row.read(0)
        post_event_path = sample_row.read(1)
        mask_path = sample_row.read(2)

        with (
            rasterio.open(pre_event_path) as pre_src,
            rasterio.open(post_event_path) as post_src,
            rasterio.open(mask_path) as mask_src,
        ):
            pre_image: np.ndarray = pre_src.read(out_dtype="float32")
            post_image: np.ndarray = post_src.read(out_dtype="float32")
            mask: np.ndarray = mask_src.read()

        image_pre = torch.from_numpy(pre_image).float()
        image_post = torch.from_numpy(post_image).float()
        mask = torch.from_numpy(mask).long()

        image_pre = self.rearrange_bands(image_pre, self.band_order)
        image_pre = self.data_normalizer(image_pre)
        image_post = self.rearrange_bands(image_post, self.band_order)
        image_post = self.data_normalizer(image_post)


        sample["image_pre"] = image_pre["image"]
        sample["image_post"] = image_post["image"]
        # We add 1 to the mask to map the current {background, building} labels to
        # the values {1, 2}. This is necessary because we add 0 padding to the
        # mask that we want to ignore in the loss function.
        sample["mask"] = mask + 1


        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
