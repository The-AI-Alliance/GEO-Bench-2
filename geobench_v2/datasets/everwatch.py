# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""EverWatch dataset."""

from torchgeo.datasets import EverWatch
import numpy as np
from PIL import Image
from torch import Tensor
import torch
from pathlib import Path

from .sensor_util import DatasetBandRegistry
from .data_util import DataUtilsMixin


class GeoBenchEverWatch(EverWatch, DataUtilsMixin):
    """ "GeoBenchEverWatch dataset with enhanced functionality.

    Allows:
    - Variable Band Selection
    - Return band wavelengths
    """

    dataset_band_config = DatasetBandRegistry.EVERWATCH
    band_default_order = ("r", "g", "b")

    normalization_stats = {
        "means": {"r": 0.0, "g": 0.0, "b": 0.0},
        "stds": {"r": 255.0, "g": 255.0, "b": 255.0},
    }

    def __init__(
        self,
        root: Path,
        split: str,
        band_order: list[str] = band_default_order,
        **kwargs,
    ) -> None:
        """Initialize EverWatch dataset.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'val', 'test'
            band_order: The order of bands to return, defaults to ['red', 'green', 'blue'], if one would
                specify ['red', 'green', 'blue', 'blue'], the dataset would return images with 4 channels
                in that order. This is useful for models that expect a certain band order, or
                test the impact of band order on model performance.
            **kwargs: Additional keyword arguments passed to ``EverWatch``
        """
        super().__init__(root=root, split=split, **kwargs)

        self.band_order = self.resolve_band_order(band_order)

        self.set_normalization_stats(self.band_order)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        sample_df = self.annot_df.loc[index]

        img_path = os.path.join(self.root, self.dir, sample_df["image_path"].iloc[0])

        image = self._load_image(img_path)

        image = self.rearrange_bands(image, self.band_orig_order, self.band_order)

        image = self.normalizer(image)

        boxes, labels = self._load_target(sample_df)

        sample = {"image": image, "bbox_xyxy": boxes, "label": labels}

        return sample
