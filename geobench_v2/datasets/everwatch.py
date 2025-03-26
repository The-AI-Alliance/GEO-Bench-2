# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""EverWatch dataset."""

import os

from typing import Type
from torchgeo.datasets import EverWatch
import numpy as np
from PIL import Image
import torch.nn as nn
from torch import Tensor
import torch
from pathlib import Path
import kornia.augmentation as K
import torch.nn.functional as F
import pandas as pd


from .sensor_util import DatasetBandRegistry
from .data_util import DataUtilsMixin, MultiModalNormalizer


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

    classes = classes = (
        'White Ibis',
        'Great Egret',
        'Great Blue Heron',
        'Snowy Egret',
        'Wood Stork',
        'Roseate Spoonbill',
        'Anhinga',
        'Unknown White'
    )

    def __init__(
        self,
        root: Path,
        split: str,
        band_order: list[str] = band_default_order,
        data_normalizer: Type[nn.Module] = MultiModalNormalizer,
        transforms: nn.Module | None = None,
    ) -> None:
        """Initialize EverWatch dataset.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'val', 'test'
            band_order: The order of bands to return, defaults to ['red', 'green', 'blue'], if one would
                specify ['red', 'green', 'blue', 'blue'], the dataset would return images with 4 channels
                in that order. This is useful for models that expect a certain band order, or
                test the impact of band order on model performance.
            data_normalizer: The data normalizer to apply to the data, defaults to :class:`data_util.MultiModalNormalizer`,
                which applies z-score normalization to each band.
            transforms:
        """
        self.root = root
        self.split = split

        self.transforms = transforms

        self.band_order = self.resolve_band_order(band_order)

        self.data_normalizer = data_normalizer(
            self.normalization_stats, self.band_order
        )

        self.annot_df = pd.read_csv(os.path.join(self.root, "annotations.csv"))

        # remove all entries where xmin == xmax or ymin == ymax
        self.annot_df = self.annot_df[
            (self.annot_df["xmin"] != self.annot_df["xmax"])
            & (self.annot_df["ymin"] != self.annot_df["ymax"])
        ].reset_index(drop=True)

        # group per image path to get all annotations for one sample
        self.annot_df["sample_index"] = pd.factorize(self.annot_df["image_path"])[0]
        self.annot_df = self.annot_df.set_index(["sample_index", self.annot_df.index])

        self.class2idx: dict[str, int] = {c: i for i, c in enumerate(self.classes)}

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        sample: dict[str, Tensor] = {}

        sample_df = self.annot_df.loc[index]

        img_path = os.path.join(self.root, "images", sample_df["image_path"].iloc[0])

        image = self._load_image(img_path).float()

        image_dict = self.rearrange_bands(image, self.band_order)

        image_dict = self.data_normalizer(image_dict)

        sample.update(image_dict)

        boxes, labels = self._load_target(sample_df)

        sample["bbox_xyxy"] = boxes
        sample["label"] = labels

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
