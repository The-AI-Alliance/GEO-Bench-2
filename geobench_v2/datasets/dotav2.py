# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""DOTAV2 dataset."""

import os
from pathlib import Path
from typing import Literal

import pandas as pd
import torch.nn as nn
from torch import Tensor
from torchgeo.datasets import DOTA

from .data_util import ClipZScoreNormalizer, DataUtilsMixin, DataNormalizer
from .sensor_util import DatasetBandRegistry


class GeoBenchDOTAV2(DOTA, DataUtilsMixin):
    """ "GeoBenchDOTAV2 dataset with enhanced functionality.

    Allows:
    - Variable Band Selection
    - Return band wavelengths
    """

    dataset_band_config = DatasetBandRegistry.DOTAV2
    band_default_order = ("red", "green", "blue")

    normalization_stats = {
        "means": {"red": 0.0, "green": 0.0, "blue": 0.0},
        "stds": {"red": 255.0, "green": 255.0, "blue": 255.0},
    }

    classes = (
        "plane",
        "ship",
        "storage-tank",
        "baseball-diamond",
        "tennis-court",
        "basketball-court",
        "ground-track-field",
        "harbor",
        "bridge",
        "large-vehicle",
        "small-vehicle",
        "helicopter",
        "roundabout",
        "soccer-ball-field",
        "swimming-pool",
        "container-crane",
        "airport",
        "helipad",
    )

    num_classes = len(classes)

    def __init__(
        self,
        root: Path,
        split: str,
        band_order: list[str] = band_default_order,
        data_normalizer: type[nn.Module] = ClipZScoreNormalizer,
        bbox_orientation: Literal["horizontal", "oriented"] = "oriented",
        transforms: nn.Module | None = None,
    ) -> None:
        """Initialize DOTAV2 dataset.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'val', 'test'
            band_order: The order of bands to return, defaults to ['red', 'green', 'blue'], if one would
                specify ['red', 'green', 'blue', 'blue'], the dataset would return images with 4 channels
                in that order. This is useful for models that expect a certain band order, or
                test the impact of band order on model performance.
            data_normalizer: The data normalizer to apply to the data, defaults to :class:`data_util.ClipZScoreNormalizer`,
                which applies z-score normalization to each band.
            transforms:
        """
        self.root = root
        self.split = split

        self.transforms = transforms

        self.band_order = self.resolve_band_order(band_order)

        self.data_df = pd.read_parquet(
            os.path.join(self.root, "geobench_dotav2_processed.parquet")
        )

        self.data_df = self.data_df[self.data_df["split"] == split].reset_index(
            drop=True
        )

        self.class2idx: dict[str, int] = {c: i for i, c in enumerate(self.classes)}

        self.bbox_orientation = bbox_orientation

        if isinstance(data_normalizer, type):
            print(f"Initializing normalizer from class: {data_normalizer.__name__}")
            if issubclass(data_normalizer, DataNormalizer):
                self.data_normalizer = data_normalizer(
                    self.normalization_stats, self.band_order
                )
            else:
                self.data_normalizer = data_normalizer()

        elif callable(data_normalizer):
            print(
                f"Using provided pre-initialized normalizer instance: {data_normalizer.__class__.__name__}"
            )
            self.data_normalizer = data_normalizer
        else:
            raise TypeError(
                f"data_normalizer must be a DataNormalizer subclass type or a callable instance. Got {type(data_normalizer)}"
            )

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        sample: dict[str, Tensor] = {}

        sample_row = self.data_df.loc[index]

        img = self._load_image(os.path.join(self.root, sample_row["processed_image"]))

        image_dict = self.rearrange_bands(img, self.band_order)
        image_dict = self.data_normalizer(image_dict)
        sample.update(image_dict)

        boxes, labels = self._load_annotations(
            os.path.join(self.root, sample_row["processed_label"])
        )

        sample["bbox_xyxy"] = boxes
        sample["label"] = labels

        # TODO kornia does not work with oriented bboxes
        # if self.transforms is not None:
        #     sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data_df)
