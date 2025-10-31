# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""EverWatch dataset."""

import io
import re
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

import geopandas as gpd
import rasterio
import torch
import torch.nn as nn
from rasterio.errors import NotGeoreferencedWarning
from torch import Tensor

from .base import GeoBenchBaseDataset
from .normalization import ZScoreNormalizer
from .sensor_util import DatasetBandRegistry


class GeoBenchEverWatch(GeoBenchBaseDataset):
    """GeoBench version of EverWatch dataset.

    Bird object detection dataset using high-resolution
    aerial RGB imagery, with bounding box annotations for multiple bird species.

    If you use this dataset in your research, please cite the following resource:

    * https://zenodo.org/records/11165946
    """

    url = "https://hf.co/datasets/aialliance/everwatch/resolve/main/{}"

    paths = ["geobench_everwatch.tortilla"]

    sha256str = ["4afb1eada24ce990c1798fa481963ab8c0a0e302ba2d4112f02261c6a8246272"]

    classes = (
        "Background",
        "White Ibis",
        "Great Egret",
        "Great Blue Heron",
        "Snowy Egret",
        "Wood Stork",
        "Roseate Spoonbill",
        "Anhinga",
        "Unknown White",
    )

    num_classes = len(classes)

    dataset_band_config = DatasetBandRegistry.EVERWATCH

    band_default_order = ("red", "green", "blue")

    normalization_stats: dict[str, dict[str, float]] = {
        "means": {
            "red": 68.09844970703125,
            "green": 83.53096771240234,
            "blue": 35.009552001953125,
        },
        "stds": {
            "red": 46.88275146484375,
            "green": 50.39387893676758,
            "blue": 29.952987670898438,
        },
    }

    def __init__(
        self,
        root: Path,
        split: Literal["train", "val", "validation", "test"],
        band_order: Sequence[str] = band_default_order,
        data_normalizer: type[nn.Module] = ZScoreNormalizer,
        transforms: nn.Module | None = None,
        metadata: Sequence[str] | None = None,
        download: bool = False,
    ) -> None:
        """Initialize EverWatch dataset.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'validation', 'test'
            band_order: The order of bands to return, defaults to ['red', 'green', 'blue'], if one would
                specify ['red', 'green', 'blue', 'blue'], the dataset would return images with 4 channels
                in that order. This is useful for models that expect a certain band order, or
                test the impact of band order on model performance.
            data_normalizer: The data normalizer to apply to the data, defaults to :class:`data_util.ZScoreNormalizer`,
                which applies z-score normalization to each band.
            transforms: The transforms to apply to the data, defaults to None.
            metadata: The metadata to return, defaults to None. If None, no metadata is returned.
            download: Whether to download the dataset
        """
        split_norm: Literal["train", "validation", "test"]
        if split == "val":
            split_norm = "validation"
        else:
            split_norm = cast(Literal["train", "validation", "test"], split)
        super().__init__(
            root=root,
            split=split_norm,
            band_order=band_order,
            data_normalizer=data_normalizer,
            transforms=transforms,
            metadata=metadata,
            download=download,
        )

        self.class2idx: dict[str, int] = {c: i for i, c in enumerate(self.classes)}

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            idx: index to return

        Returns:
            data and label at that index
        """
        sample: dict[str, Tensor] = {}

        sample_row = self.data_df.read(idx)

        image_path = sample_row.read(0)
        annot_path = sample_row.read(1)

        pattern = r"(\d+)_(\d+),(.+)"
        match = re.search(pattern, annot_path)
        if match:
            offset = int(match.group(1))
            size = int(match.group(2))
            file_name = match.group(3)

        with open(file_name, "rb") as f:
            f.seek(offset)
            data = f.read(size)
        byte_stream = io.BytesIO(data)
        annot_df = gpd.read_parquet(byte_stream)

        boxes, labels = self._load_target(annot_df)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
            with rasterio.open(image_path) as src:
                image = torch.from_numpy(src.read()).float()

        image_dict = self.rearrange_bands(image, self.band_order)

        image_dict = self.data_normalizer(image_dict)

        sample.update(image_dict)

        sample["bbox_xyxy"] = boxes
        sample["label"] = labels

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_target(self, annot_df: gpd.GeoDataFrame) -> tuple[Tensor, Tensor]:
        """Load targets from athe GeoParquet dataframe.

        Args:
            annot_df: df subset with annotations for specific image

        Returns:
            bounding boxes and labels
        """
        boxes = torch.from_numpy(
            annot_df[["xmin", "ymin", "xmax", "ymax"]].values
        ).float()

        labels = torch.Tensor(
            [
                self.class2idx[label]
                if label is not None
                else self.class2idx["Unknown White"]
                for label in annot_df["label"].tolist()
            ]
        ).long()
        return boxes, labels
