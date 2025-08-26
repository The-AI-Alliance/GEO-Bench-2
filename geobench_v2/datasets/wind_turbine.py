# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""WindTurbine dataset."""

import io
import re
import warnings
from pathlib import Path

import geopandas as gpd
import rasterio
import torch
import torch.nn as nn
from rasterio.errors import NotGeoreferencedWarning
from torch import Tensor

from .base import GeoBenchBaseDataset
from .normalization import ZScoreNormalizer
from .sensor_util import DatasetBandRegistry


class GeoBenchWindTurbine(GeoBenchBaseDataset):
    """Implementation of WindTurbine dataset."""

    url = "https://hf.co/datasets/aialliance/wind_turbine/resolve/main/{}"

    paths = ["geobench_wind_turbine.tortilla"]

    sha256str = ["ce04744472ce5ef4159575a31c662aa5d4b68ba068c7079f26ac60816ddfe23c"]

    classes = ("windTurbine",)
    num_classes = len(classes)

    dataset_band_config = DatasetBandRegistry.WINDTURBINE
    band_default_order = ("red", "green", "blue")
    normalization_stats = {
        "means": {"red": 0.0, "green": 0.0, "blue": 0.0},
        "stds": {"red": 255.0, "green": 255.0, "blue": 255.0},
    }

    def __init__(
        self,
        root: Path,
        split: str,
        band_order: list[str] = band_default_order,
        data_normalizer: type[nn.Module] = ZScoreNormalizer,
        transforms: nn.Module | None = None,
        download: bool = False,
    ) -> None:
        """Initialize WindTurbine dataset.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'validation', 'test'
            band_order: The order of bands to return, defaults to ['red', 'green', 'blue'], if one would
                specify ['red', 'green', 'blue', 'blue'], the dataset would return images with 4 channels
                in that order. This is useful for models that expect a certain band order, or
                test the impact of band order on model performance.
            data_normalizer: The data normalizer to apply to the data, defaults to :class:`data_util.ZScoreNormalizer`,
                which applies z-score normalization to each band.
            transforms: image transformations to apply to the data, defaults to None
            download: Whether to download the dataset
        """
        super().__init__(
            root=root,
            split=split,
            band_order=band_order,
            data_normalizer=data_normalizer,
            transforms=transforms,
            metadata=None,
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
            [self.class2idx[label] for label in annot_df["label"].tolist()]
        ).long()
        return boxes, labels
