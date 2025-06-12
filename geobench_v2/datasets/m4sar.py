# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""M4SAR dataset."""

import io
import re
import warnings
from pathlib import Path
from typing import Literal

import geopandas as gpd
import rasterio
import torch
import torch.nn as nn
from rasterio.errors import NotGeoreferencedWarning
from torch import Tensor

from .base import GeoBenchBaseDataset
from .normalization import ZScoreNormalizer
from .sensor_util import DatasetBandRegistry


class GeoBenchM4SAR(GeoBenchBaseDataset):
    """Implementation of M4SAR dataset."""

    url = "https://hf.co/datasets/aialliance/m4sar/resolve/main/{}"

    paths = ["geobench_m4sar.tortilla"]

    sha256str = ["ed4f88952d1dd55cff64aeaf33d37e894666fe505699eb03b483fb951c25159c"]

    # https://github.com/wchao0601/M4-SAR/blob/cac3a3633a976c281419e1d1cc83c27df5222ddc/vis-predict-label.py#L6
    classes = ("bridge", "harbor", "oil_tank", "playground", "airport", "wind_turbine")
    num_classes = len(classes)

    dataset_band_config = DatasetBandRegistry.M4SAR
    band_default_order = {"optical": ("red", "green", "blue"), "sar": ("VH",)}
    normalization_stats = {
        "means": {"red": 0.0, "green": 0.0, "blue": 0.0, "vh": 0.0},
        "stds": {"red": 255.0, "green": 255.0, "blue": 255.0, "vh": 1.0},
    }

    def __init__(
        self,
        root: Path,
        split: str,
        band_order: list[str] = band_default_order,
        data_normalizer: type[nn.Module] = ZScoreNormalizer,
        transforms: nn.Module | None = None,
        bbox_orientation: Literal["horizontal", "oriented"] = "oriented",
        download: bool = False,
    ) -> None:
        """Initialize M4SAR dataset.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'val', 'test'
            band_order: The order of bands to return, defaults to ['red', 'green', 'blue'], if one would
                specify ['red', 'green', 'blue', 'blue'], the dataset would return images with 4 channels
                in that order. This is useful for models that expect a certain band order, or
                test the impact of band order on model performance.
            data_normalizer: The data normalizer to apply to the data, defaults to :class:`data_util.ZScoreNormalizer`,
                which applies z-score normalization to each band.
            transforms: image transformations to apply to the data, defaults to None
            bbox_orientation: The orientation of the bounding boxes, defaults to 'oriented'.
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

        self.bbox_orientation = bbox_orientation

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

        optical_path = sample_row.read(0)
        sar_path = sample_row.read(1)
        annot_path = sample_row.read(2)

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

        image_sample = {}
        if "optical" in self.band_order:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
                with rasterio.open(optical_path) as src:
                    image = torch.from_numpy(src.read()).float()
                image_sample["optical"] = image
        if "sar" in self.band_order:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
                with rasterio.open(sar_path) as src:
                    # 3 channels repeated for SAR, so just pick the first one
                    image = torch.from_numpy(src.read()).float()[0:1, :, :]
                image_sample["sar"] = image

        image_dict = self.rearrange_bands(image_sample, self.band_order)

        image_dict = self.data_normalizer(image_dict)

        sample.update(image_dict)

        if self.bbox_orientation == "horizontal":
            sample["bbox_xyxy"] = boxes
        else:
            sample["bbox_xyxyxyxy"] = boxes

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
        if self.bbox_orientation == "horizontal":
            boxes = torch.from_numpy(
                annot_df[["xmin", "ymin", "xmax", "ymax"]].values
            ).float()
        else:
            boxes = torch.from_numpy(
                annot_df[["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]].values
            ).float()

        labels = torch.Tensor(annot_df["class_id"].tolist()).long()
        return boxes, labels
