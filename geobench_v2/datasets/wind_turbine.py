# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""WindTurbine dataset."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor
from torchgeo.datasets import NonGeoDataset

import io
import re

import geopandas as gpd
import warnings
from rasterio.errors import NotGeoreferencedWarning
import rasterio


from .base import GeoBenchBaseDataset
from .data_util import DataUtilsMixin
from .normalization import DataNormalizer, ZScoreNormalizer
from .sensor_util import DatasetBandRegistry


class GeoBenchWindTurbine(GeoBenchBaseDataset):
    """Implementation of WindTurbine dataset."""

    url = "https://hf.co/datasets/aialliance/wind_turbine/resolve/main/{}"

    paths = ["geobench_wind_turbine.tortilla"]

    sha256str = ["b0f1a2c4dc4f3a0b8e2d5f1e7b6c4d9a0b8e2d5f1e7b6c4d9a0b8e2d5f1e7b6c4d9a0"]

    classes = (
        "windTurbine",
    )
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
            split: The dataset split, supports 'train', 'val', 'test'
            band_order: The order of bands to return, defaults to ['red', 'green', 'blue'], if one would
                specify ['red', 'green', 'blue', 'blue'], the dataset would return images with 4 channels
                in that order. This is useful for models that expect a certain band order, or
                test the impact of band order on model performance.
            data_normalizer: The data normalizer to apply to the data, defaults to :class:`data_util.ZScoreNormalizer`,
                which applies z-score normalization to each band.
            transforms: image transformations to apply to the data, defaults to None
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
            index: index to return

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
            annot_df[['xmin', 'ymin', 'xmax', 'ymax']].values
        ).float()
        labels = torch.Tensor(
            [self.class2idx[label] for label in annot_df['label'].tolist()]
        ).long()
        return boxes, labels

# class GeoBenchWindTurbineOld(NonGeoDataset, DataUtilsMixin):
#     """ "GeoBenchWindTurbine dataset with enhanced functionality.

#     Allows:
#     - Variable Band Selection
#     - Return band wavelengths
#     """

#     dataset_band_config = DatasetBandRegistry.WINDTURBINE
#     band_default_order = ("red", "green", "blue")

#     normalization_stats = {
#         "means": {"red": 0.0, "green": 0.0, "blue": 0.0},
#         "stds": {"red": 255.0, "green": 255.0, "blue": 255.0},
#     }

#     classes = classes = "wind_turbine"

#     num_classes = len(classes)

#     def __init__(
#         self,
#         root: Path,
#         split: str,
#         band_order: list[str] = band_default_order,
#         data_normalizer: type[nn.Module] = ZScoreNormalizer,
#         transforms: nn.Module | None = None,
#     ) -> None:
#         """Initialize WindTurbine dataset.

#         Args:
#             root: Path to the dataset root directory
#             split: The dataset split, supports 'train', 'val', 'test'
#             band_order: The order of bands to return, defaults to ['red', 'green', 'blue'], if one would
#                 specify ['red', 'green', 'blue', 'blue'], the dataset would return images with 4 channels
#                 in that order. This is useful for models that expect a certain band order, or
#                 test the impact of band order on model performance.
#             data_normalizer: The data normalizer to apply to the data, defaults to :class:`data_util.ZScoreNormalizer`,
#                 which applies z-score normalization to each band.
#             transforms: image transformations to apply to the data, defaults to None
#         """
#         self.root = root
#         self.split = split

#         self.transforms = transforms

#         self.band_order = self.resolve_band_order(band_order)

#         self.data_df = pd.read_parquet(
#             os.path.join(self.root, "geobench_wind_turbine.parquet")
#         )

#         self.data_df = self.data_df[self.data_df["split"] == split].reset_index(
#             drop=True
#         )

#         if isinstance(data_normalizer, type):
#             print(f"Initializing normalizer from class: {data_normalizer.__name__}")
#             if issubclass(data_normalizer, DataNormalizer):
#                 self.data_normalizer = data_normalizer(
#                     self.normalization_stats, self.band_order
#                 )
#             else:
#                 self.data_normalizer = data_normalizer()

#         elif callable(data_normalizer):
#             print(
#                 f"Using provided pre-initialized normalizer instance: {data_normalizer.__class__.__name__}"
#             )
#             self.data_normalizer = data_normalizer
#         else:
#             raise TypeError(
#                 f"data_normalizer must be a DataNormalizer subclass type or a callable instance. Got {type(data_normalizer)}"
#             )

#     def __len__(self) -> int:
#         """Return the length of the dataset.

#         Returns:
#             length of the dataset
#         """
#         return len(self.data_df)

#     def __getitem__(self, index: int) -> dict[str, Tensor]:
#         """Return an index within the dataset.

#         Args:
#             index: index to return

#         Returns:
#             data and label at that index
#         """
#         sample: dict[str, Tensor] = {}

#         sample_row = self.data_df.iloc[index]

#         img_path = os.path.join(self.root, sample_row["image_path"])

#         image = self._load_image(img_path).float()

#         image_dict = self.rearrange_bands(image, self.band_order)

#         image_dict = self.data_normalizer(image_dict)

#         sample.update(image_dict)

#         label_path = os.path.join(self.root, sample_row["label_path"])

#         boxes, labels = self._load_target(label_path)

#         # absolute coordinates
#         sample["bbox_xyxy"] = boxes * image.shape[1]
#         sample["label"] = labels

#         if self.transforms is not None:
#             sample = self.transforms(sample)

#         return sample

#     def _load_image(self, path: str) -> Tensor:
#         """Load an image from disk.

#         Args:
#             path: Path to the image file.

#         Returns:
#             image tensor
#         """
#         image = Image.open(path).convert("RGB")
#         image = np.array(image)
#         image = torch.from_numpy(image).permute(2, 0, 1)
#         return image.float()

#     def _load_target(self, path: str) -> tuple[Tensor, Tensor]:
#         """Load target annotations from disk.

#         Args:
#             path: path to annotation .txt file in YOLO format

#         Returns:
#             boxes: bounding boxes tensor in xyxy format
#             labels: labels tensor
#         """
#         boxes = []
#         labels = []

#         with open(path) as f:
#             for line in f:
#                 parts = line.strip().split()
#                 if len(parts) >= 5:
#                     # Extract data from YOLO format
#                     class_id = int(parts[0])
#                     x_center = float(parts[1])
#                     y_center = float(parts[2])
#                     width = float(parts[3])
#                     height = float(parts[4])

#                     # Convert from YOLO format (x_center, y_center, width, height)
#                     # to xyxy format (x_min, y_min, x_max, y_max)
#                     x_min = x_center - width / 2
#                     y_min = y_center - height / 2
#                     x_max = x_center + width / 2
#                     y_max = y_center + height / 2

#                     # Clamp to [0, 1]
#                     x_min = max(0, min(1, x_min))
#                     y_min = max(0, min(1, y_min))
#                     x_max = max(0, min(1, x_max))
#                     y_max = max(0, min(1, y_max))

#                     boxes.append([x_min, y_min, x_max, y_max])
#                     labels.append(class_id)

#         if len(boxes) == 0:
#             return torch.zeros((0, 4), dtype=torch.float32), torch.zeros(
#                 0, dtype=torch.int64
#             )

#         return torch.tensor(boxes, dtype=torch.float32), torch.tensor(
#             labels, dtype=torch.int64
#         )
