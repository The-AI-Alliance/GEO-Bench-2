# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""CaFFe Dataset."""

import os

from typing import Sequence, Type
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchgeo.datasets import CaFFe
from pathlib import Path
import pandas as pd
import torch.nn as nn
import rasterio
from shapely import wkt

from .sensor_util import DatasetBandRegistry
from .data_util import MultiModalNormalizer
from .base import GeoBenchBaseDataset


class GeoBenchCaFFe(GeoBenchBaseDataset):
    """GeoBench Caffe dataset."""

    url = "https://hf.co/datasets/aialliance/caffe/resolve/main/{}"
    paths = ["geobench_caffe.tortilla"]
    sha256str = [""]

    dataset_band_config = DatasetBandRegistry.CAFFE
    # TODO update sensor type with wavelength and resolution

    band_default_order = ("gray",)

    normalization_stats = {"means": {"gray": 0.0}, "stds": {"gray": 255.0}}

    mask_dirs = ("zones", "zones")

    classes = ("N/A", "rock", "glacier", "ocean/ice melange")

    num_classes = len(classes)

    def __init__(
        self,
        root,
        split="train",
        band_order: Sequence[float | str] = ["r", "g", "b"],
        data_normalizer: Type[nn.Module] = MultiModalNormalizer,
        transforms: nn.Module | None = None,
        metadata: Sequence[str] | None = None,
        download: bool = False,
    ):
        """Initialize Caffe dataset.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'val', 'test'
            band_order: The order of bands to return, defaults to ['gray'], if one would
                specify ['gray', 'gray', 'gray], the dataset would return the gray band three times.
            data_normalizer: The data normalizer to apply to the data, defaults to :class:`data_util.MultiModalNormalizer`,
                which applies z-score normalization to each band.
            transforms:

        Raises:
            AssertionError: If split is not in the splits
        """
        super().__init__(
            root=root,
            split=split,
            band_order=band_order,
            data_normalizer=data_normalizer,
            transforms=transforms,
            metadata=metadata,
            download=download,
        )

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            idx: index to return

        Returns:
            data and label at that index
        """
        sample: dict[str, Tensor] = {}

        sample_row = self.data_df.read(idx)

        img_path = sample_row.read(0)
        mask_path = sample_row.read(1)

        with rasterio.open(img_path) as f:
            image = f.read()
        image = torch.from_numpy(image).float()

        with rasterio.open(mask_path) as f:
            mask = f.read(1)
        mask = torch.from_numpy(mask).long()

        image_dict = self.rearrange_bands(image, self.band_order)
        image_dict = self.data_normalizer(image_dict)

        sample.update(image_dict)

        sample["mask"] = mask

        point = wkt.loads(sample_row.iloc[0]["stac:centroid"])
        lon, lat = point.x, point.y

        if "lon" in self.metadata:
            sample["lon"] = torch.tensor(lon)
        if "lat" in self.metadata:
            sample["lat"] = torch.tensor(lat)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


# class GeoBenchCaFFe(CaFFe, DataUtilsMixin):
#     """CaFFe Dataset with enhanced functionality.

#     Allows:
#     - Variable Band Selection
#     - Return band wavelengths
#     """

#     dataset_band_config = DatasetBandRegistry.CAFFE
#     # TODO update sensor type with wavelength and resolution

#     band_default_order = ("gray",)

#     normalization_stats = {"means": {"gray": 0.0}, "stds": {"gray": 255.0}}

#     mask_dirs = ("zones", "zones")

#     classes = ("N/A", "rock", "glacier", "ocean/ice melange")

#     num_classes = len(classes)

#     def __init__(
#         self,
#         root: Path,
#         split: str,
#         band_order: list[str] = band_default_order,
#         data_normalizer: Type[nn.Module] = MultiModalNormalizer,
#         transforms: nn.Module | None = None,
#         metadata: Sequence[str] | None = None,
#     ) -> None:
#         """Initialize CaFFe Dataset.

#         Args:
#             root: Path to the dataset root directory
#             split: The dataset split, supports 'train', 'val', 'test'
#             band_order: The order of bands to return, defaults to ['gray'], if one would
#                 specify ['gray', 'gray', 'gray], the dataset would return the gray band three times.
#                 This is useful for models that expect a certain band order, or
#                 test the impact of band order on model performance.
#             data_normalizer: The data normalizer to apply to the data, defaults to :class:`data_util.MultiModalNormalizer`,
#                 which applies z-score normalization to each band.
#             transforms:
#             metadata: metadata names to be returned under specified keys as part of the sample in the
#                 __getitem__ method. If None, no metadata is returned.
#         """
#         if split == "validation":
#             split = "val"
#         super().__init__(root=root, split=split)
#         self.transforms = transforms

#         self.band_order = self.resolve_band_order(band_order)
#         if metadata is None:
#             self.metadata = []
#         else:
#             self.metadata = metadata

#         self.data_normalizer = data_normalizer(
#             self.normalization_stats, self.band_order
#         )

#         self.metadata_df = pd.read_parquet(
#             os.path.join(self.root, self.data_dir, "geobench_caffe_metadata.parquet")
#         )
#         self.metadata_df = self.metadata_df[
#             self.metadata_df["split"] == self.split
#         ].reset_index(drop=True)

#     def __getitem__(self, idx: int) -> dict[str, Tensor]:
#         """Return the image and mask at the given index.

#         Args:
#             idx: index of the image and mask to return

#         Returns:
#             dict: a dict containing the image and mask
#         """
#         sample: dict[str, Tensor] = {}
#         zones_filename = os.path.basename(self.fpaths[idx])
#         sample_row = self.metadata_df.iloc[idx]
#         img_filename = sample_row["filename"]
#         zones_filename = img_filename.replace("__", "_zones__")

#         def read_tensor(path: str) -> Tensor:
#             return torch.from_numpy(np.array(Image.open(path)))

#         img_path = os.path.join(
#             self.root, self.data_dir, self.image_dir, self.split, img_filename
#         )
#         img = read_tensor(img_path).unsqueeze(0).float()

#         img_dict = self.rearrange_bands(img, self.band_order)

#         img_dict = self.data_normalizer(img_dict)

#         sample.update(img_dict)

#         zone_mask = read_tensor(
#             os.path.join(self.root, self.data_dir, "zones", self.split, zones_filename)
#         ).long()

#         zone_mask = self.ordinal_map_zones[zone_mask]

#         sample["mask"] = zone_mask

#         if "lon" in self.metadata:
#             sample["lon"] = torch.tensor(sample_row["longitude"])
#         if "lat" in self.metadata:
#             sample["lat"] = torch.tensor(sample_row["latitude"])

#         if self.transforms:
#             sample = self.transforms(sample)

#         return sample

#     def __len__(self) -> int:
#         """Return the number of images in the dataset."""
#         return len(self.metadata_df)
