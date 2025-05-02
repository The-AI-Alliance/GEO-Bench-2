# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""SpaceNet8 dataset."""

from torch import Tensor
from torchgeo.datasets import SpaceNet8
from pathlib import Path
from typing import Type, Sequence
from shapely import wkt

from .sensor_util import DatasetBandRegistry
from .data_util import ClipZScoreNormalizer
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
    0. ackground
    # TODO
    but maybe also 5 classes?
    """

    url = "https://hf.co/datasets/aialliance/spacenet8/resolve/main/{}"

    sha256str = ["1d11c38a775bafc5a0790bac3b257b02203b8f0f2c6e285bebccb2917dd3d3ed"]

    # paths = ["SpaceNet8.tortilla"]
    paths = ["geobench_spacenet8.tortilla"]

    dataset_band_config = DatasetBandRegistry.SPACENET8

    normalization_stats = {
        "means": {"red": 0.0, "green": 0.0, "blue": 0.0},
        "stds": {"red": 255.0, "green": 255.0, "blue": 255.0},
    }

    band_default_order = ("red", "green", "blue")

    classes = (
        "background",
        "road (not flooded)",
        "road (flooded)",
        "building (not flooded)",
        "building (flooded)",
    )
    num_classes = len(classes)

    valid_metadata = ("lat", "lon")

    def __init__(
        self,
        root: Path,
        split: str,
        band_order: list[str] = band_default_order,
        data_normalizer: Type[nn.Module] = ClipZScoreNormalizer,
        transforms: nn.Module = None,
        metadata: Sequence[str] | None = None,
        return_stacked_image: bool = False,
        download: bool = False,
    ) -> None:
        """Initialize SpaceNet8 dataset.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'val', 'test'
            band_order: The order of bands to return, defaults to ['red', 'green', 'blue'], if one would
                specify ['red', 'green', 'blue', 'blue', 'blue'], the dataset would return images with 5 channels
                in that order. This is useful for models that expect a certain band order, or
                test the impact of band order on model performance.
            data_normalizer: The data normalizer to apply to the data, defaults to :class:`data_util.ClipZScoreNormalizer`,
            transforms: The transforms to apply to the data, defaults to None
            metadata: metadata names to be returned as part of the sample in the
                __getitem__ method. If None, no metadata is returned.
            return_stacked_image: if true, returns a single image tensor with all modalities stacked in band_order
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
        self.return_stacked_image = return_stacked_image

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
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
        mask = torch.from_numpy(mask).long().squeeze(0)

        image_pre = self.rearrange_bands(image_pre, self.band_order)
        image_pre = self.data_normalizer(image_pre)
        image_post = self.rearrange_bands(image_post, self.band_order)
        image_post = self.data_normalizer(image_post)

        sample["image_pre"] = image_pre["image"]
        sample["image_post"] = image_post["image"]

        if self.return_stacked_image:
            sample = {
                "image": torch.cat(
                    [img for key, img in sample.items() if key.startswith("image_")], 0
                )
            }

        sample["mask"] = mask

        if self.transforms is not None:
            sample = self.transforms(sample)

        point = wkt.loads(sample_row.iloc[0]["stac:centroid"])
        lon, lat = point.x, point.y

        if "lon" in self.metadata:
            sample["lon"] = torch.tensor(lon)
        if "lat" in self.metadata:
            sample["lat"] = torch.tensor(lat)

        return sample
