# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""SpaceNet8 dataset."""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

import numpy as np
import rasterio
import torch
import torch.nn as nn
from shapely import wkt
from torch import Tensor

from .base import GeoBenchBaseDataset
from .normalization import ZScoreNormalizer
from .sensor_util import DatasetBandRegistry


class GeoBenchSpaceNet8(GeoBenchBaseDataset):
    """GeoBench version of SpaceNet8 dataset."""

    url = "https://hf.co/datasets/aialliance/spacenet8/resolve/main/{}"

    sha256str = ["1d11c38a775bafc5a0790bac3b257b02203b8f0f2c6e285bebccb2917dd3d3ed"]

    # paths = ["SpaceNet8.tortilla"]
    paths = ["geobench_spacenet8.tortilla"]

    dataset_band_config = DatasetBandRegistry.SPACENET8

    normalization_stats: dict[str, dict[str, float]] = {
        "means": {
            "red": 65.36776733398438,
            "green": 84.85777282714844,
            "blue": 57.087120056152344,
        },
        "stds": {
            "red": 44.107696533203125,
            "green": 37.45336151123047,
            "blue": 35.882049560546875,
        },
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
        split: Literal["train", "val", "validation", "test"],
        band_order: list[str] = band_default_order,
        data_normalizer: type[nn.Module] = ZScoreNormalizer,
        transforms: nn.Module | None = None,
        metadata: Sequence[str] | None = None,
        return_stacked_image: bool = False,
        time_step: Sequence[str] = ["pre", "post"],
        download: bool = False,
    ) -> None:
        """Initialize SpaceNet8 dataset.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'validation', 'test'
            band_order: The order of bands to return, defaults to ['red', 'green', 'blue'], if one would
                specify ['red', 'green', 'blue', 'blue', 'blue'], the dataset would return images with 5 channels
                in that order. This is useful for models that expect a certain band order, or
                test the impact of band order on model performance.
            data_normalizer: The data normalizer to apply to the data, defaults to :class:`data_util.ZScoreNormalizer`,
            transforms: The transforms to apply to the data, defaults to None
            metadata: metadata names to be returned as part of the sample in the
                __getitem__ method. If None, no metadata is returned.
            time_step: list of image time steps to include from the list ["pre", "post"]
            return_stacked_image: if true, returns a single image tensor with all modalities stacked in band_order
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
        self.return_stacked_image = return_stacked_image

        if len(time_step) == 0:
            raise ValueError(
                "time_step must include at least one item from  ['pre', 'post']"
            )
        for i in time_step:
            assert i in ["pre", "post"], (
                "time_step must include at least one item from  ['pre', 'post']"
            )
        self.time_step = time_step

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

        with rasterio.open(pre_event_path) as pre_src:
            pre_image: np.ndarray = pre_src.read(out_dtype="float32")
        with rasterio.open(post_event_path) as post_src:
            post_image: np.ndarray = post_src.read(out_dtype="float32")
        with rasterio.open(mask_path) as mask_src:
            mask: np.ndarray = mask_src.read()

        image_pre = torch.from_numpy(pre_image).float()
        image_post = torch.from_numpy(post_image).float()
        mask = torch.from_numpy(mask).long().squeeze(0)

        image_pre = self.rearrange_bands(image_pre, self.band_order)
        image_pre = self.data_normalizer(image_pre)
        image_post = self.rearrange_bands(image_post, self.band_order)
        image_post = self.data_normalizer(image_post)

        if "pre" in self.time_step:
            sample["image_pre"] = image_pre["image"]
        if "post" in self.time_step:
            sample["image_post"] = image_post["image"]

        if self.return_stacked_image:
            if len(self.time_step) > 1:
                sample = {  # [C, T, H, W] == [C, T, H, W]
                    "image": torch.stack(
                        [
                            img
                            for key, img in sample.items()
                            if key.startswith("image_")
                        ],
                        dim=1,
                    )
                }
            else:
                sample = {  # [C, H, W]
                    "image": torch.cat(
                        [
                            img
                            for key, img in sample.items()
                            if key.startswith("image_")
                        ],
                        0,
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
