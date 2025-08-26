# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""BRIGHT dataset."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
from shapely import wkt
from torch import Tensor

from .base import GeoBenchBaseDataset
from .normalization import ZScoreNormalizer
from .sensor_util import DatasetBandRegistry


class GeoBenchBRIGHT(GeoBenchBaseDataset):
    """BRIGHT dataset with enhanced functionality.

    Allows:
    - Variable Band Selection
    - Return band wavelengths

    3 classes: background, building or road (not flooded), building or road (flooded)
    0. ackground
    # TODO
    but maybe also 5 classes?
    """

    url = "https://hf.co/datasets/aialliance/bright/resolve/main/{}"

    paths = ["BRIGHT.tortilla"]

    sha256str = [""]
    dataset_band_config = DatasetBandRegistry.BRIGHT

    normalization_stats = {
        "means": {"r": 0.0, "g": 0.0, "b": 0.0, "sar": 0.0},
        "stds": {"r": 255.0, "g": 255.0, "b": 255.0, "sar": 255.0},
    }

    band_default_order = {"aerial": ("red", "green", "blue"), "sar": ("sar",)}

    classes = ("background", "no damage", "minor damage", "major damage", "destroyed")

    num_classes = len(classes)

    valid_metadata = ("lat", "lon")

    def __init__(
        self,
        root: Path,
        split: str,
        band_order: dict[str, Sequence[str]] = band_default_order,
        data_normalizer: type[nn.Module] = ZScoreNormalizer,
        transforms: nn.Module = None,
        metadata: Sequence[str] | None = None,
        download: bool = False,
    ) -> None:
        """Initialize BRIGHT dataset.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'validation', 'test'
            band_order: The order of bands to return, defaults to ['red', 'green', 'blue'], if one would
                specify ['red', 'green', 'blue', 'blue', 'blue'], the dataset would return images with 5 channels
                in that order. This is useful for models that expect a certain band order, or
                test the impact of band order on model performance. The aerial bands are for the pre-event image, while
                the sar bands are for the post-event image.
            data_normalizer: The normalizer to use, defaults to ZScoreNormalizer, which normalizes the data
            transforms: The transforms to apply to the data, defaults to None
            metadata: metadata names to be returned as part of the sample in the
                __getitem__ method. If None, no metadata is returned.
            download: Whether to download the dataset
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
        mask = torch.from_numpy(mask).long()

        image_dict = self.rearrange_bands(
            {"aerial": image_pre, "sar": image_post}, self.band_order
        )
        image_dict = self.data_normalizer(image_dict)

        sample["image_pre"] = image_dict["image_aerial"]
        sample["image_post"] = image_dict["image_sar"]

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
