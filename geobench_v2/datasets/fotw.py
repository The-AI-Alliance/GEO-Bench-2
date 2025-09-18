# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Fields of the World Dataset."""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import rasterio
import torch
import torch.nn as nn
from shapely import wkt
from torch import Tensor

from .base import GeoBenchBaseDataset
from .normalization import ZScoreNormalizer
from .sensor_util import DatasetBandRegistry


class GeoBenchFieldsOfTheWorld(GeoBenchBaseDataset):
    """Fields of the World Dataset with enhanced functionality.

    Allows:
    - Variable Band Selection
    - Return band wavelengths
    """

    url = "https://hf.co/datasets/aialliance/fotw/resolve/main/{}"
    # paths = [
    #     "FullFOTW.0000.part.tortilla",
    #     "FullFOTW.0001.part.tortilla",
    #     "FullFOTW.0002.part.tortilla",
    #     "FullFOTW.0003.part.tortilla",
    # ]

    paths = ["geobench_fotw.tortilla"]

    sha256str = ["7b422acc120b99f3cf4e8389a28616f257ea81016073d9ee529699fcda667763"]

    dataset_band_config = DatasetBandRegistry.FOTW

    # keys should be specified according to the sensor default values
    # defined in sensor_util.py
    band_default_order = ("red", "green", "blue", "nir")

    # Define normalization stats using canonical names
    normalization_stats: dict[str, dict[str, float]] = {
        "means": {"red": 0.0, "green": 0.0, "blue": 0.0, "nir": 0.0},
        "stds": {"red": 3000.0, "green": 3000.0, "blue": 3000.0, "nir": 3000.0},
    }

    classes = ("background", "field", "field-boundary")
    num_classes = len(classes)

    valid_metadata = ("lat", "lon")

    # TODO maybe add country argument?
    def __init__(
        self,
        root: Path,
        split: str,
        band_order: Sequence[str | float] = dataset_band_config.default_order,
        data_normalizer: type[nn.Module] = ZScoreNormalizer,
        label_type: Literal["instance_seg", "semantic_seg"] = "semantic_seg",
        transforms: nn.Module | None = None,
        metadata: Sequence[str] | None = None,
        return_stacked_image: bool = False,
        download: bool = False,
    ) -> None:
        """Initialize Fields of the World Dataset.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'validation', 'test'
            band_order: The order of bands to return, defaults to ['red', 'green', 'blue', 'nir'], if one would
                specify ['red', 'green', 'blue', 'nir', 'nir'], the dataset would return images with 5 channels
                in that order. This is useful for models that expect a certain band order, or
                test the impact of band order on model performance.
            data_normalizer: The data normalizer to apply to the data, defaults to :class:`data_util.ZScoreNormalizer`,
                which applies z-score normalization to each band.
            label_type: The type of label to return, supports 'instance_seg' or 'semantic_seg'
            transforms: The transforms to apply to the data, defaults to None
            return_stacked_image: if true, returns a single image tensor with all modalities stacked in band_order
            metadata: metadata names to be returned under specified keys as part of the sample in the
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

        self.label_type = label_type
        self.return_stacked_image = return_stacked_image

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Return the image and mask at the given index.

        Args:
            idx: index of the image and mask to return

        Returns:
            dict: a dict containing the image and mask
        """
        sample: dict[str, Tensor] = {}

        sample_row = self.data_df.read(idx)

        win_a_path = sample_row.read(0)
        win_b_path = sample_row.read(1)

        mask_path = (
            sample_row.read(2)
            if self.label_type == "instance_seg"
            else sample_row.read(3)
        )

        with rasterio.open(win_a_path) as win_a_src:
            win_a = win_a_src.read()
        with rasterio.open(win_b_path) as win_b_src:
            win_b = win_b_src.read()
        with rasterio.open(mask_path) as mask_src:
            mask = mask_src.read(1)

        win_a = torch.from_numpy(win_a).float()
        win_b = torch.from_numpy(win_b).float()
        mask = torch.from_numpy(mask).long()

        # TODO how to handle window a and b?
        win_a = self.rearrange_bands(win_a, self.band_order)
        win_a = self.data_normalizer(win_a)

        win_b = self.rearrange_bands(win_b, self.band_order)
        win_b = self.data_normalizer(win_b)

        sample["image_a"] = win_a["image"]
        sample["image_b"] = win_b["image"]

        if self.return_stacked_image:
            sample: dict[str, Tensor] = {
                "image": torch.cat([sample["image_a"], sample["image_b"]], dim=0)
            }

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
