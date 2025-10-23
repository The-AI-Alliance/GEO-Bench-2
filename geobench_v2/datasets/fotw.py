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

    Field boundary segmentation dataset using multi-temporal Sentinel-2 imagery,
    with field mask annotations across diverse regions.

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2409.16252
    """

    url = "https://hf.co/datasets/aialliance/fotw/resolve/main/{}"

    paths = ["geobench_fotw.tortilla"]

    sha256str = ["2584cdada3dd3c275792d2376c8bbca782a7b1faea54924c9627906f0296854c"]

    dataset_band_config = DatasetBandRegistry.FOTW

    band_default_order = ("red", "green", "blue", "nir")

    # Define normalization stats using canonical names
    normalization_stats: dict[str, dict[str, float]] = {
        "means": {
            "red": 862.0840454101562,
            "green": 853.3894653320312,
            "blue": 592.0079956054688,
            "nir": 2984.3017578125,
        },
        "stds": {
            "red": 681.1666870117188,
            "green": 508.64013671875,
            "blue": 454.0238952636719,
            "nir": 1043.6527099609375,
        },
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
        return_window: Sequence[str] = ["win_a", "win_b"],
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
            return_window: select which windows to return
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
        for i in return_window:
            assert i in ["win_a", "win_b"], (
                "return_window can only include items from ['win_a, , 'win_b']"
            )
        self.return_window = return_window
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

        if "win_a" in self.return_window:
            sample["image_a"] = win_a["image"]
        if "win_b" in self.return_window:
            sample["image_b"] = win_b["image"]

        if self.return_stacked_image:
            sample = {"image": torch.cat([sample[key] for key in self.sample], 0)}

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
