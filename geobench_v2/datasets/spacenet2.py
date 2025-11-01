# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""SpaceNet2 dataset."""

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


class GeoBenchSpaceNet2(GeoBenchBaseDataset):
    """GeoBench version of SpaceNet2 dataset.

    Building footprint segmentation dataset using high-resolution optical imagery,
    with binary building masks.

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2102.11958
    """

    url = "https://hf.co/datasets/aialliance/spacenet2/resolve/main/{}"

    paths = ["geobench_spacenet2.tortilla"]

    sha256str = ["e48e57654c1755a6b1c79bcfe172035a27b815ebe1738d1ce95d96d5c37214b6"]

    dataset_band_config = DatasetBandRegistry.SPACENET2

    normalization_stats: dict[str, dict[str, float]] = {
        "means": {
            "coastal": 298.7280578613281,
            "blue": 358.0099182128906,
            "green": 464.5103759765625,
            "yellow": 419.947265625,
            "red": 333.60040283203125,
            "red_edge": 408.66888427734375,
            "nir1": 475.084228515625,
            "nir2": 362.3487243652344,
            "pan": 468.57403564453125,
        },
        "stds": {
            "coastal": 106.97924041748047,
            "blue": 148.18682861328125,
            "green": 224.40948486328125,
            "yellow": 225.79014587402344,
            "red": 194.02330017089844,
            "red_edge": 208.45565795898438,
            "nir1": 234.758544921875,
            "nir2": 193.23211669921875,
            "pan": 260.8954162597656,
        },
    }

    band_default_order = {
        "worldview": (
            "coastal",
            "blue",
            "green",
            "yellow",
            "red",
            "red_edge",
            "nir1",
            "nir2",
        ),
        "pan": ("pan",),
    }

    classes = ("background", "no-building", "building")

    num_classes = len(classes)

    metadata = ("lat", "lon")

    def __init__(
        self,
        root: Path,
        split: Literal["train", "val", "validation", "test"],
        band_order: list[str] = band_default_order,
        data_normalizer: type[nn.Module] = ZScoreNormalizer,
        label_type: Literal["instance_seg", "semantic_seg"] = "semantic_seg",
        transforms: nn.Module | None = None,
        metadata: Sequence[str] | None = None,
        return_stacked_image: bool = False,
        download: bool = False,
    ) -> None:
        """Initialize SpaceNet2 dataset.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'validation', 'test'
            label_type: The type of label to return, supports 'instance_seg' or 'semantic_seg'
            band_order: The order of bands to return, defaults to ['red', 'green', 'blue'], if one would
                specify ['red', 'green', 'blue', 'blue', 'blue'], the dataset would return images with 5 channels
                in that order. This is useful for models that expect a certain band order, or
                test the impact of band order on model performance.
            data_normalizer: The data normalizer to apply to the data, defaults to :class:`data_util.ZScoreNormalizer`,
                which applies z-score normalization to each band.
            label_type: The type of label to return, supports 'instance_seg' or 'semantic_seg'
            transforms: The transforms to apply to the data, defaults to None
            metadata: metadata names to be returned as part of the sample in the
                __getitem__ method. If None, no metadata is returned.
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
        self.label_type = label_type
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

        multi_path = sample_row.read(0)
        pan_path = sample_row.read(1)
        segmentation_path = sample_row.read(2)
        instance_path = sample_row.read(3)

        with rasterio.open(multi_path) as multi_src, rasterio.open(pan_path) as pan_src:
            multi_img: np.ndarray = multi_src.read(out_dtype="float32")
            pan_img: np.ndarray = pan_src.read(out_dtype="float32")

        multi_img = torch.from_numpy(multi_img).float()
        pan_img = torch.from_numpy(pan_img).float()

        image_dict = self.rearrange_bands(
            {"worldview": multi_img, "pan": pan_img}, self.band_order
        )
        image_dict = self.data_normalizer(image_dict)

        sample.update(image_dict)

        if self.label_type == "instance_seg":
            with rasterio.open(instance_path) as instance_src:
                mask: np.ndarray = instance_src.read()
        else:
            with rasterio.open(segmentation_path) as mask_src:
                mask: np.ndarray = mask_src.read()

        # We add 1 to the mask to map the current {background, building} labels to
        # the values {1, 2}. to have a true background class.
        sample["mask"] = torch.from_numpy(mask).long().squeeze(0) + 1

        if self.transforms is not None:
            sample = self.transforms(sample)

        if self.return_stacked_image:
            sample = {
                "image": torch.cat(
                    [sample[f"image_{key}"] for key in self.band_order.keys()], 0
                ),
                "mask": sample["mask"],
            }

        point = wkt.loads(sample_row.iloc[0]["stac:centroid"])
        lon, lat = point.x, point.y

        if "lon" in self.metadata:
            sample["lon"] = torch.tensor(lon)
        if "lat" in self.metadata:
            sample["lat"] = torch.tensor(lat)

        return sample
