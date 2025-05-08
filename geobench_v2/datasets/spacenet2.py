# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""SpaceNet2 dataset."""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
import torch
import torch.nn as nn
from shapely import wkt
from torch import Tensor

from .base import GeoBenchBaseDataset
from .data_util import ClipZScoreNormalizer
from .sensor_util import DatasetBandRegistry


class GeoBenchSpaceNet2(GeoBenchBaseDataset):
    """SpaceNet2 dataset with enhanced functionality.

    Allows:
    - Variable Band Selection
    - Return band wavelengths

    2 classes: background, building
    """

    url = "https://hf.co/datasets/aialliance/spacenet2/resolve/main/{}"

    # paths = [
    #     "SpaceNet2.0000.part.tortilla",
    #     "SpaceNet2.0001.part.tortilla",
    #     "SpaceNet2.0002.part.tortilla",
    #     "SpaceNet2.0003.part.tortilla",
    #     "SpaceNet2.0004.part.tortilla",
    # ]

    paths = [
        "geobench_spacenet2.0000.part.tortilla",
        "geobench_spacenet2.0001.part.tortilla",
        "geobench_spacenet2.0002.part.tortilla",
    ]

    sha256str = ["", "", ""]

    dataset_band_config = DatasetBandRegistry.SPACENET2

    normalization_stats = {
        "means": {
            "coastal": 0.0,
            "blue": 0.0,
            "green": 0.0,
            "yellow": 0.0,
            "red": 0.0,
            "red_edge": 0.0,
            "nir1": 0.0,
            "nir2": 0.0,
            "pan": 0.0,
        },
        "stds": {
            "coastal": 3000.0,
            "blue": 3000.0,
            "green": 3000.0,
            "yellow": 3000.0,
            "red": 3000.0,
            "red_edge": 3000.0,
            "nir1": 3000.0,
            "nir2": 3000.0,
            "pan": 3000.0,
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
        split: str,
        band_order: list[str] = band_default_order,
        data_normalizer: type[nn.Module] = ClipZScoreNormalizer,
        label_type: Literal["instance_seg", "semantic_seg"] = "semantic_seg",
        transforms: nn.Module = None,
        metadata: Sequence[str] | None = None,
        return_stacked_image: bool = False,
        download: bool = False,
    ) -> None:
        """Initialize SpaceNet2 dataset.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'val', 'test'
            label_type: The type of label to return, supports 'instance_seg' or 'semantic_seg'
            band_order: The order of bands to return, defaults to ['red', 'green', 'blue'], if one would
                specify ['red', 'green', 'blue', 'blue', 'blue'], the dataset would return images with 5 channels
                in that order. This is useful for models that expect a certain band order, or
                test the impact of band order on model performance.
            data_normalizer:
            label_type:
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
                    [val for key, val in sample.items() if key.startswith("image_")], 0
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
