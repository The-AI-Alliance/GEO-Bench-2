# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

# Adapted from torchgeo dataset loader from tortilla format
# https://github.com/microsoft/torchgeo/blob/main/torchgeo/datasets/bigearthnet.py

"""BigEarthNet V2 Dataset."""

from torch import Tensor
from torchgeo.datasets import SpaceNet6
from pathlib import Path
from typing import Sequence, Type
import torch.nn as nn

from .sensor_util import DatasetBandRegistry
from .base import GeoBenchBaseDataset
from .data_util import MultiModalNormalizer
import torch.nn as nn
import rasterio
import numpy as np
import torch

from rasterio.enums import Resampling


class GeoBenchBENV2(GeoBenchBaseDataset):
    """BigEarthNet V2 Dataset with enhanced functionality.

    Allows:
    - Variable Band Selection
    - Return band wavelengths
    """

    band_default_order = {
        "s2": (
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B09",
            "B11",
            "B12",
        ),
        "s1": ("VV", "VH"),
    }

    dataset_band_config = DatasetBandRegistry.BENV2

    normalization_stats: dict[str, dict[str, float]] = {
        "means": {
            "B01": 0.0,
            "B02": 0.0,
            "B03": 0.0,
            "B04": 0.0,
            "B05": 0.0,
            "B06": 0.0,
            "B07": 0.0,
            "B08": 0.0,
            "B8A": 0.0,
            "B09": 0.0,
            "B11": 0.0,
            "B12": 0.0,
            "VH": 0.0,
            "VV": 0.0,
        },
        "stds": {
            "B01": 3000.0,
            "B02": 3000.0,
            "B03": 3000.0,
            "B04": 3000.0,
            "B05": 3000.0,
            "B06": 3000.0,
            "B07": 3000.0,
            "B08": 3000.0,
            "B8A": 3000.0,
            "B09": 3000.0,
            "B11": 3000.0,
            "B12": 3000.0,
            "VH": 3000.0,
            "VV": 3000.0,
        },
    }

    # paths: Sequence[str] = (
    #     "FullBenV2.0000.part.tortilla",
    #     "FullBenV2.0001.part.tortilla",
    #     "FullBenV2.0002.part.tortilla",
    # )

    paths: Sequence[str] = ["geobench_benv2.tortilla"]

    label_names: Sequence[str] = (
        "Urban fabric",
        "Industrial or commercial units",
        "Arable land",
        "Permanent crops",
        "Pastures",
        "Complex cultivation patterns",
        "Land principally occupied by agriculture, with significant areas of natural vegetation",
        "Agro-forestry areas",
        "Broad-leaved forest",
        "Coniferous forest",
        "Mixed forest",
        "Natural grassland and sparsely vegetated areas",
        "Moors, heathland and sclerophyllous vegetation",
        "Transitional woodland, shrub",
        "Beaches, dunes, sands",
        "Inland wetlands",
        "Coastal wetlands",
        "Inland waters",
        "Marine waters",
    )

    classes = label_names

    num_classes: int = len(label_names)

    valid_metadata: Sequence[str] = ("lat", "lon")

    def __init__(
        self,
        root: Path,
        split: str,
        band_order: dict[str, Sequence[float | str]] = band_default_order,
        data_normalizer: Type[nn.Module] = MultiModalNormalizer,
        transforms: nn.Module | None = None,
        metadata: Sequence[str] = None,
        return_stacked_image: bool = False,
    ) -> None:
        """Initialize Big Earth Net V2 Dataset.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'val', 'test'
            band_order: The order of bands to return, defaults to ['B04', 'B03', 'B02'], if one would
                specify ['B04', 'B03', 'B02], the dataset would return the red, green, and blue bands.
                This is useful for models that expect a certain band order, or
                test the impact of band order on model performance.
            data_normalizer: The data normalizer to apply to the data, defaults to :class:`data_util.MultiModalNormalizer`,
                which applies z-score normalization to each band.
            transforms: Transforms to apply to the data
            metadata: metadata names to be returned under specified keys as part of the sample in the
                __getitem__ method. If None, no metadata is returned.
            return_stacked_image: If True, return the stacked modalities across channel dimension instead of the individual modalities.
        """
        super().__init__(
            root=root,
            split=split,
            band_order=band_order,
            data_normalizer=data_normalizer,
            transforms=transforms,
            metadata=metadata,
        )

        self.return_stacked_image = return_stacked_image

        self.class2idx = {c: i for i, c in enumerate(self.label_names)}

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        sample: dict[str, Tensor] = {}

        sample_row = self.data_df.read(index)

        # order is vv, vh
        s1_path = sample_row.read(0)
        s2_path = sample_row.read(1)

        data: dict[str, Tensor] = {}

        if "s1" in self.band_order:
            with rasterio.open(s1_path) as src:
                s1_img = src.read()
            data["s1"] = torch.from_numpy(s1_img).float()
        if "s2" in self.band_order:
            with rasterio.open(s2_path) as src:
                s2_img = src.read()
            data["s2"] = torch.from_numpy(s2_img).float()

        # Rearrange bands and normalize
        img = self.rearrange_bands(data, self.band_order)
        img = self.data_normalizer(img)
        sample.update(img)

        if self.return_stacked_image:
            sample = {
                "image": torch.cat(
                    [sample[f"image_{key}"] for key in self.band_order.keys()], 0
                )
            }

        if self.transforms is not None:
            sample = self.transforms(sample)

        sample["label"] = self._load_target(sample_row.iloc[0]["labels"])

        if "lon" in self.metadata:
            sample["lon"] = torch.tensor(sample_row.iloc[0]["lon"])
        if "lat" in self.metadata:
            sample["lat"] = torch.tensor(sample_row.iloc[0]["lat"])

        return sample

    def _load_target(self, label_names: list[str]) -> Tensor:
        """Load the target mask for a single image.

        Args:
            label_names: list of labels

        Returns:
            the target label
        """
        indices = [self.class2idx[label_names] for label_names in label_names]

        image_target = torch.zeros(self.num_classes, dtype=torch.long)
        image_target[indices] = 1
        return image_target
