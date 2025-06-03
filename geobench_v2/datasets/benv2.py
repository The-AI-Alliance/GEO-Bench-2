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

    url = "https://hf.co/datasets/aialliance/benv2/resolve/main/{}"

    paths: Sequence[str] = ["geobench_benv2.tortilla"]

    sha256str: Sequence[str] = [
        "e1a3b214bd6118d39ec2c0c34b310de7b8e048b4914f8aa52aa6b24625c2b286"
    ]

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
            "B01": 355.96197509765625,
            "B02": 414.3730773925781,
            "B03": 594.096435546875,
            "B04": 559.0433959960938,
            "B05": 919.4099731445312,
            "B06": 1794.6605224609375,
            "B07": 2091.45947265625,
            "B08": 2241.517822265625,
            "B8A": 2288.0302734375,
            "B09": 2289.5380859375,
            "B11": 1556.958740234375,
            "B12": 973.8273315429688,
            "VH": -12.091922760009766,
            "VV": -18.96333885192871,
        },
        "stds": {
            "B01": 512.3419799804688,
            "B02": 541.94921875,
            "B03": 532.579833984375,
            "B04": 607.0200805664062,
            "B05": 646.341064453125,
            "B06": 1041.35009765625,
            "B07": 1231.787841796875,
            "B08": 1340.4661865234375,
            "B8A": 1316.02880859375,
            "B09": 1267.3955078125,
            "B11": 984.2933349609375,
            "B12": 753.2081909179688,
            "VH": 4.574888229370117,
            "VV": 5.396073818206787,
        },
    }

    # paths: Sequence[str] = (
    #     "FullBenV2.0000.part.tortilla",
    #     "FullBenV2.0001.part.tortilla",
    #     "FullBenV2.0002.part.tortilla",
    # )

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
        band_order: dict[str, Sequence[float | str]] = {"s2": ["B04", "B03", "B02"]},
        data_normalizer: Type[nn.Module] = MultiModalNormalizer,
        transforms: nn.Module | None = None,
        metadata: Sequence[str] = None,
        return_stacked_image: bool = False,
        download: bool = False,
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
            download=download,
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
