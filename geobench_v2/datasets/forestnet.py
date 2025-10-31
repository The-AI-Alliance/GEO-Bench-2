# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Forestnet Dataset."""

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


class GeoBenchForestnet(GeoBenchBaseDataset):
    """GeoBench Version of Forestnet Dataset.

    The Forestnet dataset is a classification dataset using Landsat data to identify deforestation events in Indonesia.

    """

    url = "https://hf.co/datasets/aialliance/forestnet/resolve/main/{}"

    paths: Sequence[str] = ["geobench_forestnet.tortilla"]

    sha256str: Sequence[str] = [
        "6ee7cb7135b4ca5d0cde52e781f5960ed0e648dcceab598982fa612802cd3ad1"
    ]

    classes: Sequence[str] = (
        "Oil palm plantation",
        "Timber plantation",
        "Other large-scale plantations",
        "Grassland/shrubland",
        "Small-scale agriculture",
        "Small-scale mixed plantation",
        "Small-scale oil palm plantation",
        "Mining",
        "Fish pond",
        "Logging road",
        "Secondary forest",
        "Other",
    )

    dataset_band_config = DatasetBandRegistry.FORESTNET
    band_default_order = dataset_band_config.default_order

    normalization_stats: dict[str, dict[str, float]] = {
        "means": {
            "B02": 72.37593078613281,
            "B03": 83.18157196044922,
            "B04": 77.08612823486328,
            "B8A": 123.54252624511719,
            "B11": 91.04833221435547,
            "B12": 74.30968475341797,
        },
        "stds": {
            "B02": 16.283870697021484,
            "B03": 15.35866928100586,
            "B04": 16.66645622253418,
            "B8A": 16.948505401611328,
            "B11": 14.280089378356934,
            "B12": 13.285400390625,
        },
    }

    label_names = classes

    num_classes: int = len(label_names)

    def __init__(
        self,
        root: Path,
        split: Literal["train", "val", "validation", "test"],
        band_order: dict[str, Sequence[float | str]] = band_default_order,
        data_normalizer: type[nn.Module] = ZScoreNormalizer,
        transforms: nn.Module | None = None,
        metadata: list[str] = None,
        download: bool = False,
    ) -> None:
        """Initialize Forestnet Dataset.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'validation', 'test'
            band_order: The order of bands to return, defaults to all s2 bands. This is useful for models that expect a certain band order, or
                test the impact of band order on model performance.
            data_normalizer: The data normalizer to apply to the data, defaults to :class:`data_util.ZScoreNormalizer`,
                which applies z-score normalization to each band.
            transforms: Transforms to apply to the data
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

        with rasterio.open(img_path) as f:
            image = f.read()
        image = torch.from_numpy(image).float()

        image_dict = self.rearrange_bands(image, self.band_order)
        image_dict = self.data_normalizer(image_dict)

        sample.update(image_dict)

        point = wkt.loads(sample_row.iloc[0]["stac:centroid"])
        lon, lat = point.x, point.y

        if "lon" in self.metadata:
            sample["lon"] = torch.tensor(lon)
        if "lat" in self.metadata:
            sample["lat"] = torch.tensor(lat)
        if "time" in self.metadata:
            sample["time_start"] = sample_row.iloc[0]["stac:time_start"]

        sample["label"] = sample_row.iloc[0]["labels"]

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
