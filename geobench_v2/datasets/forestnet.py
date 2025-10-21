# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""m-So2Sat Dataset."""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast
from shapely import wkt

import rasterio
import torch
import torch.nn as nn
from torch import Tensor

from .base import GeoBenchBaseDataset
from .normalization import MultiModalNormalizer
from .sensor_util import DatasetBandRegistry


class GeoBenchForestnet(GeoBenchBaseDataset):
    """Forestnet Dataset with enhanced functionality.

    """

    url = "https://hf.co/datasets/aialliance/forestnet/resolve/main/{}"

    paths: Sequence[str] = ["geobench_forestnet.tortilla"]

    sha256str: Sequence[str] = [
        "6ee7cb7135b4ca5d0cde52e781f5960ed0e648dcceab598982fa612802cd3ad1"
    ]

    classes: Sequence[str] = ("Oil palm plantation",
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
                              "Other")

    dataset_band_config = DatasetBandRegistry.FORESTNET
    band_default_order = dataset_band_config.default_order

    normalization_stats: dict[str, dict[str, float]] = {
        "means": {
                    "B02": 72.852258,
                    "B03": 83.677155,
                    "B04": 77.58181,
                    "B8A": 123.987442,
                    "B11": 91.536942,
                    "B12": 74.719202,
                },
        "stds": {

                    "B02": 15.837172547567825,
                    "B03": 14.788812599596188,
                    "B04": 16.100543441881086,
                    "B8A": 16.35234883118129,
                    "B11": 13.7882739778638,
                    "B12": 12.69131413539181,
                },
    }
    
    label_names = classes
    
    num_classes: int = len(label_names)

    def __init__(
        self,
        root: Path,
        split: Literal["train", "val", "validation", "test"],
        rename_modalities: dict | None = None,
        band_order: dict[str, Sequence[float | str]] = band_default_order,
        data_normalizer: type[nn.Module] = MultiModalNormalizer,
        transforms: nn.Module | None = None,
        metadata: [str] = None,
        download: bool = False,
    ) -> None:
        """Initialize Forestnet Dataset.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'validation', 'test'
            band_order: The order of bands to return, defaults to all s2 bands. This is useful for models that expect a certain band order, or
                test the impact of band order on model performance.
            data_normalizer: The data normalizer to apply to the data, defaults to :class:`data_util.MultiModalNormalizer`,
                which applies z-score normalization to each band.
            transforms: Transforms to apply to the data
            metadata: metadata names to be returned under specified keys as part of the sample in the
                __getitem__ method. If None, no metadata is returned.
            download: Whether to download the dataset
            rename_modalities: dictionary with information to rename modalities in output e.g. {image: {s1:  S1RTC, s2: S2L2A}}
        """
        split_norm: Literal["train", "validation", "test"]
        if split == "val":
            split_norm = "validation"
        else:
            split_norm = cast(Literal["train", "validation", "test"], split)

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
