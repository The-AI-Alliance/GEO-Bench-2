# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""CaFFe Dataset."""

from collections.abc import Sequence

import rasterio
import torch
import torch.nn as nn
from shapely import wkt
from torch import Tensor

from .base import GeoBenchBaseDataset
from .normalization import ZScoreNormalizer
from .sensor_util import DatasetBandRegistry


class GeoBenchCaFFe(GeoBenchBaseDataset):
    """GeoBench version of Caffe dataset.

    lacier calving front segmentation dataset using Sentinel-1 SAR imagery,
    with annotated calving front masks.

    If you use this dataset in your research, please cite the following paper:

    * https://essd.copernicus.org/articles/14/4287/2022/
    """

    url = "https://hf.co/datasets/aialliance/caffe/resolve/main/{}"
    paths = ["geobench_caffe.tortilla"]
    sha256str = ["8b2a2e1020a26a2e62080c96646c9c1f1cb35a54722739f8cef6f11122c4161e"]

    dataset_band_config = DatasetBandRegistry.CAFFE

    band_default_order = ("gray",)

    normalization_stats: dict[str, dict[str, float]] = {
        "means": {"gray": 62.682498931884766},
        "stds": {"gray": 79.8001937866211},
    }

    mask_dirs = ("zones", "zones")

    classes = ("N/A", "rock", "glacier", "ocean/ice melange")

    num_classes = len(classes)

    def __init__(
        self,
        root,
        split="train",
        band_order: Sequence[float | str] = band_default_order,
        data_normalizer: type[nn.Module] = ZScoreNormalizer,
        transforms: nn.Module | None = None,
        metadata: Sequence[str] | None = None,
        download: bool = False,
    ) -> None:
        """Initialize Caffe dataset.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'validation', 'test'
            band_order: The order of bands to return, defaults to ['gray'], if one would
                specify ['gray', 'gray', 'gray], the dataset would return the gray band three times.
            data_normalizer: The data normalizer to apply to the data, defaults to :class:`data_util.ZScoreNormalizer`,
                which applies z-score normalization to each band.
            transforms: The transforms to apply to the data, defaults to None.
            metadata: The metadata to return, defaults to None.
            download: Whether to download the dataset, defaults to False.

        Raises:
            AssertionError: If split is not in the splits
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
        mask_path = sample_row.read(1)

        with rasterio.open(img_path) as f:
            image = f.read()
        image = torch.from_numpy(image).float()

        with rasterio.open(mask_path) as f:
            mask = f.read(1)
        mask = torch.from_numpy(mask).long()

        image_dict = self.rearrange_bands(image, self.band_order)
        image_dict = self.data_normalizer(image_dict)

        sample.update(image_dict)

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
