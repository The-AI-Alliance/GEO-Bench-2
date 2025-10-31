# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Cloud12Sen Dataset."""

from collections.abc import Sequence
from typing import Literal

import numpy as np
import rasterio
import torch
import torch.nn as nn
from shapely import wkt
from torch import Tensor

from .base import GeoBenchBaseDataset
from .normalization import ZScoreNormalizer
from .sensor_util import DatasetBandRegistry


class GeoBenchCloudSen12(GeoBenchBaseDataset):
    """GeoBench version of CloudSen12 dataset.

    Cloud and shadow segmentation dataset using Sentinel-2 optical imagery, with pixel-level cloud and shadow masks.

    0. clear: Pixels without cloud and cloud shadow contamination.
    1. thick cloud: Opaque clouds that block all reflected light from Earth's surface.
    2. thin cloud: Semitransparent clouds that alter the surface spectral signal but still allow recognition of the background.
    3. cloud shadow: Dark pixels where light is occluded by thick or thin clouds.

    If you use this dataset in your research, please cite the following paper:

    * https://www.sciencedirect.com/science/article/pii/S2352340924008163
    """

    url = "https://hf.co/datasets/aialliance/cloudsen12/resolve/main/{}"

    paths = ["geobench_cloudsen12.tortilla"]

    sha256str = ["16b3c03d7b15cf42f6ef0cee6d453b6ad8ebbe7744674c4b58657511f7f5d0c0"]

    classes = ("clear", "thick cloud", "thin cloud", "cloud shadow")

    num_classes = len(classes)

    splits = ("train", "val", "test")

    dataset_band_config = DatasetBandRegistry.CLOUDSEN12

    band_default_order = DatasetBandRegistry.CLOUDSEN12.default_order

    normalization_stats: dict[str, dict[str, float]] = {
        "means": {
            "B01": 2030.244384765625,
            "B02": 2074.817138671875,
            "B03": 2209.807373046875,
            "B04": 2247.927490234375,
            "B05": 2589.593505859375,
            "B06": 3103.521240234375,
            "B07": 3277.909423828125,
            "B08": 3331.6318359375,
            "B8A": 3377.544677734375,
            "B09": 4038.193115234375,
            "B11": 2448.748046875,
            "B12": 1907.728515625,
        },
        "stds": {
            "B01": 2723.43603515625,
            "B02": 2691.302734375,
            "B03": 2539.91357421875,
            "B04": 2538.520751953125,
            "B05": 2504.328369140625,
            "B06": 2241.74462890625,
            "B07": 2145.667724609375,
            "B08": 2176.997802734375,
            "B8A": 2066.763671875,
            "B09": 3083.179931640625,
            "B11": 1595.065185546875,
            "B12": 1474.11767578125,
        },
    }

    valid_metadata = ("lat", "lon")

    def __init__(
        self,
        root,
        split: Literal["train", "validation", "test"] = "train",
        band_order: Sequence[float | str] = ["B04", "B03", "B02"],
        data_normalizer: type[nn.Module] = ZScoreNormalizer,
        transforms: nn.Module | None = None,
        metadata: Sequence[str] | None = None,
        download: bool = False,
    ) -> None:
        """Initialize a CloudSen12 dataset instance.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'test'
            band_order: The order of bands to return, defaults to ['r', 'g', 'b'], if one would
                specify ['r', 'g', 'b', 'nir'], the dataset would return images with 4 channels
            data_normalizer: The data normalizer to apply to the data, defaults to :class:`data_util.ZScoreNormalizer`,
                which applies z-score normalization to each band.
            transforms: Image resize transform on sample level
            metadata: metadata names to be returned under specified keys as part of the sample in the
                __getitem__ method. If None, no metadata is returned.
            download: Whether to download the dataset

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
        """Return the sample_row at the given index.

        Args:
            idx: Index of the sample_row to return

        Returns:
            dict containing the sample_row data
        """
        sample: dict[str, Tensor] = {}

        l2a_row = self.data_df.read(idx)

        image_path: str = l2a_row.read(0)
        target_path: str = l2a_row.read(1)

        with rasterio.open(image_path) as image_src:
            image_data: np.ndarray = image_src.read(out_dtype="float32")
        with rasterio.open(target_path) as target_src:
            target_data: np.ndarray = target_src.read()

        image = torch.from_numpy(image_data).float()
        mask = torch.from_numpy(target_data).long()

        image_dict = self.rearrange_bands(image, self.band_order)

        image = self.data_normalizer(image_dict)

        sample.update(image_dict)
        sample["mask"] = torch.squeeze(mask)

        point = wkt.loads(l2a_row.iloc[0]["stac:centroid"])
        lon, lat = point.x, point.y
        if "lon" in self.metadata:
            sample["lon"] = torch.tensor(lon)
        if "lat" in self.metadata:
            sample["lat"] = torch.tensor(lat)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
