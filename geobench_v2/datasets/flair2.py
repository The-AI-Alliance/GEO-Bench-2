# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Flair 2 Aerial Dataset."""

from collections.abc import Sequence

import rasterio
import torch
import torch.nn as nn
from shapely import wkt
from torch import Tensor

from .base import GeoBenchBaseDataset
from .data_util import ClipZScoreNormalizer
from .sensor_util import DatasetBandRegistry


class GeoBenchFLAIR2(GeoBenchBaseDataset):
    """Implementation of FLAIR 2 Aerial dataset."""

    url = "https://hf.co/datasets/aialliance/flair2/resolve/main/{}"

    sha256str = ["96d18b1e7673fa2233145d69fd67db530c53bf68027b30466f7c94fd456df689"]

    # paths: Sequence[str] = (
    #     "FullFlair2.0000.part.tortilla",
    #     "FullFlair2.0001.part.tortilla",
    #     "FullFlair2.0002.part.tortilla",
    #     "FullFlair2.0003.part.tortilla",
    #     "FullFlair2.0004.part.tortilla",
    #     "FullFlair2.0005.part.tortilla",
    # )

    paths: Sequence[str] = ["geobench_flair2.tortilla"]

    classes = (
        "building",
        "pervious surface",
        "impervious surface",
        "bare soil",
        "water",
        "coniferous",
        "deciduous",
        "brushwood",
        "vineyard",
        "herbaceous vegetation",
        "agricultural land",
        "plowed land",
        "other",
    )

    num_classes = len(classes)

    dataset_band_config = DatasetBandRegistry.FLAIR2

    normalization_stats = {
        "means": {"r": 110.30502319335938, "g": 114.79083251953125, "b": 105.6126937866211, "nir": 104.3409194946289, "elevation": 17.69650650024414},
        "stds": {"r": 50.71001052856445, "g": 44.31645584106445, "b": 43.294822692871094, "nir": 39.049617767333984, "elevation": 29.94267463684082},
    }

    band_default_order = {
        "aerial": ("red", "green", "blue", "nir"),
        "elevation": ("elevation",),
    }

    valid_metadata = ("lat", "lon")

    def __init__(
        self,
        root,
        split="train",
        band_order: dict[str, Sequence[float | str]] = band_default_order,
        data_normalizer: type[nn.Module] = ClipZScoreNormalizer,
        transforms: nn.Module | None = None,
        metadata: Sequence[str] | None = None,
        download: bool = False,
    ):
        """Initialize FLAIR 2 dataset.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'test'
            band_order: The order of bands to return, defaults to ['r', 'g', 'b'], if one would
                specify ['r', 'g', 'b', 'nir'], the dataset would return images with 4 channels
            data_normalizer: The data normalizer to apply to the data, defaults to :class:`data_util.ClipZScoreNormalizer`,
                which applies z-score normalization to each band.
            transforms:

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

        aerial_path = sample_row.read(0)
        mask_path = sample_row.read(1)

        data_dict = {}
        with rasterio.open(aerial_path) as f:
            # read aerial bands
            data = f.read()
            image = data[:-1, :, :]
            data_dict["aerial"] = torch.from_numpy(image).float()
            if "elevation" in self.band_order:
                # read elevation band
                elevation = data[-1, :, :]
                data_dict["elevation"] = (
                    torch.from_numpy(elevation).unsqueeze(0).float()
                )

        with rasterio.open(mask_path) as f:
            mask = f.read(1)
        mask = torch.from_numpy(mask).long()
        # replace values > 13 with 13 as "other" class
        mask[mask > 13] = 13
        # shift the classes to start from 0 so class values will be 0-12
        mask -= 1

        image_dict = self.rearrange_bands(data_dict, self.band_order)

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
