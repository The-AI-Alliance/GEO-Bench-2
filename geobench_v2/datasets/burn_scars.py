# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""HLS Burn Scars Dataset."""

from collections.abc import Sequence

import rasterio
import torch
import torch.nn as nn
from shapely import wkt
from torch import Tensor

from .base import GeoBenchBaseDataset
from .normalization import ZScoreNormalizer
from .sensor_util import DatasetBandRegistry


class GeoBenchBurnScars(GeoBenchBaseDataset):
    """GeoBench Burn Scars dataset.

    Burned area segmentation dataset using Harmonized Landsat-Sentinel
    (HLS: Landsat 8/9 + Sentinel-2) imagery, with multi-source burn scar annotations.

    If you use this dataset in your research, please cite the following paper:

    *
    """

    url = "https://hf.co/datasets/aialliance/burn_scars/resolve/main/{}"
    paths = ["geobench_burn_scars.tortilla"]
    sha256str = ["9f865e72bc4aeb657ee03be167a093f80ee92403ccb2c23b4d19be1189e442aa"]

    dataset_band_config = DatasetBandRegistry.BURNSCARS
    # TODO update sensor type with wavelength and resolution

    band_default_order = dataset_band_config.default_order

    # normalization_stats: dict[str, dict[str, float]] = {
    #     "means": {
    #         "B02": 0.0333497067415863,
    #         "B03": 0.0570118552053618,
    #         "B04": 0.0588974813200132,
    #         "B8A": 0.2323245113436119,
    #         "B11": 0.1972854853760658,
    #         "B12": 0.1194491422518656,
    #     },
    #     "stds": {
    #         "B02": 0.0226913556882377,
    #         "B03": 0.0268075602230702,
    #         "B04": 0.0400410984436278,
    #         "B8A": 0.0779173242367269,
    #         "B11": 0.0870873883814014,
    #         "B12": 0.0724197947743781,
    #     },
    # }

    normalization_stats: dict[str, dict[str, float]] = {
        "means": {
            "B02": 0.0333497067415863,
            "B03": 0.0570118552053618,
            "B04": 0.0588974813200132,
            "B8A": 0.2323245113436119,
            "B11": 0.1972854853760658,
            "B12": 0.1194491422518656,
        },
        "stds": {
            "B02": 0.0226913556882377,
            "B03": 0.0268075602230702,
            "B04": 0.0400410984436278,
            "B8A": 0.0779173242367269,
            "B11": 0.0870873883814014,
            "B12": 0.0724197947743781,
        },
    }

    classes = ("Background", "Burn Scar")

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
        """Initialize Burn Scars dataset.

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
        mask[mask == -1] = 2  # change no data values to 2
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
