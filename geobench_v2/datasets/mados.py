# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""MADOS dataset."""

from torch import Tensor
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


class GeoBenchMADOS(GeoBenchBaseDataset):
    """MADOS dataset.

    There are always 12 S1 time steps available but the number of S2 time steps can vary.
    """

    dataset_band_config = DatasetBandRegistry.MADOS

    band_default_order = (
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B11",
        "B12",
    )

    # https://github.com/gkakogeorgiou/mados/blob/c20a7e972bdf111126540d8e6caa45808352f138/utils/dataset.py#L29
    # bands_mean = np.array([0.0582676,  0.05223386, 0.04381474, 0.0357083,  0.03412902, 0.03680401,
    # 0.03999107, 0.03566642, 0.03965081, 0.0267993,  0.01978944]).astype('float32')

    # bands_std = np.array([0.03240627, 0.03432253, 0.0354812,  0.0375769,  0.03785412, 0.04992323,
    # 0.05884482, 0.05545856, 0.06423746, 0.04211187, 0.03019115]).astype('float32')

    normalization_stats = {
        "means": {
            "B01": 0.0582676,
            "B02": 0.05223386,
            "B03": 0.04381474,
            "B04": 0.0357083,
            "B05": 0.03412902,
            "B06": 0.03680401,
            "B07": 0.03999107,
            "B08": 0.03566642,
            "B8A": 0.03965081,
            "B11": 0.0267993,
            "B12": 0.01978944,
        },
        "stds": {
            "B01": 0.03240627,
            "B02": 0.03432253,
            "B03": 0.0354812,
            "B04": 0.0375769,
            "B05": 0.03785412,
            "B06": 0.04992323,
            "B07": 0.05884482,
            "B08": 0.05545856,
            "B8A": 0.06423746,
            "B11": 0.04211187,
            "B12": 0.03019115,
        },
    }

    paths = ["MADOS.tortilla"]

    classes = (
        "Non-annotated",
        "Marine Debris",
        "Dense Sargassum",
        "Sparse Floating Algae",
        "Natural Organic Material",
        "Ship",
        "Oil Spill",
        "Marine Water",
        "Sediment-Laden Water",
        "Foam",
        "Turbid Water",
        "Shallow Water",
        "Waves & Wakes",
        "Oil Platform",
        "Jellyfish",
        "Sea snot",
    )

    num_classes = len(classes)

    def __init__(
        self,
        root: Path,
        split: str,
        band_order: Sequence[str] = ["B04", "B03", "B02", "B08"],
        data_normalizer: Type[nn.Module] = MultiModalNormalizer,
        transforms: nn.Module | None = None,
        **kwargs,
    ) -> None:
        """Initialize MADOS dataset.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'val', 'test'
            band_order: The order of bands to return, defaults to ['red', 'green', 'blue', 'nir'], if one would
                specify ['red', 'green', 'blue', 'nir', 'nir'], the dataset would return images with 5 channels
                in that order. This is useful for models that expect a certain band order, or
                test the impact of band order on model performance.
            data_normalizer: The data normalizer to apply to the data, defaults to :class:`data_util.MultiModalNormalizer`,
                which applies z-score normalization to each band.
            transforms:
            **kwargs: Additional keyword arguments passed to ``torchgeo.datasets.MADOS``
        """
        super().__init__(
            root=root,
            split=split,
            band_order=band_order,
            data_normalizer=data_normalizer,
            transforms=transforms,
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

        images = []
        for idx, band in enumerate(self.band_default_order):
            with rasterio.open(sample_row.read(idx)) as dataset:
                array = dataset.read(
                    indexes=1, out_shape=(240, 240), resampling=Resampling.bilinear
                )
                images.append(array)

        images = torch.from_numpy(np.stack(images)).float()

        img_dict = self.rearrange_bands(images, self.band_order)
        img_dict = self.data_normalizer(img_dict)

        sample.update(img_dict)

        with rasterio.open(sample_row.read(11)) as src:
            mask = src.read()
        mask = torch.from_numpy(mask).float()
        sample["mask"] = mask

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
