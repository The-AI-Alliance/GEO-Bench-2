# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Sen4AgriNet dataset."""

from torch import Tensor
from pathlib import Path
from typing import Sequence, Type, Literal
import torch.nn as nn

from .sensor_util import DatasetBandRegistry
from torchgeo.datasets import NonGeoDataset
from .data_util import MultiModalNormalizer, DataUtilsMixin
import torch.nn.functional as F
import torch.nn as nn
import rasterio
import numpy as np
import torch
import pandas as pd
import os
import h5py


class GeoBenchSen4AgriNet(NonGeoDataset, DataUtilsMixin):
    """Sen4AgriNet dataset."""

    classes = (
        "background/other",
        "wheat",
        "maize",
        "sorghum",
        "barley",
        "rye",
        "oats",
        "grapes",
        "rapeseed",
        "sunflower",
        "potatoes",
        "peas",
    )

    num_classes = len(classes)

    SELECTED_CLASSES = [
        110,  # 'Wheat'
        120,  # 'Maize'
        140,  # 'Sorghum'
        150,  # 'Barley'
        160,  # 'Rye'
        170,  # 'Oats'
        330,  # 'Grapes'
        435,  # 'Rapeseed'
        438,  # 'Sunflower'
        510,  # 'Potatoes'
        770,  # 'Peas'
    ]

    dataset_band_config = DatasetBandRegistry.SEN4AGRINET

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
        "B09",
        "B10",
        "B11",
        "B12",
    )

    normalization_stats = {
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
            "B10": 0.0,
            "B11": 0.0,
            "B12": 0.0,
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
            "B10": 3000.0,
            "B11": 3000.0,
            "B12": 3000.0,
        },
    }

    classes = ()

    num_classes = len(classes)

    def __init__(
        self,
        root: Path,
        split: Literal["train", "validation", "test"] = "train",
        band_order: dict[str, Sequence[float | str]] = band_default_order,
        data_normalizer: Type[nn.Module] = MultiModalNormalizer,
        num_time_steps: int = 1,
        transforms: nn.Module | None = None,
    ) -> None:
        """Initialize PASTIS Dataset.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'val', 'test'
            band_order: The order of bands to return, defaults to ['red', 'green', 'blue', 'nir'], if one would
                specify ['red', 'green', 'blue', 'nir', 'nir'], the dataset would return images with 5 channels
                in that order. This is useful for models that expect a certain band order, or
                test the impact of band order on model performance.
            num_time_steps: The number of last time steps to return, defaults to 1, which returns the last time step.
                if set to 10, the latest 10 time steps will be returned. If a time series has fewer time steps than
                specified, it will be padded with zeros. A value of 1 will return a [C, H, W] tensor, while a value
                of 10 will return a [T, C, H, W] tensor.
            data_normalizer: The data normalizer to apply to the data, defaults to :class:`data_util.MultiModalNormalizer`,
                which applies z-score normalization to each band.
            transforms:

        Raises:
            AssertionError: If an invalid split is specified
        """
        self.root = root
        self.split = split

        self.band_order = self.resolve_band_order(band_order)

        self.data_normalizer = data_normalizer(
            self.normalization_stats, self.band_order
        )
        self.transforms = transforms
        self.num_time_steps = num_time_steps

        self.data_df = pd.read_parquet(
            os.path.join(root, "geobench_sen4agrinet.parquet")
        )
        self.data_df = self.data_df[self.data_df["split"] == split].reset_index(
            drop=True
        )

    def __getitem__(self, index: int) -> dict[str, any]:
        """Return an index within the dataset.

        Args:
            index: Index to return

        Returns:
            A dictionary containing the data and target
        """
        sample: dict[str, Tensor] = {}
        sample_row = self.data_df.iloc[index]

        with h5py.File(os.path.join(self.root, sample_row["path"])) as hdf5_data:
            data: list[Tensor] = []
            for band in self.band_default_order:
                # need to resample the data to 366x366
                band_data = torch.from_numpy(hdf5_data[band][band][:]).float()
                if band in ["B01", "B09", "B10"]:
                    # upsample bx 6
                    band_data = F.interpolate(
                        band_data.unsqueeze(0),
                        size=(366, 366),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)

                elif band in ["B05", "B06", "B07", "B8A", "B11", "B12"]:
                    # upsample by 2
                    band_data = F.interpolate(
                        band_data.unsqueeze(0),
                        size=(366, 366),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)

                data.append(band_data)

            mask = torch.from_numpy(hdf5_data["labels"]["labels"][:]).long()

            processed_mask = torch.zeros_like(mask)
            # map classes to enumerated classes and everything else to 0
            for i, class_id in enumerate(self.SELECTED_CLASSES, 1):
                processed_mask[mask == class_id] = i

        # [T, C, H, W]
        data = torch.stack(data, dim=1)

        if data.shape[0] < self.num_time_steps:
            padding = torch.zeros(self.num_time_steps - data.shape[0], *data.shape[1:])
            data = torch.cat((padding, data), dim=0)
        else:
            data = data[-self.num_time_steps :]

        # only return [C, H, W]
        if self.num_time_steps == 1:
            data = data.squeeze(0)

        dates = sample_row["dates"]
        # turn date strings numeric
        dates = [pd.to_datetime(date).timestamp() for date in dates]

        if len(dates) < self.num_time_steps:
            sample["dates"] = [0] * (self.num_time_steps - len(dates)) + dates
        else:
            sample["dates"] = dates[-self.num_time_steps :]

        img_dict = self.rearrange_bands(data, self.band_order)

        img_dict = self.data_normalizer(img_dict)

        sample.update(img_dict)

        sample["mask"] = processed_mask
        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            The length of the dataset
        """
        return len(self.data_df)
