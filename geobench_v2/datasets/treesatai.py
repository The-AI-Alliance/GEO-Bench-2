# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""TreesatAI dataset."""

import os
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import rasterio
import torch
import torch.nn as nn
from shapely import wkt
from torch import Tensor

from .base import GeoBenchBaseDataset
from .normalization import ZScoreNormalizer
from .sensor_util import DatasetBandRegistry


class GeoBenchTreeSatAI(GeoBenchBaseDataset):
    """TreeSatAI dataset with enhanced functionality.

    Multi-label classification dataset, should we also support standard classification
    based on majority label?

    If you use this dataset, please cite:

    *
    *
    """

    url = "https://hf.co/datasets/aialliance/treesatai/resolve/main/{}"
    # paths = ["TreeSatAI.tortilla"]
    paths = ["geobench_treesatai.tortilla"]

    sha256str = ["04435ade7d429418cf2e51db9ec493a9ca196e79aff661425d82b066bdd3a759"]

    dataset_band_config = DatasetBandRegistry.TREESATAI

    normalization_stats = {
        "means": {
            "nir": 0.0,
            "green": 0.0,
            "blue": 0.0,
            "red": 0.0,
            "vv": 0.0,
            "vh": 0.0,
            "vv/vh": 0.0,
            "B02": 0.0,
            "B03": 0.0,
            "B04": 0.0,
            "B08": 0.0,
            "B05": 0.0,
            "B06": 0.0,
            "B07": 0.0,
            "B8A": 0.0,
            "B11": 0.0,
            "B12": 0.0,
            "B01": 0.0,
            "B09": 0.0,
        },
        "stds": {
            "nir": 255.0,
            "green": 255.0,
            "blue": 255.0,
            "red": 255.0,
            "vv": 1.0,
            "vh": 1.0,
            "vv/vh": 1.0,
            "B02": 1000.0,
            "B03": 1000.0,
            "B04": 1000.0,
            "B08": 1000.0,
            "B05": 1000.0,
            "B06": 1000.0,
            "B07": 1000.0,
            "B8A": 1000.0,
            "B11": 1000.0,
            "B12": 1000.0,
            "B01": 1000.0,
            "B09": 1000.0,
        },
    }

    band_default_order = {
        "aerial": ["red", "green", "blue", "nir"],
        "s2": [
            "B02",
            "B03",
            "B04",
            "B08",
            "B05",
            "B06",
            "B07",
            "B8A",
            "B11",
            "B12",
            "B01",
            "B09",
        ],
        "s1": ["vv", "vh", "vv/vh"],
    }

    classes: Sequence[str] = (
        "Abies",
        "Acer",
        "Alnus",
        "Betula",
        "Cleared",
        "Fagus",
        "Fraxinus",
        "Larix",
        "Picea",
        "Pinus",
        "Populus",
        "Prunus",
        "Pseudotsuga",
        "Quercus",
        "Tilia",
    )

    multilabel: bool = True

    num_classes: int = len(classes)

    valid_metadata = ("lat", "lon")

    def __init__(
        self,
        root: Path,
        split: str,
        band_order: dict[str, Sequence[str]] = {
            "aerial": ["red", "green", "blue", "nir"]
        },
        data_normalizer: type[nn.Module] = ZScoreNormalizer,
        transforms: nn.Module | None = None,
        metadata: Sequence[str] | None = None,
        include_ts: bool = False,
        num_time_steps: int = None,
        return_stacked_image: bool = False,
        download: bool = False,
    ) -> None:
        """Initialize TreeSatAI dataset.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'validation', 'test'
            band_order: The order of bands to return, defaults to ['red', 'green', 'blue', 'nir'], if one would
                specify ['red', 'green', 'blue', 'nir', 'nir'], the dataset would return images with 5 channels
                in that order. This is useful for models that expect a certain band order, or
                test the impact of band order on model performance.
            data_normalizer: The data normalizer to apply to the data, defaults to :class:`data_util.ZScoreNormalizer`,
                which applies z-score normalization to each band.
            transforms: image transformations to apply to the data, defaults to None
            metadata: metadata names to be returned as part of the sample in the
                __getitem__ method. If None, no metadata is returned.
            include_ts: whether or not to return the time series in data loading
            num_time_steps: number of last time steps to return in the ts data
            return_stacked_image: if true, returns a single image tensor with all modalities stacked in band_order
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

        self.include_ts = include_ts
        self.num_time_steps = num_time_steps
        self.return_stacked_image = return_stacked_image

        if include_ts:
            if num_time_steps is None:
                raise ValueError(
                    "num_time_steps must be specified if include_ts is True"
                )

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        sample: dict[str, Tensor] = {}

        sample_row = self.data_df.read(index)

        img_dict: dict[str, Tensor] = {}

        modality_to_index = {"aerial": 0, "s1": 1, "s2": 2}

        img_dict = {}
        for modality in self.band_order:
            if modality in modality_to_index:
                file_path = sample_row.read(modality_to_index[modality])
                with rasterio.open(file_path) as src:
                    data = src.read().astype(np.float32)
                img_dict[modality] = torch.from_numpy(data)

        # img_dict = self.rearrange_bands(img_dict, self.band_order)
        img_dict = self.rearrange_bands(img_dict, self.band_order)
        img_dict = self.data_normalizer(img_dict)
        sample.update(img_dict)

        # only resize the aerial image
        if self.transforms is not None:
            sample = self.transforms(sample)

        if self.return_stacked_image:
            sample = {
                "image": torch.cat(
                    [sample[f"image_{key}"] for key in self.band_order.keys()], 0
                )
            }

        sample["label"] = self._format_label(
            sample_row.iloc[0]["species_labels"], sample_row.iloc[0]["dist_labels"]
        )

        point = wkt.loads(sample_row.iloc[0]["stac:centroid"])
        lon, lat = point.x, point.y

        if "lon" in self.metadata:
            sample["lon"] = torch.tensor(lon)
        if "lat" in self.metadata:
            sample["lat"] = torch.tensor(lat)

        if self.include_ts:
            with h5py.File(
                os.path.join(self.root, sample_row.iloc[0]["ts_path"]), "r"
            ) as h5file:
                sen_1_asc_data = h5file["sen-1-asc-data"][
                    :
                ]  # Tx2x6x6, Channels: VV, VH
                sen_1_asc_products = h5file["sen-1-asc-products"][:]
                sen_1_des_data = h5file["sen-1-des-data"][
                    :
                ]  # Tx2x6x6, Channels: VV, VH
                sen_1_des_products = h5file["sen-1-des-products"][:]
                sen_2_data = h5file["sen-2-data"][
                    :
                ]  # Tx10x6x6 B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12
                sen_2_products = h5file["sen-2-products"][:]
                sen_2_masks = h5file["sen-2-masks"][
                    :
                ]  # (Tx2x6x6), Channels: snow probability, cloud probability

            if "s1" in self.band_order:
                sample["image_s1_asc_ts"] = torch.from_numpy(sen_1_asc_data)[
                    -self.num_time_steps :
                ]
                sample["image_s1_des_ts"] = torch.from_numpy(sen_1_des_data)[
                    -self.num_time_steps :
                ]
            if "s2" in self.band_order:
                sample["image_s2_ts"] = torch.from_numpy(sen_2_data)[
                    -self.num_time_steps :
                ]

        return sample

    def _format_label(
        self, class_labels: list[str], dist_labels: list[float]
    ) -> Tensor:
        """Format label list to Tensor.

        Args:
            class_labels: list of label class names
            dist_labels: list of label distribution values

        Returns:
            label tensor
        """
        label = torch.zeros(len(self.classes))
        for name in class_labels:
            label[self.classes.index(name)] = 1
        return label
