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
            "nir": 154.289794921875,
            "g": 92.13509368896484,
            "b": 85.36317443847656,
            "r": 79.30790710449219,
            "vv": -6.364912509918213,
            "vh": -12.508633613586426,
            "vv/vh": 0.4892385005950928,
            "B02": 245.31068420410156,
            "B03": 387.63568115234375,
            "B04": 248.4667205810547,
            "B08": 2825.93603515625,
            "B05": 625.9300537109375,
            "B06": 2118.83740234375,
            "B07": 2709.37890625,
            "B8A": 2982.208740234375,
            "B11": 1316.7186279296875,
            "B12": 594.203369140625,
            "B01": 265.8070068359375,
            "B09": 2962.182373046875,
        },
        "stds": {
            "nir": 49.029109954833984,
            "g": 33.52909469604492,
            "b": 27.931865692138672,
            "r": 33.36391830444336,
            "vv": 3.5287060737609863,
            "vh": 3.2120885848999023,
            "vv/vh": 0.2582942247390747,
            "B02": 117.73491668701172,
            "B03": 130.0995635986328,
            "B04": 129.66375732421875,
            "B08": 756.8175659179688,
            "B05": 191.35238647460938,
            "B06": 517.2822265625,
            "B07": 691.1488037109375,
            "B8A": 754.9419555664062,
            "B11": 411.339111328125,
            "B12": 234.48863220214844,
            "B01": 125.9928207397461,
            "B09": 674.169189453125,
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
            split: The dataset split, supports 'train', 'val', 'test'
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
