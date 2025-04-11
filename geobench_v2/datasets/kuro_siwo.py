# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Kuro Siwo dataset."""

from torch import Tensor
from torchgeo.datasets import SpaceNet6
from pathlib import Path
from typing import Sequence, Type, Literal
import torch.nn as nn

from .sensor_util import DatasetBandRegistry
from .base import GeoBenchBaseDataset
from .data_util import MultiModalNormalizer
import torch.nn as nn
import rasterio
import numpy as np
import torch


class GeoBenchKuroSiwo(GeoBenchBaseDataset):
    """Kuro Siwo Flood Change Detection Dataset.

    Classes:
    0. NO-Water
    1. Permanent Water
    2. Flood
    3. No-Data

    """

    dataset_band_config = DatasetBandRegistry.KURO_SIWO

    band_default_order = {"sar": ("vv", "vh"), "dem": ("dem",)}

    # https://github.com/Orion-AI-Lab/KuroSiwo/blob/2b9491629ffd9e1322eea4eaaf88fbaecef6d9b3/configs/train/data_config.json#L16
    # "data_mean": [0.0953, 0.0264],
    # "data_std": [0.0427, 0.0215],
    # "dem_mean":93.4313,
    # "dem_std":1410.8382,

    normalization_stats = {
        "means": {"vv": 0.0953, "vh": 0.0264, "dem": 93.4313},
        "stds": {"vv": 0.0427, "vh": 0.0215, "dem": 1410.8382},
    }

    classes = ("No Water", "Permanent Water", "Flood", "No Data")

    # TODO should move no-data to 0 and have that as ignore_index
    num_classes = len(classes)

    paths = [
        "kurosiwo.0000.part.tortilla",
        "kurosiwo.0001.part.tortilla",
        "kurosiwo.0002.part.tortilla",
        "kurosiwo.0003.part.tortilla",
        "kurosiwo.0004.part.tortilla",
        "kurosiwo.0005.part.tortilla",
    ]

    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"],
        band_order: dict[str, Sequence[str]] = band_default_order,
        data_normalizer: Type[nn.Module] = MultiModalNormalizer,
        transforms: Type[nn.Module] = None,
        return_stacked_image: bool = False,
    ) -> None:
        """Initialize Kuro Siwo Dataset.

        Args:
            root: Path to dataset
            split: Split of dataset
            band_order: Band order for dataset
            data_normalizer: Data normalizer
            transforms: Data transforms
            return_stacked_image: if true, returns a single image tensor with all modalities stacked in band_order
        """
        super().__init__(
            root=root,
            split=split,
            band_order=band_order,
            data_normalizer=data_normalizer,
            transforms=transforms,
        )
        self.return_stacked_image = return_stacked_image

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        sample: dict[str, Tensor] = {}

        sample_row = self.data_df.read(index)

        pre_event_1_vv_path = sample_row.read(0)
        pre_event_1_vh_path = sample_row.read(1)
        pre_event_2_vv_path = sample_row.read(2)
        pre_event_2_vh_path = sample_row.read(3)
        post_event_vv_path = sample_row.read(4)
        post_event_vh_path = sample_row.read(5)
        dem_path = sample_row.read(6)
        mask_path = sample_row.read(7)
        invalid_data_path = sample_row.read(8)

        with (
            rasterio.open(pre_event_1_vv_path) as pre_event_1_vv_src,
            rasterio.open(pre_event_1_vh_path) as pre_event_1_vh_src,
            rasterio.open(pre_event_2_vv_path) as pre_event_2_vv_src,
            rasterio.open(pre_event_2_vh_path) as pre_event_2_vh_src,
            rasterio.open(post_event_vv_path) as post_event_vv_src,
            rasterio.open(post_event_vh_path) as post_event_vh_src,
            rasterio.open(dem_path) as dem_src,
            rasterio.open(mask_path) as mask_src,
            rasterio.open(invalid_data_path) as invalid_data_src,
        ):
            pre_event_1_vv: np.ndarray = pre_event_1_vv_src.read(out_dtype="float32")
            pre_event_1_vh: np.ndarray = pre_event_1_vh_src.read(out_dtype="float32")
            pre_event_2_vv: np.ndarray = pre_event_2_vv_src.read(out_dtype="float32")
            pre_event_2_vh: np.ndarray = pre_event_2_vh_src.read(out_dtype="float32")
            post_event_vv: np.ndarray = post_event_vv_src.read(out_dtype="float32")
            post_event_vh: np.ndarray = post_event_vh_src.read(out_dtype="float32")
            dem: np.ndarray = dem_src.read(out_dtype="float32")
            mask: np.ndarray = mask_src.read()
            invalid_data: np.ndarray = invalid_data_src.read()

        invalid_data_tensor = torch.from_numpy(invalid_data).long()
        sample["invalid_data"] = invalid_data_tensor
        invalid_mask = invalid_data_tensor

        def process_sar_image(vv: np.ndarray, vh: np.ndarray) -> Tensor:
            image = torch.cat([torch.from_numpy(vv), torch.from_numpy(vh)])
            image = self.rearrange_bands({"sar": image}, self.band_order["sar"])
            normalized = self.data_normalizer({"image_sar": image["image"]})
            return normalized["image_sar"] * invalid_mask

        if "sar" in self.band_order:
            sample["image_pre_1"] = process_sar_image(pre_event_1_vv, pre_event_1_vh)
            sample["image_pre_2"] = process_sar_image(pre_event_2_vv, pre_event_2_vh)
            sample["image_post"] = process_sar_image(post_event_vv, post_event_vh)

        if "dem" in self.band_order:
            image_dem = torch.from_numpy(dem)
            image_dem = self.rearrange_bands({"dem": image_dem}, self.band_order["dem"])
            image_dem = self.data_normalizer({"image_dem": image_dem["image"]})
            sample["image_dem"] = image_dem["image_dem"] * invalid_mask

        sample["mask"] = torch.from_numpy(mask).long()

        if self.transforms is not None:
            sample = self.transforms(sample)

        if self.return_stacked_image:
            modality_keys = {
                "sar": ["image_pre_1", "image_pre_2", "image_post"],
                "dem": ["image_dem"]
            }
            stacked_images = [
                sample[key] 
                for modality in self.band_order 
                for key in modality_keys.get(modality, [])
            ]
            sample = {
                "image": torch.cat(stacked_images, dim=0),
                "mask": sample["mask"]
            }

        return sample
