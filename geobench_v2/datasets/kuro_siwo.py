# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Kuro Siwo dataset."""

from collections.abc import Sequence
from typing import Literal, cast

import numpy as np
import rasterio
import torch
import torch.nn as nn
from torch import Tensor

from .base import GeoBenchBaseDataset
from .normalization import ZScoreNormalizer
from .sensor_util import DatasetBandRegistry


class GeoBenchKuroSiwo(GeoBenchBaseDataset):
    """GeoBench version of Kuro Siwo dataset.

    Flood segmentation dataset using Sentinel-1 SAR, DEM,
    and slope data, with 4-class flood extent masks.

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2311.12056
    """

    url = "https://hf.co/datasets/aialliance/kuro_siwo/resolve/main/{}"

    paths = ["geobench_kuro_siwo.tortilla"]

    sha256str = ["4830fe6f23bf9750dee0c765850724b55026bf5d47cb67162d3ef7dcb04c3bbd"]

    dataset_band_config = DatasetBandRegistry.KURO_SIWO

    band_default_order: dict[str, list[str]] = {"sar": ["vv", "vh"], "dem": ["dem"]}

    # https://github.com/Orion-AI-Lab/KuroSiwo/blob/2b9491629ffd9e1322eea4eaaf88fbaecef6d9b3/configs/train/data_config.json#L16
    # "data_mean": [0.0953, 0.0264],
    # "data_std": [0.0427, 0.0215],
    # "dem_mean":93.4313,
    # "dem_std":1410.8382,

    normalization_stats: dict[str, dict[str, float]] = {
        "means": {"vv": 0.0953, "vh": 0.0264, "dem": 93.4313},
        "stds": {"vv": 0.0427, "vh": 0.0215, "dem": 1410.8382},
    }

    classes = ("No Data", "No Water", "Permanent Water", "Flood")

    num_classes = len(classes)

    CLASS_MAPPING = {
        0: 1,  # No Water -> 1
        1: 2,  # Permanent Water -> 2
        2: 3,  # Flood -> 3
        3: 0,  # No Data -> 0
    }

    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"],
        band_order: dict[str, Sequence[str]] = band_default_order,
        data_normalizer: type[nn.Module] = ZScoreNormalizer,
        transforms: type[nn.Module] = None,
        return_stacked_image: bool = False,
        time_step: Sequence[str] = ["pre_1", "pre_2", "post"],
        download: bool = False,
    ) -> None:
        """Initialize Kuro Siwo Dataset.

        Args:
            root: Path to dataset
            split: Split of dataset
            band_order: Band order for dataset
            data_normalizer: Data normalizer
            transforms: Data transforms
            return_stacked_image: if true, returns a single image tensor with all modalities stacked in band_order
            time_step: Time step for dataset
            download: whether to download the dataset
        """
        split_norm: Literal["train", "validation", "test"]
        if split == "val":
            split_norm = "validation"
        else:
            split_norm = cast(Literal["train", "validation", "test"], split)
        super().__init__(
            root=root,
            split=split_norm,
            band_order=band_order,
            data_normalizer=data_normalizer,
            transforms=transforms,
            metadata=None,
            download=download,
        )
        self.return_stacked_image = return_stacked_image
        if len(time_step) == 0:
            raise ValueError(
                "time_step must include at least one item from  ['pre_1, , 'pre_2', 'post']"
            )
        for i in time_step:
            assert i in ["pre_1", "pre_2", "post"], (
                "time_step must include at least one item from  ['pre_1, , 'pre_2', 'post']"
            )
        self.time_step = time_step

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        sample: dict[str, Tensor] = {}

        sample_row = self.data_df.read(index)

        pre_event_1_path = sample_row.read(0)
        pre_event_2_path = sample_row.read(1)
        post_event_path = sample_row.read(2)
        dem_path = sample_row.read(3)
        mask_path = sample_row.read(4)
        invalid_data_path = sample_row.read(5)

        with rasterio.open(invalid_data_path) as src:
            invalid_data = src.read()

        invalid_data_tensor = torch.from_numpy(invalid_data).long()

        sample["invalid_data"] = invalid_data_tensor
        invalid_mask = invalid_data_tensor

        def process_sar_image(image) -> Tensor:
            image = self.rearrange_bands({"sar": image}, self.band_order["sar"])
            nan_mask = torch.isnan(image["image"])
            normalized = self.data_normalizer({"image_sar": image["image"]})
            normalized = torch.where(
                nan_mask,
                torch.zeros_like(normalized["image_sar"]),
                normalized["image_sar"],
            )
            return normalized * invalid_mask

        if "sar" in self.band_order:
            with rasterio.open(pre_event_1_path) as src:
                pre_event_1_img = src.read()
                pre_event_1_img = torch.from_numpy(pre_event_1_img)
            with rasterio.open(pre_event_2_path) as src:
                pre_event_2_img = src.read()
                pre_event_2_img = torch.from_numpy(pre_event_2_img)
            with rasterio.open(post_event_path) as src:
                post_event_img = src.read()
                post_event_img = torch.from_numpy(post_event_img)
            if "pre_1" in self.time_step:
                sample["image_pre_1"] = process_sar_image(pre_event_1_img)
            if "pre_2" in self.time_step:
                sample["image_pre_2"] = process_sar_image(pre_event_2_img)
            if "post" in self.time_step:
                sample["image_post"] = process_sar_image(post_event_img)

        if "dem" in self.band_order:
            with rasterio.open(dem_path) as src:
                dem = src.read()
                dem_nans = torch.from_numpy(np.isnan(dem))

            image_dem = torch.from_numpy(dem)
            image_dem = self.rearrange_bands({"dem": image_dem}, self.band_order["dem"])
            image_dem = self.data_normalizer({"image_dem": image_dem["image"]})
            image_dem = torch.where(
                dem_nans,
                torch.zeros_like(image_dem["image_dem"]),
                image_dem["image_dem"],
            )
            sample["image_dem"] = image_dem * invalid_mask

        with rasterio.open(mask_path) as src:
            mask = src.read()
        original_mask = torch.from_numpy(mask).long().squeeze(0)
        remapped_mask = torch.zeros_like(original_mask)
        for orig_class, new_class in self.CLASS_MAPPING.items():
            remapped_mask[original_mask == orig_class] = new_class

        sample["mask"] = remapped_mask

        if self.transforms is not None:
            sample = self.transforms(sample)

        if self.return_stacked_image:
            modality_keys = {
                "sar": ["image_pre_1", "image_pre_2", "image_post"],
                "dem": ["image_dem"],
            }
            stacked_images = [
                sample[key]
                for modality in self.band_order
                for key in modality_keys.get(modality, [])
                if key in sample
            ]
            images_sizes = [item.shape for item in stacked_images]
            assert len(set(images_sizes)) == 1, (
                f"{images_sizes=} currently only supports stacking of images/modalities with the same number of bands"
            )
            sample = {  # TODO: stack dem for each sar timestamp
                "image": torch.stack(stacked_images, dim=1),  # [C, T, H, W]
                "mask": sample["mask"],
            }
            _, t, _, _ = sample["image"].shape
            if t == 1:
                sample["image"] = sample["image"].squeeze(1)

        return sample
