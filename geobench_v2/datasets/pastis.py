# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""PASTIS Dataset."""

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Literal
import re
import io
import h5py


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torchgeo.datasets import PASTIS

from .base import GeoBenchBaseDataset
from .normalization import DataNormalizer, ZScoreNormalizer
from .sensor_util import DatasetBandRegistry


class GeoBenchPASTIS(GeoBenchBaseDataset):
    """PAStis Dataset with enhanced functionality.

    This is the PASTIS-R version.

    Allows:
    - Variable Band Selection
    - Return band wavelengths
    """

    url = "https://hf.co/datasets/aialliance/pastis/resolve/main/{}"

    paths = [
        "geobench_pastis.0000.part.tortilla",
        "geobench_pastis.0001.part.tortilla",
        "geobench_pastis.0002.part.tortilla",
    ]

    sha256str = [
        "215f404a2444a4eb3d1ad173af102144f4a847d81d172f8835a4664c51813b73",
        "92b25a4220e35104ae2e79916d506c949da16dcba136d5f37452bafc0ca8ce13",
        "f3038a4f7ced1c5faf89368ee10dae408e612fd29a60f500140ce8d315503dbb"
    ]

    dataset_band_config = DatasetBandRegistry.PASTIS

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

    band_default_order = {
        "s2": ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"),
        "s1_asc": ("VV_asc", "VH_asc", "VV/VH_asc"),
        "s1_desc": ("VV_desc", "VH_desc", "VV/VH_desc"),
    }

    valid_splits = ("train", "val", "test")

    normalization_stats = {
        "means": {
            "B02": 0.0,
            "B03": 0.0,
            "B04": 0.0,
            "B05": 0.0,
            "B06": 0.0,
            "B07": 0.0,
            "B08": 0.0,
            "B8A": 0.0,
            "B11": 0.0,
            "B12": 0.0,
            "VV_asc": 0.0,
            "VH_asc": 0.0,
            "VV/VH_asc": 0.0,
            "VV_desc": 0.0,
            "VH_desc": 0.0,
            "VV/VH_desc": 0.0,
        },
        "stds": {
            "B02": 3000.0,
            "B03": 3000.0,
            "B04": 3000.0,
            "B05": 3000.0,
            "B06": 3000.0,
            "B07": 3000.0,
            "B08": 3000.0,
            "B8A": 3000.0,
            "B11": 3000.0,
            "B12": 3000.0,
            "VV_asc": 1.0,
            "VH_asc": 1.0,
            "VV/VH_asc": 1.0,
            "VV_desc": 1.0,
            "VH_desc": 1.0,
            "VV/VH_desc": 1.0,
        },
    }

    classes = PASTIS.classes

    num_classes = len(classes)

    valid_metadata = ("lat", "lon", "dates")

    def __init__(
        self,
        root: Path,
        split: str,
        band_order: dict[str, Sequence[float | str]] = {"s2": ["B04", "B03", "B02"]},
        data_normalizer: type[nn.Module] = ZScoreNormalizer,
        num_time_steps: int = 1,
        transforms: nn.Module | None = None,
        metadata: Sequence[str] | None = None,
        label_type: Literal["instance_seg", "semantic_seg"] = "semantic_seg",
        return_stacked_image: bool = False,
        download: bool = False,
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
            data_normalizer: The data normalizer to apply to the data, defaults to :class:`data_util.ZScoreNormalizer`,
                which applies z-score normalization to each band.
            transforms: The transforms to apply to the data, defaults to None
            metadata: metadata names to be returned under specified keys as part of the sample in the
                __getitem__ method. If None, no metadata is returned.
            label_type: The type of label to return, either 'instance_seg' or 'semantic_seg'
            return_stacked_image: if true, returns a single image tensor with all modalities stacked in band_order
            download: Whether to download the dataset

        Raises:
            AssertionError: If an invalid split is specified
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

        self.label_type = label_type
        self.num_time_steps = num_time_steps
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

        data = {
            "s2": self._load_image(sample_row.read(0)),
            "s1_asc": self._load_image(sample_row.read(1)),
            "s1_desc": self._load_image(sample_row.read(2))
        }

        img_dict = self.rearrange_bands(data, self.band_order)

        img_dict = self.data_normalizer(img_dict)

        sample.update(img_dict)

        if self.label_type == "semantic_seg":
            sample["mask"] = self._load_semantic_targets(sample_row.read(3))
        elif self.label_type == "instance_seg":
            sample["mask"], sample["boxes"], sample["label"] = (
                self._load_instance_targets(
                    sample_row.read(3),
                    sample_row.read(4)
                )
            )

        dates = sample_row["dates"].iloc[0]
        if len(dates) < self.num_time_steps:
            sample_dates = [0] * (self.num_time_steps - len(dates)) + dates
        else:
            sample_dates = dates[-self.num_time_steps :]

        if self.transforms:
            sample = self.transforms(sample)

        if self.return_stacked_image:
            sample = {
                "image": torch.cat(
                    [sample[f"image_{key}"] for key in self.band_order.keys()], 0
                ),
                "mask": sample["mask"],
            }

        if "lon" in self.metadata:
            sample["lon"] = torch.Tensor([sample_row.lon.iloc[0]]).squeeze()
        if "lat" in self.metadata:
            sample["lat"] = torch.Tensor([sample_row.lat.iloc[0]]).squeeze()
        if "dates" in self.metadata:
            sample["dates"] = torch.from_numpy(sample_dates)

        return sample

    def _return_byte_stream(self, path: str):
        """Return a byte stream for a given path.

        Args:
            path: internal path to tortilla modality

        Returns:
            A byte stream of the data
        """
        pattern = r"(\d+)_(\d+),(.+)"
        match = re.search(pattern, path)
        offset = int(match.group(1))
        size = int(match.group(2))
        file_name = match.group(3)

        with open(file_name, "rb") as f:
            f.seek(offset)
            data = f.read(size)
        byte_stream = io.BytesIO(data)

        return byte_stream

    def _load_image(self, path: str) -> Tensor:
        """Load a single time-series.

        Args:
            path: path to the time-series

        Returns:
            the time-series
        """
        with h5py.File(self._return_byte_stream(path), "r") as f:
            tensor = torch.from_numpy(f["data"][:]).float()
        
        if tensor.shape[0] < self.num_time_steps:
            padding = torch.zeros(
                self.num_time_steps - tensor.shape[0], *tensor.shape[1:]
            )
            tensor = torch.cat((padding, tensor), dim=0)
        else:
            tensor = tensor[-self.num_time_steps :]

        if self.num_time_steps == 1:
            tensor = tensor.squeeze(0)

        return tensor.float()

    def _load_semantic_targets(self, path: str) -> Tensor:
        """Load the target mask for a single image.

        Args:
            path: path to the label

        Returns:
            the target mask
        """
        # See https://github.com/VSainteuf/pastis-benchmark/blob/main/code/dataloader.py#L201
        # even though the mask file is 3 bands, we just select the first band
        with h5py.File(self._return_byte_stream(path), "r") as f:
            tensor = torch.from_numpy(f["data"][:][0]).long()
        return tensor

    def _load_instance_targets(
        self, sem_path: str, instance_path: str
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Load the instance segmentation targets for a single sample.

        Args:
            path: path to the label
            instance_path: path to the instance segmentation mask

        Returns:
            the instance segmentation mask, box, and label for each instance
        """
        mask_tensor = self._load_semantic_targets(sem_path)

        with h5py.File(self._return_byte_stream(instance_path), "r") as f:
            instance_tensor = torch.from_numpy(f["data"][:]).long()

        # Convert instance mask of N instances to N binary instance masks
        instance_ids = torch.unique(instance_tensor)
        # Exclude a mask for unknown/background
        instance_ids = instance_ids[instance_ids != 0]
        instance_ids = instance_ids[:, None, None]
        masks: Tensor = instance_tensor == instance_ids

        # Parse labels for each instance
        labels_list = []
        for mask in masks:
            label = mask_tensor[mask]
            label = torch.unique(label)[0]
            labels_list.append(label)

        # Get bounding boxes for each instance
        boxes_list = []
        for mask in masks:
            pos = torch.where(mask)
            xmin = torch.min(pos[1])
            xmax = torch.max(pos[1])
            ymin = torch.min(pos[0])
            ymax = torch.max(pos[0])
            boxes_list.append([xmin, ymin, xmax, ymax])

        masks = masks.to(torch.uint8)
        boxes = torch.tensor(boxes_list).to(torch.float)
        labels = torch.tensor(labels_list).to(torch.long)

        return masks, boxes, labels

    def validate_band_order(
        self, band_order: Sequence[str | float] | dict[str, Sequence[str | float]]
    ) -> list[str | float] | dict[str, list[str | float]]:
        """Validate band order configuration for PASTIS time-series data.

        For PASTIS, we need to ensure that bands in a sequence belong to the same modality,
        since different modalities have different time-series lengths.

        Args:
            band_order: Band order specification

        Returns:
            Validated and resolved band order

        Raises:
            ValueError: If bands from different modalities are mixed in a sequence
        """
        # If it's a dictionary, each modality is handled separately
        if isinstance(band_order, dict):
            resolved = self.resolve_band_order(band_order)
            return resolved

        # For a simple sequence, ensure all bands are from the same modality
        resolved = self.resolve_band_order(band_order)

        # Check that all bands are from the same modality
        modalities = []
        for band in resolved:
            if isinstance(band, (int | float)):
                continue  # Skip fill values

            modality = self.dataset_band_config.band_to_modality.get(band)
            if modality:
                modalities.append(modality)

        if len(set(modalities)) > 1:
            raise ValueError(
                "For PASTIS dataset, bands in a sequence must all be from the same modality "
                "because different modalities have different time-series lengths. "
                f"Found bands from modalities: {set(modalities)}. "
                "Please use either a sequence with bands from only one modality, "
                "or a dictionary with modality-specific band sequences."
            )

        return resolved
