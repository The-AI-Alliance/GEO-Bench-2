# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""PASTIS Dataset."""

from torch import Tensor
from torchgeo.datasets import PASTIS
from pathlib import Path
import numpy as np
from typing import Any, Sequence, Union, Type, Literal
import torch
import os
import json
import pandas as pd
import torch.nn as nn

from .sensor_util import DatasetBandRegistry
from .data_util import DataUtilsMixin, MultiModalNormalizer


class GeoBenchPASTIS(PASTIS, DataUtilsMixin):
    """PAStis Dataset with enhanced functionality.

    This is the PASTIS-R version.

    Allows:
    - Variable Band Selection
    - Return band wavelengths
    """

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

    def __init__(
        self,
        root: Path,
        split: str,
        band_order: dict[str, Sequence[float | str]] = {"s2": ["B04", "B03", "B02"]},
        data_normalizer: Type[nn.Module] = MultiModalNormalizer,
        num_time_steps: int = 1,
        transforms: nn.Module | None = None,
        label_type: Literal["instance_seg", "semantic_seg"] = "semantic_seg",
        return_stacked_image: bool = False,
        return_metadata: bool = True,
        **kwargs,
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
            label_type: The type of label to return, either 'instance_seg' or 'semantic_seg'
            return_stacked_image: if true, returns a single image tensor with all modalities stacked in band_order
            return_metadata: if true, returns metadata as part of the image
            **kwargs: Additional keyword arguments passed to ``torchgeo.datasts.PASTIS``

        Raises:
            AssertionError: If an invalid split is specified
        """
        super().__init__(root=root)

        assert split in self.valid_splits, (
            f"Invalid split {split}. Must be one of {self.valid_splits}"
        )
        self.split = split

        self.band_order = self.validate_band_order(band_order)

        self.data_normalizer = data_normalizer(
            self.normalization_stats, self.band_order
        )
        self.transforms = transforms
        self.num_time_steps = num_time_steps

        self.label_type = label_type

        self.metadata_df = pd.read_parquet(
            os.path.join(root, "geobench_pastis.parquet")
        )
        self.metadata_df = self.metadata_df[
            self.metadata_df["split"] == split
        ].reset_index(drop=True)
        # self.metadata_df["ID_PATCH"] = self.metadata_df["ID_PATCH"].astype(str)

        # self.files_df = pd.DataFrame(self.files)
        # self.files_df["ID_PATCH"] = self.files_df["s2"].apply(lambda x: x.split("/")[-1].split("_")[-1].split(".")[0])

        # self.new_df = pd.merge(self.metadata_df, self.files_df, how="left", left_on="ID_PATCH", right_on="ID_PATCH").reset_index(drop=True)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.metadata_df)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        sample: dict[str, Tensor] = {}
        sample_row = self.metadata_df.iloc[index]
        data = {
            "s2": self._load_image(os.path.join(self.root, sample_row["s2_path"])),
            "s1_asc": self._load_image(os.path.join(self.root, sample_row["s1a_path"])),
            "s1_desc": self._load_image(
                os.path.join(self.root, sample_row["s1d_path"])
            ),
        }

        img_dict = self.rearrange_bands(data, self.band_order)

        img_dict = self.data_normalizer(img_dict)

        sample.update(img_dict)

        if self.label_type == "semantic_seg":
            sample["mask"] = self._load_semantic_targets(
                os.path.join(self.root, sample_row["semantic_path"])
            )
        elif self.label_type == "instance_seg":
            sample["mask"], sample["boxes"], sample["label"] = (
                self._load_instance_targets(
                    os.path.join(self.root, sample_row["semantic_path"]),
                    os.path.join(sample_row["instance_path"]),
                )
            )

        dates = sample_row["dates-s2"]
        if len(dates) < self.num_time_steps:
            sample["dates"] = [0] * (self.num_time_steps - len(dates)) + dates
        else:
            sample["dates"] = dates[-self.num_time_steps :]
        sample["lon"] = torch.tensor([sample_row["longitude"]])
        sample["lat"] = torch.tensor([sample_row["latitude"]])

        if self.transforms:
            sample = self.transforms(sample)

        if return_stacked_image:
            stacked_image = []
            for mod in self.band_order:
                if mod == "s1_desc":
                    stacked_image.append(sample["image_s1_desc"])
                if mod == "s1_asc":
                    stacked_image.append(sample["image_s1_asc"])
                if mod == "s2":
                    stacked_image.append(sample["image_s2"])
            output = {}
            output["image"] = torch.cat(stacked_image, 0)
            output["mask"] = sample["mask"]

        if return_metadata:
            output["dates"] = sample["dates"]
            output["lon"] = sample["lon"]
            output["lat"] = sample["lat"]

        return output

    def _load_image(self, path: str) -> Tensor:
        """Load a single time-series.

        Args:
            path: path to the time-series

        Returns:
            the time-series
        """
        tensor = torch.tensor(np.load(path).astype(np.float32))

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
        array = np.load(path)[0].astype(np.uint8).copy()
        tensor = torch.from_numpy(array).long()
        return tensor

    def _load_instance_targets(
        self, sem_path: str, instance_path
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Load the instance segmentation targets for a single sample.

        Args:
            path: path to the label

        Returns:
            the instance segmentation mask, box, and label for each instance
        """
        mask_array = np.load(sem_path)[0].copy()
        instance_array = np.load(instance_path).copy()

        mask_tensor = torch.from_numpy(mask_array)
        instance_tensor = torch.from_numpy(instance_array)

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
        self,
        band_order: Union[
            Sequence[Union[str, float]], dict[str, Sequence[Union[str, float]]]
        ],
    ) -> Union[list[Union[str, float]], dict[str, list[Union[str, float]]]]:
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
            if isinstance(band, (int, float)):
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
