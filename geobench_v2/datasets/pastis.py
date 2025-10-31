# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""PASTIS Dataset."""

import io
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

import h5py
import torch
import torch.nn as nn
from torch import Tensor
from torchgeo.datasets import PASTIS

from .base import GeoBenchBaseDataset
from .normalization import ZScoreNormalizer
from .sensor_util import DatasetBandRegistry


class GeoBenchPASTIS(GeoBenchBaseDataset):
    """GeoBench version of PASTIS dataset.

    Crop type and parcel segmentation dataset using
    multi-temporal Sentinel-1 and Sentinel-2 imagery, with 19-class parcel-level labels.

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2112.07558
    """

    url = "https://hf.co/datasets/aialliance/pastis/resolve/main/{}"

    paths = [
        "geobench_pastis.0000.part.tortilla",
        "geobench_pastis.0001.part.tortilla",
        "geobench_pastis.0002.part.tortilla",
    ]

    sha256str = [
        "56b1490c6dc7345fdff79e94d9132753ee28d8504bb061d8db39d19e888f7ca3",
        "7d0463a695a0822a1f25638598b2d54daf28f387d6a80d353be7c323069060db",
        "22d27389b1ccee4f250f4a187d034d12dbf15c5610a5dd8d502ee8783e94c81e",
    ]

    dataset_band_config = DatasetBandRegistry.PASTIS

    band_default_order = {
        "s2": ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"),
        "s1_asc": ("VV_asc", "VH_asc", "VV/VH_asc"),
        "s1_desc": ("VV_desc", "VH_desc", "VV/VH_desc"),
    }

    normalization_stats: dict[str, dict[str, float]] = {
        "means": {
            "B02": 1369.9984130859375,
            "B03": 1583.14794921875,
            "B04": 1627.649658203125,
            "B05": 1930.8377685546875,
            "B06": 2921.8388671875,
            "B07": 3284.9306640625,
            "B08": 3421.798828125,
            "B8A": 3544.233642578125,
            "B11": 2564.71435546875,
            "B12": 1708.5986328125,
            "VV_asc": -10.283859252929688,
            "VH_asc": -16.86566734313965,
            "VV/VH_asc": 6.581782817840576,
            "VV_desc": -10.348858833312988,
            "VH_desc": -16.90220069885254,
            "VV/VH_desc": 6.553304672241211,
        },
        "stds": {
            "B02": 2247.75537109375,
            "B03": 2179.169921875,
            "B04": 2255.17626953125,
            "B05": 2142.72216796875,
            "B06": 1928.7330322265625,
            "B07": 1900.8660888671875,
            "B08": 1890.31640625,
            "B8A": 1873.0811767578125,
            "B11": 1409.2015380859375,
            "B12": 1189.0947265625,
            "VV_asc": 3.0927364826202393,
            "VH_asc": 3.026491403579712,
            "VV/VH_asc": 3.3431670665740967,
            "VV_desc": 3.216468334197998,
            "VH_desc": 3.0307400226593018,
            "VV/VH_desc": 3.3312063217163086,
        },
    }

    classes = PASTIS.classes

    num_classes = len(classes)

    valid_metadata = ("lat", "lon", "dates")

    def __init__(
        self,
        root: Path,
        split: Literal["train", "val", "validation", "test"],
        rename_modalities: dict | None = None,
        band_order: dict[str, Sequence[float | str]] = {"s2": ["B04", "B03", "B02"]},
        data_normalizer: type[nn.Module] = ZScoreNormalizer,
        num_time_steps: int = 1,
        transforms: nn.Module | None = None,
        metadata: Sequence[str] | None = None,
        label_type: Literal["instance_seg", "semantic_seg"] = "semantic_seg",
        return_stacked_image: bool = False,
        temporal_aggregation: Literal["mean", "median"] = None,
        temporal_output_format: Literal["TCHW", "CTHW"] = "CTHW",
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
            rename_modalities: dictionary with information to rename modalities in output e.g. {image: {sar:  S1RTC, rgbn: S2L2A}}
            download: Whether to download the dataset
            temporal_aggregation: whether apply temporal aggregation [mean, median]
            temporal_output_format: what temporal format the data should be in [TCHW, CTHW]

        Raises:
            AssertionError: If an invalid split is specified
        """
        split_norm: Literal["train", "validation", "test"]
        if split == "val":
            split_norm = "validation"
        else:
            split_norm = cast(Literal["train", "validation", "test"], split)

        band_order = self.validate_band_order(band_order)
        if metadata is None:
            metadata = []
        else:
            metadata = metadata

        super().__init__(
            root=root,
            split=split_norm,
            band_order=band_order,
            data_normalizer=data_normalizer,
            transforms=transforms,
            metadata=metadata,
            download=download,
        )

        if split == "validation":
            split = "val"

        self.split = split

        self.band_order = self.validate_band_order(band_order)

        self.transforms = transforms
        self.num_time_steps = num_time_steps
        self.label_type = label_type
        if return_stacked_image:
            assert rename_modalities is None, (
                "Cannot return a stacked image if modalities are renamed"
            )
        self.return_stacked_image = return_stacked_image
        self.rename_modalities = rename_modalities

        if metadata is None:
            self.metadata = []
        else:
            self.metadata = metadata

        self.temporal_aggregation = temporal_aggregation
        self.temporal_output_format = temporal_output_format

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data_df)

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
            "s1_desc": self._load_image(sample_row.read(2)),
        }

        img_dict = self.rearrange_bands(data, self.band_order)

        img_dict = self.data_normalizer(img_dict)

        sample.update(img_dict)

        if self.label_type == "semantic_seg":
            sample["mask"] = self._load_semantic_targets(sample_row.read(3))
        elif self.label_type == "instance_seg":
            sample["mask"], sample["boxes"], sample["label"] = (
                self._load_instance_targets(sample_row.read(3), sample_row.read(4))
            )

        dates = sample_row["dates"].iloc[0]
        if len(dates) < self.num_time_steps:
            sample_dates = [0] * (self.num_time_steps - len(dates)) + dates
        else:
            sample_dates = dates[-self.num_time_steps :]
        if self.transforms:
            sample = self.transforms(sample)

        if self.temporal_output_format == "CTHW":
            for key in sample:
                if "image" in key and len(sample[key].shape) == 4:  # [T, C, H, W]
                    sample[key] = sample[key].permute(1, 0, 2, 3)  # C, T, H, W

        if self.return_stacked_image:
            sample = {
                "image": torch.cat(
                    [sample[f"image_{key}"] for key in self.band_order.keys()], 0
                ),
                "mask": sample["mask"],
            }
            if self.num_time_steps == 1:
                sample["image"] = sample["image"].squeeze(1)

        if self.rename_modalities is not None:
            for key, value in self.rename_modalities.items():
                if isinstance(value, dict):
                    sample[key] = {}
                    for old_sub_key in value:
                        if old_sub_key in self.band_order:
                            new_sub_key = value[old_sub_key]
                            if new_sub_key in sample[key]:
                                # Note that this overwrites key order in self.band_order,
                                # so order of self.rename_modalities should follow final desired order
                                sample[key][new_sub_key] = torch.cat(
                                    [
                                        sample[key][new_sub_key],
                                        sample[f"image_{old_sub_key}"],
                                    ],
                                    0,
                                )
                            else:
                                sample[key][new_sub_key] = sample[
                                    f"image_{old_sub_key}"
                                ]
                            del sample[f"image_{old_sub_key}"]
                else:
                    if key in self.band_order:
                        new_sub_key = value
                        sample[new_sub_key] = sample[f"image_{key}"]
                        del sample[f"image_{key}"]
                    else:
                        raise ValueError(
                            "rename_modalities must include names that exist in the dataset"
                        )

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
        if match:
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
            step = tensor.shape[0] / self.num_time_steps
            indexes = [int(i * step) for i in range(self.num_time_steps)]
            tensor = tensor[indexes, :, :, :]

        if self.temporal_aggregation is not None:
            if self.temporal_aggregation == "mean":
                tensor = torch.mean(tensor, 0)
            if self.temporal_aggregation == "median":
                tensor = torch.median(tensor, 0).values

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
            sem_path: path to the label
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

        mask_tensor = mask_tensor.to(torch.int16)
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
            if xmin == xmax:
                xmax = xmax + 1
            if ymin == ymax:
                ymax = ymax + 1
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
