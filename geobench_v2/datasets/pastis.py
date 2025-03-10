# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""PASTIS Dataset."""

from torch import Tensor
from torchgeo.datasets import PASTIS
from pathlib import Path
import numpy as np
from typing import Any, Sequence, Union
import torch

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
        "VV_asc",
        "VH_asc",
        "VV/VH_asc",
        "VV_desc",
        "VH_desc",
        "VV/VH_desc",
    )

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
            "VV_asc": 3000.0,
            "VH_asc": 3000.0,
            "VV/VH_asc": 3000.0,
            "VV_desc": 3000.0,
            "VH_desc": 3000.0,
            "VV/VH_desc": 3000.0,
        },
    }

    def __init__(
        self,
        root: Path,
        split: str,
        band_order: Sequence[float | str] | dict[str, Sequence[float | str]] = [
            "B04",
            "B03",
            "B02",
        ],
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
            **kwargs: Additional keyword arguments passed to ``PASTIS``

        Raises:
            AssertionError: If an invalid split is specified
        """
        super().__init__(root=root, **kwargs)

        assert split in self.valid_splits, (
            f"Invalid split {split}. Must be one of {self.valid_splits}"
        )
        self.split = split

        self.band_order = self.validate_band_order(band_order)

        self.normalizer = MultiModalNormalizer(
            self.normalization_stats, self.band_order
        )

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        sample: dict[str, Tensor] = {}
        image_s2 = self._load_image(index, "s2")
        image_s1a = self._load_image(index, "s1a")
        image_s1d = self._load_image(index, "s1d")

        # each of them is a time series, so we concatenate them along the channel dimension
        data = {"s2": image_s2, "s1_asc": image_s1a, "s1_desc": image_s1d}
        img_dict = self.rearrange_bands(data, self.band_order)

        img_dict = self.normalizer(img_dict)

        sample.update(img_dict)

        if self.mode == "semantic":
            mask = self._load_semantic_targets(index)
            sample["mask"] = mask
        elif self.mode == "instance":
            mask, boxes, labels = self._load_instance_targets(index)
            sample["mask"] = mask
            sample["boxes"] = boxes
            sample["label"] = labels

        return sample

    def _load_image(self, index: int, bands: str) -> Tensor:
        """Load a single time-series.

        Args:
            index: index to return
            bands: bands to internally load from torchgeo

        Returns:
            the time-series
        """
        path = self.files[index][bands]
        array = np.load(path)
        tensor = torch.from_numpy(array)  # [T, C, H, W]
        # TODO fix later, but atm only return the latest time step
        return tensor[-1]

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
