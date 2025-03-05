# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Big Earth Net V2 Dataset."""

from torchgeo.datasets import BigEarthNetV2
from torch import Tensor
from pathlib import Path
from typing import List, Union, Optional
from .sensor_util import BandRegistry, SatelliteType
from .data_util import DataUtilsMixin
import torch


class GeoBenchBENV2(BigEarthNetV2, DataUtilsMixin):
    """Big Earth Net V2 Dataset with enhanced functionality.

    Allows:
    - Variable Band Selection
    - Return band wavelengths
    """

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
        "B11",
        "B12",
        "VV",
        "VH",
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
            "B11": 0.0,
            "B12": 0.0,
            "VV": 0.0,
            "VH": 0.0,
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
            "B11": 3000.0,
            "B12": 3000.0,
            "VV": 3000.0,
            "VH": 3000.0,
        },
    }

    def __init__(
        self,
        root: Path,
        split: str,
        band_order: list["str"] = ["B04", "B03", "B02"],
        **kwargs,
    ) -> None:
        """Initialize Big Earth Net V2 Dataset.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'val', 'test'
            band_order: The order of bands to return, defaults to ['B04', 'B03', 'B02'], if one would
                specify ['B04', 'B03', 'B02], the dataset would return the red, green, and blue bands.
                This is useful for models that expect a certain band order, or
                test the impact of band order on model performance.
            **kwargs: Additional keyword arguments passed to ``BigEarthNetV2``
        """
        super().__init__(root=root, split=split, bands="all", **kwargs)

        # Resolve band names at init time
        self.band_order = []
        for band in band_order or self.band_default_order:
            if isinstance(band, (int, float)):
                self.band_order.append(band)
            else:
                # For multimodal, keep modality prefix
                if "_" in band:
                    mod, band_name = band.split("_", 1)
                    canonical = f"{mod}_{BandRegistry.resolve_band(band_name)}"
                else:
                    canonical = BandRegistry.resolve_band(band)
                self.band_order.append(canonical)

        self.set_normalization_stats(self.band_order)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        sample: dict[str, Tensor] = {}

        img = torch.cat(
            [self._load_image(index, "s1"), self._load_image(index, "s2")], dim=0
        )

        img = self.rearrange_bands(img, self.band_default_order, self.band_order)

        sample["image"] = self.normalizer(img)

        # subselect_band_order

        sample["mask"] = self._load_map(index)
        sample["label"] = self._load_target(index)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
