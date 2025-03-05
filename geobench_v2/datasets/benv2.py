# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Big Earth Net V2 Dataset."""

from torchgeo.datasets import BigEarthNetV2
from torch import Tensor
from pathlib import Path


class GeoBenchBENV2(BigEarthNetV2):
    """Big Earth Net V2 Dataset with enhanced functionality.

    Allows:
    - Variable Band Selection
    - Return band wavelengths
    """

    band_default_order = {
        "B01": 0,
        "B02": 1,
        "B03": 2,
        "B04": 3,
        "B05": 4,
        "B06": 5,
        "B07": 6,
        "B08": 7,
        "B8A": 8,
        "B09": 9,
        "B11": 10,
        "B12": 11,
        "VV": 0,
        "VH": 1,
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

        # TODO allow input of blank channels
        assert all(band in self.band_default_order.keys() for band in band_order), (
            f"Invalid bands in {band_order}. Must be among {list(self.band_default_order.keys())}"
        )

        self.band_order = band_order

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        sample: dict[str, Tensor] = {}

        match self.bands:
            case "s1":
                sample["image"] = self._load_image(index, "s1")
            case "s2":
                sample["image"] = self._load_image(index, "s2")
            case "all":
                sample["image_s1"] = self._load_image(index, "s1")
                sample["image_s2"] = self._load_image(index, "s2")

        # subselect_band_order

        sample["mask"] = self._load_map(index)
        sample["label"] = self._load_target(index)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
