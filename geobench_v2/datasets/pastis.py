# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""PASTIS Dataset."""

from torch import Tensor
from torchgeo.datasets import PASTIS
from pathlib import Path


class GeoBenchPASTIS(PASTIS):
    """PAStis Dataset with enhanced functionality.

    This is the PASTIS-R version.

    Allows:
    - Variable Band Selection
    - Return band wavelengths
    """

    band_default_order = {
        "red": 0,
        "green": 1,
        "blue": 2,
        "nir": 3,
        "swir1": 4,
        "swir2": 5,
        "pan": 6,
    }

    def __init__(
        self,
        root: Path,
        split: str,
        band_order: list[str] = ["red", "green", "blue", "nir"],
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
        """
        super().__init__(root=root, split=split, **kwargs)
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
        # TODO need to allow multiple modalities
        # TODO need to allow band order
        image = self._load_image(index)
        if self.mode == "semantic":
            mask = self._load_semantic_targets(index)
            sample = {"image": image, "mask": mask}
        elif self.mode == "instance":
            mask, boxes, labels = self._load_instance_targets(index)
            sample = {"image": image, "mask": mask, "boxes": boxes, "label": labels}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
