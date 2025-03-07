# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""PASTIS Dataset."""

from torch import Tensor
from torchgeo.datasets import PASTIS
from pathlib import Path
import numpy as np
import torch


class GeoBenchPASTIS(PASTIS):
    """PAStis Dataset with enhanced functionality.

    This is the PASTIS-R version.

    Allows:
    - Variable Band Selection
    - Return band wavelengths
    """

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
        "vv_asc",
        "vh_asc",
        "vv/vh_asc",
        "vv_desc",
        "vh_desc",
        "vv/vh_desc",
    )

    valid_splits = ("train", "val", "test")

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

        Raises:
            AssertionError: If an invalid split is specified
        """
        super().__init__(root=root, **kwargs)
        # TODO allow input of blank channels

        assert split in self.valid_splits, (
            f"Invalid split {split}. Must be one of {self.valid_splits}"
        )
        self.split = split

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
        image_s2 = self._load_image(index, "s2")
        image_s1a = self._load_image(index, "s1a")
        image_s1a = self._load_image(index, "s1b")

        # each of them is a time series, so we concatenate them along the channel dimension
        image = torch.cat([image_s2, image_s1a, image_s1b], dim=1)
        import pdb

        pdb.set_trace()
        if self.mode == "semantic":
            mask = self._load_semantic_targets(index)
            sample = {"image": image, "mask": mask}
        elif self.mode == "instance":
            mask, boxes, labels = self._load_instance_targets(index)
            sample = {"image": image, "mask": mask, "boxes": boxes, "label": labels}

        if self.transforms is not None:
            sample = self.transforms(sample)

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

        tensor = torch.from_numpy(array)
        return tensor
