# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Resisc45 Dataset."""

import torch
from torch import Tensor
from torchgeo.datasets import RESISC45
from pathlib import Path


class GeoBenchRESISC45(RESISC45):
    """Resisc45 Dataset with enhanced functionality.

    Allows:
    - Variable Band Selection
    - Return band wavelengths
    """

    band_orig_order = {"r": 0, "g": 1, "b": 2}

    def __init__(
        self, root: Path, split: str, band_order: list["str"] = ["r", "g", "b"], **kwargs
    ):
        """Initialize Resisc45 Dataset.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'val', 'test'
            band_order: The order of bands to return, defaults to ['r', 'g', 'b'], if one would
                specify ['g', 'r', 'b', 'b], the dataset would return the green band first, then the red band,
                and then the blue band twice. This is useful for models that expect a certain band order, or
                test the impact of band order on model performance.
            **kwargs: Additional keyword arguments passed to ``RESISC45``
        """
        super().__init__(root=root, split=split, **kwargs)

        # TODO: allow input of blank channels
        assert all(band in self.band_orig_order.keys() for band in band_order), (
            f"Invalid bands in {band_order}. Must be among {list(self.band_orig_order.keys())}"
        )

        self.band_order = band_order

    def _load_image(self, index: int) -> tuple[Tensor, Tensor]:
        """Load a single image and its class label.

        Args:
            index: index to return

        Returns:
            the image and class label
        """
        image, label = super()._load_image(index)

        # Reorder bands according to band_order
        def _reorder_bands(image: Tensor) -> Tensor:
            return torch.stack(
                [image[self.band_orig_order[band]] for band in self.band_order]
            )

        return self._reorder_bands(image), label
