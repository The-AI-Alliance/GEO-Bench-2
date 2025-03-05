# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""EverWatch dataset."""

from torchgeo.datasets import EverWatch
import numpy as np
from PIL import Image
from torch import Tensor
import torch
from pathlib import Path


class GeoBenchEverWatch(EverWatch):
    """ "GeoBenchEverWatch dataset with enhanced functionality.

    Allows:
    - Variable Band Selection
    - Return band wavelengths
    """

    band_default_order = {"red": 0, "green": 1, "blue": 2}

    def __init__(
        self,
        root: Path,
        split: str,
        band_order: list[str] = ["red", "green", "blue"],
        **kwargs,
    ) -> None:
        """Initialize EverWatch dataset.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'val', 'test'
            band_order: The order of bands to return, defaults to ['red', 'green', 'blue'], if one would
                specify ['red', 'green', 'blue', 'blue'], the dataset would return images with 4 channels
                in that order. This is useful for models that expect a certain band order, or
                test the impact of band order on model performance.
            **kwargs: Additional keyword arguments passed to ``EverWatch``
        """
        super().__init__(root=root, split=split, **kwargs)
        # TODO allow input of blank channels
        assert all(band in self.band_default_order.keys() for band in band_order), (
            f"Invalid bands in {band_order}. Must be among {list(self.band_default_order.keys())}"
        )

        self.band_order = band_order

    def _load_image(self, path: Path) -> Tensor:
        """Load a single image.

        Args:
            path: path to the image

        Returns:
            the image
        """
        with Image.open(path) as img:
            array: np.typing.NDArray[np.uint8] = np.array(img)
            tensor = torch.from_numpy(array)
            # Convert from HxWxC to CxHxW
            tensor = tensor.permute((2, 0, 1))

        # variable band selection
        tensor = torch.stack(
            [tensor[self.band_default_order[band]] for band in self.band_order]
        )
        return tensor
