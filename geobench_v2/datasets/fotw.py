# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Fields of the World Dataset."""

import torch
from torch import Tensor
from torchgeo.datasets import FieldsOfTheWorld


class GeoBenchFieldsOfTheWorld(FieldsOfTheWorld):
    """Fields of the World Dataset with enhanced functionality.

    Allows:
    - Variable Band Selection
    - Return band wavelengths
    """

    band_default_order = {"red": 0, "green": 1, "blue": 2, "nir": 3}

    def __init__(
        self,
        root: str,
        split: str,
        band_order: list[str] = ["red", "green", "blue", "nir"],
        **kwargs,
    ) -> None:
        """Initialize Fields of the World Dataset.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'val', 'test'
            band_order: The order of bands to return, defaults to ['red', 'green', 'blue', 'nir'], if one would
                specify ['red', 'green', 'blue', 'nir', 'nir'], the dataset would return images with 5 channels
                in that order. This is useful for models that expect a certain band order, or
                test the impact of band order on model performance.
            **kwargs: Additional keyword arguments passed to ``FieldsOfTheWorld``
        """
        super().__init__(root=root, split=split, **kwargs)
        # TODO allow input of blank channels
        assert all(band in self.band_default_order.keys() for band in band_order), (
            f"Invalid bands in {band_order}. Must be among {list(self.band_default_order.keys())}"
        )

        self.band_order = band_order

    def __getitem_(self, idx: int) -> dict[str, Tensor]:
        """Return the image and mask at the given index.

        Args:
            idx: index of the image and mask to return

        Returns:
            dict: a dict containing the image and mask
        """
        win_a_fn = self.files[idx]["win_a"]
        win_b_fn = self.files[idx]["win_b"]
        mask_fn = self.files[idx]["mask"]

        win_a = self._load_image(win_a_fn)
        win_b = self._load_image(win_b_fn)

        # adapt img according to band_order
        win_a = torch.stack(
            [win_a[self.band_default_order[band]] for band in self.band_order]
        )
        win_b = torch.stack(
            [win_b[self.band_default_order[band]] for band in self.band_order]
        )
        mask = self._load_target(mask_fn)

        # concat or return two separate images?
        image = torch.cat((win_a, win_b), dim=0)
        sample = {"image": image, "mask": mask}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
