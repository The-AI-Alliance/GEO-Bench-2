# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""SpaceNet8 dataset."""

from torch import Tensor
from torchgeo.datasets import SpaceNet8
from pathlib import Path


class GeoBenchSpaceNet8(SpaceNet8):
    """SpaceNet8 dataset with enhanced functionality.

    Allows:
    - Variable Band Selection
    - Return band wavelengths
    """

    band_default_order = {"red": 0, "green": 1, "blue": 2, "nir": 3}

    def __init__(
        self,
        root: Path,
        split: str,
        band_order: list[str] = ["red", "green", "blue", "nir"],
        **kwargs,
    ) -> None:
        """Initialize SpaceNet8 dataset.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'val', 'test'
            band_order: The order of bands to return, defaults to ['red', 'green', 'blue', 'nir'], if one would
                specify ['red', 'green', 'blue', 'nir', 'nir'], the dataset would return images with 5 channels
                in that order. This is useful for models that expect a certain band order, or
                test the impact of band order on model performance.
            **kwargs: Additional keyword arguments passed to ``SpaceNet6``
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
        image_path = self.images[index]
        img, tfm, raster_crs = self._load_image(image_path)
        h, w = img.shape[1:]

        # adapt img according to band order

        img = torch.stack(
            [img[self.band_default_order[band]] for band in self.band_order]
        )

        sample = {"image": img}

        if self.split == "train":
            mask_path = self.masks[index]
            mask = self._load_mask(mask_path, tfm, raster_crs, (h, w))
            sample["mask"] = mask

        # We add 1 to the mask to map the current {background, building} labels to
        # the values {1, 2}. This is necessary because we add 0 padding to the
        # mask that we want to ignore in the loss function.
        if "mask" in sample:
            sample["mask"] += 1

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
