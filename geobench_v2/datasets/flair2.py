# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Flair 2 Aerial Dataset."""

import os

import numpy as np
import rasterio
from typing import Sequence, ClassVar, Union, Type
import torch
import torch.nn as nn
from torch import Tensor
from torchgeo.datasets import NonGeoDataset
from torchgeo.datasets.utils import percentile_normalization
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import glob

from .sensor_util import DatasetBandRegistry
from .data_util import DataUtilsMixin, MultiModalNormalizer


class GeoBenchFLAIR2(NonGeoDataset, DataUtilsMixin):
    """Implementation of FLAIR 2 Aerial dataset."""

    classes = [
        "background",
        "building",
        "pervious surface",
        "impervious surface",
        "bare soil",
        "water",
        "coniferous",
        "deciduous",
        "vineyard",
        "herbaceous vegetation",
        "agricultural land",
        "plowed land",
        "swimming_pool",
        "snow",
        "clear cut",
        "mixed",
        "ligneous",
        "greenhouse",
        "other",
    ]

    url_prefix: str = "https://storage.gra.cloud.ovh.net/v1/AUTH_366279ce616242ebb14161b7991a8461/defi-ia/flair_data_2"
    md5s: ClassVar[dict[str, str]] = {
        "flair-2_centroids_sp_to_patch": "f8ba3b176197c254b6c165c97e93c759",
        "flair_aerial_train": "0f575b360800f58add19c08f05e18429",
        "flair_labels_train": "80d3cd2ee117a61128faa08cbb842c0c",
        "flair_2_aerial_test": "a647e0ba7e5345b28c48d7887ee79888",
        "flair_2_labels_test": "394a769ffcb4a783335eecd3f8baef57",
    }

    dir_names: ClassVar[dict[str, dict[str, str]]] = {
        "train": {"images": "flair_aerial_train", "masks": "flair_labels_train"},
        "test": {"images": "flair_2_aerial_test", "masks": "flair_2_labels_test"},
    }
    globs: ClassVar[dict[str, str]] = {"images": "IMG_*.tif", "masks": "MSK_*.tif"}

    splits = ("train", "val", "test")

    dataset_band_config = DatasetBandRegistry.FLAIR2

    normalization_stats = {
        "means": {"r": 0.0, "g": 0.0, "b": 0.0, "nir": 0.0, "elevation": 0.0},
        "stds": {"r": 255.0, "g": 255.0, "b": 255.0, "nir": 255.0, "elevation": 255.0},
    }

    band_default_order = ("r", "g", "b", "nir", "elevation")

    def __init__(
        self,
        root,
        split="train",
        band_order: Sequence[float | str] = ["r", "g", "b"],
        data_normalizer: Type[nn.Module] = MultiModalNormalizer,
        transforms: nn.Module | None = None,
    ):
        """Initialize FLAIR 2 dataset.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'test'
            band_order: The order of bands to return, defaults to ['r', 'g', 'b'], if one would
                specify ['r', 'g', 'b', 'nir'], the dataset would return images with 4 channels
            data_normalizer: The data normalizer to apply to the data, defaults to :class:`data_util.MultiModalNormalizer`,
                which applies z-score normalization to each band.
            transforms:

        Raises:
            AssertionError: If split is not in the splits
        """
        assert split in self.splits, f"split must be one of {self.splits}"

        self.transforms = transforms

        self.root = root
        self.split = split

        self.band_order = self.resolve_band_order(band_order)

        self.data_normalizer = data_normalizer(
            self.normalization_stats, self.band_order
        )

        self.samples = self._load_files()

    def _load_files(self) -> list[dict[str, str]]:
        """Return the paths of the files in the dataset.

        Returns:
            list of dicts containing paths for each pair of image, masks
        """
        images = sorted(
            glob.glob(
                os.path.join(
                    self.root,
                    self.dir_names[self.split]["images"],
                    "**",
                    self.globs["images"],
                ),
                recursive=True,
            )
        )

        masks = sorted(
            glob.glob(
                os.path.join(
                    self.root,
                    self.dir_names[self.split]["masks"],
                    "**",
                    self.globs["masks"],
                ),
                recursive=True,
            )
        )

        files = [dict(image=image, mask=mask) for image, mask in zip(images, masks)]

        return files

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        sample: dict[str, Tensor] = {}
        path = self.samples[index]["image"]
        image = self.load_image(path)

        image_dict = self.rearrange_bands(image, self.band_order)

        image_dict = self.data_normalizer(image_dict)
        sample.update(image_dict)

        path = self.samples[index]["mask"]
        mask = self.load_mask(path)

        sample["mask"] = mask

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def load_image(self, path: str) -> Tensor:
        """Load an image from a file.

        Args:
            path: path to the image file

        Returns:
            the image as a tensor
        """
        with rasterio.open(path) as f:
            x = f.read()
        x = torch.from_numpy(x).to(torch.float32)
        return x

    def load_mask(self, path: str) -> Tensor:
        """Load a mask from a file.

        Args:
            path: path to the mask file

        Returns:
            the mask as a tensor
        """
        with rasterio.open(path) as f:
            x = f.read(1)
        # TODO replace values > 13 with 13 as "other" class
        x[x > 13] = 13
        # shift the classes to start from 0
        x -= 1
        x = torch.from_numpy(x).to(torch.long)
        return x

    def __len__(self) -> int:
        """Return length of the dataset."""
        return len(self.samples)

    # TODO add automatic download

    def plot(
        self,
        sample: dict[str, torch.Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        image = np.rollaxis(sample["image"][:3].numpy(), 0, 3)
        image = percentile_normalization(image, lower=0, upper=100, axis=(0, 1))

        ncols = 1
        show_mask = "mask" in sample
        show_predictions = "prediction" in sample

        if show_mask:
            mask = sample["mask"].numpy()
            ncols += 1

        if show_predictions:
            prediction = sample["prediction"].numpy()
            ncols += 1

        fig, axs = plt.subplots(ncols=ncols, figsize=(ncols * 8, 8))
        if not isinstance(axs, np.ndarray):
            axs = [axs]
        axs[0].imshow(image)
        axs[0].axis("off")
        if show_titles:
            axs[0].set_title("Image")

        if show_mask:
            axs[1].imshow(mask, interpolation="none")
            axs[1].axis("off")
            if show_titles:
                axs[1].set_title("Label")

        if show_predictions:
            axs[2].imshow(prediction, interpolation="none")
            axs[2].axis("off")
            if show_titles:
                axs[2].set_title("Prediction")

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
