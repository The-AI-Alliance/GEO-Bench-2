# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Base dataset."""

import hashlib
import os
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union, Literal, cast

import rasterio
import tacoreader
import torch
import torch.nn as nn
from torch import Tensor
from torchgeo.datasets import DatasetNotFoundError, NonGeoDataset
from torchvision.datasets.utils import download_url

from .data_util import DataUtilsMixin
from .normalization import DataNormalizer, ZScoreNormalizer


class GeoBenchBaseDataset(NonGeoDataset, DataUtilsMixin):
    """Base dataset for classification tasks."""

    url = ""
    paths: Sequence[str] = []
    sha256str: Sequence[str] = []

    # Normalization stats should follow: {"means"|"stds": {modality: {band: value}}}
    normalization_stats: dict[str, dict[str, float]] = {}
    # Allow subclasses to define a default band order (shape flexible)
    band_default_order: Any = ()

    def __init__(
        self,
        root: Path,
        split: str,
        band_order: Sequence[str] | Mapping[str, Sequence[str]],
        data_normalizer: type[nn.Module] = ZScoreNormalizer,
        transforms: Optional[nn.Module] = None,
        metadata: Optional[Sequence[str]] = None,
        download: bool = False,
    ) -> None:
        """Initialize the dataset.

        Args:
            root: Root directory where the dataset can be found
            split: The dataset split, supports 'train', 'validation', 'test', 'extra_test'. Also accepts 'val' as an alias for 'validation'.
            band_order: List of bands to return
            data_normalizer: Normalization strategy. Can be:
                             - A class type inheriting from DataNormalizer (e.g., ZScoreNormalizer)
                               or a basic callable class (e.g., nn.Identity - default).
                               It will be initialized appropriately (using stats/band_order if needed).
                             - An initialized callable instance (e.g., a custom nn.Module or nn.Identity()).
                               It will be used directly.
            transforms: A composition of transformations to apply to the data
            metadata: metadata names to be returned as part of the sample in the
                __getitem__ method. If None, no metadata is returned.
            download: If True, download the dataset .
        """
        super().__init__()
        self.root = root
        self.split = split
        self.band_order = band_order
        self.transforms = transforms
        self.download = download
        self.dataset_verification()

        # Normalize split value and restrict to the expected literals
        split_norm: Literal["train", "validation", "test"]
        if split == "val":
            split_norm = "validation"
        elif split in ("train", "validation", "test"):
            split_norm = cast(Literal["train", "validation", "test"], split)
        else:
            raise ValueError(
                "split must be one of {'train', 'val', 'validation', 'test'}"
            )
        self.split = split_norm

        # Store metadata as a list of strings on the instance
        self.metadata: list[str] = list(metadata) if metadata is not None else []

        self.band_order = self.resolve_band_order(band_order)

        self.data_df = tacoreader.load([os.path.join(root, f) for f in self.paths])
        self.data_df = self.data_df[
            (self.data_df["tortilla:data_split"] == self.split)
        ].reset_index(drop=True)

        # Initialize normalizer
        self.data_normalizer = data_normalizer(
            self.normalization_stats, self.band_order
        )
        self.transforms = transforms

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return an index within the dataset.

        Args:
            index: Index to return

        Returns:
            A dictionary containing the data and target
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            The length of the dataset
        """
        return len(self.data_df)

    def _load_tiff(self, path: str) -> Tensor:
        """Load a TIFF file.

        Args:
            path: Path to the TIFF file

        Return:
            The image tensor
        """
        with rasterio.open(path) as src:
            img = src.read()

        tensor = torch.from_numpy(img)
        return tensor

    def dataset_verification(self) -> None:
        """Verify the dataset."""
        exists = [os.path.exists(os.path.join(self.root, path)) for path in self.paths]
        if all(exists):
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        for path, sha256str in zip(self.paths, self.sha256str):
            if not os.path.exists(os.path.join(self.root, path)):
                download_url(self.url.format(path), self.root, filename=path)
                if not self.verify_sha256str(os.path.join(self.root, path), sha256str):
                    raise ValueError(
                        f"sha256str verification failed for {path}. "
                        "The file may be corrupted or incomplete."
                    )

        # TODO check for other band stats etc files

    def verify_sha256str(self, file_path, expected_sha256str):
        """Verify file integrity using sha256str hash.

        Args:
            file_path: Path to the file to verify
            expected_sha256str: Expected sha256str hash

        Returns:
            bool: True if the file is valid, False otherwise
        """
        if not os.path.isfile(file_path):
            return False
        sha256str_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256str_hash.update(chunk)

        calculated_hash = sha256str_hash.hexdigest()
        print(calculated_hash)
        return calculated_hash == expected_sha256str
