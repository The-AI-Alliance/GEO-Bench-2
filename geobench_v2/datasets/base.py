# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Base dataset."""

import hashlib
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

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
    """Base dataset for GeoBench datasets."""

    url = ""
    paths: Sequence[str] = []
    sha256str: Sequence[str] = []

    normalization_stats: dict[str, dict[str, float]] = {}
    band_default_order: Any = ()

    def __init__(
        self,
        root: Path,
        split: str,
        band_order: Sequence[str] | Mapping[str, Sequence[str]],
        data_normalizer: type[nn.Module] = ZScoreNormalizer,
        transforms: nn.Module | None = None,
        metadata: list[str] | None = None,
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
        self.transforms = transforms
        self.download = download
        self.dataset_verification()

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
        self.metadata: list[str] = metadata if metadata is not None else []

        self.band_order = self.resolve_band_order(band_order)

        self.data_df = tacoreader.load([os.path.join(root, f) for f in self.paths])
        self.data_df = self.data_df[
            (self.data_df["tortilla:data_split"] == self.split)
        ].reset_index(drop=True)

        if isinstance(data_normalizer, type):
            print(f"Initializing normalizer from class: {data_normalizer.__name__}")
            if issubclass(data_normalizer, (DataNormalizer, ZScoreNormalizer)):
                self.data_normalizer = data_normalizer(
                    self.normalization_stats, self.band_order
                )
            else:
                self.data_normalizer = data_normalizer()

        elif callable(data_normalizer):
            print(
                f"Using provided pre-initialized normalizer instance: {data_normalizer.__class__.__name__}"
            )
            self.data_normalizer = data_normalizer
        else:
            raise TypeError(
                f"data_normalizer must be a DataNormalizer subclass type or a callable instance. Got {type(data_normalizer)}"
            )

    def __getitem__(self, index: int) -> dict[str, any]:
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

        return calculated_hash == expected_sha256str
