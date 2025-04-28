# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Base dataset."""

import torch.nn as nn
from torch import Tensor
from typing import Type, Literal, Sequence, Callable, Union
import rasterio
from torchgeo.datasets import NonGeoDataset
import tacoreader
import os
import torch
import torch.nn as nn


from .data_util import DataUtilsMixin, MultiModalNormalizer, DataNormalizer


class GeoBenchBaseDataset(NonGeoDataset, DataUtilsMixin):
    """Base dataset for classification tasks."""

    paths: list[str] = []

    normalization_stats = {"means": {}, "stds": {}}
    band_default_order: dict[str, list[str]] = {}

    def __init__(
        self,
        root: str,
        split: Literal["train", "validation", "test"],
        band_order: list[str],
        data_normalizer: Union[
            Type[DataNormalizer], Callable[[dict[str, Tensor]], dict[str, Tensor]]
        ] = nn.Identity,
        transforms: nn.Module = None,
        metadata: Sequence[str] | None = None,
    ) -> None:
        """Initialize the dataset.
        Args:
            root: Root directory where the dataset can be found
            split: The dataset split, supports 'train', 'val', 'test'
            band_order:
            data_normalizer: Normalization strategy. Can be:
                             - A class type inheriting from DataNormalizer (e.g., MultiModalNormalizer)
                               or a basic callable class (e.g., nn.Identity - default).
                               It will be initialized appropriately (using stats/band_order if needed).
                             - An initialized callable instance (e.g., a custom nn.Module or nn.Identity()).
                               It will be used directly.
            transform: A composition of transformations to apply to the data
            metadata: metadata names to be returned as part of the sample in the
                __getitem__ method. If None, no metadata is returned.
        """
        super().__init__()
        self.root = root
        self.split = split
        self.band_order = band_order
        self.transforms = transforms
        if metadata is None:
            self.metadata = []
        else:
            self.metadata = metadata

        self.data_df = tacoreader.load([os.path.join(root, f) for f in self.paths])
        effective_split = "validation" if split == "val" else split
        self.data_df = self.data_df[
            self.data_df["tortilla:data_split"] == effective_split
        ].reset_index(drop=True)

        self.band_order = self.resolve_band_order(band_order)

        if isinstance(data_normalizer, type):
            print(f"Initializing normalizer from class: {data_normalizer.__name__}")
            if issubclass(data_normalizer, DataNormalizer):
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

        Return
            The image tensor
        """
        with rasterio.open(path) as src:
            img = src.read()

        tensor = torch.from_numpy(img)
        return tensor
