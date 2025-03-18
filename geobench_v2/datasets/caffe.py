# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""CaFFe Dataset."""

import os

from typing import Sequence
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchgeo.datasets import CaFFe
from pathlib import Path
import pandas as pd

from .sensor_util import DatasetBandRegistry
from .data_util import DataUtilsMixin, MultiModalNormalizer


class GeoBenchCaFFe(CaFFe, DataUtilsMixin):
    """CaFFe Dataset with enhanced functionality.

    Allows:
    - Variable Band Selection
    - Return band wavelengths
    """

    dataset_band_config = DatasetBandRegistry.CAFFE
    # TODO update sensor type with wavelength and resolution

    band_default_order = ("gray",)

    normalization_stats = {"means": {"gray": 0.0}, "stds": {"gray": 255.0}}

    mask_dirs = ("zones", "zones")

    def __init__(
        self,
        root: Path,
        split: str,
        band_order: list["str"] = band_default_order,
        **kwargs,
    ) -> None:
        """Initialize CaFFe Dataset.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'val', 'test'
            band_order: The order of bands to return, defaults to ['gray'], if one would
                specify ['gray', 'gray', 'gray], the dataset would return the gray band three times.
                This is useful for models that expect a certain band order, or
                test the impact of band order on model performance.
            **kwargs: Additional keyword arguments passed to ``CaFFe``
        """
        super().__init__(root=root, split=split, **kwargs)
        self.band_order = self.resolve_band_order(band_order)

        self.normalizer = MultiModalNormalizer(
            self.normalization_stats, self.band_order
        )

        self.metadata_df = pd.read_parquet(
            os.path.join(self.root, self.data_dir, "geobench_caffe_metadata.parquet")
        )
        self.metadata_df = self.metadata_df[
            self.metadata_df["split"] == self.split
        ].reset_index(drop=True)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Return the image and mask at the given index.

        Args:
            idx: index of the image and mask to return

        Returns:
            dict: a dict containing the image and mask
        """
        sample: dict[str, Tensor] = {}
        zones_filename = os.path.basename(self.fpaths[idx])
        sample_row = self.metadata_df.iloc[idx]
        img_filename = sample_row["filename"]
        zones_filename = img_filename.replace("__", "_zones__")

        def read_tensor(path: str) -> Tensor:
            return torch.from_numpy(np.array(Image.open(path)))

        img_path = os.path.join(
            self.root, self.data_dir, self.image_dir, self.split, img_filename
        )
        img = read_tensor(img_path).unsqueeze(0).float()

        img_dict = self.rearrange_bands(img, self.band_order)

        img_dict = self.normalizer(img_dict)

        sample.update(img_dict)

        zone_mask = read_tensor(
            os.path.join(self.root, self.data_dir, "zones", self.split, zones_filename)
        ).long()

        zone_mask = self.ordinal_map_zones[zone_mask]

        sample["mask"] = zone_mask

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.metadata_df)
