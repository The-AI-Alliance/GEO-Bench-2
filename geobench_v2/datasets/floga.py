# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Forest wiLdfire Observations for the Greek Area (FLOGA) Dataset."""

import os
from collections.abc import Sequence
from typing import ClassVar, Literal

import h5py
import pandas as pd
import torch
from torch import Tensor
from torchgeo.datasets import NonGeoDataset

from .data_util import DataUtilsMixin
from .sensor_util import DatasetBandRegistry


# TODO need to add automatic download
class GeoBenchFLOGA(NonGeoDataset, DataUtilsMixin):
    """Implementation of FLOGA dataset.

    FLOGA is a dataset for forest wildfire detection that provides Sentinel-2 Level-2A and MODIS (Bands 1-7) imagery.

    Classes:

    0: for non-burnt pixels
    1: for burnt pixels
    2: for pixels burnt in other fire events of the same year, these can be excluded/ignored in the
        training/eval procedure

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2311.03339
    """

    splits = ("train", "val", "test")

    dataset_band_config = DatasetBandRegistry.FLOGA

    default_order = (
        ["B02", "B03", "B04", "B08", "M01", "M02", "M03", "M04", "M05", "M06", "M07"],
    )

    normalization_stats: ClassVar[dict[str, dict[str, float]]] = {
        "means": {
            "B02": 0.0,
            "B03": 0.0,
            "B04": 0.0,
            "B07": 0.0,
            "B08": 0.0,
            "M01": 0.0,
            "M02": 0.0,
            "M03": 0.0,
            "M04": 0.0,
            "M05": 0.0,
            "M06": 0.0,
            "M07": 0.0,
        },
        "stds": {
            "B02": 1.0,
            "B03": 1.0,
            "B04": 1.0,
            "B08": 1.0,
            "M01": 1.0,
            "M02": 1.0,
            "M03": 1.0,
            "M04": 1.0,
            "M05": 1.0,
            "M06": 1.0,
            "M07": 1.0,
        },
    }

    def __init__(
        self,
        root,
        split: Literal["train", "val", "test"] = "train",
        band_order: Sequence[float | str] = ["B04", "B03", "B02"],
        transforms=None,
    ) -> None:
        """Initialize a FLOGA dataset instance.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'test'
            band_order: The order of bands to return, defaults to ['r', 'g', 'b'], if one would
                specify ['r', 'g', 'b', 'nir'], the dataset would return images with 4 channels
            transforms: A composition of transforms to apply to the sample_row

        Raises:
            AssertionError: If *split* is not in the splits
        """
        assert split in self.splits, f"split must be one of {self.splits}"

        self.root = root
        self.split = split
        self.transforms = transforms

        self.band_order = self.resolve_band_order(band_order)

        self.normalizer = ClipZScoreNormalizer(
            self.normalization_stats, self.band_order
        )

        # TODO update
        self.metadata_df = pd.read_parquet(
            "/mnt/rg_climate_benchmark/data/datasets_segmentation/floga/dataset/allEvents_60-20-20_r1_all.parquet"
        )
        self.metadata_df = self.metadata_df[
            self.metadata_df["split"] == split
        ].reset_index(drop=True)

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            The number of samples in the dataset
        """
        return len(self.metadata_df)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Return the sample_row at the given index.

        Args:
            idx: Index of the sample_row to return

        Returns:
            dict containing the sample_row data, for each given band modality a pre and post image is returned
        """
        sample: dict[str, Tensor] = {}
        sample_row = self.metadata_df.iloc[idx]

        h5_path = sample_row["filepath"]

        image_pre = {}
        image_post = {}
        with h5py.File(os.path.join(self.root, h5_path), "r") as f:
            if "s2" in self.band_order:
                s2_pre = f["sen2_10_pre"][:]
                image_pre["s2"] = torch.from_numpy(s2_pre).float()
                s2_post = f["sen2_10_post"][:]
                image_post["s2"] = torch.from_numpy(s2_post).float()
            if "modis" in self.band_order:
                modis_pre = f["mod_500_pre"][:]
                image_pre["modis"] = torch.from_numpy(modis_pre).float()
                modis_post = f["mod_500_post"][:]
                image_post["modis"] = torch.from_numpy(modis_post).float()

            mask = f["label"][:]

        image_pre = self.rearrange_bands(image_pre, self.band_order)
        image_pre = self.normalizer(image_pre)
        image_post = self.rearrange_bands(image_post, self.band_order)
        image_post = self.normalizer(image_post)

        sample.update({f"{k}_pre": v for k, v in image_pre.items()})
        sample.update({f"{k}_post": v for k, v in image_post.items()})
        sample.update({"mask": torch.from_numpy(mask).long()})

        if self.transforms:
            sample = self.transforms(sample)

        return sample
