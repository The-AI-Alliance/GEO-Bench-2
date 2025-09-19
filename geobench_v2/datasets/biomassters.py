# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Biomassters dataset."""

from collections.abc import Sequence
from pathlib import Path

import einops
import rasterio
import torch
import torch.nn as nn
from torch import Tensor

from .base import GeoBenchBaseDataset
from .normalization import ZScoreNormalizer
from .sensor_util import DatasetBandRegistry


class GeoBenchBioMassters(GeoBenchBaseDataset):
    """GeoBench version of BioMassters dataset."""

    url = "https://hf.co/datasets/aialliance/biomassters/resolve/main/{}"

    paths = [
        "geobench_biomassters.0000.part.tortilla"
        # "geobench_biomassters.0001.part.tortilla",
        # "geobench_biomassters.0002.part.tortilla",
        # "geobench_biomassters.0003.part.tortilla",
        # "geobench_biomassters.0004.part.tortilla",
        # "geobench_biomassters.0005.part.tortilla",
        # "geobench_biomassters.0006.part.tortilla",
    ]

    sha256str: Sequence[str] = [
        "77682ec73a9d496eb694b6a6e65c2ee793ed9f326e6b37a9dea1b065177334ff"
    ]

    dataset_band_config = DatasetBandRegistry.BIOMASSTERS

    normalization_stats: dict[str, dict[str, float]] = {
        "means": {
            "VV_asc": 0.0,
            "VH_asc": 0.0,
            "VV_desc": 0.0,
            "VH_desc": 0.0,
            "B02": 0.0,
            "B03": 0.0,
            "B04": 0.0,
            "B05": 0.0,
            "B06": 0.0,
            "B07": 0.0,
            "B08": 0.0,
            "B8A": 0.0,
            "B11": 0.0,
            "B12": 0.0,
            "AGB": 0.0,  # 2 percentile
        },
        "stds": {
            "VV_asc": 1.0,
            "VH_asc": 1.0,
            "VV_desc": 1.0,
            "VH_desc": 1.0,
            "B02": 3000.0,
            "B03": 3000.0,
            "B04": 3000.0,
            "B05": 3000.0,
            "B06": 3000.0,
            "B07": 3000.0,
            "B08": 3000.0,
            "B8A": 3000.0,
            "B11": 3000.0,
            "B12": 3000.0,
            "AGB": 289.89,  # 98 percentile
        },
    }

    band_default_order = {
        "s1": {"VV_asc", "VH_asc", "VV_desc", "VH_desc"},
        "s2": {"B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"},
    }

    valid_metadata: Sequence[str] = "time"

    def __init__(
        self,
        root: Path,
        split: str,
        band_order: dict[str, Sequence[str]] = {
            "s1": ["VV_asc", "VH_asc"],
            "s2": ["B04", "B03", "B02", "B08"],
        },
        data_normalizer: type[nn.Module] = ZScoreNormalizer,
        transforms: nn.Module | None = None,
        metadata: Sequence[str] | None = None,
        num_time_steps: int = 1,
        download: bool = False,
    ) -> None:
        """Initialize BioMassters dataset.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'validation', 'test'
            band_order: The order of bands to return, defaults to ['red', 'green', 'blue', 'nir'], if one would
                specify ['red', 'green', 'blue', 'nir', 'nir'], the dataset would return images with 5 channels
                in that order. This is useful for models that expect a certain band order, or
                test the impact of band order on model performance.
            data_normalizer: The data normalizer to apply to the data, defaults to :class:`data_util.ZScoreNormalizer`,
                which applies z-score normalization to each band.
            transforms: The transforms to apply to the data, defaults to None.
            metadata: metadata names to be returned under specified keys as part of the sample in the
                __getitem__ method. If None, no metadata is returned.
            num_time_steps: Number of last time steps to include in the dataset, maximum is 12, for S2
                missing time steps are filled with zeros.
            download: Whether to download the dataset

        Raises:
            AssertionError: If the number of time steps is greater than 12
        """
        super().__init__(
            root=root,
            split=split,
            band_order=band_order,
            data_normalizer=data_normalizer,
            transforms=transforms,
            metadata=metadata,
            download=download,
        )
        assert num_time_steps <= 12, (
            "Number of time steps must be less than or equal to 12"
        )
        self.num_time_steps = num_time_steps

        # data does not have georeferencing information, yet is a Gtiff, that the tacoreader can only read with rasterio
        import warnings

        from rasterio.errors import NotGeoreferencedWarning

        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            idx: index to return

        Returns:
            data and label at that index

        If num_time_steps is 1, the dataset will return image samples with shape [C, H, W],
        if num_time_steps is greater than 1, the dataset will return image samples with shape [T, C, H, W],
        where T is the number of time steps.
        """
        sample: dict[str, Tensor] = {}
        sample_row = self.data_df.read(idx)

        img_dict: dict[str, Tensor] = {}

        spatial_mask = None

        if "s1" in self.band_order:
            sample_s1_row = sample_row[sample_row["modality"] == "S1"]
            s1_data = []
            for i in sample_s1_row.index[: self.num_time_steps]:
                s1_step = sample_row.read(i)
                with rasterio.open(s1_step) as src:
                    img = src.read()
                img = torch.from_numpy(img)
                s1_data.append(img)
            s1_data = torch.stack(s1_data)

            # for single time step only return [C, H, W]
            if self.num_time_steps == 1:
                s1_data = s1_data[0]

            # replace -9999 with 0
            s1_mask = s1_data == -9999
            s1_data[s1_mask] = 0.0
            # Create a spatial mask that ignores channels/timesteps
            if s1_mask.dim() == 3:  # [C, H, W]
                spatial_mask = s1_mask.any(dim=0)  # [H, W]
            else:  # [T, C, H, W]
                spatial_mask = s1_mask.any(dim=(1))  # [T, H, W]
            img_dict["s1"] = s1_data

        if "s2" in self.band_order:
            sample_s2_row = sample_row[sample_row["modality"] == "S2"]
            s2_data = []
            for i in sample_s2_row.index[: self.num_time_steps]:
                s2_step = sample_row.read(i)
                with rasterio.open(s2_step) as src:
                    img = src.read()
                img = torch.from_numpy(img).float()

                s2_data.append(img)

            s2_data = torch.stack(s2_data)

            if s2_data.shape[0] < self.num_time_steps:
                padding = torch.zeros(
                    self.num_time_steps - s2_data.shape[0], *s2_data.shape[1:]
                )
                s2_data = torch.cat((padding, s2_data), dim=0)

            # for single time step only return [C, H, W]
            if self.num_time_steps == 1:
                s2_data = s2_data[0]

            img_dict["s2"] = s2_data

        img_dict = self.rearrange_bands(img_dict, self.band_order)
        img_dict = self.data_normalizer(img_dict)

        # after normalization replace the no-data pixels with 0 again
        if "s1" in self.band_order:
            if img_dict["image_s1"].dim() == 3:  # [C, H, W]
                img_dict["image_s1"][:, spatial_mask] = 0.0
            else:  # [T, C, H, W]
                img_dict["image_s1"][
                    einops.repeat(
                        spatial_mask,
                        "t h w -> t c h w",
                        c=img_dict["image_s1"].shape[1],
                    )
                ] = 0.0

        if "s2" in self.band_order and spatial_mask is not None:
            if img_dict["image_s2"].dim() == 3:  # [C, H, W]
                img_dict["image_s2"][:, spatial_mask] = 0.0
            else:  # [T, C, H, W]
                # unsqueeze channel dim for broadcasting
                img_dict["image_s2"][
                    einops.repeat(
                        spatial_mask,
                        "t h w -> t c h w",
                        c=img_dict["image_s2"].shape[1],
                    )
                ] = 0.0

        sample.update(img_dict)

        # last entry is the agb label
        agb_path = sample_row.read(-1)

        with rasterio.open(agb_path) as src:
            agb = src.read()
        agb = torch.from_numpy(agb).float()

        agb = (
            agb - self.normalization_stats["means"]["AGB"]
        ) / self.normalization_stats["stds"]["AGB"]

        sample["mask"] = agb

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
