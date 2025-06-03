# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Biomassters dataset."""

from torch import Tensor
from pathlib import Path
from typing import Sequence, Type
import torch.nn as nn

from .sensor_util import DatasetBandRegistry
from .base import GeoBenchBaseDataset
from .data_util import MultiModalNormalizer
import torch.nn as nn
import rasterio
import numpy as np
import torch
import einops
from PIL import Image


class GeoBenchBioMassters(GeoBenchBaseDataset):
    """BioMassters dataset.

    There are always 12 S1 time steps available but the number of S2 time steps can vary.

    Dataset does not include geospatial information.
    """

    url = "https://hf.co/datasets/aialliance/biomassters/resolve/main/{}"
    # paths = [
    #     "BioMassters.0000.part.tortilla",
    #     "BioMassters.0001.part.tortilla",
    #     "BioMassters.0002.part.tortilla",
    #     "BioMassters.0003.part.tortilla",
    #     "BioMassters.0004.part.tortilla",
    #     "BioMassters.0005.part.tortilla",
    #     "BioMassters.0006.part.tortilla",
    #     "BioMassters.0007.part.tortilla",
    #     "BioMassters.0008.part.tortilla",
    #     "BioMassters.0009.part.tortilla",
    #     "BioMassters.0010.part.tortilla",
    #     "BioMassters.0011.part.tortilla",
    #     "BioMassters.0012.part.tortilla",
    #     "BioMassters.0013.part.tortilla",
    #     "BioMassters.0014.part.tortilla",
    #     "BioMassters.0015.part.tortilla",
    # ]

    paths = [
        "geobench_biomassters.0000.part.tortilla",
        "geobench_biomassters.0001.part.tortilla",
        "geobench_biomassters.0002.part.tortilla",
        "geobench_biomassters.0003.part.tortilla",
        "geobench_biomassters.0004.part.tortilla",
        "geobench_biomassters.0005.part.tortilla",
        "geobench_biomassters.0006.part.tortilla",
    ]

    sha256str: Sequence[str] = [""]

    dataset_band_config = DatasetBandRegistry.BIOMASSTERS

    normalization_stats = {
        "means": {
            "VH_desc": -16.17842674255371,
            "VH_asc": -16.246776580810547,
            "VV_asc": -10.21057415008545,
            "VH_desc": -10.160239219665527,
            "B04": 1976.1279296875,
            "B05": 2247.41552734375,
            "B8A": 2724.071044921875,
            "B02": 2058.060546875,
            "B12": 614.978759765625,
            "B08": 2813.352294921875,
            "B03": 1962.2747802734375,
            "B07": 2676.554931640625,
            "B06": 2639.328857421875,
            "B11": 814.9059448242188,
            "AGB": 0.0,  # 2 percentile
        },
        "stds": {
            "VH_desc": 7.192081451416016,
            "VH_asc": 7.049084186553955,
            "VV_asc": 4.686783313751221,
            "VV_desc": 4.753581523895264,
            "B04": 2660.7744140625,
            "B05": 2692.405517578125,
            "B8A": 2371.029296875,
            "B02": 2772.856201171875,
            "B12": 843.781494140625,
            "B08": 2548.47265625,
            "B03": 2582.9853515625,
            "B07": 2445.801025390625,
            "B06": 2556.077392578125,
            "B11": 993.0784912109375,
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
        data_normalizer: Type[nn.Module] = MultiModalNormalizer,
        transforms: nn.Module | None = None,
        metadata: Sequence[str] | None = None,
        num_time_steps: int = 1,
        download: bool = False,
        return_stacked_image: bool = False,
    ) -> None:
        """Initialize BioMassters dataset.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'val', 'test'
            band_order: The order of bands to return, defaults to ['red', 'green', 'blue', 'nir'], if one would
                specify ['red', 'green', 'blue', 'nir', 'nir'], the dataset would return images with 5 channels
                in that order. This is useful for models that expect a certain band order, or
                test the impact of band order on model performance.
            data_normalizer: The data normalizer to apply to the data, defaults to :class:`data_util.MultiModalNormalizer`,
                which applies z-score normalization to each band.
            transforms:
            metadata: metadata names to be returned under specified keys as part of the sample in the
                __getitem__ method. If None, no metadata is returned.
            num_time_steps: Number of last time steps to include in the dataset, maximum is 12, for S2
                missing time steps are filled with zeros.
            **kwargs: Additional keyword arguments passed to ``torchgeo.datasets.BioMassters``

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

        self.return_stacked_image = return_stacked_image

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
                img_dict["image_s1"] = img_dict["image_s1"].permute(1, 0, 2, 3) #C, T, H, W

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
                img_dict["image_s2"] = img_dict["image_s2"].permute(1, 0, 2, 3) #C, T, H, W

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

        if self.return_stacked_image:
            sample = {
                "image": torch.cat(
                    [sample[f"image_{key}"] for key in self.band_order.keys()], 0
                ),
                "mask": sample["mask"],
            }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
