# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""So2Sat Dataset."""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

import rasterio
import torch
import torch.nn as nn
from torch import Tensor

from .base import GeoBenchBaseDataset
from .normalization import ZScoreNormalizer
from .sensor_util import DatasetBandRegistry


class GeoBenchSo2Sat(GeoBenchBaseDataset):
    """GeoBench Version of So2Sat Dataset.

    Local Climate Zone classification dataset using Sentinel-1 SAR (VV, VH)
    and Sentinel-2 optical (10 bands) imagery.
    """

    url = "https://hf.co/datasets/aialliance/so2sat/resolve/main/{}"

    paths: Sequence[str] = ["geobench_so2sat.tortilla"]

    sha256str: Sequence[str] = [
        "2f9aa3a0cbf7f5071d2fafee24156a6041a9c732f0f979881b8db582201aa7bc"
    ]

    band_default_order = {
        "s2": ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"),
        "s1": ("VV", "VH"),
    }

    dataset_band_config = DatasetBandRegistry.SO2SAT

    normalization_stats: dict[str, dict[str, float]] = {
        "means": {
            "B02": 0.12951241433620453,
            "B03": 0.11725383251905441,
            "B04": 0.1138240396976471,
            "B05": 0.12717732787132263,
            "B06": 0.17068715393543243,
            "B07": 0.19281364977359772,
            "B08": 0.1855016052722931,
            "B8A": 0.20730900764465332,
            "B11": 0.17685027420520782,
            "B12": 0.12849648296833038,
            "VV": -0.000024043731173151173,
            "VH": -0.000029148337489459664,
        },
        "stds": {
            "B02": 0.041423603892326355,
            "B03": 0.05196256563067436,
            "B04": 0.0733252465724945,
            "B05": 0.06936437636613846,
            "B06": 0.07505552470684052,
            "B07": 0.0855887159705162,
            "B08": 0.0865049883723259,
            "B8A": 0.09397122263908386,
            "B11": 0.10238894075155258,
            "B12": 0.09227467328310013,
            "VV": 0.21561050415039062,
            "VH": 0.5443007946014404,
        },
    }

    normalization_stats: dict[str, dict[str, float]] = {
        "means": {
            "B02": 0.12951050698757172,
            "B03": 0.11724399030208588,
            "B04": 0.11381018161773682,
            "B05": 0.12716519832611084,
            "B06": 0.17067235708236694,
            "B07": 0.19281364977359772,
            "B08": 0.185484379529953,
            "B8A": 0.20729145407676697,
            "B11": 0.1768450140953064,
            "B12": 0.12849585711956024,
            "VH": -0.000024043731173151173,
            "VV": -0.000029148337489459664,
        },
        "stds": {
            "B02": 0.041423603892326355,
            "B03": 0.05196256563067436,
            "B04": 0.0733252465724945,
            "B05": 0.06936437636613846,
            "B06": 0.07505552470684052,
            "B07": 0.0855887159705162,
            "B08": 0.0865049883723259,
            "B8A": 0.09397122263908386,
            "B11": 0.10238894075155258,
            "B12": 0.09227467328310013,
            "VH": 0.21561050415039062,
            "VV": 0.5443007946014404,
        },
    }

    classes: Sequence[str] = (
        "Compact high-rise",
        "Compact middle-rise",
        "Compact low-rise",
        "Open high-rise",
        "Open middle-rise",
        "Open low-rise",
        "Lightweight low-rise",
        "Large low-rise",
        "Sparsely built",
        "Heavy industry",
        "Dense Trees",
        "Scattered trees",
        "Bush, scrub",
        "Low plants",
        "Bare rock or paved",
        "Bare soil or sand",
        "Water",
    )

    label_names = classes

    num_classes: int = len(label_names)

    multilabel: bool = False

    def __init__(
        self,
        root: Path,
        split: Literal["train", "val", "validation", "test"],
        rename_modalities: dict | None = None,
        band_order: dict[str, Sequence[float | str]] = {"s2": band_default_order["s2"]},
        data_normalizer: type[nn.Module] = ZScoreNormalizer,
        transforms: nn.Module | None = None,
        return_stacked_image: bool = False,
        download: bool = False,
    ) -> None:
        """Initialize Big Earth Net V2 Dataset.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'validation', 'test'
            band_order: The order of bands to return, defaults to all s2 bands. This is useful for models that expect a certain band order, or
                test the impact of band order on model performance.
            data_normalizer: The data normalizer to apply to the data, defaults to :class:`data_util.ZScoreNormalizer`,
                which applies z-score normalization to each band.
            transforms: Transforms to apply to the data
            metadata: metadata names to be returned under specified keys as part of the sample in the
                __getitem__ method. If None, no metadata is returned.
            return_stacked_image: If True, return the stacked modalities across channel dimension instead of the individual modalities.
            download: Whether to download the dataset
            rename_modalities: dictionary with information to rename modalities in output e.g. {image: {s1:  S1RTC, s2: S2L2A}}
        """
        split_norm: Literal["train", "validation", "test"]
        if split == "val":
            split_norm = "validation"
        else:
            split_norm = cast(Literal["train", "validation", "test"], split)
        super().__init__(
            root=root,
            split=split_norm,
            band_order=band_order,
            data_normalizer=data_normalizer,
            transforms=transforms,
            download=download,
        )
        if return_stacked_image:
            assert rename_modalities is None, (
                "Cannot return a stacked image if modalities are renamed"
            )
        self.return_stacked_image = return_stacked_image
        self.rename_modalities = rename_modalities

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        sample: dict[str, Tensor] = {}

        sample_row = self.data_df.read(index)

        # order is vv, vh
        s1_path = sample_row.read(0)
        s2_path = sample_row.read(1)

        data: dict[str, Tensor] = {}

        if "s1" in self.band_order:
            with rasterio.open(s1_path) as src:
                s1_img = src.read()
            data["s1"] = torch.from_numpy(s1_img).float()
        if "s2" in self.band_order:
            with rasterio.open(s2_path) as src:
                s2_img = src.read()
            data["s2"] = torch.from_numpy(s2_img).float()

        # Rearrange bands and normalize
        img = self.rearrange_bands(data, self.band_order)
        img = self.data_normalizer(img)
        sample.update(img)

        if self.return_stacked_image:
            sample = {
                "image": torch.cat(
                    [sample[f"image_{key}"] for key in self.band_order.keys()], 0
                )
            }

        if self.transforms is not None:
            sample = self.transforms(sample)

        if self.rename_modalities is not None:
            for key, value in self.rename_modalities.items():
                if isinstance(value, dict):
                    sample[key] = {}
                    for old_sub_key in value:
                        if old_sub_key in self.band_order:
                            new_sub_key = value[old_sub_key]
                            sample[key][new_sub_key] = sample[f"image_{old_sub_key}"]
                            del sample[f"image_{old_sub_key}"]
                else:
                    if key in self.band_order:
                        new_sub_key = value
                        sample[new_sub_key] = sample[f"image_{key}"]
                        del sample[f"image_{key}"]
                    else:
                        raise ValueError(
                            "rename_modalities must include names that exist in the dataset"
                        )

        sample["label"] = sample_row.iloc[0]["labels"]

        return sample
