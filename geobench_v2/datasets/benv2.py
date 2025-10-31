# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

# Adapted from torchgeo dataset loader from tortilla format
# https://github.com/microsoft/torchgeo/blob/main/torchgeo/datasets/bigearthnet.py

"""BigEarthNet V2 Dataset."""

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


class GeoBenchBENV2(GeoBenchBaseDataset):
    """GeoBench Version of BigEarthNet V2 Dataset.

    Multi-label land cover classification dataset using Sentinel-1 SAR (VV, VH)
    and Sentinel-2 optical (12 bands) imagery, with
    CORINE Land Cover-based hierarchical multi-label annotations.


    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2407.03653
    """

    url = "https://hf.co/datasets/aialliance/benv2/resolve/main/{}"

    paths: Sequence[str] = ["geobench_benv2.tortilla"]

    sha256str: Sequence[str] = [
        "821c2f429c3e85c158c758bbb215bf61170a2451a11284efaf0f89cef97e468a"
    ]

    band_default_order = {
        "s2": (
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B09",
            "B11",
            "B12",
        ),
        "s1": ("VV", "VH"),
    }

    dataset_band_config = DatasetBandRegistry.BENV2

    normalization_stats: dict[str, dict[str, float]] = {
        "means": {
            "B01": 355.96197509765625,
            "B02": 414.3730773925781,
            "B03": 594.096435546875,
            "B04": 559.0433959960938,
            "B05": 919.4099731445312,
            "B06": 1794.6605224609375,
            "B07": 2091.45947265625,
            "B08": 2241.517822265625,
            "B8A": 2288.0302734375,
            "B09": 2289.5380859375,
            "B11": 1556.958740234375,
            "B12": 973.8273315429688,
            "VH": -12.091922760009766,
            "VV": -18.96333885192871,
        },
        "stds": {
            "B01": 512.3419799804688,
            "B02": 541.94921875,
            "B03": 532.579833984375,
            "B04": 607.0200805664062,
            "B05": 646.341064453125,
            "B06": 1041.35009765625,
            "B07": 1231.787841796875,
            "B08": 1340.4661865234375,
            "B8A": 1316.02880859375,
            "B09": 1267.3955078125,
            "B11": 984.2933349609375,
            "B12": 753.2081909179688,
            "VH": 4.574888229370117,
            "VV": 5.396073818206787,
        },
    }

    label_names: Sequence[str] = (
        "Urban fabric",
        "Industrial or commercial units",
        "Arable land",
        "Permanent crops",
        "Pastures",
        "Complex cultivation patterns",
        "Land principally occupied by agriculture, with significant areas of natural vegetation",
        "Agro-forestry areas",
        "Broad-leaved forest",
        "Coniferous forest",
        "Mixed forest",
        "Natural grassland and sparsely vegetated areas",
        "Moors, heathland and sclerophyllous vegetation",
        "Transitional woodland, shrub",
        "Beaches, dunes, sands",
        "Inland wetlands",
        "Coastal wetlands",
        "Inland waters",
        "Marine waters",
    )

    classes = label_names

    num_classes: int = len(label_names)

    multilabel: bool = True

    valid_metadata: Sequence[str] = ("lat", "lon")

    def __init__(
        self,
        root: Path,
        split: Literal["train", "val", "validation", "test"],
        rename_modalities: dict | None = None,
        band_order: dict[str, Sequence[float | str]] = {"s2": ["B04", "B03", "B02"]},
        data_normalizer: type[nn.Module] = ZScoreNormalizer,
        transforms: nn.Module | None = None,
        metadata: Sequence[str] | None = None,
        return_stacked_image: bool = False,
        download: bool = False,
    ) -> None:
        """Initialize Big Earth Net V2 Dataset.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'validation', 'test'
            band_order: The order of bands to return, defaults to ['B04', 'B03', 'B02'], if one would
                specify ['B04', 'B03', 'B02], the dataset would return the red, green, and blue bands.
                This is useful for models that expect a certain band order, or
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
            metadata=metadata,
            download=download,
        )
        if return_stacked_image:
            assert rename_modalities is None, (
                "Cannot return a stacked image if modalities are renamed"
            )
        self.return_stacked_image = return_stacked_image
        self.rename_modalities = rename_modalities
        self.class2idx = {c: i for i, c in enumerate(self.label_names)}

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

        sample["label"] = self._load_target(sample_row.iloc[0]["labels"])

        if "lon" in self.metadata:
            sample["lon"] = torch.tensor(sample_row.iloc[0]["lon"])
        if "lat" in self.metadata:
            sample["lat"] = torch.tensor(sample_row.iloc[0]["lat"])

        return sample

    def _load_target(self, label_names: list[str]) -> Tensor:
        """Load the target mask for a single image.

        Args:
            label_names: list of labels

        Returns:
            the target label
        """
        indices = [self.class2idx[label_names] for label_names in label_names]

        image_target = torch.zeros(self.num_classes, dtype=torch.long)
        image_target[indices] = 1
        return image_target
