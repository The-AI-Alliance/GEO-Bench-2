# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""m-So2Sat Dataset."""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

import rasterio
import torch
import torch.nn as nn
from torch import Tensor

from .base import GeoBenchBaseDataset
from .normalization import MultiModalNormalizer
from .sensor_util import DatasetBandRegistry


class GeoBenchSo2Sat(GeoBenchBaseDataset):
    """m-So2Sat Dataset with enhanced functionality.

    Allows:
    - Variable Band Selection
    - Return band wavelengths
    """

    url = "https://hf.co/datasets/aialliance/mso2sat/resolve/main/{}"

    paths: Sequence[str] = ["geobench_so2sat.tortilla"]

    sha256str: Sequence[str] = [
        "1c1eea74f352b02e49a8549e2605f89c23bce2474c6bc728849d0ce97b50eb2b"
    ]
        

    band_default_order = {
        "s2": (
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B11",
            "B12",
        ),
        "s1": ("VV", "VH"),
    }

    dataset_band_config = DatasetBandRegistry.SO2SAT

    normalization_stats: dict[str, dict[str, float]] = {
        "means": {
                    "B02": 0.12951652705669403,
                    "B03": 0.11734361201524734,
                    "B04": 0.11374464631080627,
                    "B05": 0.12693354487419128,
                    "B06": 0.16917912662029266,
                    "B07": 0.19080990552902222,
                    "B08": 0.18381330370903015,
                    "B8A": 0.20517952740192413,
                    "B11": 0.1762811541557312,
                    "B12": 0.1286638230085373,
                    "VH": 0.00030114364926703274,
                    "VV": 0.000289927760604769,

                },
        "stds": {

                    "B02": 0.040680479258298874,
                    "B03": 0.05125178396701813,
                    "B04": 0.07254913449287415,
                    "B05": 0.06872648745775223,
                    "B06": 0.07402216643095016,
                    "B07": 0.08412779122591019,
                    "B08": 0.08534552156925201,
                    "B8A": 0.09248979389667511,
                    "B11": 0.10270608961582184,
                    "B12": 0.09284552931785583,
                    "VH": 0.20626230537891388,
                    "VV": 0.5187134146690369,
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
        "Water"
    )

    num_classes: int = len(classes)

    multilabel: bool = False

    def __init__(
        self,
        root: Path,
        split: Literal["train", "val", "validation", "test"],
        rename_modalities: dict | None = None,
        band_order: dict[str, Sequence[float | str]] = {"s2": band_default_order['s2']},
        data_normalizer: type[nn.Module] = MultiModalNormalizer,
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
            data_normalizer: The data normalizer to apply to the data, defaults to :class:`data_util.MultiModalNormalizer`,
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
