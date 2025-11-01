# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""NZCattle dataset."""

import io
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

import h5py
import rasterio
import torch
import torch.nn as nn
from torch import Tensor

from geobench_v2.datasets.sensor_util import DatasetBandRegistry

from .base import GeoBenchBaseDataset
from .normalization import ZScoreNormalizer


class GeoBenchNZCattle(GeoBenchBaseDataset):
    """GeoBench version of NzCattle dataset."""

    url = "https://hf.co/datasets/aialliance/nzcattle/resolve/main/{}"

    paths = ["geobench_nzcattle.tortilla"]

    sha256str = ["70ca3b78af3f5b17868dd856b8e31b102a03e74439d58960a69c77b1efcd31c1"]

    dataset_band_config = DatasetBandRegistry.NZCATTLE
    band_default_order = ("red", "green", "blue")

    normalization_stats = {
        "means": {
            "red": 126.21480560302734,
            "green": 130.08578491210938,
            "blue": 106.48361206054688,
        },
        "stds": {
            "red": 18.657546997070312,
            "green": 23.068553924560547,
            "blue": 19.50484848022461,
        },
    }

    classes = ["background", "cattle"]

    num_classes = len(classes)

    def __init__(
        self,
        root: Path,
        split: Literal["train", "val", "validation", "test"],
        band_order: Sequence[str] = band_default_order,
        data_normalizer: type[nn.Module] = ZScoreNormalizer,
        transforms: nn.Module | None = None,
        download: bool = False,
    ) -> None:
        """Initialize nzCattle dataset.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'validation', 'test'
            band_order: The order of bands to return, defaults to ['red', 'green', 'blue'], if one would
                specify ['red', 'green', 'blue', 'blue'], the dataset would return images with 4 channels
                in that order. This is useful for models that expect a certain band order, or
                test the impact of band order on model performance.
            data_normalizer: The data normalizer to apply to the data, defaults to :class:`ZScoreNormalizer`,
                which applies z-score normalization to each band.
            transforms: image transformations to apply to the data, defaults to None
            download: Whether to download the dataset
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
            metadata=None,
            download=download,
        )

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        sample_row = self.data_df.read(index)

        image_path = sample_row["internal:subfile"].values[0]
        anno_path = sample_row["internal:subfile"].values[1]

        sample: dict[str, Tensor] = {}

        ## load image
        image = self._load_image(image_path)

        image_dict = self.rearrange_bands(image, self.band_order)
        image_dict = self.data_normalizer(image_dict)
        sample.update(image_dict)

        ## load annotations

        boxes, labels = self._load_target(anno_path)

        sample["bbox_xyxy"] = boxes
        sample["label"] = labels

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_image(self, path: str) -> Tensor:
        """Load an image from disk.

        Args:
            path: Path to the image file.

        Returns:
            image tensor
        """
        with rasterio.open(path) as src:
            image = src.read(out_dtype="float32")

        return torch.tensor(image)

    def _load_target(self, path: str) -> tuple[Tensor, Tensor]:
        """Load target annotations from disk.

        Args:
            path: path to annotation tortilla file

        Returns:
            boxes: bounding boxes tensor in xyxy format
            labels: labels tensor
        """
        pattern = r"(\d+)_(\d+),(.+)"
        match = re.search(pattern, path)
        if match:
            offset = int(match.group(1))
            size = int(match.group(2))
            file_name = match.group(3)

        with open(file_name, "rb") as f:
            f.seek(offset)
            data = f.read(size)
        byte_stream = io.BytesIO(data)

        with h5py.File(byte_stream, "r") as f:
            annotations = json.loads(f.attrs["annotation"])

        annotations = annotations["boxes"]

        boxes = []
        labels = []

        for anno in annotations:
            labels.append(anno["category_id"])

            x, y, width, height = anno["bbox"]

            boxes.append([x, y, x + width, y + height])

        if len(boxes) == 0:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros(
                0, dtype=torch.int64
            )

        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(
            labels, dtype=torch.int64
        )
