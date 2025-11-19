# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Substation dataset."""

import io
import json
import re
from pathlib import Path

import h5py
import numpy as np
import rasterio
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torch import Tensor

from geobench_v2.datasets.sensor_util import DatasetBandRegistry

from .base import GeoBenchBaseDataset
from .normalization import ZScoreNormalizer


def polygon_to_mask(vertices, width=228, height=228):
    """Convert a polygon defined by a flat vertex list into a binary mask.

    Args:
        vertices (list): Flat list of coordinates [x1, y1, x2, y2, ..., xn, yn]
        width (int): Mask width (default: 228)
        height (int): Mask height (default: 228)

    Returns:
        np.ndarray: Binary mask (dtype=np.uint8) with 1s inside the polygon.
    """
    # Convert flat list to list of (x, y) tuples
    polygon = [(vertices[i], vertices[i + 1]) for i in range(0, len(vertices), 2)]

    # Create blank image and draw filled polygon
    img = Image.new("L", (width, height), 0)  # 'L' mode = 8-bit grayscale
    draw = ImageDraw.Draw(img)
    draw.polygon(polygon, fill=1)  # Fill polygon with 1 (white)

    # Convert to NumPy array
    return np.array(img, dtype=np.uint8)


class GeoBenchSubstation(GeoBenchBaseDataset):
    """GeoBench version Substation dataset."""

    url = "https://hf.co/datasets/aialliance/substation/resolve/main/{}"

    paths = ["geobench_substation.tortilla"]

    sha256str = ["7f12cd5b510fca4a153b8e77b786d7fc7f7c4e04aece1507121e9c24ff1d47a4"]

    dataset_band_config = DatasetBandRegistry.SUBSTATION
    band_default_order = dataset_band_config.default_order

    normalization_stats = {
        "means": {
            "B01": 1439.5198974609375,
            "B02": 1238.3345947265625,
            "B03": 1204.8643798828125,
            "B04": 1188.5020751953125,
            "B05": 1428.9481201171875,
            "B06": 2204.446533203125,
            "B07": 2580.017333984375,
            "B08": 2501.617431640625,
            "B8A": 2803.1650390625,
            "B09": 824.6425170898438,
            "B10": 16.955820083618164,
            "B11": 2201.2177734375,
            "B12": 1500.3538818359375,
        },
        "stds": {
            "B01": 281.2446594238281,
            "B02": 384.88836669921875,
            "B03": 446.6835632324219,
            "B04": 665.450927734375,
            "B05": 592.2904663085938,
            "B06": 605.96630859375,
            "B07": 750.5545043945312,
            "B08": 766.072509765625,
            "B8A": 832.4408569335938,
            "B09": 326.4425048828125,
            "B10": 16.070602416992188,
            "B11": 835.5003662109375,
            "B12": 814.5209350585938,
        },
    }

    classes = ["background", "power_station"]

    num_classes = len(classes)

    def __init__(
        self,
        root: Path,
        split: str,
        band_order: list[str] = band_default_order,
        data_normalizer: type[nn.Module] = ZScoreNormalizer,
        transforms: nn.Module | None = None,
        download: bool = False,
    ) -> None:
        """Initialize Substation dataset.

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
        super().__init__(
            root=root,
            split=split,
            band_order=band_order,
            data_normalizer=data_normalizer,
            transforms=transforms,
            metadata=None,
            download=download,
        )

        self.band_indexes = [
            [i for i, y in enumerate(self.band_default_order) if y == x][0]
            for x in self.band_order
        ]
        if len(self.band_indexes) != len(self.band_order):
            assert "Invalid element in band_order"

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
        image_dict = {"image": self._load_image(image_path)}
        image_dict = self.data_normalizer(image_dict)
        sample.update(image_dict)

        ## load annotations

        boxes, labels, masks = self._load_target(anno_path)

        sample["bbox_xyxy"] = boxes
        sample["label"] = labels
        sample["mask"] = masks

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

        image = image[self.band_indexes, :, :]

        tensor_image = torch.from_numpy(image)
        tensor_image = tensor_image.float()

        return tensor_image

    def _load_target(self, path: str) -> tuple[Tensor, Tensor]:
        """Load target annotations from disk.

        Args:
            path: path to annotation tortilla file

        Returns:
            boxes: bounding boxes tensor in xyxy format
            labels: labels tensor
            masks: masks tensor
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

        annotations = annotations["sample_annotations"]

        boxes = []
        labels = []
        masks = []

        for anno in annotations:
            labels.append(anno["category_id"])

            x, y, width, height = anno["bbox"]

            boxes.append([x, y, x + width, y + height])

            masks.append(polygon_to_mask(anno["mask"][0]))

        if len(boxes) == 0:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros(
                0, dtype=torch.int64
            )

        return (
            torch.tensor(np.array(boxes), dtype=torch.float32),
            torch.tensor(np.array(labels), dtype=torch.int64),
            torch.tensor(np.array(masks), dtype=torch.int64),
        )
