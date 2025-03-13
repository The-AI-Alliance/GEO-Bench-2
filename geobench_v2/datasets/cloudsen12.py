# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Cloud12Sen Dataset."""

import os
import numpy as np
import rasterio


from typing import Sequence, ClassVar, Union
import torch
from torch import Tensor
from torchgeo.datasets import NonGeoDataset

from .sensor_util import DatasetBandRegistry
from .data_util import DataUtilsMixin, MultiModalNormalizer
import tacoreader
import numpy as np


class GeoBenchCloudSen12(NonGeoDataset, DataUtilsMixin):
    """Implementation of CloudSen12 dataset Sentinel 2 L1C.


    CloudSen12 is a dataset for cloud segmentation that provides humanly annotated Sentinel-2 L1C imagery.

    The dataset contains four semantic segmentation classes:

    0. clear: Pixels without cloud and cloud shadow contamination.
    1. thick cloud: Opaque clouds that block all reflected light from Earth's surface.
    2. thin cloud: Semitransparent clouds that alter the surface spectral signal but still allow recognition of the background.
    3. cloud shadow: Dark pixels where light is occluded by thick or thin clouds.

    If you use this dataset in your research, please cite the following paper:

    * link
    """

    classes = ("clear", "thick cloud", "thin cloud", "cloud shadow")

    splits = ("train", "val", "test")

    dataset_band_config = DatasetBandRegistry.CLOUDSEN12

    normalization_stats = {
        "means": {
            "B01": 0.0,
            "B02": 0.0,
            "B03": 0.0,
            "B04": 0.0,
            "B05": 0.0,
            "B06": 0.0,
            "B07": 0.0,
            "B08": 0.0,
            "B8A": 0.0,
            "B09": 0.0,
            "B11": 0.0,
            "B12": 0.0,
        },
        "stds": {
            "B01": 1.0,
            "B02": 1.0,
            "B03": 1.0,
            "B04": 1.0,
            "B05": 1.0,
            "B06": 1.0,
            "B07": 1.0,
            "B08": 1.0,
            "B8A": 1.0,
            "B09": 1.0,
            "B11": 1.0,
            "B12": 1.0,
        },
    }

    classes = ("clear", "thick cloud", "thin cloud", "cloud shadow")

    # taco_files = [
    #     "cloudsen12-l1c.0000.part.taco",
    #     "cloudsen12-l1c.0001.part.taco",
    #     "cloudsen12-l1c.0002.part.taco",
    #     "cloudsen12-l1c.0003.part.taco",
    #     "cloudsen12-l1c.0004.part.taco",
    # ]

    taco_name = "geobench_cloudsen12.taco"

    def __init__(
        self,
        root,
        split="train",
        band_order: Sequence[float | str] = ["B04", "B03", "B02"],
        transforms=None,
    ) -> None:
        """Initialize a CloudSen12 dataset instance.

        Args:
            root: Path to the dataset root directory
            split: The dataset split, supports 'train', 'test'
            band_order: The order of bands to return, defaults to ['r', 'g', 'b'], if one would
                specify ['r', 'g', 'b', 'nir'], the dataset would return images with 4 channels
            transforms: A composition of transforms to apply to the sample_row

        Raises:
            AssertionError: If split is not in the splits
        """

        assert split in self.splits, f"split must be one of {self.splits}"

        self.root = root
        self.split = split
        self.transforms = transforms

        self.band_order = self.resolve_band_order(band_order)

        self.normalizer = MultiModalNormalizer(
            self.normalization_stats, self.band_order
        )

        self.metadata_df = tacoreader.load(self.taco_name)
        self.metadata_df = self.metadata_df[self.metadata_df["tortilla:data_split"] == split].reset_index(drop=True)


    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Return the sample_row at the given index.

        Args:
            idx: Index of the sample_row to return

        Returns:
            dict containing the sample_row data
        """
        sample: dict[str, Tensor] = {}
        sample_row = self.metadata_df.read(idx)

        image_path: str = sample_row.read(0)
        target_path: str = sample_row.read(1)

        with (
            rasterio.open(image_path) as image_src,
            rasterio.open(target_path) as target_src,
        ):
            # TODO check
            image_data: np.ndarray = image_src.read(out_dtype="float32")
            target_data: np.ndarray = target_src.read()

        image = torch.from_numpy(image_data).float()
        mask = torch.from_numpy(target_data).long()

        image_dict = self.rearrange_bands(image, self.band_order)

        image = self.normalizer(image_dict)

        sample.update(image_dict)
        sample.update({"mask": mask})

        return sample

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            The number of samples in the dataset
        """
        return len(self.metadata_df)
