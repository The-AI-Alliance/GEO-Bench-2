# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Cloud12Sen Tests."""

import pytest
from geobench_v2.datamodules import GeoBenchCloudSen12DataModule


@pytest.fixture
def data_root():
    """Path to test data directory."""
    return "/mnt/rg_climate_benchmark/data/datasets_segmentation/cloudsen12"


@pytest.fixture
def band_order():
    """Test band configuration with RGBN bands."""
    return ["B01", "B02", "B08", "B02", 0.0]


@pytest.fixture
def datamodule(data_root, band_order):
    """Initialize CloudSen12 datamodule with test configuration."""
    return GeoBenchCloudSen12DataModule(
        img_size=74,
        batch_size=32,
        eval_batch_size=64,
        num_workers=0,
        pin_memory=False,
        band_order=band_order,
        root=data_root,
    )


class TestCloudSen122DataModule:
    """Test cases for CloudSen12 datamodule functionality."""

    def test_batch_dimensions(self, datamodule):
        """Test if batches have correct dimensions."""
        datamodule.setup("fit")
        train_batch = next(iter(datamodule.train_dataloader()))
        assert train_batch["image"].shape[0] == datamodule.batch_size
        assert train_batch["image"].shape[1] == len(datamodule.band_order)
        assert train_batch["image"].shape[2] == datamodule.img_size
        assert train_batch["image"].shape[3] == 74

    def test_band_order_resolution(self, datamodule):
        """Test if band order is correctly resolved."""
        assert len(datamodule.band_order) == 5
        assert isinstance(datamodule.band_order[4], float)
        assert datamodule.band_order[4] == 0.0
