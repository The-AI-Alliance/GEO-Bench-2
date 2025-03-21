# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.


import pytest
from geobench_v2.datamodules import GeoBenchSpaceNet8DataModule


@pytest.fixture
def data_root():
    """Path to test data directory."""
    return "/mnt/rg_climate_benchmark/data/geobenchV2/spacenet8"


@pytest.fixture
def band_order():
    """Test band configuration with fill values."""
    return ["red", 0.0, "blue"]


@pytest.fixture
def datamodule(data_root, band_order):
    """Initialize SpaceNet8 datamodule with test configuration."""
    return GeoBenchSpaceNet8DataModule(
        img_size=74,
        batch_size=32,
        eval_batch_size=64,
        num_workers=0,
        pin_memory=False,
        band_order=band_order,
        root=data_root,
    )


class TestSpaceNet8DataModule:
    """Test cases for SpaceNet8 datamodule functionality."""

    def test_batch_dimensions(self, datamodule):
        """Test if batches have correct dimensions."""
        datamodule.setup("fit")
        train_batch = next(iter(datamodule.train_dataloader()))
        assert train_batch["image_pre"].shape[0] == datamodule.batch_size
        assert train_batch["image_pre"].shape[1] == len(datamodule.band_order)
        assert train_batch["image_pre"].shape[2] == 74
        assert train_batch["image_pre"].shape[3] == datamodule.img_size

        assert train_batch["mask"].shape[0] == datamodule.batch_size
        assert train_batch["mask"].shape[2] == 74
        assert train_batch["mask"].shape[3] == datamodule.img_size

    def test_band_order_resolution(self, datamodule):
        """Test if band order is correctly resolved."""
        assert len(datamodule.band_order) == 3
        assert isinstance(datamodule.band_order[1], float)
        assert datamodule.band_order[1] == 0.0