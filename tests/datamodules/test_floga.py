# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""GeoBench FLOGA Tests."""

import pytest
from geobench_v2.datamodules import GeoBenchFLOGADataModule


@pytest.fixture
def data_root():
    """Path to test data directory."""
    return "/mnt/rg_climate_benchmark/data/datasets_segmentation/floga/dataset"


@pytest.fixture
def band_order():
    """Test band configuration with bands."""
    return {"s2": ["B02", "B08", "B02", 0.0], "modis": ["M01", "M02", 0.0]}


@pytest.fixture
def datamodule(data_root, band_order):
    """Initialize FLOGA datamodule with test configuration."""
    return GeoBenchFLOGADataModule(
        img_size=256,
        batch_size=32,
        eval_batch_size=64,
        num_workers=0,
        pin_memory=False,
        band_order=band_order,
        root=data_root,
    )


class TestFLOGADataModule:
    """Test cases for FLOGA datamodule functionality."""

    def test_batch_dimensions(self, datamodule):
        """Test if batches have correct dimensions."""
        datamodule.setup("fit")
        train_batch = next(iter(datamodule.train_dataloader()))
        assert train_batch["image_s2_pre"].shape[0] == datamodule.batch_size
        assert train_batch["image_s2_pre"].shape[1] == len(datamodule.band_order["s2"])

    def test_band_order_resolution(self, datamodule):
        """Test if band order is correctly resolved."""
        assert len(datamodule.band_order["s2"]) == 4
        assert len(datamodule.band_order["modis"]) == 3
        assert datamodule.band_order["s2"][3] == 0.0
        assert datamodule.band_order["modis"][2] == 0.0
