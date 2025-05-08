# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

import pytest

from geobench_v2.datamodules import GeoBenchSen4AgriNetDataModule


@pytest.fixture
def data_root():
    """Path to test data directory."""
    return "/mnt/rg_climate_benchmark/data/geobenchV2/sen4agrinet"


@pytest.fixture
def band_order():
    """Test band configuration with RGB bands."""
    return ["red", "green", 1.0, "blue", "B07"]


@pytest.fixture
def datamodule(data_root, band_order):
    """Initialize Sen4AgriNet datamodule with test configuration."""
    return GeoBenchSen4AgriNetDataModule(
        img_size=74,
        batch_size=4,
        eval_batch_size=4,
        num_workers=0,
        pin_memory=False,
        band_order=band_order,
        root=data_root,
        num_time_steps=3,
    )


class TestSen4AgriNetDataModule:
    """Test cases for Sen4AgriNet datamodule functionality."""

    def test_batch_dimensions(self, datamodule):
        """Test if batches have correct dimensions."""
        datamodule.setup("fit")
        train_batch = next(iter(datamodule.train_dataloader()))

        assert train_batch["image"].shape[0] == datamodule.batch_size
        assert train_batch["image"].shape[1] == datamodule.train_dataset.num_time_steps
        assert train_batch["image"].shape[2] == len(datamodule.band_order)
        assert train_batch["image"].shape[3] == 74
        assert train_batch["image"].shape[4] == datamodule.img_size

    def test_band_order_resolution(self, datamodule):
        """Test if band order is correctly resolved."""
        assert len(datamodule.band_order) == 5
        assert isinstance(datamodule.band_order[2], float)
        assert datamodule.band_order[2] == 1.0
