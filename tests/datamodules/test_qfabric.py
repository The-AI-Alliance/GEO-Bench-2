# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

import pytest
import os
import matplotlib.pyplot as plt
from geobench_v2.datamodules import GeoBenchQFabricDataModule


"""Test QFabric."""


@pytest.fixture
def data_root():
    """Path to test data directory."""
    return "/mnt/rg_climate_benchmark/data/geobenchV2/qfabric"


@pytest.fixture
def band_order():
    """Test band configuration with RGBN bands."""
    return ["red", "green", "blue", 0.0]


@pytest.fixture
def datamodule(data_root, band_order):
    """Initialize QFabric datamodule with test configuration."""
    dm = GeoBenchQFabricDataModule(
        img_size=1024,
        batch_size=7,
        eval_batch_size=8,
        num_workers=0,
        pin_memory=False,
        band_order=band_order,
        root=data_root,
        time_steps=[0, 3, 2],
    )
    dm.setup("fit")
    dm.setup("test")
    return dm


class TestQFabricDataModule:
    """Test cases for QFabri datamodule functionality."""

    def test_batch_dimensions(self, datamodule):
        """Test if batches have correct dimensions."""
        train_batch = next(iter(datamodule.train_dataloader()))
        assert train_batch["image"].shape[0] == datamodule.batch_size
        assert train_batch["image"].shape[1] == 3  # timesteps
        assert train_batch["image"].shape[2] == len(datamodule.band_order)
        assert train_batch["image"].shape[3] == datamodule.img_size
        assert train_batch["image"].shape[4] == datamodule.img_size

    def test_band_order_resolution(self, datamodule):
        """Test if band order is correctly resolved."""
        assert len(datamodule.band_order) == 4
        assert isinstance(datamodule.band_order[3], float)
        assert datamodule.band_order[3] == 0.0

    def test_batch_visualization(self, datamodule):
        """Test batch visualization."""
        fig, batch = datamodule.visualize_batch("train")
        assert isinstance(fig, plt.Figure)
        assert isinstance(batch, dict)

        fig.savefig(os.path.join("tests", "data", "qfabric", "test_batch.png"))
