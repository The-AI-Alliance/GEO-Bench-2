# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

import os

import matplotlib.pyplot as plt
import pytest

from geobench_v2.datamodules import GeoBenchWindTurbineDataModule


@pytest.fixture
def data_root():
    """Path to test data directory."""
    return "/mnt/rg_climate_benchmark/data/geobenchV2/wind_turbine"


@pytest.fixture
def band_order():
    """Test band configuration with RGB and fill value."""
    return ["red", "green", 0, "blue"]


@pytest.fixture
def datamodule(data_root, band_order):
    """Initialize WindTurbine datamodule with test configuration."""
    dm = GeoBenchWindTurbineDataModule(
        img_size=512,
        batch_size=8,
        eval_batch_size=4,
        num_workers=0,
        pin_memory=False,
        band_order=band_order,
        root=data_root,
    )
    dm.setup("fit")
    dm.setup("test")
    return dm


class TestWindTurbineDataModule:
    """Test cases for WindTurbine datamodule functionality."""

    def test_batch_dimensions(self, datamodule):
        """Test if batches have correct dimensions."""
        train_batch = next(iter(datamodule.train_dataloader()))
        assert train_batch["image"].shape[0] == datamodule.batch_size
        assert train_batch["image"].shape[1] == len(datamodule.band_order)
        assert train_batch["image"].shape[2] == datamodule.img_size
        assert train_batch["image"].shape[3] == datamodule.img_size

        assert "bbox_xyxy" in train_batch
        assert "label" in train_batch

    def test_band_order_resolution(self, datamodule):
        """Test if band order is correctly resolved."""
        assert len(datamodule.band_order) == 4
        assert datamodule.band_order[0] == "r"

    def test_batch_visualization(self, datamodule):
        """Test batch visualization."""
        fig, batch = datamodule.visualize_batch("train")
        assert isinstance(fig, plt.Figure)
        assert isinstance(batch, dict)

        fig.savefig(os.path.join("tests", "data", "windturbine", "test_batch.png"))
