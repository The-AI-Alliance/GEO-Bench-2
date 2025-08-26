# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""GeoBench DOTAV2 Tests."""

import os

import matplotlib.pyplot as plt
import pytest

from geobench_v2.datamodules import GeoBenchDOTAV2DataModule


@pytest.fixture
def data_root():
    """Path to test data directory."""
    return "/mnt/rg_climate_benchmark/data/geobenchV2/dotav2"


@pytest.fixture
def band_order():
    """Test band configuration with RGB and fill value."""
    return ["red", "green", "blue", 0, "green"]


@pytest.fixture
def datamodule(data_root, band_order):
    """Initialize DOTAV2 datamodule with test configuration."""
    dm = GeoBenchDOTAV2DataModule(
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


class TestDOTAV2DataModule:
    """Test cases for DOTAV2 datamodule functionality."""

    def test_batch_dimensions(self, datamodule):
        """Test if batches have correct dimensions."""
        datamodule.setup("fit")
        train_batch = next(iter(datamodule.train_dataloader()))
        assert train_batch["image"].shape[0] == datamodule.batch_size
        assert train_batch["image"].shape[1] == len(datamodule.band_order)

        # TODO resize augmentations do not work with oriented bboxes
        assert len(train_batch["bbox_xyxy"]) == datamodule.batch_size
        assert len(train_batch["label"]) == datamodule.batch_size

    def test_band_order_resolution(self, datamodule):
        """Test if band order is correctly resolved."""
        assert len(datamodule.band_order) == 5
        assert isinstance(datamodule.band_order[3], int)
        assert datamodule.band_order[3] == 0

    def test_batch_visualization(self, datamodule):
        """Test batch visualization."""
        fig, batch = datamodule.visualize_batch("train")
        assert isinstance(fig, plt.Figure)
        assert isinstance(batch, dict)

        fig.savefig(os.path.join("tests", "data", "dotav2", "test_batch.png"))
