# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""EverWatch Tests."""

import os
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from pytest import MonkeyPatch

from geobench_v2.datamodules import GeoBenchEverWatchDataModule
from geobench_v2.datasets import GeoBenchEverWatch


@pytest.fixture(params=[["red", "green", "blue", 0, "green"]])
def band_order(request):
    """Parameterized band configuration with different configurations."""
    return request.param


@pytest.fixture
def datamodule(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    band_order: dict[str, Sequence[str | float]],
):
    """Initialize EverWatch datamodule with test configuration."""
    monkeypatch.setattr(GeoBenchEverWatch, "paths", ["everwatch.tortilla"])
    monkeypatch.setattr(
        GeoBenchEverWatch, "url", os.path.join("tests", "data", "everwatch", "{}")
    )
    monkeypatch.setattr(
        GeoBenchEverWatch,
        "sha256str",
        ["24f32265b005b047caa5046e70e2c5a1b8e76b0234a30b5736545a342449749b"],
    )
    dm = GeoBenchEverWatchDataModule(
        img_size=256,
        batch_size=2,
        eval_batch_size=1,
        num_workers=0,
        pin_memory=False,
        band_order=band_order,
        root=tmp_path,
        download=True,
    )
    dm.setup("fit")
    dm.setup("test")

    return dm


class TestEverWatchDataModule:
    """Test cases for EverWatch datamodule functionality."""

    def test_batch_dimensions(self, datamodule):
        """Test if batches have correct dimensions."""
        datamodule.setup("fit")
        train_batch = next(iter(datamodule.train_dataloader()))
        assert train_batch["image"].shape[0] == datamodule.batch_size
        assert train_batch["image"].shape[1] == len(datamodule.band_order)
        assert train_batch["image"].shape[2] == datamodule.img_size
        assert train_batch["image"].shape[3] == datamodule.img_size

    def test_band_order_resolution(self, datamodule):
        """Test if band order is correctly resolved."""
        assert len(datamodule.band_order) == 5
        assert isinstance(datamodule.band_order[3], int)
        assert datamodule.band_order[3] == 0
    
    def test_torchvision_compatibility(self, datamodule):
        """Test if torchvision compatibility increases label value by 1"""
        old_sample = datamodule.train_dataset[0]
        datamodule.train_dataset.torchvision_detection_compatible = True
        new_sample = datamodule.train_dataset[0]
        assert old_sample["label"][0]+1 == new_sample["label"][0]
        
        

    def test_batch_visualization(self, datamodule):
        """Test batch visualization."""
        fig, batch = datamodule.visualize_batch("train")
        assert isinstance(fig, plt.Figure)
        assert isinstance(batch, dict)

        fig.savefig(os.path.join("tests", "data", "everwatch", "test_batch.png"))
