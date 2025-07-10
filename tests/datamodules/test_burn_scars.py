
# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Test Burn Scars Dataset."""

import os
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from pytest import MonkeyPatch
from torchgeo.datasets import DatasetNotFoundError

from geobench_v2.datamodules import GeoBenchBurnScarsDataModule
from geobench_v2.datasets.burn_scars import GeoBenchBurnScars
import pdb

@pytest.fixture
def band_order():
    """Test band configuration with fill values."""
    return ["red", "green", "blue"]


@pytest.fixture
def datamodule(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    band_order: dict[str, Sequence[str | float]],
):
    """Initialize Burn Scars datamodule with test configuration."""
    monkeypatch.setattr(GeoBenchBurnScars, "paths", ["burn_scars.tortilla"])
    monkeypatch.setattr(
        GeoBenchBurnScars, "url", os.path.join("tests", "data", "burn_scars", "{}")
    )
    monkeypatch.setattr(
        GeoBenchBurnScars,
        "sha256str",
        ["dd5c9aca65a65ef21325fa9de130a7dab9fda7b2ae9f669dd306d4571ddc7c20"],
    )
    dm = GeoBenchBurnScarsDataModule(
        img_size=512,
        batch_size=2,
        eval_batch_size=1,
        num_workers=0,
        pin_memory=False,
        band_order=band_order,
        root=tmp_path,
        download=True,
        metadata=["lon", "lat"],
    )
    dm.setup("fit")
    dm.setup("test")
    # pdb.set_trace()
    return dm


class TestBurnScarsDataModule:
    """Test cases for Burn Scars datamodule functionality."""

    def test_loaders(self, datamodule):
        """Test if dataloaders are created successfully."""
        assert len(datamodule.train_dataloader()) > 0
        assert len(datamodule.val_dataloader()) > 0
        assert len(datamodule.test_dataloader()) > 0

    def test_batch_dimensions(self, datamodule):
        """Test if batches have correct dimensions."""
        train_batch = next(iter(datamodule.train_dataloader()))
        assert train_batch["image"].shape[0] == datamodule.batch_size
        assert train_batch["image"].shape[1] == len(datamodule.band_order)
        assert train_batch["image"].shape[3] == datamodule.img_size

        assert train_batch["mask"].shape == (
            datamodule.batch_size,
            datamodule.img_size,
            datamodule.img_size,
        )

        assert "lon" in train_batch
        assert "lat" in train_batch
        assert train_batch["lon"].shape == (datamodule.batch_size,)
        assert train_batch["lat"].shape == (datamodule.batch_size,)

    def test_band_order_resolution(self, datamodule):
        """Test if band order is correctly resolved."""
        assert len(datamodule.band_order) == 3
        assert isinstance(datamodule.band_order[1], str)
        assert datamodule.band_order[1] == 'B03'

    def test_batch_visualization(self, datamodule):
        """Test batch visualization."""
        fig, batch = datamodule.visualize_batch(split="train")
        assert isinstance(fig, plt.Figure)
        assert isinstance(batch, dict)

        fig.savefig(os.path.join("tests", "data", "burn_scars", "test_batch.png"))

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match="Dataset not found"):
            GeoBenchBurnScars(tmp_path, split="train")
