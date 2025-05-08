# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Test CaFFe Dataset."""

import os
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from pytest import MonkeyPatch
from torchgeo.datasets import DatasetNotFoundError

from geobench_v2.datamodules import GeoBenchCaFFeDataModule
from geobench_v2.datasets.caffe import GeoBenchCaFFe


@pytest.fixture
def band_order():
    """Test band configuration with fill values."""
    return ["gray", 0.0, "gray"]


@pytest.fixture
def datamodule(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    band_order: dict[str, Sequence[str | float]],
):
    """Initialize CaFFe datamodule with test configuration."""
    monkeypatch.setattr(GeoBenchCaFFe, "paths", ["caffe.tortilla"])
    monkeypatch.setattr(
        GeoBenchCaFFe, "url", os.path.join("tests", "data", "benv2", "{}")
    )
    monkeypatch.setattr(
        GeoBenchCaFFe,
        "sha256str",
        ["94a1bf150f7a25df6acd16c7f46ddc9b0b0e4d581e40fd282e22853115e26023"],
    )
    dm = GeoBenchCaFFeDataModule(
        img_size=512,
        batch_size=4,
        eval_batch_size=2,
        num_workers=0,
        pin_memory=False,
        band_order=band_order,
        root=tmp_path,
        download=True,
        metadata=["lon", "lat"],
    )
    dm.setup("fit")
    dm.setup("test")

    return dm


class TestCaFFeDataModule:
    """Test cases for CaFFe datamodule functionality."""

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
        assert isinstance(datamodule.band_order[1], float)
        assert datamodule.band_order[1] == 0.0

    def test_batch_visualization(self, datamodule):
        """Test batch visualization."""
        fig, batch = datamodule.visualize_batch("train")
        assert isinstance(fig, plt.Figure)
        assert isinstance(batch, dict)

        fig.savefig(os.path.join("tests", "data", "caffe", "test_batch.png"))

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match="Dataset not found"):
            GeoBenchCaFFe(tmp_path, split="train")
