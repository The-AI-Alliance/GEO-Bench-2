# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

import os
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from pytest import MonkeyPatch

from geobench_v2.datamodules import GeoBenchM4SARDataModule
from geobench_v2.datasets import GeoBenchM4SAR


@pytest.fixture(params=[{"optical": ("red", "green", 0, "blue"), "sar": ("VV", "VH")}])
def band_order(request):
    """Parameterized band configuration with different configurations."""
    return request.param


@pytest.fixture
def datamodule(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    band_order: dict[str, Sequence[str | float]],
):
    """Initialize M4SAR datamodule with test configuration."""
    monkeypatch.setattr(GeoBenchM4SAR, "paths", ["m4sar.tortilla"])
    monkeypatch.setattr(
        GeoBenchM4SAR, "url", os.path.join("tests", "data", "m4sar", "{}")
    )
    monkeypatch.setattr(
        GeoBenchM4SAR,
        "sha256str",
        ["0617c9fe1de89a216a08ef41d42bed1e6a31dd214711eb38c0e357c0ff8bd60e"],
    )
    dm = GeoBenchM4SARDataModule(
        img_size=512,
        batch_size=2,
        eval_batch_size=1,
        num_workers=0,
        pin_memory=False,
        band_order=band_order,
        root=tmp_path,
        bbox_orientation="horizontal",
        download=True,
    )
    dm.setup("fit")
    dm.setup("test")

    return dm


class TestM4SARDataModule:
    """Test cases for M4SAR datamodule functionality."""

    def test_batch_dimensions(self, datamodule):
        """Test if batches have correct dimensions."""
        train_batch = next(iter(datamodule.train_dataloader()))
        assert train_batch["image_optical"].shape[0] == datamodule.batch_size
        assert train_batch["image_optical"].shape[1] == len(
            datamodule.band_order["optical"]
        )
        assert train_batch["image_sar"].shape[2] == datamodule.img_size
        assert train_batch["image_sar"].shape[3] == datamodule.img_size

        assert "bbox_xyxy" in train_batch
        assert "label" in train_batch

    def test_band_order_resolution(self, datamodule):
        """Test if band order is correctly resolved."""
        assert len(datamodule.band_order["optical"]) == 4
        assert datamodule.band_order["optical"][0] == "red"

    def test_batch_visualization(self, datamodule):
        """Test batch visualization."""
        fig, batch = datamodule.visualize_batch("train")
        assert isinstance(fig, plt.Figure)
        assert isinstance(batch, dict)

        fig.savefig(os.path.join("tests", "data", "m4sar", "test_batch.png"))
