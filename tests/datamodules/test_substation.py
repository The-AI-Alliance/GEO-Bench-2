# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

import os
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from pytest import MonkeyPatch

from geobench_v2.datamodules import GeoBenchSubstationDataModule
from geobench_v2.datasets import GeoBenchSubstation


@pytest.fixture(params=[["B02", "B03", "B11", "B08", "B04"]])
def band_order(request):
    """Parameterized band configuration with different configurations."""
    return request.param


@pytest.fixture
def datamodule(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    band_order: dict[str, Sequence[str | float]],
):
    """Initialize Substation datamodule with test configuration."""
    monkeypatch.setattr(GeoBenchSubstation, "paths", ["substation.tortilla"])
    monkeypatch.setattr(
        GeoBenchSubstation, "url", os.path.join("tests", "data", "substation", "{}")
    )
    monkeypatch.setattr(
        GeoBenchSubstation,
        "sha256str",
        ["bd8eaa5f156279529c481e70699ba92e6905862d3a5c4c2cf54a1b5d361e2c89"],
    )

    dm = GeoBenchSubstationDataModule(
        img_size=228,
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


class TestSubstationDataModule:
    """Test cases for Substation datamodule functionality."""

    def test_batch_dimensions(self, datamodule):
        """Test if batches have correct dimensions."""
        train_batch = next(iter(datamodule.train_dataloader()))
        assert train_batch["image"].shape[0] == datamodule.batch_size
        assert train_batch["image"].shape[1] == len(datamodule.band_order)
        assert train_batch["image"].shape[2] == datamodule.img_size
        assert train_batch["image"].shape[3] == datamodule.img_size

        assert "bbox_xyxy" in train_batch
        assert "label" in train_batch
        assert "mask" in train_batch

    def test_band_order_resolution(self, datamodule):
        """Test if band order is correctly resolved."""
        assert len(datamodule.band_order) == 5
        assert datamodule.band_order[0] == "B02"

    def test_batch_visualization(self, datamodule):
        """Test batch visualization."""
        fig, batch = datamodule.visualize_batch("train")
        assert isinstance(fig, plt.Figure)
        assert isinstance(batch, dict)

        fig.savefig(os.path.join("tests", "data", "substation", "test_batch.png"))
        plt.close(fig)
