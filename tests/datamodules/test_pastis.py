# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""PASTIS DataModule Tests."""

import os
from collections.abc import Sequence

import matplotlib.pyplot as plt
import pytest
from pytest import MonkeyPatch
from pathlib import Path

from torchgeo.datasets import DatasetNotFoundError

from geobench_v2.datamodules import GeoBenchPASTISDataModule
from geobench_v2.datasets import GeoBenchPASTIS


@pytest.fixture(
    params=[
        {
            "s2": ["B02", "B03", 0.0, "B08", "B04"],
            "s1_asc": ["VV_asc", "VH_asc", 1.0],
            "s1_desc": ["VV_desc", "VH_desc", -1.0],
        }
    ]
)
def band_order(request):
    """Parameterized band configuration with different configurations."""
    return request.param


@pytest.fixture
def invalid_mixed_band_order():
    """Test invalid band configuration with mixed modality bands."""
    return ["B02", "VV_asc", "B08"]  # Mixes S2 and S1 bands


@pytest.fixture
def datamodule(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    band_order: dict[str, Sequence[str | float]],
):
    monkeypatch.setattr(GeoBenchPASTIS, "paths", ["pastis.tortilla"])
    monkeypatch.setattr(
        GeoBenchPASTIS, "url", os.path.join("tests", "data", "pastis", "{}")
    )
    monkeypatch.setattr(
        GeoBenchPASTIS,
        "sha256str",
        ["8ec713be2d99fe2785d902545642346759466fe4bfd85d7e45fe0cbb55f0a882"],
    )
    datamodule = GeoBenchPASTISDataModule(
        img_size=74,
        batch_size=2,
        eval_batch_size=1,
        num_workers=0,
        band_order=band_order,
        root=tmp_path,
        num_time_steps=2,
        metadata=["lon", "lat"],
        pin_memory=False,
        download=True,
    )
    datamodule.setup("fit")
    datamodule.setup("test")
    return datamodule


class TestPASTISDataModule:
    """Test cases for PASTIS datamodule functionality."""

    def test_loaders(self, datamodule):
        """Test if dataloaders are created successfully."""
        assert len(datamodule.train_dataloader()) > 0
        assert len(datamodule.val_dataloader()) > 0
        assert len(datamodule.test_dataloader()) > 0
        assert len(datamodule.extra_test_dataloader()) > 0

    def test_multimodal_band_order(self, datamodule):
        """Test batch retrieval with modality-specific band sequences."""

        batch = next(iter(datamodule.train_dataloader()))

        # Check modality-specific outputs
        assert "image_s2" in batch
        assert "image_s1_asc" in batch
        assert "image_s1_desc" in batch
        assert "mask" in batch

        # TODO handle correctly both cases
        # Check dimensions - these are time-series shapes [batch_size, T, C, H, W]
        assert batch["image_s2"].shape[0] == datamodule.batch_size
        assert batch["image_s2"].shape[1] == 2  # num time steps
        assert batch["image_s2"].shape[2] == len(datamodule.band_order["s2"])
        assert batch["image_s2"].shape[3] == datamodule.img_size
        assert batch["image_s2"].shape[4] == datamodule.img_size

        assert batch["image_s1_asc"].shape[0] == datamodule.batch_size
        assert batch["image_s1_asc"].shape[1] == 2
        assert batch["image_s1_asc"].shape[2] == len(datamodule.band_order["s1_asc"])
        assert batch["image_s1_asc"].shape[3] == datamodule.img_size
        assert batch["image_s1_asc"].shape[4] == datamodule.img_size

        assert batch["image_s1_desc"].shape[0] == datamodule.batch_size
        assert batch["image_s1_desc"].shape[1] == 2
        assert batch["image_s1_desc"].shape[2] == len(datamodule.band_order["s1_desc"])
        assert batch["image_s1_desc"].shape[3] == datamodule.img_size
        assert batch["image_s1_desc"].shape[4] == datamodule.img_size

        assert "lon" in batch
        assert "lat" in batch
        assert batch["lon"].shape == (datamodule.batch_size,)
        assert batch["lat"].shape == (datamodule.batch_size,)

    def test_batch_visualization(self, datamodule):
        """Test batch visualization."""
        fig, batch = datamodule.visualize_batch("train")
        assert isinstance(fig, plt.Figure)
        assert isinstance(batch, dict)

        fig.savefig(os.path.join("tests", "data", "pastis", "test_batch.png"))

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match="Dataset not found"):
            GeoBenchPASTIS(tmp_path, split="train")
