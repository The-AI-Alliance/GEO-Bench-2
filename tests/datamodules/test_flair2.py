# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""FLAIR2 Tests."""

import pytest
import os
from pathlib import Path
from pytest import MonkeyPatch
from typing import Sequence
import torch
import matplotlib.pyplot as plt
from geobench_v2.datasets import GeoBenchFLAIR2
from geobench_v2.datamodules import GeoBenchFLAIR2DataModule
from torchgeo.datasets import DatasetNotFoundError


@pytest.fixture(params=[["r", 1.0, "g", "b"]])
def band_order(request):
    """Parameterized band configuration with different configurations."""
    return request.param


@pytest.fixture
def datamodule(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    band_order: dict[str, Sequence[str | float]],
):
    """Initialize FLAIR2 datamodule with test configuration."""
    monkeypatch.setattr(GeoBenchFLAIR2, "paths", ["flair2.tortilla"])
    monkeypatch.setattr(
        GeoBenchFLAIR2, "url", os.path.join("tests", "data", "flair2", "{}")
    )
    monkeypatch.setattr(
        GeoBenchFLAIR2,
        "sha256str",
        ["98347cdf1d9597b843e225392126b98aa381a2201e38d83fdb720ea2c44d8df9"],
    )
    dm = GeoBenchFLAIR2DataModule(
        img_size=256,
        batch_size=2,
        eval_batch_size=1,
        num_workers=0,
        pin_memory=False,
        band_order=band_order,
        root=tmp_path,
        metadata=["lon", "lat"],
        download=True,
    )
    dm.setup("fit")
    dm.setup("test")

    return dm


class TestFlAIR2DataModule:
    """Test cases for FLAIR2 datamodule functionality."""

    def test_loaders(self, datamodule):
        """Test if dataloaders are created successfully."""
        assert len(datamodule.train_dataloader()) > 0
        assert len(datamodule.val_dataloader()) > 0
        assert len(datamodule.test_dataloader()) > 0

    def test_load_batch_and_check_dims(self, datamodule):
        """Test loading a batch."""
        train_batch = next(iter(datamodule.train_dataloader()))
        assert isinstance(train_batch, dict)

        assert train_batch["image"].shape[0] == datamodule.batch_size
        assert train_batch["image"].shape[1] == len(datamodule.band_order)
        assert train_batch["image"].shape[2] == datamodule.img_size

        assert train_batch["mask"].shape[0] == datamodule.batch_size
        assert train_batch["mask"].shape[1] == datamodule.img_size

        assert torch.isclose(train_batch["image"][:, 1], torch.tensor(1.0)).all()

        assert "lon" in train_batch
        assert "lat" in train_batch
        assert train_batch["lon"].shape == (datamodule.batch_size,)
        assert train_batch["lat"].shape == (datamodule.batch_size,)

    def test_batch_visualization(self, datamodule):
        """Test batch visualization."""
        fig, batch = datamodule.visualize_batch("train")
        assert isinstance(fig, plt.Figure)
        assert isinstance(batch, dict)

        fig.savefig(os.path.join("tests", "data", "flair2", "test_batch.png"))

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match="Dataset not found"):
            GeoBenchFLAIR2(tmp_path, split="train")
