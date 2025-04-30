# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""MADOS Tests."""

import pytest
import os
from pytest import MonkeyPatch
from typing import Sequence
import torch
from pathlib import Path
from torchgeo.datasets import DatasetNotFoundError
import matplotlib.pyplot as plt
from geobench_v2.datasets import GeoBenchMADOS
from geobench_v2.datamodules import GeoBenchMADOSDataModule


@pytest.fixture(params=[["red", "blue", "green", "B07", 0.0]])
def band_order(request):
    """Parameterized band configuration with different configurations."""
    return request.param


@pytest.fixture
def datamodule(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    band_order: dict[str, Sequence[str | float]],
):
    """Initialize MADOS datamodule with test configuration."""
    monkeypatch.setattr(GeoBenchMADOS, "paths", ["mados.tortilla"])
    monkeypatch.setattr(
        GeoBenchMADOS, "url", os.path.join("tests", "data", "mados", "{}")
    )
    monkeypatch.setattr(
        GeoBenchMADOS,
        "sha256str",
        ["b21daa927a7cdbba96394465f8973c1b5e0757f0956f8767db3f2f26e28427b5"],
    )
    dm = GeoBenchMADOSDataModule(
        img_size=224,
        batch_size=4,
        eval_batch_size=2,
        num_workers=0,
        pin_memory=False,
        band_order=band_order,
        root=tmp_path,
        download=True,
    )
    dm.setup("fit")
    dm.setup("test")

    return dm


class TestMADOSDataModule:
    """Test cases for MADOS datamodule functionality."""

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
        assert train_batch["image"].shape[3] == datamodule.img_size

        assert train_batch["mask"].shape[0] == datamodule.batch_size
        assert train_batch["mask"].shape[1] == datamodule.img_size
        assert train_batch["mask"].shape[2] == datamodule.img_size

        assert torch.isclose(train_batch["image"][:, 4], torch.tensor(0.0)).all()

    def test_batch_visualization(self, datamodule):
        """Test batch visualization."""
        fig, batch = datamodule.visualize_batch("train")
        assert isinstance(fig, plt.Figure)
        assert isinstance(batch, dict)

        fig.savefig(os.path.join("tests", "data", "mados", "test_batch.png"))

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match="Dataset not found"):
            GeoBenchMADOS(tmp_path, split="train")
