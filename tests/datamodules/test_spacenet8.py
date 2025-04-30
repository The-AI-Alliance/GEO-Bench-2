# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""SpaceNet8 Tests."""

import pytest
import os
from pytest import MonkeyPatch
from typing import Sequence
import torch
from pathlib import Path
from torchgeo.datasets import DatasetNotFoundError
import matplotlib.pyplot as plt
from geobench_v2.datasets import GeoBenchSpaceNet8
from geobench_v2.datamodules import GeoBenchSpaceNet8DataModule


@pytest.fixture(params=[["red", 0.0, "blue", "g"]])
def band_order(request):
    """Parameterized band configuration with different configurations."""
    return request.param


@pytest.fixture
def datamodule(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    band_order: dict[str, Sequence[str | float]],
):
    """Initialize SpaceNet8 datamodule with test configuration."""
    monkeypatch.setattr(GeoBenchSpaceNet8, "paths", ["spacenet8.tortilla"])
    monkeypatch.setattr(
        GeoBenchSpaceNet8, "url", os.path.join("tests", "data", "spacenet8", "{}")
    )
    monkeypatch.setattr(
        GeoBenchSpaceNet8,
        "sha256str",
        ["084714af0af5e087e583f27605ea83cf6955637f936697c13c4484e028de91fa"],
    )
    dm = GeoBenchSpaceNet8DataModule(
        img_size=256,
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


class TestSpaceNet8DataModule:
    """Test cases for SpaceNet 8 datamodule functionality."""

    def test_loaders(self, datamodule):
        """Test if dataloaders are created successfully."""
        assert len(datamodule.train_dataloader()) > 0
        assert len(datamodule.val_dataloader()) > 0
        assert len(datamodule.test_dataloader()) > 0

    def test_load_batch_and_check_dims(self, datamodule):
        """Test loading a batch."""
        train_batch = next(iter(datamodule.train_dataloader()))
        assert isinstance(train_batch, dict)
        for key in ["image_pre", "image_post"]:
            assert train_batch[key].shape[0] == datamodule.batch_size
            assert train_batch[key].shape[1] == len(datamodule.band_order)
            assert train_batch[key].shape[2] == datamodule.img_size
            assert train_batch[key].shape[3] == datamodule.img_size
            assert torch.isclose(train_batch[key][:, 1], torch.tensor(0.0)).all()

        assert train_batch["mask"].shape[0] == datamodule.batch_size
        assert train_batch["mask"].shape[1] == datamodule.img_size
        assert train_batch["mask"].shape[2] == datamodule.img_size

        assert "lon" in train_batch
        assert "lat" in train_batch
        assert train_batch["lon"].shape == (datamodule.batch_size,)
        assert train_batch["lat"].shape == (datamodule.batch_size,)

    def test_batch_visualization(self, datamodule):
        """Test batch visualization."""
        fig, batch = datamodule.visualize_batch("train")
        assert isinstance(fig, plt.Figure)
        assert isinstance(batch, dict)

        fig.savefig(os.path.join("tests", "data", "spacenet8", "test_batch.png"))

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match="Dataset not found"):
            GeoBenchSpaceNet8(tmp_path, split="train")
