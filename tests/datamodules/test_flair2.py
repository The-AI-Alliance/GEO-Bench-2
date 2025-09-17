# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""FLAIR2 Tests."""

import os
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
from pytest import MonkeyPatch
from torchgeo.datasets import DatasetNotFoundError

from geobench_v2.datamodules import GeoBenchFLAIR2DataModule
from geobench_v2.datasets import GeoBenchFLAIR2


@pytest.fixture(params=[{"aerial": ["r", 1.0, "g", "b"], "elevation": ["elevation"]}])
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
        ["581c7a853a2b2acdca2b9d55d4a49fa6668cdb25d9b72198dc0933eb100d90fd"],
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

        assert train_batch["image_aerial"].shape[0] == datamodule.batch_size
        assert train_batch["image_aerial"].shape[1] == len(
            datamodule.band_order["aerial"]
        )
        assert train_batch["image_aerial"].shape[2] == datamodule.img_size

        assert train_batch["mask"].shape[0] == datamodule.batch_size
        assert train_batch["mask"].shape[1] == datamodule.img_size

        assert torch.isclose(train_batch["image_aerial"][:, 1], torch.tensor(1.0)).all()

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
        plt.close(fig)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match="Dataset not found"):
            GeoBenchFLAIR2(tmp_path, split="train")
