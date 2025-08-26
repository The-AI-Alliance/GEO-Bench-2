# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""BRIGHT Tests."""

import os
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch
from pytest import MonkeyPatch
from torchgeo.datasets import DatasetNotFoundError

from geobench_v2.datamodules import GeoBenchBRIGHTDataModule
from geobench_v2.datasets import GeoBenchBRIGHT


@pytest.fixture(params=[{"aerial": ["red", 0.0, "blue"], "sar": ["sar", 1.0]}])
def band_order(request):
    """Parameterized band configuration with different configurations."""
    return request.param


@pytest.fixture
def datamodule(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    band_order: dict[str, Sequence[str | float]],
):
    """Initialize BRIGHT datamodule with test configuration."""
    monkeypatch.setattr(GeoBenchBRIGHT, "paths", ["bright.tortilla"])
    monkeypatch.setattr(
        GeoBenchBRIGHT, "url", os.path.join("tests", "data", "bright", "{}")
    )
    monkeypatch.setattr(
        GeoBenchBRIGHT,
        "sha256str",
        ["c8dd9d7fb449311deb8595abd50cfa0a37f16d3a320c131aa91890596bfd2c30"],
    )
    dm = GeoBenchBRIGHTDataModule(
        img_size=74,
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

    return dm


class TestBRIGHTDataModule:
    """Test cases for BRIGHT datamodule functionality."""

    def test_loaders(self, datamodule):
        """Test if dataloaders are created successfully."""
        assert len(datamodule.train_dataloader()) > 0
        assert len(datamodule.val_dataloader()) > 0
        assert len(datamodule.test_dataloader()) > 0

    def test_load_batch_and_check_dims(self, datamodule):
        """Test loading a batch."""
        train_batch = next(iter(datamodule.train_dataloader()))
        assert isinstance(train_batch, dict)

        assert train_batch["image_pre"].shape[0] == datamodule.batch_size
        assert train_batch["image_pre"].shape[1] == len(datamodule.band_order["aerial"])
        assert train_batch["image_pre"].shape[2] == datamodule.img_size
        assert train_batch["image_pre"].shape[3] == 74
        assert torch.isclose(train_batch["image_pre"][:, 1], torch.tensor(0.0)).all()

        assert train_batch["image_post"].shape[0] == datamodule.batch_size
        assert train_batch["image_post"].shape[1] == len(datamodule.band_order["sar"])
        assert train_batch["image_post"].shape[2] == datamodule.img_size
        assert train_batch["image_post"].shape[3] == 74
        assert torch.isclose(train_batch["image_post"][:, 1], torch.tensor(1.0)).all()

        assert train_batch["mask"].shape[0] == datamodule.batch_size
        assert train_batch["mask"].shape[1] == 1
        assert train_batch["mask"].shape[2] == datamodule.img_size
        assert train_batch["mask"].shape[3] == 74

        assert "lon" in train_batch
        assert "lat" in train_batch
        assert train_batch["lon"].shape == (datamodule.batch_size,)
        assert train_batch["lat"].shape == (datamodule.batch_size,)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match="Dataset not found"):
            GeoBenchBRIGHT(tmp_path, split="train")
