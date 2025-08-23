# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""CloudSen12 Tests."""

import os
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch
from pytest import MonkeyPatch
from torchgeo.datasets import DatasetNotFoundError

from geobench_v2.datamodules import GeoBenchCloudSen12DataModule
from geobench_v2.datasets import GeoBenchCloudSen12


@pytest.fixture(params=[["B01", "B02", "B08", "B02", 0.0]])
def band_order(request):
    """Parameterized band configuration with different configurations."""
    return request.param


@pytest.fixture
def datamodule(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    band_order: dict[str, Sequence[str | float]],
):
    """Initialize CloudSen12 datamodule with test configuration."""
    monkeypatch.setattr(GeoBenchCloudSen12, "paths", ["cloudsen12.tortilla"])
    monkeypatch.setattr(
        GeoBenchCloudSen12, "url", os.path.join("tests", "data", "cloudsen12", "{}")
    )
    monkeypatch.setattr(
        GeoBenchCloudSen12,
        "sha256str",
        ["be92ca1e65987bb7a35193a7fad4b5d8e45b7d7005a4e963d1fea1ded2b1d6fc"],
    )
    dm = GeoBenchCloudSen12DataModule(
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


class TestCloudSen12DataModule:
    """Test cases for CloudSen12 datamodule functionality."""

    def test_loaders(self, datamodule):
        """Test if dataloaders are created successfully."""
        assert len(datamodule.train_dataloader()) > 0
        assert len(datamodule.val_dataloader()) > 0
        assert len(datamodule.test_dataloader()) > 0
        assert len(datamodule.extra_test_dataloader()) > 0

    def test_load_batch_and_check_dims(self, datamodule):
        """Test loading a batch."""
        train_batch = next(iter(datamodule.train_dataloader()))
        assert isinstance(train_batch, dict)

        assert train_batch["image"].shape[0] == datamodule.batch_size
        assert train_batch["image"].shape[1] == len(datamodule.band_order)
        assert train_batch["image"].shape[2] == datamodule.img_size
        assert train_batch["image"].shape[3] == 74

        assert train_batch["mask"].shape[0] == datamodule.batch_size
        assert train_batch["mask"].shape[1] == 1
        assert train_batch["mask"].shape[2] == datamodule.img_size
        assert train_batch["mask"].shape[3] == 74

        assert torch.isclose(train_batch["image"][:, 4], torch.tensor(0.0)).all()

        assert "lon" in train_batch
        assert "lat" in train_batch
        assert train_batch["lon"].shape == (datamodule.batch_size,)
        assert train_batch["lat"].shape == (datamodule.batch_size,)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match="Dataset not found"):
            GeoBenchCloudSen12(tmp_path, split="train")
