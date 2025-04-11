# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""FieldsOfTheWorld Tests."""

import pytest
import os
from pytest import MonkeyPatch
from typing import Sequence
import torch
from geobench_v2.datasets import GeoBenchFieldsOfTheWorld
from geobench_v2.datamodules import GeoBenchFieldsOfTheWorldDataModule


@pytest.fixture(params=[["red", "blue", "nir", 0.0]])
def band_order(request):
    """Parameterized band configuration with different configurations."""
    return request.param


@pytest.fixture
def datamodule(monkeypatch: MonkeyPatch, band_order: dict[str, Sequence[str | float]]):
    """Initialize FOTW datamodule with test configuration."""
    monkeypatch.setattr(GeoBenchFieldsOfTheWorld, "paths", ["fotw.tortilla"])
    dm = GeoBenchFieldsOfTheWorldDataModule(
        img_size=74,
        batch_size=4,
        eval_batch_size=2,
        num_workers=0,
        pin_memory=False,
        band_order=band_order,
        root=os.path.join("tests", "data", "fotw"),
    )
    dm.setup("fit")
    dm.setup("test")

    return dm


class TestFieldsOfTheWorldDataModule:
    """Test cases for FOTW datamodule functionality."""

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
        assert train_batch["image"].shape[3] == 74

        assert train_batch["mask"].shape[0] == datamodule.batch_size
        assert train_batch["mask"].shape[1] == datamodule.img_size
        assert train_batch["mask"].shape[2] == 74

        assert torch.isclose(train_batch["image"][:, 3], torch.tensor(0.0)).all()

        assert "lon" in train_batch
        assert "lat" in train_batch
        assert train_batch["lon"].shape == (datamodule.batch_size,)
        assert train_batch["lat"].shape == (datamodule.batch_size,)
