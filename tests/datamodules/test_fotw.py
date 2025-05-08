# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""FieldsOfTheWorld Tests."""

import os
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch
from pytest import MonkeyPatch
from torchgeo.datasets import DatasetNotFoundError

from geobench_v2.datamodules import GeoBenchFieldsOfTheWorldDataModule
from geobench_v2.datasets import GeoBenchFieldsOfTheWorld


@pytest.fixture(params=[["red", "blue", "nir", 0.0]])
def band_order(request):
    """Parameterized band configuration with different configurations."""
    return request.param


@pytest.fixture
def datamodule(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    band_order: dict[str, Sequence[str | float]],
):
    """Initialize FOTW datamodule with test configuration."""
    monkeypatch.setattr(GeoBenchFieldsOfTheWorld, "paths", ["fotw.tortilla"])
    monkeypatch.setattr(
        GeoBenchFieldsOfTheWorld, "url", os.path.join("tests", "data", "fotw", "{}")
    )
    monkeypatch.setattr(
        GeoBenchFieldsOfTheWorld,
        "sha256str",
        ["ac574b468d1303858056edf6943f7b4e1a73c9e828597331d9f101387ca40898"],
    )
    dm = GeoBenchFieldsOfTheWorldDataModule(
        img_size=74,
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

        expected_dims = {
            "image_a": (
                datamodule.batch_size,
                len(datamodule.band_order),
                datamodule.img_size,
                datamodule.img_size,
            ),
            "image_b": (
                datamodule.batch_size,
                len(datamodule.band_order),
                datamodule.img_size,
                datamodule.img_size,
            ),
            "mask": (datamodule.batch_size, datamodule.img_size, datamodule.img_size),
        }

        for key, expected_shape in expected_dims.items():
            assert train_batch[key].shape == expected_shape, (
                f"Wrong shape for {key}: got {train_batch[key].shape}, expected {expected_shape}"
            )

        assert torch.isclose(train_batch["image_a"][:, 3], torch.tensor(0.0)).all()

        assert "lon" in train_batch
        assert "lat" in train_batch
        assert train_batch["lon"].shape == (datamodule.batch_size,)
        assert train_batch["lat"].shape == (datamodule.batch_size,)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match="Dataset not found"):
            GeoBenchFieldsOfTheWorld(tmp_path, split="train")
