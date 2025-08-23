# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""KuroSiwo Tests."""

import os
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
from pytest import MonkeyPatch
from torchgeo.datasets import DatasetNotFoundError

from geobench_v2.datamodules import GeoBenchKuroSiwoDataModule
from geobench_v2.datasets import GeoBenchKuroSiwo


@pytest.fixture(params=[{"sar": ["vv", "vh", 0.2], "dem": [0.1, "dem"]}])
def band_order(request):
    """Parameterized band configuration with different configurations."""
    return request.param


@pytest.fixture
def datamodule(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    band_order: dict[str, Sequence[str | float]],
):
    """Initialize KuroSiwo datamodule with test configuration."""
    monkeypatch.setattr(GeoBenchKuroSiwo, "paths", ["kuro_siwo.tortilla"])
    monkeypatch.setattr(
        GeoBenchKuroSiwo, "url", os.path.join("tests", "data", "kuro_siwo", "{}")
    )
    monkeypatch.setattr(
        GeoBenchKuroSiwo,
        "sha256str",
        ["4d0b48543e669f7df45e4195d1872fe7cf2c52c0810a9fe9872ee5daaec3a22b"],
    )
    dm = GeoBenchKuroSiwoDataModule(
        img_size=256,
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


@pytest.fixture
def stacked_datamodule(
    monkeypatch: MonkeyPatch, band_order: dict[str, Sequence[str | float]]
):
    """Initialize KuroSiwo datamodule with test configuration."""
    monkeypatch.setattr(GeoBenchKuroSiwo, "paths", ["kuro_siwo.tortilla"])
    dm = GeoBenchKuroSiwoDataModule(
        img_size=256,
        batch_size=4,
        eval_batch_size=2,
        num_workers=0,
        pin_memory=False,
        band_order=band_order,
        root=os.path.join("tests", "data", "kuro_siwo"),
        return_stacked_image=True,
    )
    dm.setup("fit")
    dm.setup("test")

    return dm


class TestKuroSiwoDataModule:
    """Test cases for KuroSiwo datamodule functionality."""

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

        # Define expected dimensions for testing
        expected_dims = {
            "image_sar_pre_1": (
                datamodule.batch_size,
                len(datamodule.band_order["sar"]),
                datamodule.img_size,
                datamodule.img_size,
            ),
            "image_sar_pre_2": (
                datamodule.batch_size,
                len(datamodule.band_order["sar"]),
                datamodule.img_size,
                datamodule.img_size,
            ),
            "image_sar_post": (
                datamodule.batch_size,
                len(datamodule.band_order["sar"]),
                datamodule.img_size,
                datamodule.img_size,
            ),
            "image_dem": (
                datamodule.batch_size,
                len(datamodule.band_order["dem"]),
                datamodule.img_size,
                datamodule.img_size,
            ),
            "mask": (datamodule.batch_size, datamodule.img_size, datamodule.img_size),
        }

        for key, expected_shape in expected_dims.items():
            assert train_batch[key].shape == expected_shape, (
                f"Wrong shape for {key}: got {train_batch[key].shape}, expected {expected_shape}"
            )

    def test_constant_values(self, datamodule):
        """Test if constant values in band_order are correctly applied."""
        train_batch = next(iter(datamodule.train_dataloader()))

        # Check constants in sar bands if present
        if any(
            isinstance(band, (int | float)) for band in datamodule.band_order["sar"]
        ):
            for i, band in enumerate(datamodule.band_order["sar"]):
                if isinstance(band, (int | float)):
                    assert torch.isclose(
                        train_batch["image_sar_pre_1"][:, i], torch.tensor(band)
                    ).all(), f"Constant value mismatch for image_pre_1 channel {i}"

                    assert torch.isclose(
                        train_batch["image_sar_pre_2"][:, i], torch.tensor(band)
                    ).all(), f"Constant value mismatch for image_pre_2 channel {i}"

                    assert torch.isclose(
                        train_batch["image_sar_post"][:, i], torch.tensor(band)
                    ).all(), f"Constant value mismatch for image_post channel {i}"

        # Check constants in dem bands if present
        if any(
            isinstance(band, (int | float)) for band in datamodule.band_order["dem"]
        ):
            for i, band in enumerate(datamodule.band_order["dem"]):
                if isinstance(band, (int | float)):
                    assert torch.isclose(
                        train_batch["image_dem"][:, i], torch.tensor(band)
                    ).all(), f"Constant value mismatch for image_dem channel {i}"

    def test_stacked_batch(self, stacked_datamodule):
        """Test batch of stacked mode."""
        train_batch = next(iter(stacked_datamodule.train_dataloader()))

        assert isinstance(train_batch, dict)

        assert "image" in train_batch
        assert "mask" in train_batch

        # for stacked mode, multiply sar channels by 3 (pre_1, pre_2, post) plus the dem channels
        num_channels = 3 * len(stacked_datamodule.band_order["sar"]) + len(
            stacked_datamodule.band_order["dem"]
        )
        assert train_batch["image"].shape == (
            stacked_datamodule.batch_size,
            num_channels,
            stacked_datamodule.img_size,
            stacked_datamodule.img_size,
        )

    def test_batch_visualization(self, datamodule):
        """Test batch visualization."""
        fig, batch = datamodule.visualize_batch(split="train")
        assert isinstance(fig, plt.Figure)
        assert isinstance(batch, dict)

        fig.savefig(os.path.join("tests", "data", "kuro_siwo", "test_batch.png"))

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match="Dataset not found"):
            GeoBenchKuroSiwo(tmp_path, split="train")
