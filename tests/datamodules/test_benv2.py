# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""BenV2 Tests."""

import os

import pytest
from typing import Sequence
import torch
import matplotlib.pyplot as plt
from pytest import MonkeyPatch
from geobench_v2.datasets import GeoBenchBENV2
from geobench_v2.datamodules import GeoBenchBENV2DataModule


@pytest.fixture(
    params=[
        {"s2": ["B02", "B03", "B04", 0.0], "s1": ["VV", "VH", -1.0]},
        {"s2": ["B04", "B03", "B02"]},
    ]
)
def band_order(request):
    """Parameterized band configuration with different configurations."""
    return request.param


@pytest.fixture
def datamodule(monkeypatch: MonkeyPatch, band_order: dict[str, Sequence[str | float]]):
    """Initialize BENV2 datamodule with test configuration."""
    monkeypatch.setattr(GeoBenchBENV2, "paths", ["benv2.tortilla"])
    dm = GeoBenchBENV2DataModule(
        img_size=74,
        batch_size=4,
        eval_batch_size=2,
        num_workers=0,
        pin_memory=False,
        band_order=band_order,
        root=os.path.join("tests", "data", "benv2"),
        metadata=["lon", "lat"],
    )
    dm.setup("fit")
    dm.setup("test")

    return dm


class TestBENV2DataModule:
    """Test cases for BENV2 datamodule functionality."""

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
            f"image_{modality}": (
                datamodule.batch_size,
                len(band_names),
                datamodule.img_size,
                datamodule.img_size,
            )
            for modality, band_names in datamodule.band_order.items()
        }
        expected_dims["label"] = (datamodule.batch_size, datamodule.num_classes)

        for key, expected_shape in expected_dims.items():
            assert train_batch[key].shape == expected_shape, (
                f"Wrong shape for {key}: got {train_batch[key].shape}, expected {expected_shape}"
            )

        assert "lon" in train_batch
        assert "lat" in train_batch
        assert train_batch["lon"].shape == (datamodule.batch_size,)
        assert train_batch["lat"].shape == (datamodule.batch_size,)

    def test_batch_visualization(self, datamodule):
        """Test batch visualization."""
        fig, batch = datamodule.visualize_batch("train")
        assert isinstance(fig, plt.Figure)
        assert isinstance(batch, dict)

        fig.savefig(os.path.join("tests", "data", "benv2", "test_batch.png"))
