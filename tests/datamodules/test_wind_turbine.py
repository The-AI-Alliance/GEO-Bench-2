# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

import os
from collections.abc import Sequence
from pathlib import Path
import torch

import matplotlib.pyplot as plt
import pytest
from pytest import MonkeyPatch

from geobench_v2.datamodules import GeoBenchWindTurbineDataModule
from geobench_v2.datasets import GeoBenchWindTurbine


@pytest.fixture(params=[["red", "green", 0, "blue"]])
def band_order(request):
    """Parameterized band configuration with different configurations."""
    return request.param


@pytest.fixture
def datamodule(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    band_order: dict[str, Sequence[str | float]],
):
    """Initialize WindTurbine datamodule with test configuration."""
    monkeypatch.setattr(GeoBenchWindTurbine, "paths", ["wind_turbine.tortilla"])
    monkeypatch.setattr(
        GeoBenchWindTurbine, "url", os.path.join("tests", "data", "wind_turbine", "{}")
    )
    monkeypatch.setattr(
        GeoBenchWindTurbine,
        "sha256str",
        ["5e92fe98d665dcb08efaa8cb749b72c6890ff9b7a63987b675a7fc0795085a37"],
    )
    dm = GeoBenchWindTurbineDataModule(
        img_size=256,
        batch_size=2,
        eval_batch_size=1,
        num_workers=0,
        pin_memory=False,
        band_order=band_order,
        root=tmp_path,
        download=True,
    )
    dm.setup("fit")
    dm.setup("test")

    return dm


class TestWindTurbineDataModule:
    """Test cases for WindTurbine datamodule functionality."""

    def test_batch_dimensions(self, datamodule):
        """Test if batches have correct dimensions."""
        train_batch = next(iter(datamodule.train_dataloader()))
        assert train_batch["image"].shape[0] == datamodule.batch_size
        assert train_batch["image"].shape[1] == len(datamodule.band_order)
        assert train_batch["image"].shape[2] == datamodule.img_size
        assert train_batch["image"].shape[3] == datamodule.img_size

        assert "bbox_xyxy" in train_batch
        assert "label" in train_batch

    def test_band_order_resolution(self, datamodule):
        """Test if band order is correctly resolved."""
        assert len(datamodule.band_order) == 4
        assert datamodule.band_order[0] == "red"

    def test_batch_visualization(self, datamodule):
        """Test batch visualization."""
        fig, batch = datamodule.visualize_batch(split="train")
        assert isinstance(fig, plt.Figure)
        assert isinstance(batch, dict)

        fig.savefig(os.path.join("tests", "data", "wind_turbine", "test_batch.png"))

    def test_batch_visualization_with_pred(self, datamodule):
        """Test batch visualization with predictions."""
        # Simulate a prediction in the batch
        train_batch = next(iter(datamodule.train_dataloader()))
        
        train_batch["pred_boxes"] = [
            train_batch["bbox_xyxy"][i] + torch.ones_like(train_batch["bbox_xyxy"][i]) * 0.1 for i in range(len(train_batch["bbox_xyxy"]))
        ]
        train_batch["pred_labels"] = [
            train_batch["label"][i] for i in range(len(train_batch["label"]))
        ]

        fig, batch = datamodule.visualize_batch(batch=train_batch)

        fig.savefig(os.path.join("tests", "data", "wind_turbine", "test_batch_with_pred.png"))
