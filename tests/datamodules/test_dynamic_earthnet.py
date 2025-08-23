# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Dynamic EarthNet."""

import os
from collections.abc import Sequence
from pathlib import Path

import pytest
import matplotlib.pyplot as plt
import torch
from pytest import MonkeyPatch
from torchgeo.datasets import DatasetNotFoundError

from geobench_v2.datamodules import GeoBenchDynamicEarthNetDataModule
from geobench_v2.datasets import GeoBenchDynamicEarthNet


@pytest.fixture(params=[{"s2": ["B02", "B03", 0.0, "B04"], "planet": ["r", "b", 0.0, "g"]}])
def band_order(request):
    """Parameterized band configuration with different configurations."""
    return request.param


@pytest.fixture
def datamodule(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    band_order: dict[str, Sequence[str | float]],
):
    """Initialize DynamicEarthNet datamodule with test configuration."""
    monkeypatch.setattr(GeoBenchDynamicEarthNet, "paths", ["dynamic_earthnet.tortilla"])
    monkeypatch.setattr(
        GeoBenchDynamicEarthNet,
        "url",
        os.path.join("tests", "data", "dynamic_earthnet", "{}"),
    )
    monkeypatch.setattr(
        GeoBenchDynamicEarthNet,
        "sha256str",
        ["5b7ff7005ccde198df47e03d40ce39bd2b99162473c5ebbcfb2289ff382d3cdf"],
    )
    dm = GeoBenchDynamicEarthNetDataModule(
        img_size=74,
        batch_size=2,
        eval_batch_size=1,
        num_workers=0,
        pin_memory=False,
        band_order=band_order,
        root=tmp_path,
        download=True,
        temporal_setting="weekly",
        metadata=["lon", "lat"],
    )
    dm.setup("fit")
    dm.setup("test")

    return dm


class TestDynamicEarthNetDataModule:
    """Test cases for DynamicEarthNet datamodule functionality."""

    def test_loaders(self, datamodule):
        """Test if dataloaders are created successfully."""
        assert len(datamodule.train_dataloader()) > 0
        assert len(datamodule.val_dataloader()) > 0
        assert len(datamodule.test_dataloader()) > 0
        assert len(datamodule.extra_test_dataloader()) > 0

    def test_load_batch_and_check_dims(self, datamodule):
        """Test loading a batch."""
        train_loader = datamodule.train_dataloader()
        train_batch = next(iter(train_loader))
        assert isinstance(train_batch, dict)

        num_time_steps = 6 # for weekly setting
        expected_dims = {}
        for modality, band_names in datamodule.band_order.items():
            if modality == "s2":
                expected_dims[f"image_{modality}"] = (
                    datamodule.batch_size,
                    len(band_names),
                    datamodule.img_size,
                    datamodule.img_size,
                )
            else:
                expected_dims[f"image_{modality}"] = (
                    datamodule.batch_size,
                    num_time_steps,
                    len(band_names),
                    datamodule.img_size,
                    datamodule.img_size,
                )

        expected_dims["mask"] = (
            datamodule.batch_size,
            1,
            datamodule.img_size,
            datamodule.img_size,
        )

        for key, expected_shape in expected_dims.items():
            assert train_batch[key].shape == expected_shape, (
                f"Wrong shape for {key}: got {train_batch[key].shape}, expected {expected_shape}"
            )

        # check that constant values are correct
        for modality, band_names in datamodule.band_order.items():
            for i, band in enumerate(band_names):
                if isinstance(band, (int | float)):
                    key = f"image_{modality}"
                    if modality == "s2":
                        assert torch.isclose(
                            train_batch[key][:, i], torch.tensor(band)
                        ).all(), f"Constant value mismatch for {key} channel {i}"
                    else:
                        # for planet, we need to check the time dimension
                        assert torch.isclose(
                            train_batch[key][:, :, i], torch.tensor(band)
                        ).all(), f"Constant value mismatch for {key} channel {i}"

        assert "lon" in train_batch
        assert "lat" in train_batch
        assert train_batch["lon"].shape == (datamodule.batch_size,)
        assert train_batch["lat"].shape == (datamodule.batch_size,)

    def test_batch_visualization(self, datamodule):
        """Test batch visualization."""
        fig, batch = datamodule.visualize_batch("train")
        assert isinstance(fig, plt.Figure)
        assert isinstance(batch, dict)

        fig.savefig(os.path.join("tests", "data", "dynamic_earthnet", "test_batch.png"))

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match="Dataset not found"):
            GeoBenchDynamicEarthNet(tmp_path, split="train")
