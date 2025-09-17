# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""BioMassters Tests."""

import os
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from pytest import MonkeyPatch

from geobench_v2.datamodules import GeoBenchBioMasstersDataModule
from geobench_v2.datasets import GeoBenchBioMassters


@pytest.fixture(
    params=[
        {"s1": ["VV_asc", "VH_desc", -1.0, "VH_asc"], "s2": ["B02", "B03", "B04", 0.0]},
        {"s2": ["B02", "B03", "B04"]},
    ]
)
def band_order(request):
    """Parameterized band configuration with different configurations."""
    return request.param


@pytest.fixture
def datamodule(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    band_order: dict[str, Sequence[str | float]],
):
    """Initialize BioMassters datamodule with test configuration."""
    monkeypatch.setattr(GeoBenchBioMassters, "paths", ["biomassters.tortilla"])
    monkeypatch.setattr(
        GeoBenchBioMassters, "url", os.path.join("tests", "data", "biomassters", "{}")
    )
    monkeypatch.setattr(
        GeoBenchBioMassters,
        "sha256str",
        ["4e130ac568584ebb72b8798b91775756a7b78730531e2f3cddc75af94602dd12"],
    )
    dm = GeoBenchBioMasstersDataModule(
        img_size=74,
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
def ts_datamodule(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    band_order: dict[str, Sequence[str | float]],
):
    """Initialize BioMassters datamodule with time series support."""
    monkeypatch.setattr(GeoBenchBioMassters, "paths", ["biomassters.tortilla"])
    monkeypatch.setattr(
        GeoBenchBioMassters, "url", os.path.join("tests", "data", "biomassters", "{}")
    )
    monkeypatch.setattr(
        GeoBenchBioMassters,
        "sha256str",
        ["4e130ac568584ebb72b8798b91775756a7b78730531e2f3cddc75af94602dd12"],
    )
    dm = GeoBenchBioMasstersDataModule(
        img_size=74,
        batch_size=4,
        eval_batch_size=2,
        num_workers=0,
        pin_memory=False,
        band_order=band_order,
        root=tmp_path,
        download=True,
        num_time_steps=3,
    )
    dm.setup("fit")
    dm.setup("test")

    return dm


class TestBioMasstersDataModule:
    """Test cases for BioMassters datamodule functionality."""

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

    def test_time_series_output(self, ts_datamodule):
        """Test time series output dimensions."""
        train_batch = next(iter(ts_datamodule.train_dataloader()))

        # For time series data, dimensions should be [batch, time, channels, height, width]
        for modality, band_names in ts_datamodule.band_order.items():
            key = f"image_{modality}"
            assert train_batch[key].dim() == 5
            assert train_batch[key].shape[0] == ts_datamodule.batch_size
            assert (
                train_batch[key].shape[1] == ts_datamodule.train_dataset.num_time_steps
            )
            assert train_batch[key].shape[2] == len(band_names)
            assert train_batch[key].shape[3] == ts_datamodule.img_size
            assert train_batch[key].shape[4] == ts_datamodule.img_size

        assert train_batch["mask"].shape == (
            ts_datamodule.batch_size,
            1,
            ts_datamodule.img_size,
            ts_datamodule.img_size,
        )

    def test_batch_visualization(self, datamodule):
        """Test batch visualization."""
        fig, batch = datamodule.visualize_batch("train")
        assert isinstance(fig, plt.Figure)
        assert isinstance(batch, dict)

        fig.savefig(os.path.join("tests", "data", "biomassters", "test_batch.png"))
        plt.close(fig)
