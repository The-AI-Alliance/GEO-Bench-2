# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""TreeSatAI Tests."""

import os
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
from pytest import MonkeyPatch
from torchgeo.datasets import DatasetNotFoundError

from geobench_v2.datamodules import GeoBenchTreeSatAIDataModule
from geobench_v2.datasets import GeoBenchTreeSatAI


@pytest.fixture(
    params=[
        # {
        #     "aerial": ["r", "g", "b", "nir"],
        #     "s2": ["B02", "B03", "B04", 0.0],
        #     "s1": ["VV", "VH", -1.0],
        # },
        {"aerial": ["r", "g", 0.0, "b"]}
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
    """Initialize TreeSatAI datamodule with test configuration."""
    monkeypatch.setattr(GeoBenchTreeSatAI, "paths", ["treesatai.tortilla"])
    monkeypatch.setattr(
        GeoBenchTreeSatAI, "url", os.path.join("tests", "data", "benv2", "{}")
    )
    monkeypatch.setattr(
        GeoBenchTreeSatAI,
        "sha256str",
        ["157322209c9f91be4ca4603d58d26688cac7409c081fc78d5327147ae615ccec"],
    )
    dm = GeoBenchTreeSatAIDataModule(
        img_size=74,
        batch_size=4,
        eval_batch_size=2,
        num_workers=0,
        pin_memory=False,
        band_order=band_order,
        root=os.path.join("tests", "data", "treesatai"),
        metadata=["lon", "lat"],
    )
    dm.setup("fit")
    dm.setup("test")

    return dm


# @pytest.fixture
# def ts_datamodule(monkeypatch: MonkeyPatch, band_order: dict[str, Sequence[str | float]]):
#     """Initialize TreeSatAI datamodule with time series enabled."""
#     monkeypatch.setattr(GeoBenchTreeSatAI, "paths", ["treesatai.tortilla"])
#     dm = GeoBenchTreeSatAIDataModule(
#         img_size=74,
#         batch_size=4,
#         eval_batch_size=2,
#         num_workers=0,
#         pin_memory=False,
#         band_order=band_order,
#         root=os.path.join("tests", "data", "treesatai"),
#         include_ts=True,
#         num_time_steps=12,
#     )
#     dm.setup("fit")
#     dm.setup("test")

#     return dm


class TestTreeSatAIDataModule:
    """Test cases for TreeSatAI datamodule functionality."""

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

        # check that constant values are correct
        for modality, band_names in datamodule.band_order.items():
            for i, band in enumerate(band_names):
                if isinstance(band, (int | float)):
                    key = f"image_{modality}"
                    assert torch.isclose(
                        train_batch[key][:, i], torch.tensor(band)
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

        fig.savefig(os.path.join("tests", "data", "treesatai", "test_batch.png"))

    # def test_time_series_output(self, ts_datamodule):
    #     """Test time series output dimensions."""
    #     train_batch = next(iter(ts_datamodule.train_dataloader()))

    #     for modality in ts_datamodule.band_order.keys():
    #         ts_key = f"image_{modality}_ts"
    #         assert ts_key in train_batch

    #         expected_shape = (
    #             ts_datamodule.batch_size,
    #             ts_datamodule.num_time_steps,
    #             len([b for b in ts_datamodule.band_order[modality] if isinstance(b, str)]),
    #             ts_datamodule.img_size,
    #             ts_datamodule.img_size,
    #         )

    #         assert train_batch[ts_key].shape == expected_shape, (
    #             f"Wrong shape for {ts_key}: got {train_batch[ts_key].shape}, expected {expected_shape}"
    #         )

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match="Dataset not found"):
            GeoBenchTreeSatAI(tmp_path, split="train")
