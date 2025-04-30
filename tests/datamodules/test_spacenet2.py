# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""SpaceNet2 Tests."""

import os
import pytest
from typing import Sequence, Dict, Union
import torch
from pathlib import Path
from torchgeo.datasets import DatasetNotFoundError
import matplotlib.pyplot as plt
from pytest import MonkeyPatch
from geobench_v2.datasets import GeoBenchSpaceNet2
from geobench_v2.datamodules import GeoBenchSpaceNet2DataModule


@pytest.fixture(params=[{"worldview": ["r", "g", "b", 0.2], "pan": ["pan", 1.0]}])
def band_order(request):
    """Parameterized band configuration with different configurations."""
    return request.param


@pytest.fixture
def datamodule(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    band_order: Dict[str, Sequence[Union[str, float]]],
):
    """Initialize SpaceNet2 datamodule with test configuration."""
    monkeypatch.setattr(GeoBenchSpaceNet2, "paths", ["spacenet2.tortilla"])
    monkeypatch.setattr(
        GeoBenchSpaceNet2, "url", os.path.join("tests", "data", "spacenet2", "{}")
    )
    monkeypatch.setattr(
        GeoBenchSpaceNet2,
        "sha256str",
        ["73753fdebe48ea7aeaf9dcd086c47635ed1c8fd84f4ef39a3cfe12455e5fd265"],
    )
    dm = GeoBenchSpaceNet2DataModule(
        img_size=256,
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


class TestSpaceNet2DataModule:
    """Test cases for SpaceNet2 datamodule functionality."""

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
                if isinstance(band, (int, float)):
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

        fig.savefig(os.path.join("tests", "data", "spacenet2", "test_batch.png"))

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match="Dataset not found"):
            GeoBenchSpaceNet2(tmp_path, split="train")
