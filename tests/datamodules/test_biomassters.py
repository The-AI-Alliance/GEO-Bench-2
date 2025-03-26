# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""BioMassters Tests."""

import pytest
import torch
from geobench_v2.datamodules import GeoBenchBioMasstersDataModule


@pytest.fixture
def data_root():
    """Path to test data directory."""
    return "/mnt/rg_climate_benchmark/data/geobenchV2/biomassters"


@pytest.fixture
def multimodal_band_order():
    """Test band configuration with separate modality sequences."""
    return {"s1": ["VV_asc", "VH_desc", -1.0], "s2": ["B02", "B03", 0.0]}


@pytest.fixture
def datamodule(data_root, band_order):
    """Initialize BioMassters datamodule with test configuration."""
    return GeoBenchBioMasstersDataModule(
        img_size=256,
        batch_size=32,
        eval_batch_size=64,
        num_workers=0,
        pin_memory=False,
        band_order=band_order,
        root=data_root,
    )


class TestBioMasstersDataModule:
    """Test cases for BioMassters datamodule functionality."""

    def test_multimodal_band_order(self, data_root, multimodal_band_order):
        """Test batch retrieval with modality-specific band sequences."""
        dm = GeoBenchBioMasstersDataModule(
            img_size=74, batch_size=32, band_order=multimodal_band_order, root=data_root
        )
        dm.setup("fit")
        batch = next(iter(dm.train_dataloader()))

        # Check modality-specific outputs
        assert "image_s2" in batch
        assert "image_s1" in batch
        assert batch["image_s2"].shape[:2] == (dm.batch_size, 3)  # S2 bands
        assert batch["image_s1"].shape[:2] == (dm.batch_size, 3)  # S1 bands
        assert batch["image_s2"].shape[2] == 74
        assert batch["image_s2"].shape[3] == dm.img_size
        assert torch.isclose(batch["image_s1"][:, 2], torch.tensor(-1.0)).all()

    def test_multimodal_ts_band_order(self, data_root, multimodal_band_order):
        """Test batch retrieval with modality-specific band sequences."""
        dm = GeoBenchBioMasstersDataModule(
            img_size=74,
            batch_size=32,
            band_order=multimodal_band_order,
            root=data_root,
            num_time_steps=7,
        )
        dm.setup("fit")
        batch = next(iter(dm.train_dataloader()))

        # Check modality-specific outputs

        assert "image_s2" in batch

        assert batch["image_s1"].dim() == 5
        assert batch["image_s2"].dim() == 5
        assert batch["image_s1"].shape[2] == 7
        assert batch["image_s2"].shape[2] == 7
        assert batch["image_s1"].shape[3] == 74
        assert batch["image_s2"].shape[3] == 74
        assert batch["image_s1"].shape[4] == dm.img_size
        assert batch["image_s2"].shape[4] == dm.img_size
