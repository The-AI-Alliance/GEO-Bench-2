# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""BenV2 Tests."""

import pytest
import torch
from geobench_v2.datamodules import GeoBenchBENV2DataModule


@pytest.fixture
def data_root():
    """Path to test data directory."""
    return "/mnt/rg_climate_benchmark/data/geobenchV2/benv2"


@pytest.fixture
def multimodal_band_order():
    """Test band configuration with separate modality sequences."""
    return {"s2": ["B02", "B03", 0.0], "s1": ["VV", "VH", -1.0]}


@pytest.fixture
def datamodule(data_root, band_order):
    """Initialize BENV2 datamodule with test configuration."""
    return GeoBenchBENV2DataModule(
        img_size=256,
        batch_size=32,
        eval_batch_size=64,
        num_workers=0,
        pin_memory=False,
        band_order=band_order,
        root=data_root,
    )


class TestBENV2DataModule:
    """Test cases for BENV2 datamodule functionality."""

    def test_multimodal_band_order(self, data_root, multimodal_band_order):
        """Test batch retrieval with modality-specific band sequences."""
        dm = GeoBenchBENV2DataModule(
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
