# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Dynamic EarthNet."""

import pytest
import torch
from geobench_v2.datamodules import GeoBenchDynamicEarthNetDataModule


@pytest.fixture
def data_root():
    """Path to test data directory."""
    return "/mnt/rg_climate_benchmark/data/geobenchV2/dynamic_earthnet"


@pytest.fixture
def multimodal_band_order():
    """Test band configuration with separate modality sequences."""
    return {"s2": ["B02", "B03", 0.0], "planet": ["r", "b", 0.0]}


@pytest.fixture
def datamodule(data_root, multimodal_band_order):
    """Initialize DynamicEarthNet datamodule with test configuration."""
    return GeoBenchDynamicEarthNetDataModule(
        img_size=74,
        batch_size=2,
        eval_batch_size=2,
        num_workers=0,
        pin_memory=False,
        band_order=multimodal_band_order,
        root=data_root,
        temporal_setting="weekly",
    )


class TestDynamicEarthNetDataModule:
    """Test cases for DynamicEarthNet datamodule functionality."""

    def test_multimodal_band_order(self, datamodule, multimodal_band_order):
        """Test batch retrieval with modality-specific band sequences."""
        datamodule.setup("fit")
        batch = next(iter(datamodule.train_dataloader()))

        # Check modality-specific outputs
        assert "image_s2" in batch
        assert "image_planet" in batch

        import pdb

        pdb.set_trace()

        print(0)
