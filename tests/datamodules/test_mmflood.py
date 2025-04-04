# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.


import pytest
from geobench_v2.datamodules import GeoBenchMMFloodDataModule


@pytest.fixture
def data_root():
    """Path to test data directory."""
    return "/mnt/rg_climate_benchmark/data/geobenchV2/mmflood"


@pytest.fixture
def band_order():
    """Test band configuration with fill values."""
    return {"s1": ["vv", 0.2, "vh"], "hydro": ["hydro", 1.0], "dem": ["dem", 0.2]}


@pytest.fixture
def datamodule(data_root, band_order):
    """Initialize MMFlood datamodule with test configuration."""
    return GeoBenchMMFloodDataModule(
        img_size=74,
        batch_size=32,
        eval_batch_size=64,
        num_workers=0,
        pin_memory=False,
        band_order=band_order,
        root=data_root,
    )


class TestMMFloodDataModule:
    """Test cases for MMFlood datamodule functionality."""

    def test_batch_dimensions(self, datamodule):
        """Test if batches have correct dimensions."""
        datamodule.setup("fit")
        train_batch = next(iter(datamodule.train_dataloader()))

        # Define expected dimensions for testing
        expected_dims = {
            "image_s1": (
                datamodule.batch_size,
                len(datamodule.band_order["s1"]),
                74,
                datamodule.img_size,
            ),
            "image_dem": (
                datamodule.batch_size,
                len(datamodule.band_order["dem"]),
                74,
                datamodule.img_size,
            ),
            "image_hydro": (
                datamodule.batch_size,
                len(datamodule.band_order["hydro"]),
                74,
                datamodule.img_size,
            ),
            "mask": (datamodule.batch_size, 1, 74, datamodule.img_size),
        }

        for key, expected_shape in expected_dims.items():
            assert train_batch[key].shape == expected_shape, (
                f"Wrong shape for {key}: got {train_batch[key].shape}, expected {expected_shape}"
            )

    def test_band_order_resolution(self, datamodule):
        """Test if band order is correctly resolved."""
        assert len(datamodule.band_order["s1"]) == 3
        assert isinstance(datamodule.band_order["s1"][1], float)
        assert datamodule.band_order["s1"][1] == 0.2
