# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.


import pytest
from geobench_v2.datamodules import GeoBenchSpaceNet2DataModule


@pytest.fixture
def data_root():
    """Path to test data directory."""
    return "/mnt/rg_climate_benchmark/data/geobenchV2/spacenet2"


@pytest.fixture
def band_order():
    """Test band configuration with fill values."""
    return {"worldview": ["r", "g", 0.2], "pan": ["pan", 1.0]}


@pytest.fixture
def datamodule(data_root, band_order):
    """Initialize SpaceNet2 datamodule with test configuration."""
    return GeoBenchSpaceNet2DataModule(
        img_size=74,
        batch_size=32,
        eval_batch_size=64,
        num_workers=0,
        pin_memory=False,
        band_order=band_order,
        root=data_root,
    )


class TestSpaceNet2DataModule:
    """Test cases for SpaceNet2 datamodule functionality."""

    def test_batch_dimensions(self, datamodule):
        """Test if batches have correct dimensions."""
        datamodule.setup("fit")
        train_batch = next(iter(datamodule.train_dataloader()))
        expected_dims = {
            "image_worldview": (
                datamodule.batch_size,
                len(datamodule.band_order["worldview"]),
                74,
                datamodule.img_size,
            ),
            "image_pan": (
                datamodule.batch_size,
                len(datamodule.band_order["pan"]),
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
        assert len(datamodule.band_order["worldview"]) == 3
        assert isinstance(datamodule.band_order["worldview"][2], float)
        assert datamodule.band_order["pan"][1] == 1.0
