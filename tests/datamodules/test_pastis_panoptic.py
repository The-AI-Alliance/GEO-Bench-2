# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""PASTIS DataModule Tests."""

import os

import matplotlib.pyplot as plt
import pytest

from geobench_v2.datamodules import GeoBenchPASTISPanopticDataModule
from geobench_v2.datamodules.pastis_panoptic import pastis_collate_fn as collate
import pdb

@pytest.fixture(
    params=[
        {
            "s2": ["B02", "B03", 0.0, "B08", "B04"],
            "s1_asc": ["VV_asc", "VH_asc", 1.0],
            "s1_desc": ["VV_desc", "VH_desc", -1.0],
        }
    ]
)
def band_order(request):
    """Parameterized band configuration with different configurations."""
    return request.param


@pytest.fixture
def invalid_mixed_band_order():
    """Test invalid band configuration with mixed modality bands."""
    return ["B02", "VV_asc", "B08"]  # Mixes S2 and S1 bands


@pytest.fixture
def datamodule(band_order):
    datamodule = GeoBenchPASTISPanopticDataModule(
        img_size=256,
        batch_size=8,
        num_workers=0,
        band_order=band_order,
        root="/opt/app-root/src/fm-geospatial/data/PASTIS/",
        metadata=["lon", "lat"],
        return_stacked_image=False,
        label_type="instance_seg",
        collate_fn=collate,
        num_time_steps= 12,
    )
    datamodule.setup("fit")
    datamodule.setup("test")
    return datamodule


class TestPASTISPanopticDataModule:
    """Test cases for PASTIS datamodule functionality."""

    def test_multimodal_band_order(self, datamodule):
        """Test batch retrieval with modality-specific band sequences."""

        batch = next(iter(datamodule.train_dataloader()))

        # Check modality-specific outputs
        assert "image_s2" in batch
        assert "image_s1_asc" in batch
        assert "image_s1_desc" in batch
        assert "mask" in batch

        # TODO handle correctly both cases
        # Check dimensions - these are time-series shapes [batch_size, bands, time_steps, H, W]
        # or [batch_size, bands, H, W] depending on how they're processed
        assert batch["image_s2"].shape[0] == datamodule.batch_size
        assert batch["image_s2"].shape[1] == datamodule.train_dataset.num_time_steps
        assert batch["image_s2"].shape[2] == len(datamodule.band_order["s2"])
        assert batch["image_s2"].shape[3] == datamodule.img_size
        assert batch["image_s2"].shape[4] == datamodule.img_size
        
        assert batch["image_s1_asc"].shape[0] == datamodule.batch_size
        assert batch["image_s1_asc"].shape[1] == datamodule.train_dataset.num_time_steps
        assert batch["image_s1_asc"].shape[2] == len(datamodule.band_order["s1_asc"])
        assert batch["image_s1_asc"].shape[3] == datamodule.img_size
        assert batch["image_s1_asc"].shape[4] == datamodule.img_size
        
        assert batch["image_s1_desc"].shape[0] == datamodule.batch_size
        assert batch["image_s1_desc"].shape[1] == datamodule.train_dataset.num_time_steps
        assert batch["image_s1_desc"].shape[2] == len(datamodule.band_order["s1_desc"])
        assert batch["image_s1_desc"].shape[3] == datamodule.img_size
        assert batch["image_s1_desc"].shape[4] == datamodule.img_size

        assert "lon" in batch
        assert "lat" in batch
        assert len(batch["lon"]) == datamodule.batch_size
        assert len(batch["lat"]) == datamodule.batch_size

    def test_batch_visualization(self, datamodule):
        """Test batch visualization."""
        fig, batch = datamodule.visualize_batch("train")
        assert isinstance(fig, plt.Figure)
        assert isinstance(batch, dict)

        fig.savefig(os.path.join("tests", "data", "pastis", "test_batch.png"))

    def test_time_series(self, band_order):
        """Test batch retrieval with time series."""
        num_time_steps = 3
        datamodule = GeoBenchPASTISPanopticDataModule(
            img_size=74,
            batch_size=4,
            num_time_steps=num_time_steps,
            band_order=band_order,
            root="/opt/app-root/src/fm-geospatial/data/PASTIS/",
        )
        datamodule.setup("fit")
        batch = next(iter(datamodule.train_dataloader()))

        # Check single tensor output - only S2 bands
        assert batch["image_s2"].shape[0] == datamodule.batch_size
        assert batch["image_s2"].shape[1] == num_time_steps
        assert batch["image_s2"].shape[2] == len(datamodule.band_order["s2"])
        assert batch["image_s2"].shape[3] == datamodule.img_size
        assert batch["image_s2"].shape[4] == datamodule.img_size
