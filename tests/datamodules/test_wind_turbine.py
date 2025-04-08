import pytest
from geobench_v2.datamodules import GeoBenchWindTurbineDataModule


@pytest.fixture
def data_root():
    """Path to test data directory."""
    return "/mnt/rg_climate_benchmark/data/geobenchV2/wind_turbine"


@pytest.fixture
def band_order():
    """Test band configuration with RGB and fill value."""
    # return ["red", "green", "blue", 0, "green"]
    return ["red", "green", "blue", 0]


@pytest.fixture
def datamodule(data_root, band_order):
    """Initialize WindTurbine datamodule with test configuration."""
    return GeoBenchWindTurbineDataModule(
        img_size=74,
        batch_size=16,
        eval_batch_size=64,
        num_workers=0,
        pin_memory=False,
        band_order=band_order,
        root=data_root,
    )


class TestWindTurbineDataModule:
    """Test cases for WindTurbine datamodule functionality."""

    def test_batch_dimensions(self, datamodule):
        """Test if batches have correct dimensions."""
        datamodule.setup("fit")
        train_batch = next(iter(datamodule.train_dataloader()))
        assert train_batch["image"].shape[0] == datamodule.batch_size
        assert train_batch["image"].shape[1] == len(datamodule.band_order)
        assert train_batch["image"].shape[2] == datamodule.img_size
        assert train_batch["image"].shape[3] == datamodule.img_size

        assert "bboxes_xyxy" in train_batch
        assert "labels" in train_batch

    def test_band_order_resolution(self, datamodule):
        """Test if band order is correctly resolved."""
        # assert len(datamodule.band_order) == 5
        assert len(datamodule.band_order) == 4
        assert datamodulue.band_order[0] == "red"
