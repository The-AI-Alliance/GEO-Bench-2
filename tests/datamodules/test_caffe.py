import pytest
from geobench_v2.datamodules import GeoBenchCaFFeDataModule


@pytest.fixture
def data_root():
    """Path to test data directory."""
    return "/mnt/rg_climate_benchmark/data/datasets_segmentation/Caffe"


@pytest.fixture
def band_order():
    """Test band configuration with fill values."""
    return ["gray", 0.0, "gray"]


@pytest.fixture
def datamodule(data_root, band_order):
    """Initialize CaFFe datamodule with test configuration."""
    return GeoBenchCaFFeDataModule(
        img_size=256,
        batch_size=32,
        eval_batch_size=64,
        num_workers=0,
        pin_memory=False,
        band_order=band_order,
        root=data_root,
    )


class TestCaFFeDataModule:
    """Test cases for CaFFe datamodule functionality."""

    def test_batch_dimensions(self, datamodule):
        """Test if batches have correct dimensions."""
        datamodule.setup("fit")
        train_batch = next(iter(datamodule.train_dataloader()))
        assert train_batch["image"].shape[0] == datamodule.batch_size
        assert train_batch["image"].shape[1] == len(datamodule.band_order)

    def test_band_order_resolution(self, datamodule):
        """Test if band order is correctly resolved."""
        assert len(datamodule.band_order) == 3
        assert isinstance(datamodule.band_order[1], float)
        assert datamodule.band_order[1] == 0.0
