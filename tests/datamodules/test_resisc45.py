import pytest

from geobench_v2.datamodules import GeoBenchRESISC45DataModule


@pytest.fixture
def data_root():
    """Path to test data directory."""
    return "/mnt/rg_climate_benchmark/data/datasets_classification/resisc45"


@pytest.fixture
def band_order():
    """Test band configuration with RGB bands."""
    return ["red", "green", 1.0, "blue"]


@pytest.fixture
def datamodule(data_root, band_order):
    """Initialize RESISC45 datamodule with test configuration."""
    return GeoBenchRESISC45DataModule(
        img_size=74,
        batch_size=32,
        eval_batch_size=64,
        num_workers=0,
        pin_memory=False,
        band_order=band_order,
        root=data_root,
    )


class TestRESISC45DataModule:
    """Test cases for RESISC45 datamodule functionality."""

    def test_batch_dimensions(self, datamodule):
        """Test if batches have correct dimensions."""
        datamodule.setup("fit")
        train_batch = next(iter(datamodule.train_dataloader()))
        assert train_batch["image"].shape[0] == datamodule.batch_size
        assert train_batch["image"].shape[1] == len(datamodule.band_order)
        assert train_batch["image"].shape[2] == 74
        assert train_batch["image"].shape[3] == datamodule.img_size

    def test_band_order_resolution(self, datamodule):
        """Test if band order is correctly resolved."""
        assert len(datamodule.band_order) == 4
        assert isinstance(datamodule.band_order[2], float)
        assert datamodule.band_order[2] == 1.0
