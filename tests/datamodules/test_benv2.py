import pytest
from geobench_v2.datamodules import GeoBenchBENV2DataModule


@pytest.fixture
def data_root():
    """Path to test data directory."""
    return "/mnt/rg_climate_benchmark/data/datasets_classification/benv2"


@pytest.fixture
def band_order():
    """Test band configuration with mix of bands and fill values."""
    return ["VV", "B01", "B02", 1.5, "B03"]


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

    def test_batch_dimensions(self, datamodule):
        """Test if batches have correct dimensions."""
        datamodule.setup("fit")
        train_batch = next(iter(datamodule.train_dataloader()))

        assert train_batch["image"].shape[0] == datamodule.batch_size
        assert train_batch["image"].shape[1] == len(datamodule.band_order)

    def test_band_order_resolution(self, datamodule):
        """Test if band order is correctly resolved."""
        assert len(datamodule.band_order) == 5
        assert isinstance(datamodule.band_order[3], float)
        assert datamodule.band_order[3] == 1.5

    def test_sequence_band_order(self, data_root):
        """Test batch retrieval with sequence of bands."""
        dm = GeoBenchBENV2DataModule(
            img_size=256,
            batch_size=32,
            band_order=["VV", "B01", "B02", 1.5, "B03"],
            root=data_root,
        )
        dm.setup("fit")
        batch = next(iter(dm.train_dataloader()))

        # Check single tensor output
        assert "image" in batch
        assert batch["image"].shape[0] == dm.batch_size
        assert batch["image"].shape[1] == 5  # Number of bands in sequence

    def test_multimodal_band_order(self, data_root, multimodal_band_order):
        """Test batch retrieval with modality-specific band sequences."""
        dm = GeoBenchBENV2DataModule(
            img_size=256,
            batch_size=32,
            band_order=multimodal_band_order,
            root=data_root,
        )
        dm.setup("fit")
        batch = next(iter(dm.train_dataloader()))

        # Check modality-specific outputs
        assert "image_s2" in batch
        assert "image_s1" in batch
        assert batch["image_s2"].shape[:2] == (dm.batch_size, 3)  # S2 bands
        assert batch["image_s1"].shape[:2] == (dm.batch_size, 3)  # S1 bands
