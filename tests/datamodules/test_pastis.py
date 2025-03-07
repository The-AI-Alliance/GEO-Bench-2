import pytest
from geobench_v2.datamodules import GeoBenchPASTISDataModule


@pytest.fixture
def data_root():
    """Path to test data directory."""
    return "/mnt/rg_climate_benchmark/data/datasets_segmentation/pastis_r"


@pytest.fixture
def s2_only_band_order():
    """Test band configuration with only S2 bands."""
    return ["B02", "B03", 0.0, "B08"]


@pytest.fixture
def s1_asc_only_band_order():
    """Test band configuration with only S1 ascending bands."""
    return ["VV_asc", "VH_asc", 1.0]


@pytest.fixture
def multimodal_band_order():
    """Test band configuration with separate modality sequences."""
    return {
        "s2": ["B02", "B03", 0.0, "B08"],
        "s1_asc": ["VV_asc", "VH_asc", 1.0],
        "s1_desc": ["VV_desc", "VH_desc", -1.0],
    }


@pytest.fixture
def invalid_mixed_band_order():
    """Test invalid band configuration with mixed modality bands."""
    return ["B02", "VV_asc", "B08"]  # Mixes S2 and S1 bands


class TestPASTISDataModule:
    """Test cases for PASTIS datamodule functionality."""

    def test_single_modality_band_order(self, data_root, s2_only_band_order):
        """Test batch retrieval with bands from a single modality."""
        dm = GeoBenchPASTISDataModule(
            img_size=256, batch_size=32, band_order=s2_only_band_order, root=data_root
        )
        dm.setup("fit")
        batch = next(iter(dm.train_dataloader()))

        # Check single tensor output - only S2 bands
        assert "image" in batch
        assert batch["image"].shape[0] == dm.batch_size
        assert batch["image"].shape[1] == len(
            s2_only_band_order
        )  # Number of bands in sequence

    def test_multimodal_band_order(self, data_root, multimodal_band_order):
        """Test batch retrieval with modality-specific band sequences."""
        dm = GeoBenchPASTISDataModule(
            img_size=256,
            batch_size=32,
            band_order=multimodal_band_order,
            root=data_root,
        )
        dm.setup("fit")
        batch = next(iter(dm.train_dataloader()))

        # Check modality-specific outputs
        assert "image_s2" in batch
        assert "image_s1_asc" in batch
        assert "image_s1_desc" in batch

        # TODO handle correctly both cases
        # Check dimensions - remember these are time-series shapes [batch_size, bands, time_steps, H, W]
        # or [batch_size, bands, H, W] depending on how they're processed
        assert batch["image_s2"].shape[0] == dm.batch_size
        assert batch["image_s2"].shape[1] == len(multimodal_band_order["s2"])

        assert batch["image_s1_asc"].shape[0] == dm.batch_size
        assert batch["image_s1_asc"].shape[1] == len(multimodal_band_order["s1_asc"])

        assert batch["image_s1_desc"].shape[0] == dm.batch_size
        assert batch["image_s1_desc"].shape[1] == len(multimodal_band_order["s1_desc"])

    def test_invalid_mixed_band_order(self, data_root, invalid_mixed_band_order):
        """Test that validation rejects band sequences with mixed modalities."""
        with pytest.raises(
            ValueError, match="bands in a sequence must all be from the same modality"
        ):
            dm = GeoBenchPASTISDataModule(
                img_size=256,
                batch_size=32,
                band_order=invalid_mixed_band_order,
                root=data_root,
            )
            dm.setup("fit")

    def test_alternative_single_modality_band_order(
        self, data_root, s1_asc_only_band_order
    ):
        """Test batch retrieval with bands from a single S1 modality."""
        dm = GeoBenchPASTISDataModule(
            img_size=256,
            batch_size=32,
            band_order=s1_asc_only_band_order,
            root=data_root,
        )
        dm.setup("fit")
        batch = next(iter(dm.train_dataloader()))

        # Check single tensor output - only S1 ascending bands
        assert "image" in batch
        assert batch["image"].shape[0] == dm.batch_size
        assert batch["image"].shape[1] == len(s1_asc_only_band_order)
