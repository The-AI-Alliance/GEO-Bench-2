import pytest
from geobench_v2.datasets.sensor_util import (
    BandConfig,
    ModalityConfig,
    MultiModalConfig,
    SensorType,
    SensorBandRegistry,
    DatasetBandRegistry,
)


class TestBandConfig:
    """Test band configuration functionality."""

    def test_basic_config(self):
        config = BandConfig("red", ["r", "red", "RED"], wavelength=0.665)
        assert config.canonical_name == "red"
        assert "r" in config.aliases
        assert config.wavelength == 0.665
        assert config.resolution is None

    def test_full_config(self):
        config = BandConfig(
            canonical_name="coastal",
            aliases=["coastal_aerosol", "b01"],
            wavelength=0.443,
            resolution=60,
        )
        assert config.canonical_name == "coastal"
        assert "coastal_aerosol" in config.aliases
        assert config.wavelength == 0.443
        assert config.resolution == 60

    def test_alias_matching(self):
        config = BandConfig("red", ["r", "red", "RED"], wavelength=0.665)
        assert config.matches_alias("r")
        assert config.matches_alias("red")
        assert config.matches_alias("RED")
        assert not config.matches_alias("blue")


class TestModalityConfig:
    """Test modality configuration functionality."""

    def test_rgb_config(self):
        config = SensorBandRegistry.RGB
        assert len(config.bands) == 3
        assert "r" in config.bands
        assert config.bands["r"].wavelength == 0.665
        assert config.default_order == ["r", "g", "b"]
        assert config.native_resolution is None

    def test_rgbn_config(self):
        config = SensorBandRegistry.RGBN
        assert len(config.bands) == 4
        assert all(b in config.bands for b in ["r", "g", "b", "nir"])
        assert config.bands["nir"].wavelength == 0.842
        assert config.default_order == ["r", "g", "b", "nir"]

    def test_sentinel2_config(self):
        config = SensorBandRegistry.SENTINEL2
        # Test band count (excluding B10/cirrus)
        assert len(config.bands) == 12
        # Test resolutions
        assert config.bands["B02"].resolution == 10  # 10m bands
        assert config.bands["B05"].resolution == 20  # 20m bands
        assert config.bands["B01"].resolution == 60  # 60m bands
        # Test wavelengths
        assert config.bands["B04"].wavelength == 0.665  # Red
        assert config.bands["B08"].wavelength == 0.842  # NIR
        # Test native resolution
        assert config.native_resolution == 10

    def test_sentinel1_config(self):
        config = SensorBandRegistry.SENTINEL1
        assert len(config.bands) == 2
        assert all(band in config.bands for band in ["VV", "VH"])
        assert config.bands["VV"].canonical_name == "vv"
        assert config.bands["VH"].canonical_name == "vh"
        assert config.native_resolution == 10

    def test_grayscale_config(self):
        config = SensorBandRegistry.GRAYSCALE
        assert len(config.bands) == 1
        assert "gray" in config.bands
        assert config.default_order == ["gray"]


class TestMultiModalConfig:
    """Test multi-modal configuration functionality."""

    def test_benv2_config(self):
        config = DatasetBandRegistry.BENV2
        # Test modalities
        assert set(config.modalities.keys()) == {"s1", "s2"}
        # Test band mapping
        assert config.band_to_modality["B02"] == "s2"
        assert config.band_to_modality["VV"] == "s1"
        # Test all S2 bands are mapped
        s2_bands = [
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B09",
            "B11",
            "B12",
        ]
        assert all(config.band_to_modality[b] == "s2" for b in s2_bands)
        # Test S1 bands are mapped
        assert all(config.band_to_modality[b] == "s1" for b in ["VV", "VH"])

    def test_pastis_config(self):
        config = DatasetBandRegistry.PASTIS
        # Test modalities
        assert set(config.modalities.keys()) == {"s1", "s2"}
        # Test S1 complex band names
        s1_bands = ["VV_asc", "VH_asc", "VV/VH_asc", "VV_desc", "VH_desc", "VV/VH_desc"]
        assert all(b in config.band_to_modality for b in s1_bands)
        assert all(config.band_to_modality[b] == "s1" for b in s1_bands)
        # Test default order matches band mapping
        assert all(b in config.band_to_modality for b in config.default_order)


class TestBandResolution:
    """Test band name resolution functionality."""

    @pytest.mark.parametrize(
        "band_spec,expected",
        [
            # S2 bands and aliases
            ("B02", "B02"),
            ("blue", "B02"),
            ("b02", "B02"),
            ("B04", "B04"),
            ("red", "B04"),
            # S1 bands and aliases
            ("VV", "VV"),
            ("co_pol", "VV"),
            ("VH", "VH"),
            ("cross_pol", "VH"),
            # RGB bands and aliases
            ("r", "r"),
            ("RED", "r"),
            ("g", "g"),
            ("GREEN", "g"),
            # Complex cases
            ("coastal_aerosol", "B01"),
            ("near_infrared", "B08"),
            ("wv", "B09"),
        ],
    )
    def test_band_resolution(self, band_spec, expected):
        for modality in ["s1", "s2", "rgb"]:
            config = SensorBandRegistry.get_modality_config(modality)
            try:
                result = config.resolve_band(band_spec)
                assert result == expected
                break
            except ValueError:
                continue

    def test_resolution_errors(self):
        config = SensorBandRegistry.get_modality_config("s2")
        with pytest.raises(ValueError, match="Band invalid not found in configuration"):
            config.resolve_band("invalid")


class TestDatasetRegistry:
    """Test dataset registry functionality."""

    def test_single_modality_datasets(self):
        single_modal = [
            ("CAFFE", ["gray"]),
            ("EVERWATCH", ["r", "g", "b"]),
            ("RESISC45", ["r", "g", "b"]),
            ("FOTW", ["r", "g", "b", "nir"]),
        ]
        for name, expected_bands in single_modal:
            config = DatasetBandRegistry.get_dataset_config(name)
            assert isinstance(config, ModalityConfig)
            assert config.default_order == expected_bands

    def test_multi_modal_datasets(self):
        multi_modal = ["BENV2", "PASTIS"]
        for name in multi_modal:
            config = DatasetBandRegistry.get_dataset_config(name)
            assert isinstance(config, MultiModalConfig)
            assert len(config.modalities) > 1
            assert len(config.band_to_modality) > 0
            assert all(
                isinstance(m, ModalityConfig) for m in config.modalities.values()
            )

    def test_invalid_dataset(self):
        with pytest.raises(AttributeError):
            DatasetBandRegistry.get_dataset_config("INVALID")
