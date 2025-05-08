import pytest

from geobench_v2.datasets.sensor_util import (
    BandConfig,
    DatasetBandRegistry,
    ModalityConfig,
    MultiModalConfig,
    SensorBandRegistry,
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
        assert set(config.modalities.keys()) == {"s2", "s1_asc", "s1_desc"}
        # Test band mappings for different modalities
        s2_bands = [
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B11",
            "B12",
        ]
        assert all(config.band_to_modality[b] == "s2" for b in s2_bands)

        # Test S1 ascending/descending band mappings
        assert config.band_to_modality["VV_asc"] == "s1_asc"
        assert config.band_to_modality["VH_asc"] == "s1_asc"
        assert config.band_to_modality["VV/VH_asc"] == "s1_asc"
        assert config.band_to_modality["VV_desc"] == "s1_desc"
        assert config.band_to_modality["VH_desc"] == "s1_desc"
        assert config.band_to_modality["VV/VH_desc"] == "s1_desc"

        # Test default order includes bands from all modalities
        assert len(config.default_order) == len(s2_bands) + 6  # S2 bands + 6 S1 bands


class TestBandResolution:
    """Test band name resolution functionality."""

    @pytest.mark.parametrize(
        "modality,band_spec,expected",
        [
            # S2 bands and aliases
            ("s2", "B02", "B02"),
            ("s2", "blue", "B02"),
            ("s2", "b02", "B02"),
            ("s2", "B04", "B04"),
            ("s2", "red", "B04"),
            # S1 bands and aliases
            ("s1", "VV", "VV"),
            ("s1", "co_pol", "VV"),
            ("s1", "VH", "VH"),
            ("s1", "cross_pol", "VH"),
            # RGB bands and aliases
            ("rgb", "r", "r"),
            ("rgb", "RED", "r"),
            ("rgb", "g", "g"),
            ("rgb", "GREEN", "g"),
        ],
    )
    def test_band_resolution_by_modality(self, modality, band_spec, expected):
        """Test band resolution within specific modalities."""
        config = SensorBandRegistry.get_modality_config(modality)
        result = config.resolve_band(band_spec)
        assert result == expected

    def test_invalid_band_resolution(self):
        """Test that invalid band names return None."""
        config = SensorBandRegistry.RGB
        assert config.resolve_band("invalid_band") is None

        config = SensorBandRegistry.SENTINEL2
        assert config.resolve_band("not_a_band") is None
        assert config.resolve_band("b99") is None

    def test_band_resolution_case_sensitivity(self):
        """Test case sensitivity in band resolution."""
        config = SensorBandRegistry.SENTINEL2
        # Same band with different case formats
        assert config.resolve_band("B02") == "B02"
        assert config.resolve_band("b02") == "B02"
        assert config.resolve_band("blue") == "B02"

        # RGB traditionally has lowercase names
        rgb_config = SensorBandRegistry.RGB
        assert rgb_config.resolve_band("r") == "r"
        assert rgb_config.resolve_band("red") == "r"

    @pytest.mark.parametrize(
        "dataset_name,modality,band_name,expected",
        [
            # BENV2 dataset with different bands
            ("BENV2", "s2", "B02", "B02"),
            ("BENV2", "s2", "blue", "B02"),
            ("BENV2", "s1", "VV", "VV"),
            ("BENV2", "s1", "co_pol", "VV"),
            # PASTIS with separated S1 modalities
            ("PASTIS", "s2", "B02", "B02"),
            ("PASTIS", "s1_asc", "VV_asc", "VV_asc"),
            ("PASTIS", "s1_desc", "VH_desc", "VH_desc"),
            # RGB datasets
            ("EVERWATCH", "self", "r", "r"),
            ("FOTW", "self", "nir", "nir"),
        ],
    )
    def test_band_resolution_across_datasets(
        self, dataset_name, modality, band_name, expected
    ):
        """Test band resolution across different dataset configurations."""
        dataset_config = getattr(DatasetBandRegistry, dataset_name)
        mod_config = dataset_config.modalities[modality]
        assert mod_config.resolve_band(band_name) == expected


class TestDatasetRegistry:
    """Test dataset registry functionality."""

    def test_single_modality_datasets(self):
        """Test single modality dataset configurations."""
        single_modal = [
            ("CAFFE", ["gray"]),
            ("EVERWATCH", ["r", "g", "b"]),
            ("RESISC45", ["r", "g", "b"]),
            ("FOTW", ["r", "g", "b", "nir"]),
            ("SPACENET6", ["r", "g", "b", "nir"]),
        ]
        for name, expected_bands in single_modal:
            config = getattr(DatasetBandRegistry, name)
            assert isinstance(config, ModalityConfig)
            assert config.default_order == expected_bands
            assert "self" in config.modalities
            assert config.modalities["self"] is config  # Should point to itself

    def test_multi_modal_datasets(self):
        """Test multi-modal dataset configurations."""
        multi_modal = {"BENV2": {"s1", "s2"}, "PASTIS": {"s2", "s1_asc", "s1_desc"}}

        for name, expected_modalities in multi_modal.items():
            config = getattr(DatasetBandRegistry, name)
            assert isinstance(config, MultiModalConfig)
            assert set(config.modalities.keys()) == expected_modalities
            assert len(config.band_to_modality) > 0
            assert all(
                isinstance(m, ModalityConfig) for m in config.modalities.values()
            )

            # Get the actual default_order strings (skipping any non-string values)
            default_bands = [b for b in config.default_order if isinstance(b, str)]

            # Test that all string bands in default_order are in band_to_modality
            for band in default_bands:
                assert band in config.band_to_modality, (
                    f"Band {band} not in band_to_modality for {name}"
                )

    @pytest.mark.parametrize(
        "dataset_name,band_count",
        [
            ("CAFFE", 1),  # Grayscale - 1 band
            ("EVERWATCH", 3),  # RGB - 3 bands
            ("FOTW", 4),  # RGBN - 4 bands
            ("BENV2", 14),  # S1(2) + S2(12) = 14 bands
            ("PASTIS", 16),  # S2(10) + S1_asc(3) + S1_desc(3) = 16 bands
        ],
    )
    def test_dataset_band_counts(self, dataset_name, band_count):
        """Test that datasets have the correct number of bands."""
        config = getattr(DatasetBandRegistry, dataset_name)

        # Count unique bands across all modalities
        unique_bands = 0
        if isinstance(config, MultiModalConfig):
            for mod_config in config.modalities.values():
                unique_bands += len(mod_config.bands)
        else:
            unique_bands = len(config.bands)

        assert unique_bands == band_count

    @pytest.mark.parametrize(
        "dataset_name,expected_resolution",
        [
            ("EVERWATCH", None),  # RGB doesn't specify resolution
            ("BENV2", 10),  # BENV2 is 10m resolution
            ("PASTIS", 10),  # PASTIS is 10m resolution
        ],
    )
    def test_dataset_resolution(self, dataset_name, expected_resolution):
        """Test that datasets have the correct native resolution."""
        config = getattr(DatasetBandRegistry, dataset_name)

        if isinstance(config, MultiModalConfig):
            for mod_name, mod_config in config.modalities.items():
                assert mod_config.native_resolution == expected_resolution
        else:
            assert config.native_resolution == expected_resolution
