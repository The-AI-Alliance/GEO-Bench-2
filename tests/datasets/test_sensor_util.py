import pytest

from geobench_v2.datasets.sensor_util import (
    BandConfig,
    DatasetBandRegistry,
    ModalityConfig,
    MultiModalConfig,
    SensorBandRegistry,
)


class TestBandConfig:
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
    def test_rgb_config(self):
        config = SensorBandRegistry.RGB
        assert len(config.bands) == 3
        assert "red" in config.bands
        assert config.bands["red"].wavelength == 0.665
        assert config.default_order == ["red", "green", "blue"]
        assert config.native_resolution is None

    def test_rgbn_config(self):
        config = SensorBandRegistry.RGBN
        assert len(config.bands) == 4
        assert all(b in config.bands for b in ["red", "green", "blue", "nir"])
        assert config.bands["nir"].wavelength == 0.842
        assert config.default_order == ["red", "green", "blue", "nir"]

    def test_sentinel2_config(self):
        config = SensorBandRegistry.SENTINEL2
        # Includes B10 -> 13 bands
        assert len(config.bands) == 13
        assert config.bands["B02"].resolution == 10
        assert config.bands["B05"].resolution == 20
        assert config.bands["B01"].resolution == 60
        assert config.bands["B04"].wavelength == 0.665
        assert config.bands["B08"].wavelength == 0.842
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
    def test_benv2_config(self):
        config = DatasetBandRegistry.BENV2
        assert set(config.modalities.keys()) == {"s1", "s2"}
        assert config.band_to_modality["B02"] == "s2"
        assert config.band_to_modality["VV"] == "s1"
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
        assert all(config.band_to_modality[b] == "s1" for b in ["VV", "VH"])

    def test_pastis_config(self):
        config = DatasetBandRegistry.PASTIS
        assert set(config.modalities.keys()) == {"s2", "s1_asc", "s1_desc"}
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
        assert config.band_to_modality["VV_asc"] == "s1_asc"
        assert config.band_to_modality["VH_asc"] == "s1_asc"
        assert config.band_to_modality["VV/VH_asc"] == "s1_asc"
        assert config.band_to_modality["VV_desc"] == "s1_desc"
        assert config.band_to_modality["VH_desc"] == "s1_desc"
        assert config.band_to_modality["VV/VH_desc"] == "s1_desc"


class TestBandResolution:
    @pytest.mark.parametrize(
        "modality,band_spec,expected",
        [
            ("s2", "B02", "B02"),
            ("s2", "blue", "B02"),
            ("s2", "b02", "B02"),
            ("s2", "B04", "B04"),
            ("s2", "red", "B04"),
            ("s1", "VV", "VV"),
            ("s1", "co_pol", "VV"),
            ("s1", "VH", "VH"),
            ("s1", "cross_pol", "VH"),
            ("rgb", "red", "red"),
            ("rgb", "RED", "red"),
            ("rgb", "green", "green"),
            ("rgb", "GREEN", "green"),
        ],
    )
    def test_band_resolution_by_modality(self, modality, band_spec, expected):
        config = SensorBandRegistry.get_modality_config(modality)
        result = config.resolve_band(band_spec)
        assert result == expected

    def test_invalid_band_resolution(self):
        config = SensorBandRegistry.RGB
        assert config.resolve_band("invalid_band") is None
        config = SensorBandRegistry.SENTINEL2
        assert config.resolve_band("not_a_band") is None
        assert config.resolve_band("b99") is None

    def test_band_resolution_case_sensitivity(self):
        config = SensorBandRegistry.SENTINEL2
        assert config.resolve_band("B02") == "B02"
        assert config.resolve_band("b02") == "B02"
        assert config.resolve_band("blue") == "B02"
        rgb_config = SensorBandRegistry.RGB
        assert rgb_config.resolve_band("red") == "red"
        assert rgb_config.resolve_band("RED") == "red"

    @pytest.mark.parametrize(
        "dataset_name,modality,band_name,expected",
        [
            ("BENV2", "s2", "B02", "B02"),
            ("BENV2", "s2", "blue", "B02"),
            ("BENV2", "s1", "VV", "VV"),
            ("BENV2", "s1", "co_pol", "VV"),
            ("PASTIS", "s2", "B02", "B02"),
            ("PASTIS", "s1_asc", "VV_asc", "VV_asc"),
            ("PASTIS", "s1_desc", "VH_desc", "VH_desc"),
            ("EVERWATCH", "self", "red", "red"),
            ("FOTW", "self", "nir", "nir"),
        ],
    )
    def test_band_resolution_across_datasets(
        self, dataset_name, modality, band_name, expected
    ):
        dataset_config = getattr(DatasetBandRegistry, dataset_name)
        mod_config = dataset_config.modalities[modality]
        assert mod_config.resolve_band(band_name) == expected


class TestDatasetRegistry:
    def test_single_modality_datasets(self):
        single_modal = [
            ("CAFFE", ["gray"]),
            ("EVERWATCH", ["red", "green", "blue"]),
            ("FOTW", ["red", "green", "blue", "nir"]),
        ]
        for name, expected_bands in single_modal:
            config = getattr(DatasetBandRegistry, name)
            assert isinstance(config, ModalityConfig)
            assert config.default_order == expected_bands
            assert "self" in config.modalities
            assert config.modalities["self"] is config

    def test_multi_modal_datasets(self):
        multi_modal = {"BENV2": {"s1", "s2"}, "PASTIS": {"s2", "s1_asc", "s1_desc"}}
        for name, expected_modalities in multi_modal.items():
            config = getattr(DatasetBandRegistry, name)
            assert isinstance(config, MultiModalConfig)
            assert set(config.modalities.keys()) == expected_modalities
            assert len(config.band_to_modality) > 0
            for modality, bands in config.modalities.items():
                for band in bands.bands:
                    assert config.band_to_modality[band] == modality
                # default_bands = [b for b in config.default_order if isinstance(b, str)]
                # for band in default_bands:
                #     if band not in config.band_to_modality:
                #         import pdb
                #         pdb.set_trace()
                assert band in config.band_to_modality

    @pytest.mark.parametrize(
        "dataset_name,band_count",
        [
            ("CAFFE", 1),
            ("EVERWATCH", 3),
            ("FOTW", 4),
            # Adjust these if dataset definitions include/exclude B10 or subsets:
            ("BENV2", 14),  # S1(2) + S2(12)
            ("PASTIS", 16),  # As originally (depends on subset definition)
        ],
    )
    def test_dataset_band_counts(self, dataset_name, band_count):
        config = getattr(DatasetBandRegistry, dataset_name)
        unique_bands = 0
        if isinstance(config, MultiModalConfig):
            for mod_config in config.modalities.values():
                unique_bands += len(mod_config.bands)
        else:
            unique_bands = len(config.bands)
        assert unique_bands == band_count

    @pytest.mark.parametrize(
        "dataset_name,expected_resolution",
        [("EVERWATCH", None), ("BENV2", 10), ("PASTIS", 10)],
    )
    def test_dataset_resolution(self, dataset_name, expected_resolution):
        config = getattr(DatasetBandRegistry, dataset_name)
        if isinstance(config, MultiModalConfig):
            for mod_config in config.modalities.values():
                assert mod_config.native_resolution == expected_resolution
        else:
            assert config.native_resolution == expected_resolution
