import pytest
import torch
from geobench_v2.datasets.data_util import DataUtilsMixin, MultiModalNormalizer
from geobench_v2.datasets.sensor_util import (
    ModalityConfig,
    MultiModalConfig,
    BandConfig,
    DatasetBandRegistry,
    SensorBandRegistry,
)


class MockMultiModal_S1_S2(MultiModalConfig):
    """S1+S2 configuration testing standard satellite combination."""

    def __init__(self):
        super().__init__(
            modalities={
                "s2": SensorBandRegistry.SENTINEL2,
                "s1": SensorBandRegistry.SENTINEL1,
            },
            default_order=["B02", "B03", "B04", "VV", "VH"],
            band_to_modality={
                "B01": "s2",
                "B02": "s2",
                "B03": "s2",
                "B04": "s2",
                "B05": "s2",
                "B06": "s2",
                "B07": "s2",
                "B08": "s2",
                "B8A": "s2",
                "B09": "s2",
                "B11": "s2",
                "B12": "s2",
                "VV": "s1",
                "VH": "s1",
            },
        )


class MockMultiModal_S2_RGB(MultiModalConfig):
    """S2+RGB configuration testing overlapping band names."""

    def __init__(self):
        super().__init__(
            modalities={
                "s2": SensorBandRegistry.SENTINEL2,
                "rgb": SensorBandRegistry.RGB,
            },
            default_order=["B02", "B03", "B04", "r", "g", "b"],
            band_to_modality={
                # S2 bands with standard names
                "B02": "s2",
                "B03": "s2",
                "B04": "s2",
                # RGB bands with aliases
                "r": "rgb",
                "red": "rgb",
                "g": "rgb",
                "green": "rgb",
                "b": "rgb",
                "blue": "rgb",
            },
        )


class MockMultiModal_AllSensors(MultiModalConfig):
    """Complex configuration with all sensor types."""

    def __init__(self):
        super().__init__(
            modalities={
                "s2": SensorBandRegistry.SENTINEL2,
                "s1": SensorBandRegistry.SENTINEL1,
                "rgb": SensorBandRegistry.RGB,
                "rgbn": SensorBandRegistry.RGBN,
            },
            default_order=["B02", "r", "VV", "nir"],
            band_to_modality={
                # S2 bands
                "B02": "s2",
                "blue": "s2",
                "B03": "s2",
                "green": "s2",
                "B04": "s2",
                "red": "s2",
                # S1 bands
                "VV": "s1",
                "VH": "s1",
                # RGB bands
                "r": "rgb",
                "g": "rgb",
                "b": "rgb",
                # RGBN specific
                "nir": "rgbn",
            },
        )


class TestDatasetS1S2:
    """Test cases for S1+S2 combination."""

    @pytest.fixture
    def s1s2_dataset(self):
        class Dataset(DataUtilsMixin):
            dataset_band_config = MockMultiModal_S1_S2()

        return Dataset()

    @pytest.fixture
    def s1s2_data(self):
        """Create enumerated test data for each band."""
        s2_data = torch.arange(12).float().reshape(12, 1, 1).expand(-1, 32, 32)
        s1_data = torch.arange(2).float().reshape(2, 1, 1).expand(-1, 32, 32) + 100
        return {
            "s2": s2_data,  # Bands 0-11 for S2
            "s1": s1_data,  # Bands 100-101 for S1
        }

    def test_band_resolution(self, s1s2_dataset):
        """Test basic band resolution."""
        resolved = s1s2_dataset.resolve_band_order(["B02", "VV"])
        assert resolved == ["B02", "VV"]

        # Test with aliases
        resolved = s1s2_dataset.resolve_band_order(["b02", "co_pol"])
        assert resolved == ["B02", "VV"]

    def test_band_rearrangement(self, s1s2_dataset, s1s2_data):
        """Test band selection and ordering."""
        result = s1s2_dataset.rearrange_bands(s1s2_data, ["B02", "VV", "B04"])
        expected = torch.stack(
            [
                s1s2_data["s2"][1],  # B02 = 1
                s1s2_data["s1"][0],  # VV = 100
                s1s2_data["s2"][3],  # B04 = 3
            ]
        )
        assert torch.allclose(result["image"], expected)

    def test_modality_dict_output(self, s1s2_dataset, s1s2_data):
        """Test dictionary output mode."""
        result = s1s2_dataset.rearrange_bands(
            s1s2_data, {"s2": ["B02", "B03"], "s1": ["VV", "VH"]}
        )
        assert torch.allclose(result["image_s2"], s1s2_data["s2"][[1, 2]])
        assert torch.allclose(result["image_s1"], s1s2_data["s1"])


class TestDatasetS2RGB:
    """Test cases for S2+RGB combination with overlapping bands."""

    @pytest.fixture
    def s2rgb_dataset(self):
        class Dataset(DataUtilsMixin):
            dataset_band_config = MockMultiModal_S2_RGB()

        return Dataset()

    @pytest.fixture
    def s2rgb_data(self):
        """Create enumerated test data."""
        s2_data = torch.arange(12).float().reshape(12, 1, 1).expand(-1, 32, 32)
        rgb_data = torch.arange(3).float().reshape(3, 1, 1).expand(-1, 32, 32) + 200
        return {
            "s2": s2_data,  # Bands 0-11
            "rgb": rgb_data,  # Bands 200-202
        }

    def test_overlapping_bands(self, s2rgb_dataset, s2rgb_data):
        """Test handling of overlapping band names."""
        # Using RGB names should select from RGB modality
        result = s2rgb_dataset.rearrange_bands(s2rgb_data, ["r", "g", "b"])
        expected = s2rgb_data["rgb"]
        assert torch.allclose(result["image"], expected)

        # Using S2 names should select from S2
        result = s2rgb_dataset.rearrange_bands(s2rgb_data, ["B02", "B03", "B04"])
        expected = s2rgb_data["s2"][[1, 2, 3]]
        assert torch.allclose(result["image"], expected)


class TestAllSensors:
    """Test cases for complex configuration with all sensors."""

    @pytest.fixture
    def all_sensors_dataset(self):
        class Dataset(DataUtilsMixin):
            dataset_band_config = MockMultiModal_AllSensors()

        return Dataset()

    @pytest.fixture
    def all_sensors_data(self):
        """Create enumerated test data."""
        return {
            "s2": torch.arange(12).float().reshape(12, 1, 1).expand(-1, 32, 32),
            "s1": torch.arange(2).float().reshape(2, 1, 1).expand(-1, 32, 32) + 100,
            "rgb": torch.arange(3).float().reshape(3, 1, 1).expand(-1, 32, 32) + 200,
            "rgbn": torch.arange(4).float().reshape(4, 1, 1).expand(-1, 32, 32) + 300,
        }

    def test_mixed_band_selection(self, all_sensors_dataset, all_sensors_data):
        """Test selecting bands across all modalities."""
        result = all_sensors_dataset.rearrange_bands(
            all_sensors_data, ["B02", "r", "VV", "nir"]
        )
        expected = torch.stack(
            [
                all_sensors_data["s2"][1],  # B02 = 1
                all_sensors_data["rgb"][0],  # r = 200
                all_sensors_data["s1"][0],  # VV = 100
                all_sensors_data["rgbn"][3],  # nir = 303
            ]
        )
        assert torch.allclose(result["image"], expected)

    def test_fill_values(self, all_sensors_dataset, all_sensors_data):
        """Test mixing fill values with real bands."""
        result = all_sensors_dataset.rearrange_bands(
            all_sensors_data, ["B02", 0.0, "VV", -999.0, "nir"]
        )["image"]
        assert result.shape == (5, 32, 32)
        assert torch.allclose(result[1], torch.zeros_like(result[1]))
        assert torch.allclose(result[3], torch.full_like(result[3], -999.0))

    @pytest.mark.parametrize(
        "invalid_input",
        [
            ["invalid_band"],
            ["B02_invalid"],
            ["rgb_B02"],  # Wrong modality prefix
            ["VV", "invalid", "B02"],
        ],
    )
    def test_invalid_inputs(self, all_sensors_dataset, all_sensors_data, invalid_input):
        """Test various invalid input combinations."""
        with pytest.raises(ValueError):
            all_sensors_dataset.rearrange_bands(all_sensors_data, invalid_input)

    def test_multimodal_dict_output(self, all_sensors_dataset, all_sensors_data):
        """Test dictionary output with multiple modalities."""
        result = all_sensors_dataset.rearrange_bands(
            all_sensors_data,
            {
                "s2": ["B02", "B03", 0.0],
                "s1": ["VV", -1.0, "VH"],
                "rgb": ["r", "g", "b"],
                "rgbn": ["nir"],
            },
        )

        # Check all modalities are present
        assert set(result.keys()) == {"image_s2", "image_s1", "image_rgb", "image_rgbn"}

        # Check dimensions
        assert result["image_s2"].shape == (3, 32, 32)
        assert result["image_s1"].shape == (3, 32, 32)
        assert result["image_rgb"].shape == (3, 32, 32)
        assert result["image_rgbn"].shape == (1, 32, 32)

        # Check values including fill values
        assert torch.allclose(result["image_s2"][0:2], all_sensors_data["s2"][[1, 2]])
        assert torch.allclose(
            result["image_s2"][2], torch.zeros_like(result["image_s2"][2])
        )
        assert torch.allclose(result["image_s1"][[0, 2]], all_sensors_data["s1"])
        assert torch.allclose(
            result["image_s1"][1], torch.full_like(result["image_s1"][1], -1.0)
        )
        assert torch.allclose(result["image_rgb"], all_sensors_data["rgb"])
        assert torch.allclose(result["image_rgbn"], all_sensors_data["rgbn"][3:4])

    @pytest.mark.parametrize(
        "invalid_dict",
        [
            {"invalid_modality": ["B02"]},
            {"s2": ["invalid_band"]},
            {"s1": ["VV"], "s2": ["invalid"]},
            {"rgb": ["r", "invalid", "b"]},
        ],
    )
    def test_invalid_multimodal_dict(
        self, all_sensors_dataset, all_sensors_data, invalid_dict
    ):
        """Test error handling for invalid dictionary configurations."""
        with pytest.raises(ValueError):
            all_sensors_dataset.rearrange_bands(all_sensors_data, invalid_dict)

    def test_empty_modality_sequence(self, all_sensors_dataset, all_sensors_data):
        """Test proper error handling for empty band sequences."""
        with pytest.raises(
            ValueError, match="Empty band sequence provided for modality s1"
        ):
            all_sensors_dataset.rearrange_bands(
                all_sensors_data, {"s2": ["B02", "B03"], "s1": [], "rgb": ["r"]}
            )


class TestMultiModalNormalizer:
    """Test cases for MultiModalNormalizer functionality."""

    @pytest.fixture
    def normalization_stats(self):
        """Create test normalization statistics."""
        return {
            "means": {
                "B02": 1000.0,
                "B03": 1200.0,
                "B04": 1400.0,
                "VV": -10.0,
                "VH": -15.0,
                "r": 128.0,
                "g": 127.0,
                "b": 126.0,
                "nir": 2000.0,
            },
            "stds": {
                "B02": 500.0,
                "B03": 600.0,
                "B04": 700.0,
                "VV": 5.0,
                "VH": 3.0,
                "r": 64.0,
                "g": 63.0,
                "b": 62.0,
                "nir": 1000.0,
            },
        }

    def test_single_tensor_normalization(self, normalization_stats):
        """Test normalization of single tensor data."""
        # Create test data: mean-std, mean, mean+std for each band
        data = torch.zeros(3, 4, 8, 8)

        # B02, B03, fill=0, B04
        data[0] = torch.tensor([500.0, 600.0, 0.0, 700.0]).reshape(4, 1, 1)
        data[1] = torch.tensor([1000.0, 1200.0, 0.0, 1400.0]).reshape(4, 1, 1)
        data[2] = torch.tensor([1500.0, 1800.0, 0.0, 2100.0]).reshape(4, 1, 1)
        data = data.expand(-1, -1, 8, 8)

        # Expected: -1, 0, 1 for each normalized band, 0 for fill
        expected = (
            torch.tensor(
                [[-1.0, -1.0, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 1.0]]
            )
            .reshape(3, 4, 1, 1)
            .expand(-1, -1, 8, 8)
        )

        normalizer = MultiModalNormalizer(
            normalization_stats, ["B02", "B03", 0.0, "B04"]
        )
        result = normalizer({"image": data})

        assert torch.allclose(result["image"], expected, rtol=1e-5)

    def test_multimodal_normalization(self, normalization_stats):
        """Test normalization of multi-modal data."""
        # Create test data for two modalities
        s2_data = torch.zeros(3, 3, 8, 8)
        s1_data = torch.zeros(3, 3, 8, 8)

        # B02, B03, fill values for S2
        s2_values = (
            torch.tensor(
                [
                    [500.0, 600.0, 0.0],  # mean-std
                    [1000.0, 1200.0, 0.0],  # mean
                    [1500.0, 1800.0, 0.0],  # mean+std
                ]
            )
            .reshape(3, 3, 1, 1)
            .expand(-1, -1, 8, 8)
        )
        s2_data.copy_(s2_values)

        # VV, VH, fill values for S1
        s1_values = (
            torch.tensor(
                [
                    [-15.0, -18.0, -999.0],  # mean-std
                    [-10.0, -15.0, -999.0],  # mean
                    [-5.0, -12.0, -999.0],  # mean+std
                ]
            )
            .reshape(3, 3, 1, 1)
            .expand(-1, -1, 8, 8)
        )
        s1_data.copy_(s1_values)

        # Expected normalized values: -1, 0, 1 for real bands
        expected_s2 = (
            torch.tensor([[-1.0, -1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
            .reshape(3, 3, 1, 1)
            .expand(-1, -1, 8, 8)
        )

        expected_s1 = (
            torch.tensor([[-1.0, -1.0, -999.0], [0.0, 0.0, -999.0], [1.0, 1.0, -999.0]])
            .reshape(3, 3, 1, 1)
            .expand(-1, -1, 8, 8)
        )

        band_order = {"s2": ["B02", "B03", 0.0], "s1": ["VV", "VH", -999.0]}
        normalizer = MultiModalNormalizer(normalization_stats, band_order)
        result = normalizer({"image_s2": s2_data, "image_s1": s1_data})

        assert torch.allclose(result["image_s2"], expected_s2, rtol=1e-5)
        assert torch.allclose(result["image_s1"], expected_s1, rtol=1e-5)

    def test_multimodal_unnormalize(self, normalization_stats):
        """Test that unnormalization is the inverse of normalization for unbatched data."""
        # Create test data for two modalities (unbatched: C×H×W)
        s2_data = torch.zeros(3, 8, 8)
        s1_data = torch.zeros(3, 8, 8)

        # B02, B03, fill values for S2
        s2_data[0, :, :] = 500.0  # B02: mean-std
        s2_data[1, :, :] = 600.0  # B03: mean-std
        s2_data[2, :, :] = 0.0  # fill value

        # Second sample at mean values
        second_s2 = torch.zeros(3, 8, 8)
        second_s2[0, :, :] = 1000.0  # B02: mean
        second_s2[1, :, :] = 1200.0  # B03: mean
        second_s2[2, :, :] = 0.0  # fill value

        # Third sample at mean+std
        third_s2 = torch.zeros(3, 8, 8)
        third_s2[0, :, :] = 1500.0  # B02: mean+std
        third_s2[1, :, :] = 1800.0  # B03: mean+std
        third_s2[2, :, :] = 0.0  # fill value

        # VV, VH, fill values for S1
        s1_data[0, :, :] = -15.0  # VV: mean-std
        s1_data[1, :, :] = -18.0  # VH: mean-std
        s1_data[2, :, :] = -999.0  # fill value

        # Second sample at mean
        second_s1 = torch.zeros(3, 8, 8)
        second_s1[0, :, :] = -10.0  # VV: mean
        second_s1[1, :, :] = -15.0  # VH: mean
        second_s1[2, :, :] = -999.0  # fill value

        # Third sample at mean+std
        third_s1 = torch.zeros(3, 8, 8)
        third_s1[0, :, :] = -5.0  # VV: mean+std
        third_s1[1, :, :] = -12.0  # VH: mean+std
        third_s1[2, :, :] = -999.0  # fill value

        band_order = {"s2": ["B02", "B03", 0.0], "s1": ["VV", "VH", -999.0]}
        normalizer = MultiModalNormalizer(normalization_stats, band_order)

        # Test the first set of samples (mean-std)
        original_s2_data = s2_data.clone()
        original_s1_data = s1_data.clone()

        # Normalize the data (unbatched)
        normalized = normalizer({"image_s2": s2_data, "image_s1": s1_data})
        unnormalized = normalizer.unnormalize(normalized)

        # Check that unnormalized data is equal to original data
        assert torch.allclose(unnormalized["image_s2"], original_s2_data, rtol=1e-5)
        assert torch.allclose(unnormalized["image_s1"], original_s1_data, rtol=1e-5)

        # Test the second set of samples (mean values)
        original_second_s2 = second_s2.clone()
        original_second_s1 = second_s1.clone()

        # Normalize
        normalized2 = normalizer({"image_s2": second_s2, "image_s1": second_s1})
        unnormalized2 = normalizer.unnormalize(normalized2)

        # Check unnormalized values match original
        assert torch.allclose(unnormalized2["image_s2"], original_second_s2, rtol=1e-5)
        assert torch.allclose(unnormalized2["image_s1"], original_second_s1, rtol=1e-5)

        # Test the third set of samples (mean+std)
        original_third_s2 = third_s2.clone()
        original_third_s1 = third_s1.clone()

        # Normalize
        normalized3 = normalizer({"image_s2": third_s2, "image_s1": third_s1})
        unnormalized3 = normalizer.unnormalize(normalized3)
        assert torch.allclose(unnormalized3["image_s2"], original_third_s2, rtol=1e-5)
        assert torch.allclose(unnormalized3["image_s1"], original_third_s1, rtol=1e-5)
