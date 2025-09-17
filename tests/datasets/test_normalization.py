import pytest
import torch

from geobench_v2.datasets.normalization import SatMAENormalizer, ZScoreNormalizer


class TestSatMAENormalizer:
    """
    Tests the SatMAENormalizer class.

    Verifies correct SatMAE-style normalization (clipping to mean +/- 2*std, then scaling)
    and denormalization across different output ranges ([0,1], [0,255], [-1,1]) and
    input tensor dimensions (3D, 4D, 5D). Also tests that fill values specified in the
    band order are correctly ignored during both normalization and denormalization.
    """

    TEST_STATS = {"means": {"B1": 100.0, "B2": -10.0}, "stds": {"B1": 20.0, "B2": 5.0}}

    B1_MIN_CLIP = TEST_STATS["means"]["B1"] - 2 * TEST_STATS["stds"]["B1"]
    B1_MAX_CLIP = TEST_STATS["means"]["B1"] + 2 * TEST_STATS["stds"]["B1"]
    B2_MIN_CLIP = TEST_STATS["means"]["B2"] - 2 * TEST_STATS["stds"]["B2"]
    B2_MAX_CLIP = TEST_STATS["means"]["B2"] + 2 * TEST_STATS["stds"]["B2"]

    TEST_VALUES_B1 = [
        (40.0, B1_MIN_CLIP, 0.0),
        (60.0, B1_MIN_CLIP, 0.0),
        (100.0, 100.0, 0.5),
        (140.0, B1_MAX_CLIP, 1.0),
        (160.0, B1_MAX_CLIP, 1.0),
    ]
    TEST_VALUES_B2 = [
        (-30.0, B2_MIN_CLIP, 0.0),
        (-20.0, B2_MIN_CLIP, 0.0),
        (-10.0, -10.0, 0.5),
        (0.0, B2_MAX_CLIP, 1.0),
        (10.0, B2_MAX_CLIP, 1.0),
    ]

    INPUT_VALS = torch.tensor(
        [[v[0] for v in TEST_VALUES_B1], [v[0] for v in TEST_VALUES_B2]]
    )
    EXPECTED_CLIPPED = torch.tensor(
        [[v[1] for v in TEST_VALUES_B1], [v[1] for v in TEST_VALUES_B2]]
    )
    EXPECTED_NORM_0_1 = torch.tensor(
        [[v[2] for v in TEST_VALUES_B1], [v[2] for v in TEST_VALUES_B2]]
    )

    @staticmethod
    def scale_to_range(tensor_0_1, output_range):
        """Helper to scale tensor from [0, 1] to target range."""
        if output_range == "zero_255":
            return tensor_0_1 * 255.0
        elif output_range == "neg_one_one":
            return tensor_0_1 * 2.0 - 1.0
        else:
            return tensor_0_1

    @pytest.fixture
    def stats(self):
        """Return the test statistics."""
        return self.TEST_STATS.copy()

    @pytest.mark.parametrize("output_range", ["zero_one", "zero_255", "neg_one_one"])
    @pytest.mark.parametrize(
        "tensor_dim, shape_template, permute_dims",
        [
            (3, (2, 5, 1, 1), (1, 0, 2, 3)),
            (4, (2, 2, 1, 5), None),
            (5, (2, 2, 2, 1, 5), None),
        ],
    )
    def test_normalization_values(
        self, stats, output_range, tensor_dim, shape_template, permute_dims
    ):
        """Verify exact normalized values for known inputs across dimensions and ranges."""
        normalizer = SatMAENormalizer(stats, ["B1", "B2"], output_range=output_range)

        input_base = self.INPUT_VALS.clone()
        expected_base = self.scale_to_range(
            self.EXPECTED_NORM_0_1.clone(), output_range
        )

        if tensor_dim == 3:
            input_reshaped = input_base.unsqueeze(-1).unsqueeze(-1)
            expected_reshaped = expected_base.unsqueeze(-1).unsqueeze(-1)
            test_tensor = input_reshaped.permute(permute_dims)
            expected_tensor = expected_reshaped.permute(permute_dims)
        elif tensor_dim == 4:
            input_reshaped = input_base.unsqueeze(0).unsqueeze(2)
            expected_reshaped = expected_base.unsqueeze(0).unsqueeze(2)
            test_tensor = input_reshaped.expand(shape_template)
            expected_tensor = expected_reshaped.expand(shape_template)
        elif tensor_dim == 5:
            input_reshaped = input_base.unsqueeze(0).unsqueeze(0).unsqueeze(3)
            expected_reshaped = expected_base.unsqueeze(0).unsqueeze(0).unsqueeze(3)
            test_tensor = input_reshaped.expand(shape_template)
            expected_tensor = expected_reshaped.expand(shape_template)
        else:
            pytest.fail(f"Unsupported tensor_dim: {tensor_dim}")

        result = normalizer({"image": test_tensor})
        normalized_tensor = result["image"]

        assert normalized_tensor.shape == expected_tensor.shape
        assert torch.allclose(normalized_tensor, expected_tensor, atol=1e-6), (
            f"Normalization failed for dim={tensor_dim}, range={output_range}"
        )

    @pytest.mark.parametrize("output_range", ["zero_one", "zero_255", "neg_one_one"])
    def test_denormalization_roundtrip(self, stats, output_range):
        """Verify denormalization returns the *clipped* original values."""
        normalizer = SatMAENormalizer(stats, ["B1", "B2"], output_range=output_range)

        input_tensor = self.INPUT_VALS.unsqueeze(-1).unsqueeze(-1)
        expected_roundtrip = self.EXPECTED_CLIPPED.unsqueeze(-1).unsqueeze(-1)

        test_tensor = input_tensor.permute(1, 0, 2, 3)
        expected_tensor = expected_roundtrip.permute(1, 0, 2, 3)

        normalized_result = normalizer({"image": test_tensor})
        denormalized_result = normalizer.unnormalize(normalized_result)
        denormalized_tensor = denormalized_result["image"]

        tolerance = 1e-5

        if not torch.allclose(denormalized_tensor, expected_tensor, atol=tolerance):
            print(f"\nDenorm roundtrip failed for range={output_range}")
            print("Expected (clipped):")
            print(expected_tensor.squeeze())
            print("Actual (denormalized):")
            print(denormalized_tensor.squeeze())
        assert torch.allclose(denormalized_tensor, expected_tensor, atol=tolerance), (
            f"Denormalization roundtrip failed for range={output_range}"
        )

    def test_fill_values_ignored(self, stats):
        """Verify that fill values in band_order are ignored during norm/denorm."""
        fill_value = -999.0
        normalizer = SatMAENormalizer(
            stats, ["B1", fill_value, "B2"], output_range="zero_one"
        )

        input_vals = (
            torch.tensor(
                [
                    [self.TEST_STATS["means"]["B1"]],
                    [fill_value],
                    [self.TEST_STATS["means"]["B2"]],
                ]
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        test_tensor = input_vals.permute(1, 0, 2, 3)

        normalized_result = normalizer({"image": test_tensor})
        normalized_tensor = normalized_result["image"]

        expected_norm = torch.tensor([[[[0.5]], [[fill_value]], [[0.5]]]])

        if not torch.allclose(normalized_tensor, expected_norm, atol=1e-6):
            print("\nFill value norm failed")
            print("Expected:")
            print(expected_norm)
            print("Actual:")
            print(normalized_tensor)
        assert torch.allclose(normalized_tensor, expected_norm, atol=1e-6), (
            "Fill value was altered during normalization"
        )

        denormalized_result = normalizer.unnormalize(normalized_result)
        denormalized_tensor = denormalized_result["image"]

        expected_denorm = test_tensor.clone()

        if not torch.allclose(denormalized_tensor, expected_denorm, atol=1e-6):
            print("\nFill value denorm failed")
            print("Expected:")
            print(expected_denorm)
            print("Actual:")
            print(denormalized_tensor)
        assert torch.allclose(denormalized_tensor, expected_denorm, atol=1e-6), (
            "Fill value was altered during denormalization"
        )

    def test_second_stage_normalization(self, stats):
        """Test second-stage normalization (ImageNet-style)."""
        normalizer = SatMAENormalizer(
            stats, ["B1", "B2"], output_range="zero_one", apply_second_stage=False
        )

        stats_with_norm = stats.copy()
        stats_with_norm["norm_mean"] = {"B1": 0.5, "B2": 0.5}
        stats_with_norm["norm_std"] = {"B1": 0.25, "B2": 0.25}

        normalizer_with_stats = SatMAENormalizer(
            stats_with_norm,
            ["B1", "B2"],
            output_range="zero_one",
            apply_second_stage=True,
        )

        # Correct shape: (B, T, C, H, W) -> (1, 1, 2, 1, 1)
        input_tensor = torch.tensor([100.0, -10.0]).view(1, 1, 2, 1, 1)

        result1 = normalizer({"image": input_tensor})
        result2 = normalizer_with_stats({"image": input_tensor})

        assert abs(result1["image"][0, 0, 0, 0, 0].item() - 0.5) < 1e-5
        assert abs(result2["image"][0, 0, 0, 0, 0].item() - 0.0) < 1e-5

        roundtrip1 = normalizer.unnormalize(result1)
        roundtrip2 = normalizer_with_stats.unnormalize(result2)

        assert (
            abs(
                roundtrip1["image"][0, 0, 0, 0, 0].item()
                - self.TEST_STATS["means"]["B1"]
            )
            < 1e-5
        )
        assert (
            abs(
                roundtrip2["image"][0, 0, 0, 0, 0].item()
                - self.TEST_STATS["means"]["B1"]
            )
            < 1e-5
        )


class TestZScoreNormalizer:
    """
    Tests the ZScoreNormalizer class with sequential clip then z-score logic.

    Verifies correct normalization (clip then z-score) and denormalization (inverse z-score only)
    behavior, handling of fill values, and both single-tensor and multi-modal dictionary inputs.
    """

    TEST_STATS = {
        "means": {"B1": 100.0, "B2": -10.0, "B3": 50.0},
        "stds": {"B1": 20.0, "B2": 5.0, "B3": 10.0},
        # clip_* present but ignored by ZScoreNormalizer
        "clip_min": {"B1": 70.0},
        "clip_max": {"B1": 130.0},
    }

    TEST_VALUES_B1_IN = [60.0, 70.0, 100.0, 130.0, 140.0]
    TEST_VALUES_B2_IN = [-20.0, -15.0, -10.0, -5.0, 0.0]
    TEST_VALUES_B3_IN = [30.0, 40.0, 50.0, 60.0, 70.0]

    # Pure z-score (no clipping) -> (x - mean)/std
    TEST_VALUES_B1_NORM = [-2.0, -1.5, 0.0, 1.5, 2.0]
    TEST_VALUES_B2_NORM = [-2.0, -1.0, 0.0, 1.0, 2.0]
    TEST_VALUES_B3_NORM = [-2.0, -1.0, 0.0, 1.0, 2.0]

    # Roundtrip should recover ORIGINAL (not clipped) values
    TEST_VALUES_B1_ROUNDTRIP = TEST_VALUES_B1_IN
    TEST_VALUES_B2_ROUNDTRIP = TEST_VALUES_B2_IN
    TEST_VALUES_B3_ROUNDTRIP = TEST_VALUES_B3_IN

    FILL_VALUE = -999.0

    INPUT_VALS = torch.tensor(
        [TEST_VALUES_B1_IN, TEST_VALUES_B2_IN, [FILL_VALUE] * 5, TEST_VALUES_B3_IN]
    )

    EXPECTED_NORM = torch.tensor(
        [
            TEST_VALUES_B1_NORM,
            TEST_VALUES_B2_NORM,
            [FILL_VALUE] * 5,
            TEST_VALUES_B3_NORM,
        ]
    )

    EXPECTED_ROUNDTRIP = torch.tensor(
        [
            TEST_VALUES_B1_ROUNDTRIP,
            TEST_VALUES_B2_ROUNDTRIP,
            [FILL_VALUE] * 5,
            TEST_VALUES_B3_ROUNDTRIP,
        ]
    )

    BAND_ORDER = ["B1", "B2", FILL_VALUE, "B3"]

    @pytest.fixture
    def stats(self):
        """Return the test statistics."""
        return self.TEST_STATS.copy()

    def test_normalization_values(self, stats):
        """Verify normalization applies clip then z-score correctly per band."""
        normalizer = ZScoreNormalizer(stats, self.BAND_ORDER)

        input_tensor = self.INPUT_VALS.unsqueeze(-1).unsqueeze(-1)
        test_tensor = input_tensor.permute(1, 0, 2, 3)
        # Use the pre-calculated EXPECTED_NORM
        expected_tensor = (
            self.EXPECTED_NORM.unsqueeze(-1).unsqueeze(-1).permute(1, 0, 2, 3)
        )

        result = normalizer({"image": test_tensor})
        normalized_tensor = result["image"]

        assert normalized_tensor.shape == expected_tensor.shape
        assert torch.allclose(normalized_tensor, expected_tensor, atol=1e-5), (
            "Normalization failed"
        )

    def test_denormalization_roundtrip(self, stats):
        """Verify denormalization reverses only the z-score step."""
        normalizer = ZScoreNormalizer(stats, self.BAND_ORDER)

        input_tensor = self.INPUT_VALS.unsqueeze(-1).unsqueeze(-1)
        test_tensor = input_tensor.permute(1, 0, 2, 3)

        # Use the pre-calculated EXPECTED_ROUNDTRIP (clipped values)
        expected_tensor = (
            self.EXPECTED_ROUNDTRIP.unsqueeze(-1).unsqueeze(-1).permute(1, 0, 2, 3)
        )

        normalized_result = normalizer({"image": test_tensor})
        denormalized_result = normalizer.unnormalize(normalized_result)
        denormalized_tensor = denormalized_result["image"]

        tolerance = 1e-5

        if not torch.allclose(denormalized_tensor, expected_tensor, atol=tolerance):
            print("\nMultiModal Denorm roundtrip failed")
            print("Input (Original):")
            print(test_tensor.squeeze())
            print("Expected (Roundtrip - Clipped):")
            print(expected_tensor.squeeze())
            print("Actual (Denormalized):")
            print(denormalized_tensor.squeeze())
            print("Difference:")
            print((denormalized_tensor - expected_tensor).squeeze())

        assert torch.allclose(denormalized_tensor, expected_tensor, atol=tolerance), (
            "MultiModal Denormalization roundtrip failed"
        )

    def test_multimodal_input_dict(self, stats):
        """Test sequential normalization and denormalization with dict input."""
        band_order_dict = {"mod1": ["B1", self.FILL_VALUE], "mod2": ["B2", "B3"]}
        normalizer = ZScoreNormalizer(stats, band_order_dict)

        input_mod1 = (
            torch.tensor([[self.TEST_VALUES_B1_IN[0]], [self.FILL_VALUE]])
            .unsqueeze(-1)
            .unsqueeze(-1)
            .permute(1, 0, 2, 3)
        )
        input_mod2 = (
            torch.tensor([[self.TEST_VALUES_B2_IN[0]], [self.TEST_VALUES_B3_IN[0]]])
            .unsqueeze(-1)
            .unsqueeze(-1)
            .permute(1, 0, 2, 3)
        )

        input_data = {"image_mod1": input_mod1, "image_mod2": input_mod2}

        expected_mod1 = torch.tensor(
            [[[[self.TEST_VALUES_B1_NORM[0]]], [[self.FILL_VALUE]]]]
        )
        expected_mod2 = torch.tensor(
            [[[[self.TEST_VALUES_B2_NORM[0]]], [[self.TEST_VALUES_B3_NORM[0]]]]]
        )

        normalized_result = normalizer(input_data)
        assert torch.allclose(normalized_result["image_mod1"], expected_mod1, atol=1e-6)
        assert torch.allclose(normalized_result["image_mod2"], expected_mod2, atol=1e-6)

        denormalized_result = normalizer.unnormalize(normalized_result)

        # Roundtrip returns original (no clipping)
        expected_roundtrip_mod1 = torch.tensor(
            [[[[self.TEST_VALUES_B1_IN[0]]], [[self.FILL_VALUE]]]]
        )
        expected_roundtrip_mod2 = torch.tensor(
            [[[[self.TEST_VALUES_B2_IN[0]]], [[self.TEST_VALUES_B3_IN[0]]]]]
        )

        assert torch.allclose(
            denormalized_result["image_mod1"], expected_roundtrip_mod1, atol=1e-5
        )
        assert torch.allclose(
            denormalized_result["image_mod2"], expected_roundtrip_mod2, atol=1e-5
        )
