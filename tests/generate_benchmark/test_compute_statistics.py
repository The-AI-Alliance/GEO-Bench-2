# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Unit test for statistics computation."""

import tempfile

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from geobench_v2.generate_benchmark.compute_statistics import (
    ClassificationDatasetStatistics,
    SegmentationDatasetStatistics,
)


class MockBandRegistry:
    """Mock implementation of DatasetBandRegistry for testing."""

    def __init__(self, band_names=None):
        self.band_names = band_names or ["red", "green", "blue"]

    def get_band_names(self):
        return self.band_names


class MockClassificationDataset(Dataset):
    """Mock classification dataset with controlled class distribution."""

    def __init__(self, size=100, num_classes=3, class_weights=None, img_size=32):
        """Initialize mock classification dataset.

        Args:
            size: Total dataset size
            num_classes: Number of classes
            class_weights: Optional weights for class distribution (will be normalized)
            img_size: Size of generated images
        """
        self.size = size
        self.num_classes = num_classes

        if class_weights is None:
            class_weights = torch.ones(num_classes)
        else:
            class_weights = torch.tensor(class_weights, dtype=torch.float)

        class_weights = class_weights / class_weights.sum()

        self.expected_counts = (class_weights * size).round().long()

        diff = self.expected_counts.sum().item() - size
        if diff != 0:
            max_idx = self.expected_counts.argmax()
            self.expected_counts[max_idx] -= diff

        self.labels = []
        for c in range(num_classes):
            self.labels.extend([c] * self.expected_counts[c].item())

        self.images = torch.randn(size, 3, img_size, img_size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {"image": self.images[idx], "label": self.labels[idx]}


class MockSegmentationDataset(Dataset):
    """Mock segmentation dataset with controlled class distribution."""

    def __init__(self, size=20, num_classes=3, img_size=64, class_weights=None):
        """Initialize mock segmentation dataset.

        Args:
            size: Total dataset size
            num_classes: Number of classes
            img_size: Size of generated images
            class_weights: Optional weights for pixel distribution (will be normalized)
        """
        self.size = size
        self.num_classes = num_classes
        self.img_size = img_size

        if class_weights is None:
            class_weights = torch.ones(num_classes)
        else:
            class_weights = torch.tensor(class_weights, dtype=torch.float)

        class_weights = class_weights / class_weights.sum()

        self.images = torch.randn(size, 3, img_size, img_size)

        self.masks = []
        self.class_presence = torch.zeros(num_classes)

        for i in range(size):
            mask = torch.zeros(img_size, img_size, dtype=torch.long)
            present_in_this_mask = torch.zeros(num_classes, dtype=torch.bool)
            pixels_per_class = (class_weights * img_size * img_size).round().long()
            remaining_pixels = img_size * img_size

            pixel_idx = 0
            for c in range(num_classes):
                num_pixels = min(pixels_per_class[c].item(), remaining_pixels)

                if num_pixels > 0:
                    mask.view(-1)[pixel_idx : pixel_idx + num_pixels] = c
                    pixel_idx += num_pixels
                    remaining_pixels -= num_pixels
                    present_in_this_mask[c] = True

            flat_mask = mask.view(-1)
            perm = torch.randperm(flat_mask.size(0))
            flat_mask = flat_mask[perm]
            mask = flat_mask.view(img_size, img_size)

            self.masks.append(mask)
            self.class_presence += present_in_this_mask

        self.expected_pixel_counts = torch.zeros(num_classes)
        for mask in self.masks:
            for c in range(num_classes):
                self.expected_pixel_counts[c] += (mask == c).sum().item()

        total_pixels = size * img_size * img_size
        self.expected_pixel_distribution = self.expected_pixel_counts / total_pixels
        self.expected_class_presence_ratio = self.class_presence / size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {"image": self.images[idx], "mask": self.masks[idx]}


@pytest.fixture
def band_registry():
    """Create a mock band registry."""
    return MockBandRegistry()


@pytest.fixture
def temp_directory():
    """Create a temporary directory for test output."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


def test_classification_statistics_balanced(band_registry, temp_directory):
    """Test classification statistics with a balanced dataset."""
    num_classes = 4
    dataset_size = 100
    dataset = MockClassificationDataset(
        size=dataset_size, num_classes=num_classes, class_weights=None
    )

    stats_computer = ClassificationDatasetStatistics(
        dataset=dataset,
        dataset_band_config=band_registry,
        num_classes=num_classes,
        target_key="label",
        input_keys=["image"],
        batch_size=10,
        device="cpu",
        save_dir=temp_directory,
    )

    _, target_stats = stats_computer.compute_statistics()

    expected_count_per_class = dataset_size // num_classes
    expected_frequency = 1.0 / num_classes

    np.testing.assert_allclose(
        target_stats["class_counts"],
        [expected_count_per_class] * num_classes,
        rtol=1e-5,
    )

    np.testing.assert_allclose(
        target_stats["class_frequencies"], [expected_frequency] * num_classes, rtol=1e-5
    )

    assert target_stats["total_samples"] == dataset_size
    assert target_stats["num_classes"] == num_classes


def test_classification_statistics_imbalanced(band_registry, temp_directory):
    """Test classification statistics with an imbalanced dataset."""
    num_classes = 3
    dataset_size = 100
    class_weights = [0.7, 0.2, 0.1]

    dataset = MockClassificationDataset(
        size=dataset_size, num_classes=num_classes, class_weights=class_weights
    )

    expected_counts = dataset.expected_counts.numpy()
    expected_frequencies = expected_counts / dataset_size

    stats_computer = ClassificationDatasetStatistics(
        dataset=dataset,
        dataset_band_config=band_registry,
        num_classes=num_classes,
        target_key="label",
        input_keys=["image"],
        batch_size=10,
        device="cpu",
        save_dir=temp_directory,
    )

    _, target_stats = stats_computer.compute_statistics()

    np.testing.assert_allclose(target_stats["class_counts"], expected_counts, rtol=1e-5)

    np.testing.assert_allclose(
        target_stats["class_frequencies"], expected_frequencies, rtol=1e-5
    )

    assert target_stats["total_samples"] == dataset_size


def test_segmentation_statistics_balanced(band_registry, temp_directory):
    """Test segmentation statistics with balanced classes."""
    num_classes = 3
    dataset_size = 10
    img_size = 32

    dataset = MockSegmentationDataset(
        size=dataset_size,
        num_classes=num_classes,
        img_size=img_size,
        class_weights=None,
    )

    stats_computer = SegmentationDatasetStatistics(
        dataset=dataset,
        dataset_band_config=band_registry,
        num_classes=num_classes,
        target_key="mask",
        input_keys=["image"],
        batch_size=5,
        device="cpu",
        save_dir=temp_directory,
    )

    _, target_stats = stats_computer.compute_statistics()

    np.testing.assert_allclose(
        target_stats["pixel_counts"], dataset.expected_pixel_counts.numpy(), rtol=1e-5
    )

    np.testing.assert_allclose(
        target_stats["pixel_distribution"],
        dataset.expected_pixel_distribution.numpy(),
        rtol=1e-5,
    )

    np.testing.assert_allclose(
        target_stats["class_presence_ratio"],
        dataset.expected_class_presence_ratio.numpy(),
        rtol=1e-5,
    )

    assert target_stats["total_images"] == dataset_size
    assert target_stats["total_pixels"] == dataset_size * img_size * img_size
    assert target_stats["num_classes"] == num_classes
