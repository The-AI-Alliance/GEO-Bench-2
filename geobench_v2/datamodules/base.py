# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Base DataModules."""

from ligthning import LightningDataModule


class GeoBenchDataModule(LightningDataModule):
    """GeoBench DataModule."""

    def __init__(self):
        """Initialize GeoBench DataModule."""
        super().__init__()


class GeoBenchClassificationDataModule(GeoBenchDataModule):
    """GeoBench Classification DataModule."""

    def __init__(self):
        """Initialize GeoBench Classification DataModule."""
        super().__init__()


class GeoBenchSegmentationDataModule(GeoBenchDataModule):
    """GeoBench Segmentation DataModule."""

    def __init__(self):
        """Initialize GeoBench Segmentation DataModule."""
        super().__init__()


class GeoBenchPixelRegressionDataModule(GeoBenchDataModule):
    """GeoBench Pixel Regression DataModule."""

    def __init__(self):
        """Initialize GeoBench Pixel Regression DataModule."""
        super().__init__()


class GeoBenchObjectDetectionDataModule(GeoBenchDataModule):
    """GeoBench Object Detection DataModule."""

    def __init__(self):
        """Initialize GeoBench Object Detection DataModule."""
        super().__init__()
