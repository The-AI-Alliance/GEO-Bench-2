# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Base DataModules."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Sequence

import kornia.augmentation as K
import pandas as pd
import torch
import torch.nn as nn
from lightning import LightningDataModule
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


# TODO come up with an expected metadata file scheme
# with common names etc. so a standardization
# - datamodules have functions to create nice visualizations of data distribution etc
# - datasets return an id that can be used to link back to all metadata available
# - datasets return lat/lon, if available time, and wavelength information
# - show how to allow for more elaborate analysis of predictions etc.


class GeoBenchDataModule(LightningDataModule, ABC):
    """GeoBench DataModule."""

    def __init__(
        self,
        dataset_class: Dataset,
        img_size: int,
        band_order: Sequence[float | str],
        batch_size: int = 32,
        eval_batch_size: int = 64,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        train_augmentations: nn.Module | None = None,
        eval_augmentations: nn.Module | None = None,
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize GeoBench DataModule.

        Args:
            dataset_class: Dataset class to use in the DataModule
            img_size: Desired image input size for the model
            band_order: band order of the image sample to be returned
            batch_size: Batch size during training
            eval_batch_size: Batch size during evaluation, can usually be larger than batch_size,
                to speed up evaluation.
            num_workers: Number of workers for dataloaders
            collate_fn: Collate function that can reformat samples to the needs of the model.
            train_augmentations: Transforms/Augmentations to apply during training, they will be applied
                at the sample level and should *not* include normalization, normalization happens on the dataset level for each
                sample, while geometric and color augmentations will be applied on a batch of data
            eval_augmentations: Transforms/Augmentations to apply during evaluation, they will be applied
                at the sample level and should *not* include normalization, normalization happens on the dataset level for each
                sample, while geometric and color augme]ntations will be applied on a batch of data
            pin_memory: whether to pin memory in dataloaders
            **kwargs: Additional keyword arguments passed to ``dataset_class``
        """
        super().__init__()

        self.dataset_class = dataset_class
        self.img_size = img_size
        self.band_order = band_order
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.kwargs = kwargs
        self.train_augmentations = train_augmentations
        self.eval_augmentations = eval_augmentations

        self.define_augmentations()

    def prepare_data(self) -> None:
        """Download and prepare data, only for distributed setup."""
        if self.kwargs.get("download", False):
            self.dataset_class(**self.kwargs)

    def setup(self, stage: str | None = None) -> None:
        """Setup data for train, val, test.

        Args:
            stage: One of 'fit', 'validate', 'test', or 'predict'.
        """
        self.train_dataset = self.dataset_class(
            split="train", band_order=self.band_order, **self.kwargs
        )
        self.val_dataset = self.dataset_class(
            split="val", band_order=self.band_order, **self.kwargs
        )
        self.test_dataset = self.dataset_class(
            split="test", band_order=self.band_order, **self.kwargs
        )

    @abstractmethod
    def load_metadata(self) -> pd.DataFrame:
        """Load metadata file.

        Returns:
            pandas DataFrame with metadata.
        """
        pass

    @abstractmethod
    def visualize_batch(
        self, split: str = "train"
    ) -> tuple[plt.Figure, dict[str, Tensor]]:
        """Visualize a batch of data.

        Args:
            split: One of 'train', 'val', 'test'

        Returns:
            The matplotlib figure and the batch of data
        """
        pass

    @abstractmethod
    def define_augmentations(self) -> None:
        """Define augmentations for the dataset and task."""
        pass

    @abstractmethod
    def visualize_geolocation_distribution(self) -> None:
        """Visualize the geolocation distribution of the dataset."""
        pass

    def train_dataloader(self) -> DataLoader:
        """Return train dataloader.

        Returns:
            Train Dataloader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader.

        Returns:
            Validation Dataloader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader.

        Returns:
            Test Dataloader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )


class GeoBenchClassificationDataModule(GeoBenchDataModule):
    """GeoBench Classification DataModule.

    By default, will yield a batch of images and their corresponding labels as
    a dictionary with keys 'image' and 'label'.
    """

    def __init__(
        self,
        dataset_class: Dataset,
        img_size: int,
        band_order: Sequence[float | str],
        batch_size: int = 32,
        eval_batch_size: int = 64,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        train_augmentations: nn.Module | None = None,
        eval_augmentations: nn.Module | None = None,
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize GeoBench Classification DataModule.

        Args:
            dataset_class: Dataset class to use in the DataModule
            img_size: Desired image input size for the model
            band_order: band order of the image sample to be returned
            batch_size: Batch size during training
            eval_batch_size: Batch size during evaluation, can usually be larger than batch_size,
                to speed up evaluation.
            num_workers: Number of workers for dataloaders
            collate_fn: Collate function that can reformat samples to the needs of the model.
            train_augmentations: Transforms/Augmentations to apply during training, they will be applied
                at the sample level and should include normalization. See :method:`define_augmentations`
                for the default transformation.
            eval_augmentations: Transforms/Augmentations to apply during evaluation, they will be applied
                at the sample level and should include normalization. See :method:`define_augmentations`
                for the default transformation.
            pin_memory: whether to pin memory in dataloaders
            **kwargs: Additional keyword arguments passed to ``dataset_class``
        """
        super().__init__(
            dataset_class=dataset_class,
            img_size=img_size,
            band_order=band_order,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            train_augmentations=train_augmentations,
            eval_augmentations=eval_augmentations,
            pin_memory=pin_memory,
            **kwargs,
        )

    def define_augmentations(self) -> None:
        """Define data transform/augmentations for the dataset and task."""
        if self.train_augmentations is not None:
            self.train_transform = nn.Sequential(
                K.Resize(size=self.img_size, align_corners=True),
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.5),
            )

        if self.eval_augmentations is not None:
            self.eval_transform = nn.Sequential(
                K.Resize(size=self.img_size, align_corners=True)
            )


class GeoBenchSegmentationDataModule(GeoBenchDataModule):
    """GeoBench Segmentation DataModule.

    By default, will yield a batch of images and their corresponding masks as
    a dictionary with keys 'image' and 'mask'.
    """

    def __init__(
        self,
        dataset_class: Dataset,
        img_size: int,
        band_order: Sequence[float | str],
        batch_size: int = 32,
        eval_batch_size: int = 64,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        train_augmentations: nn.Module | None = None,
        eval_augmentations: nn.Module | None = None,
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize GeoBench Segmentation DataModule.

        Args:
            dataset_class: Dataset class to use in the DataModule
            img_size: Desired image input size for the model
            band_order: band order of the image sample to be returned
            batch_size: Batch size during training
            eval_batch_size: Batch size during evaluation, can usually be larger than batch_size,
                to speed up evaluation.
            num_workers: Number of workers for dataloaders
            collate_fn: Collate function that can reformat samples to the needs of the model.
            train_augmentations: Transforms/Augmentations to apply during training, they will be applied
                at the sample level and should include normalization. See :method:`define_augmentations`
                for the default transformation.
            eval_augmentations: Transforms/Augmentations to apply during evaluation, they will be applied
                at the sample level and should include normalization. See :method:`define_augmentations`
                for the default transformation.
            pin_memory: whether to pin memory in dataloaders
            **kwargs: Additional keyword arguments passed to ``dataset_class``
        """
        super().__init__(
            dataset_class=dataset_class,
            img_size=img_size,
            band_order=band_order,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            train_augmentations=train_augmentations,
            eval_augmentations=eval_augmentations,
            pin_memory=pin_memory,
            **kwargs,
        )

    def define_augmentations(self) -> None:
        """Define augmentations for the dataset and task."""
        self.train_augmentations = K.AugmentationSequential(
            K.Resize(size=self.img_size, align_corners=True),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=["image", "mask"],
        )

        self.eval_transform = K.AugmentationSequential(
            # K.Normalize(mean=self.mean, std=self.std),
            K.Resize(size=self.img_size, align_corners=True),
            data_keys=["image", "mask"],
        )


class GeoBenchObjectDetectionDataModule(GeoBenchDataModule):
    """GeoBench Object Detection DataModule.

    By default, will yield a batch of images and their corresponding bounding boxes and labels as
    a dictionary with keys 'image', 'boxes_xyxy', and 'labels'.
    """

    def __init__(
        self,
        dataset_class: Dataset,
        img_size: int,
        band_order: Sequence[float | str],
        batch_size: int = 32,
        eval_batch_size: int = 64,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        train_augmentations: nn.Module | None = None,
        eval_augmentations: nn.Module | None = None,
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize GeoBench Object Detection DataModule.

        Args:
            dataset_class: Dataset class to use in the DataModule
            img_size: Desired image input size for the model
            batch_size: Batch size during training
            eval_batch_size: Batch size during evaluation, can usually be larger than batch_size,
                to speed up evaluation.
            num_workers: Number of workers for dataloaders
            collate_fn: Collate function that can reformat samples to the needs of the model.
            train_augmentations: Transforms/Augmentations to apply during training, they will be applied
                at the sample level and should include normalization. See :method:`define_augmentations`
                for the default transformation.
            eval_augmentations: Transforms/Augmentations to apply during evaluation, they will be applied
                at the sample level and should include normalization. See :method:`define_augmentations`
                for the default transformation.
            pin_memory: whether to pin memory in dataloaders
            **kwargs: Additional keyword arguments passed to ``dataset_class``
        """
        super().__init__(
            dataset_class=dataset_class,
            img_size=img_size,
            band_order=band_order,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            train_augmentations=train_augmentations,
            eval_augmentations=eval_augmentations,
            pin_memory=pin_memory,
            **kwargs,
        )

    def define_augmentations(self) -> None:
        """Define augmentations for the dataset and task."""
        self.train_transform = K.AugmentationSequential(
            K.Resize(size=self.img_size, align_corners=True),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=["image", "bbox_xyxy", "label"],
        )

        self.eval_transform = K.AugmentationSequential(
            K.Resize(size=self.img_size, align_corners=True),
            data_keys=["image", "bbox_xyxy", "label"],
        )
