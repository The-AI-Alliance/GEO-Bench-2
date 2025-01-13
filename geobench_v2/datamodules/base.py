# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Base DataModules."""

from collections.abc import Callable
from typing import Any

import kornia.augmentation as K
import torch
import torch.nn as nn
from ligthning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchgeo.transforms import AugmentationSequential


class GeoBenchDataModule(LightningDataModule):
    """GeoBench DataModule."""

    def __init__(
        self,
        dataset_class: Dataset,
        img_size: int,
        batch_size: int = 32,
        eval_batch_size: int = 64,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize GeoBench DataModule.

        Args:
            dataset_class: Dataset class to use in the DataModule
            img_size: Desired image input size for the model
            batch_size: Batch size during training
            eval_batch_size: Batch size during evaluation, can usually be larger than batch_size,
                to speed up evaluation.
            num_workers: Number of workers for dataloaders
            collate_fn: Collate function that can reformat samples to the needs of the model.
            pin_memory: whether to pin memory in dataloaders
            **kwargs: Additional keyword arguments passed to ``dataset_class``
        """
        super().__init__()

        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.kwargs = kwargs

        self.define_augmentations()
        self.set_normalization_stats()

    def setup(self, stage: str | None = None) -> None:
        """Setup data for train, val, test.

        Args:
            stage: One of 'fit', 'validate', 'test', or 'predict'.
        """
        raise NotImplementedError(
            "This method should be implemented in task-specific classes"
        )

    def set_normalization_stats(self):
        """Set normalization statistics for the input images of the dataset according to the band order."""
        if "band_order" in self.kwargs:
            band_order = self.kwargs["band_order"]
        else:
            band_order = self.dataset_class.band_order

        self.mean = torch.Tensor([self.band_means[band] for band in band_order])
        self.std = torch.Tensor([self.band_stds[band] for band in band_order])

    def define_augmentations(self) -> None:
        """Define augmentations for the dataset and task."""
        raise NotImplementedError(
            "This method should be implemented in task-specific classes"
        )

    def visualize_geolocation_distribution(self) -> None:
        """Visualize the geolocation distribution of the dataset."""
        raise NotImplementedError(
            "This method should be implemented in task-specific classes"
        )

    def train_dataloader(self) -> DataLoader:
        """Return train dataloader."""
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
        """Return validation dataloader."""
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
        """Return test dataloader."""
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
        batch_size: int = 32,
        eval_batch_size: int = 64,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize GeoBench Classification DataModule.

        Args:
            dataset_class: Dataset class to use in the DataModule
            img_size: Desired image input size for the model
            batch_size: Batch size during training
            eval_batch_size: Batch size during evaluation, can usually be larger than batch_size,
                to speed up evaluation.
            num_workers: Number of workers for dataloaders
            collate_fn: Collate function that can reformat samples to the needs of the model.
            pin_memory: whether to pin memory in dataloaders
            **kwargs: Additional keyword arguments passed to ``dataset_class``
        """
        super().__init__(
            dataset_class=dataset_class,
            img_size=img_size,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            kwargs=kwargs,
        )

    def define_augmentations(self) -> None:
        """Define augmentations for the dataset and task."""
        self.train_transform = nn.Sequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.Resize(size=self.img_size, align_corners=True),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
        )

        self.eval_transform = nn.Sequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.Resize(size=self.img_size, align_corners=True),
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
        batch_size: int = 32,
        eval_batch_size: int = 64,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize GeoBench Segmentation DataModule.

        Args:
            dataset_class: Dataset class to use in the DataModule
            img_size: Desired image input size for the model
            batch_size: Batch size during training
            eval_batch_size: Batch size during evaluation, can usually be larger than batch_size,
                to speed up evaluation.
            num_workers: Number of workers for dataloaders
            collate_fn: Collate function that can reformat samples to the needs of the model.
            pin_memory: whether to pin memory in dataloaders
            **kwargs: Additional keyword arguments passed to ``dataset_class``
        """
        super().__init__(
            dataset_class=dataset_class,
            img_size=img_size,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            kwargs=kwargs,
        )

    def define_augmentations(self) -> None:
        """Define augmentations for the dataset and task."""
        self.train_transform = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.Resize(size=self.img_size, align_corners=True),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=["image", "mask"],
        )

        self.eval_transform = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.Resize(size=self.img_size, align_corners=True),
            data_keys=["image", "mask"],
        )
