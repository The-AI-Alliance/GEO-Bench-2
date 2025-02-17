# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Base DataModules."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import kornia.augmentation as K
import pandas as pd
import torch
import torch.nn as nn
from lightning import LightningDataModule
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchgeo.transforms import AugmentationSequential


class GeoBenchDataModule(LightningDataModule, ABC):
    """GeoBench DataModule."""

    def __init__(
        self,
        dataset_class: Dataset,
        img_size: int,
        batch_size: int = 32,
        eval_batch_size: int = 64,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        train_transforms: nn.Module | None = None,
        eval_transforms: nn.Module | None = None,
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
            train_transforms: Transforms/Augmentations to apply during training, they will be applied
                at the sample level and should include normalization
            eval_transforms: Transforms/Augmentations to apply during evaluation, they will be applied
                at the sample level and should include normalization
            pin_memory: whether to pin memory in dataloaders
            **kwargs: Additional keyword arguments passed to ``dataset_class``
        """
        super().__init__()

        self.dataset_class = dataset_class
        self.img_size = img_size
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.kwargs = kwargs
        self.train_transforms = train_transforms
        self.eval_transforms = eval_transforms

        self.set_normalization_stats()
        self.define_transformations()

    def prepare_data(self) -> None:
        """Download and prepare data, only for distributed setup."""
        if self.kwargs.get("download", False):
            self.dataset_class(**self.kwargs)

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
            band_order = self.dataset_class.band_default_order

        self.mean = torch.Tensor([self.band_means[band] for band in band_order])
        self.std = torch.Tensor([self.band_stds[band] for band in band_order])

    @abstractmethod
    def define_transformations(self) -> None:
        """Define transformations/augmentations for the dataset and task."""
        pass

    # move to dataset class instead and make it accesible on datamodule level
    # perhaps combining the dfs across the splits
    @abstractmethod
    def collect_metadata(self) -> pd.DataFrame:
        """Collect metadata of the dataset into a pandas DataFrame."""
        pass

    @abstractmethod
    def visualize_geolocation_distribution(self) -> None:
        """Visualize the geolocation distribution of the dataset."""
        pass

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
        train_transforms: nn.Module | None = None,
        eval_transforms: nn.Module | None = None,
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
            train_transforms: Transforms/Augmentations to apply during training, they will be applied
                at the sample level and should include normalization. See :method:`define_transformations`
                for the default transformation.
            eval_transforms: Transforms/Augmentations to apply during evaluation, they will be applied
                at the sample level and should include normalization. See :method:`define_transformations`
                for the default transformation.
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
            train_transforms=train_transforms,
            eval_transforms=eval_transforms,
            pin_memory=pin_memory,
            **kwargs,
        )

    def define_transformations(self) -> None:
        """Define data transform/augmentations for the dataset and task."""
        if self.train_transforms is not None:
            self.train_transform = nn.Sequential(
                K.Normalize(mean=self.mean, std=self.std),
                K.Resize(size=self.img_size, align_corners=True),
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.5),
            )

        if self.eval_transforms is not None:
            self.eval_transform = nn.Sequential(
                K.Normalize(mean=self.mean, std=self.std),
                K.Resize(size=self.img_size, align_corners=True),
            )

    def visualize_batch(
        self, split: str = "train"
    ) -> tuple[plt.Figure, dict[str, Tensor]]:
        """Visualize a batch of data.

        Args:
            split: One of 'train', 'val', 'test'

        Returns:
            The matplotlib figure and the batch of data
        """
        # subsample 8 examples from the batch
        # plot image and mask
        # add batch["image"] statistics to the figure


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
        train_transforms: nn.Module | None = None,
        eval_transforms: nn.Module | None = None,
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
            train_transforms: Transforms/Augmentations to apply during training, they will be applied
                at the sample level and should include normalization. See :method:`define_transformations`
                for the default transformation.
            eval_transforms: Transforms/Augmentations to apply during evaluation, they will be applied
                at the sample level and should include normalization. See :method:`define_transformations`
                for the default transformation.
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
            train_transforms=train_transforms,
            eval_transforms=eval_transforms,
            pin_memory=pin_memory,
            **kwargs,
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
