# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Utilities for computing and storing input and target statistics."""

import os
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from torch import Tensor
from abc import ABC, abstractmethod

from geobench_v2.datasets.sensor_util import DatasetBandRegistry


# Using Caleb Robinson's implementation: https://gist.github.com/calebrob6/1ef1e64bd62b1274adf2c6f91e20d215
class ImageSatistics(torch.nn.Module):
    def __init__(self, shape, dims):
        """Initializes the ImageSatistics method.

        A PyTorch module that can be put on the GPU and calculate the multidimensional
        mean and variance of inputs online in a numerically stable way. This is useful
        for calculating the channel-wise mean and variance of a big dataset because you
        don't have to load the entire dataset into memory.

        Uses the "Parallel algorithm" from: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        Similar implementation here: https://github.com/openai/baselines/blob/master/baselines/common/running_mean_std.py#L5

        Access the mean, variance, and standard deviation of the inputs with the
        `mean`, `var`, and `std` attributes.

        Example:
        ```
        rs = ImageSatistics((12,), [0, 2, 3])
        for inputs, _ in dataloader:
            rs(inputs)
        print(rs.mean)
        print(rs.var)
        print(rs.std)
        ```

        Args:
            shape: The shape of resulting mean and variance. For example, if you
                are calculating the mean and variance over the 0th, 2nd, and 3rd
                dimensions of inputs of size (64, 12, 256, 256), this should be 12.
            dims: The dimensions of your input to calculate the mean and variance
                over. In the above example, this should be [0, 2, 3].
        """
        super(ImageSatistics, self).__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("min", torch.ones(shape))
        self.register_buffer("max", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("std", torch.ones(shape))
        self.register_buffer("count", torch.zeros(1))
        self.dims = dims

    def update(self, x: Tensor) -> None:
        """Update the mean and variance with a new batch of inputs.

        Args:
            x: tensor over which to compute statistics
        """
        with torch.no_grad():
            batch_mean = torch.mean(x, dim=self.dims)
            batch_var = torch.var(x, dim=self.dims)
            batch_count = torch.tensor(x.shape[self.dims[0]], dtype=torch.float)

            n_ab = self.count + batch_count
            m_a = self.mean * self.count
            m_b = batch_mean * batch_count
            M2_a = self.var * self.count
            M2_b = batch_var * batch_count

            delta = batch_mean - self.mean

            self.mean = (m_a + m_b) / (n_ab)
            self.var = (M2_a + M2_b + delta**2 * self.count * batch_count / (n_ab)) / (
                n_ab
            )
            self.count += batch_count
            self.std = torch.sqrt(self.var + 1e-8)

            min_vals = x
            max_vals = x
            for dim in sorted(self.dims, reverse=True):
                min_vals = min_vals.min(dim=dim, keepdim=True)[0]
                max_vals = max_vals.max(dim=dim, keepdim=True)[0]

            min_vals = min_vals.squeeze()
            max_vals = max_vals.squeeze()

            self.min = torch.min(self.min, min_vals)
            self.max = torch.max(self.max, max_vals)

    def forward(self, x: Tensor) -> Tensor:
        """Update the statistics with a new batch of inputs and return the inputs.

        Args:
            x: tensor over which to compute statistics
        """
        self.update(x)
        return x


class DatasetStatistics(ABC):
    """Base class for computing dataset statistics."""

    def __init__(
        self,
        dataset: Dataset,
        dataset_band_config: DatasetBandRegistry,
        input_keys: list[str] = ["image"],
        target_key: str = "label",
        batch_size: int = 32,
        num_workers: int = 4,
        device: str = "cpu",
        save_dir: Optional[str] = None,
        **kwargs,
    ):
        """Initialize statistics computer.

        Args:
            dataset: Dataset to analyze
            dataset_band_config: Band configuration for the dataset
            input_keys: Keys for input data in batch dict, can compute statistics for multi-modal inputs
            target_key: Key for target data in batch dict, assume only single target
            batch_size: Batch size for dataloader
            num_workers: Number of workers for dataloader
            device: Device for computation
            save_dir: Directory to save statistics
            **kwargs: Additional task-specific arguments
        """
        self.dataset = dataset
        self.dataset_band_config = dataset_band_config
        self.input_keys = input_keys
        self.target_key = target_key
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.save_dir = save_dir

        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True if self.device == "cuda" else False,
        )

        self.input_stats = {key: {} for key in self.input_keys}

    def compute_batch_image_statistics(
        self, batch: Dict[str, Tensor]
    ) -> Dict[str, Dict[str, Any]]:
        """Compute statistics for input data using ImageSatistics.

        Args:
            batch: Batch of input data
        """
        for key in self.running_stats:
            input_data = batch[key]
            if torch.is_tensor(input_data):
                input_data = input_data.to(self.device)
                self.running_stats[key](input_data)

    @abstractmethod
    def compute_batch_target_statistics(
        self, targets: Tensor
    ) -> Dict[str, Dict[str, Any]]:
        """Compute statistics for target data.

        Args:
            targets: Target data tensor
        """
        pass

    @abstractmethod
    def aggregate_target_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate target statistics."""
        pass

    def aggregate_image_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate image input statistics."""
        for key in self.running_stats:
            stats = self.running_stats[key]
            self.input_stats[key].update(
                {
                    "mean": stats.mean.cpu().numpy(),
                    "std": stats.std.cpu().numpy(),
                    "var": stats.var.cpu().numpy(),
                    "count": stats.count.cpu().item(),
                }
            )

        return self.input_stats

    def aggregate_statistics(
        self,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        """Aggregate all statistics.

        Returns:
            image input statistics and task dependent target statistics
        """
        return self.aggregate_image_statistics(), self.aggregate_target_statistics()

    def compute_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Compute statistics for input data using ImageSatistics.

        Returns:
            Dictionary with input statistics for each input key
        """
        self.running_stats: dict[str, ImageSatistics] = {}

        for key in self.input_keys:
            batch = next(iter(self.dataloader))
            input_data = batch[key]

            if input_data.dim() == 4:
                band_names = (
                    self.dataset_band_config.get_band_names()
                    if hasattr(self.dataset_band_config, "get_band_names")
                    else None
                )
                num_channels = input_data.size(1)

                if band_names and len(band_names) == num_channels:
                    self.input_stats[key]["band_names"] = band_names

                shape = (num_channels,)
                dims = [0, 2, 3]

            elif input_data.dim() == 3:
                shape = (1,)
                dims = [0, 1, 2]

            else:
                if input_data.dim() >= 2:
                    shape = (input_data.size(1),)
                    dims = [0] + list(range(2, input_data.dim()))
                else:
                    shape = (1,)
                    dims = [0]

            self.running_stats[key] = ImageSatistics(shape, dims).to(self.device)

        for batch in tqdm(self.dataloader, desc="Computing dataset statistics"):
            self.compute_batch_image_statistics(batch)
            self.compute_batch_target_statistics(batch[self.target_key])

        return self.aggregate_statistics()


class ClassificationDatasetStatistics(DatasetStatistics):
    """Compute statistics for a classification dataset."""

    def __init__(
        self,
        dataset: Dataset,
        dataset_band_config: DatasetBandRegistry,
        num_classes: int,
        input_keys: list[str] = ["image"],
        target_key: str = "label",
        batch_size: int = 32,
        num_workers: int = 4,
        device: str = "cpu",
        save_dir: Optional[str] = None,
        **kwargs,
    ):
        """Initialize classification statistics computer.

        Args:
            dataset: Dataset to analyze
            dataset_band_config: Band configuration for the dataset
            num_classes: Number of classes
            input_keys: Keys for input data in batch dict, can compute statistics for multi-modal inputs
            target_key: Key for target data in batch dict, assume only single target
            batch_size: Batch size for dataloader
            num_workers: Number of workers for dataloader
            device: Device for computation
            save_dir: Directory to save statistics
            **kwargs: Additional task-specific arguments
        """
        super().__init__(
            dataset=dataset,
            dataset_band_config=dataset_band_config,
            input_keys=input_keys,
            target_key=target_key,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            save_dir=save_dir,
            **kwargs,
        )
        self.num_classes = num_classes
        self.class_counts = torch.zeros(self.num_classes, device=self.device)
        self.total_samples = 0

    def compute_batch_target_statistics(
        self, targets: Tensor
    ) -> Dict[str, Dict[str, Any]]:
        """Compute classification statistics for target data.

        Args:
            targets: Target data tensor
        """
        for c in range(self.num_classes):
            class_count = (targets == c).sum().item()
            self.class_counts[c] += class_count

        self.total_samples += targets.shape[0]

    def aggregate_target_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate classification target statistics."""
        class_frequencies = self.class_counts.float() / self.total_samples
        self.target_stats = {
            "class_counts": self.class_counts.cpu().numpy(),
            "total_samples": self.total_samples,
            "num_classes": self.num_classes,
            "class_frequencies": class_frequencies.cpu().numpy(),
        }
        return self.target_stats


class SegmentationDatasetStatistics(DatasetStatistics):
    """Compute statistics for a segmentation dataset."""

    def __init__(
        self,
        dataset: Dataset,
        dataset_band_config: DatasetBandRegistry,
        num_classes: int,
        input_keys: list[str] = ["image"],
        target_key: str = "mask",
        batch_size: int = 16,
        num_workers: int = 4,
        device: str = "cpu",
        save_dir: Optional[str] = None,
        **kwargs,
    ):
        """Initialize segmentation statistics computer.

        Args:
            dataset: Dataset to analyze
            dataset_band_config: Band configuration for the dataset
            num_classes: Number of classes
            input_keys: Keys for input data in batch dict
            target_key: Key for target data in batch dict (typically 'mask')
            batch_size: Batch size for dataloader
            num_workers: Number of workers for dataloader
            device: Device for computation
            save_dir: Directory to save statistics
            **kwargs: Additional task-specific arguments
        """
        super().__init__(
            dataset=dataset,
            dataset_band_config=dataset_band_config,
            input_keys=input_keys,
            target_key=target_key,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            save_dir=save_dir,
            **kwargs,
        )
        self.num_classes = num_classes
        self.pixel_counts = torch.zeros(self.num_classes, device=self.device)
        self.class_presence = torch.zeros(self.num_classes, device=self.device)
        self.total_pixels = 0
        self.total_images = 0

    def compute_batch_target_statistics(
        self, targets: Tensor
    ) -> Dict[str, Dict[str, Any]]:
        """Compute segmentation statistics for target data.

        Args:
            targets: Target data tensor of segmentation masks
        """
        targets = targets.to(self.device)

        if targets.dim() == 4 and targets.size(1) == 1:
            targets = targets.squeeze(1)

        batch_size = targets.size(0)
        for i in range(batch_size):
            mask = targets[i]

            for c in range(self.num_classes):
                class_pixels = (mask == c).sum().item()
                self.pixel_counts[c] += class_pixels

                if class_pixels > 0:
                    self.class_presence[c] += 1

            self.total_pixels += mask.numel()
            self.total_images += 1

    def aggregate_target_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate segmentation target statistics."""
        pixel_distribution = (
            self.pixel_counts.float() / self.total_pixels
            if self.total_pixels > 0
            else self.pixel_counts.float()
        )
        class_presence_ratio = (
            self.class_presence.float() / self.total_images
            if self.total_images > 0
            else self.class_presence.float()
        )

        self.target_stats = {
            "pixel_counts": self.pixel_counts.cpu().numpy(),
            "pixel_distribution": pixel_distribution.cpu().numpy(),
            "class_presence_counts": self.class_presence.cpu().numpy(),
            "class_presence_ratio": class_presence_ratio.cpu().numpy(),
            "total_pixels": self.total_pixels,
            "total_images": self.total_images,
            "num_classes": self.num_classes,
        }

        return self.target_stats
