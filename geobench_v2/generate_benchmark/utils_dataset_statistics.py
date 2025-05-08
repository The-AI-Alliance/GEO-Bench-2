# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Utilities for computing and storing input and target statistics."""

import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from lightning import LightningDataModule
from torch import Tensor
from tqdm.auto import tqdm


class NoNormalization(nn.Module):
    """No normalization applied to the input batch, used to replace
    the datamodule or dataset normalization scheme to compute original stas
    """

    def __init__(self, stats, band_order):
        super().__init__()

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        return batch


# Using Caleb Robinson's implementation: https://gist.github.com/calebrob6/1ef1e64bd62b1274adf2c6f91e20d215
class ImageSatistics(torch.nn.Module):
    def __init__(
        self,
        shape: tuple[int],
        dims: list[int],
        bins: int = 1000,
        range_vals: tuple[float, float] = (0, 100),
        compute_quantiles: bool = False,
        clip_min_val: Tensor | None = None,
        clip_max_val: Tensor | None = None,
    ):
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
        rs.mean)
        print(rs.var)
        print(rs.std)
        ```

        Args:
            shape: The shape of resulting mean and variance. For example, if you
                are calculating the mean and variance over the 0th, 2nd, and 3rd
                dimensions of inputs of size (64, 12, 256, 256), this should be 12.
            dims: The dimensions of your input to calculate the mean and variance
                over. In the above example, this should be [0, 2, 3].
            bins: Number of bins for histogram
            range_vals: Range for histogram
            compute_quantiles: Whether to compute 2nd and 98th percentiles
            clip_min_vals: Minimum values for clipping
            clip_max_val: Maximum values for clipping
        """
        super(ImageSatistics, self).__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("min", torch.full(shape, float("inf")))
        self.register_buffer("max", torch.full(shape, float("-inf")))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("std", torch.ones(shape))
        self.register_buffer("count", torch.zeros(1))
        if compute_quantiles:
            self.register_buffer("pct_02", torch.zeros(shape))
            self.register_buffer("pct_98", torch.zeros(shape))
        self.dims = dims
        self.compute_quantiles = compute_quantiles
        self.bins = bins
        self.range_min, self.range_max = range_vals
        self.register_buffer("hist", torch.zeros(shape[0], bins))

        if clip_min_val is not None and clip_max_val is not None:
            self.register_buffer("clip_min_val", torch.tensor(clip_min_val))
            self.register_buffer("clip_max_val", torch.tensor(clip_max_val))
            self.clipping_enabled = True
        else:
            # Still register as None buffers to avoid attribute errors if accessed
            self.register_buffer("clip_min_val", None)
            self.register_buffer("clip_max_val", None)
            self.clipping_enabled = False

    def update(self, x: Tensor) -> None:
        """Update the mean and variance with a new batch of inputs.

        Args:
            x: tensor over which to compute statistics
        """
        with torch.no_grad():
            # first optional clipping:
            if self.clipping_enabled:
                x = torch.clamp(x, min=self.clip_min_val, max=self.clip_max_val)

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

            # compute channel histograms
            # Determine the channel dimension as the unique dimension not in dims.
            all_dims = set(range(x.ndim))
            dims_set = set(self.dims)
            channel_dims = list(all_dims - dims_set)
            if len(channel_dims) != 1:
                raise ValueError(
                    "Could not determine unique channel dimension from dims."
                )
            channel_dim = channel_dims[0]
            channels = self.hist.shape[0]

            # Loop over channels to compute histogram for each.
            for i in range(channels):
                channel_data = x.select(dim=channel_dim, index=i).flatten()
                # Compute the histogram of the channel's data.
                hist_channel = torch.histc(
                    channel_data, bins=self.bins, min=self.range_min, max=self.range_max
                )
                self.hist[i] += hist_channel
                if self.compute_quantiles:
                    self.pct_02[i] = torch.quantile(channel_data, 0.02)
                    self.pct_98[i] = torch.quantile(channel_data, 0.98)

    def forward(self, x: Tensor) -> Tensor:
        """Update the statistics with a new batch of inputs and return the inputs.

        Args:
            x: tensor over which to compute statistics
        """
        self.update(x)
        return x

    def extra_repr(self) -> str:
        """Return a string representation of the ImageSatistics object."""
        return (
            f"ImageSatistics(mean={self.mean}, var={self.var}, std={self.std}, "
            f"min={self.min}, max={self.max}, count={self.count}, bins={self.bins}, dims={self.dims}, clip_min_val={self.clip_min_val}, clip_max_val={self.clip_max_val}))"
        )


class DatasetStatistics(ABC):
    """Base class for computing dataset statistics."""

    def __init__(
        self,
        datamodule: LightningDataModule,
        bins: int = 500,
        range_vals: dict[str, tuple[float, float]] | tuple[float, float] = (0.0, 1.0),
        clip_min_vals: dict[str, float] | None = None,
        clip_max_vals: dict[str, float] | None = None,
        input_keys: list[str] = ["image"],
        target_key: str = "label",
        device: str = "cpu",
        save_dir: str | None = None,
        **kwargs,
    ):
        """Initialize statistics computer.

        Args:
            datamodule: lightning datamodule which will choose train loader for statistics
            bins: Number of bins for histogram
            range_vals: Range for histogram
            clip_min_vals: Minimum values for clipping per input_key
            clip_max_vals: Maximum values for clipping per input_key
            input_keys: Keys for input data in batch dict, can compute statistics for multi-modal inputs
            target_key: Key for target data in batch dict, assume only single target
            device: Device for computation
            save_dir: Directory to save statistics
            **kwargs: Additional task-specific arguments
        """
        self.input_keys = input_keys
        self.target_key = target_key
        self.device = device
        self.save_dir = save_dir
        self.bins = bins
        self.range_vals = range_vals
        self.clip_min_vals = clip_min_vals
        self.clip_max_vals = clip_max_vals

        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

        datamodule.setup("fit")

        self.datamodule = datamodule

        self.dataset_band_config = datamodule.dataset_band_config

        self.dataloader = datamodule.train_dataloader()
        self.input_stats = {key: {} for key in self.input_keys}

        self.initialize_running_stats()

    def compute_batch_image_statistics(
        self, batch: dict[str, Tensor]
    ) -> dict[str, dict[str, Any]]:
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
    ) -> dict[str, dict[str, Any]]:
        """Compute statistics for target data.

        Args:
            targets: Target data tensor
        """
        pass

    @abstractmethod
    def aggregate_target_statistics(self) -> dict[str, dict[str, Any]]:
        """Aggregate target statistics."""
        pass

    def aggregate_image_statistics(self) -> dict[str, dict[str, Any]]:
        """Aggregate image input statistics."""
        # Extract statistics from running_stats
        for key in self.running_stats:
            stats = self.running_stats[key]
            update_dict = {
                "mean": stats.mean.cpu().numpy(),
                "std": stats.std.cpu().numpy(),
                "var": stats.var.cpu().numpy(),
                "min": stats.min.cpu().numpy(),  # Min value *after* potential clipping
                "max": stats.max.cpu().numpy(),  # Max value *after* potential clipping
                "count": stats.count.cpu().item(),
                "histograms": stats.hist.cpu().numpy(),
                "histogram_bins": torch.linspace(
                    stats.range_min, stats.range_max, stats.bins + 1
                )
                .cpu()
                .numpy(),
                "pct_02": stats.pct_02.cpu().numpy()
                if hasattr(stats, "pct_02")
                and stats.pct_02 is not None  # Check existence and not None
                else None,
                "pct_98": stats.pct_98.cpu().numpy()
                if hasattr(stats, "pct_98")
                and stats.pct_98 is not None  # Check existence and not None
                else None,
            }
            # Add used clip values if clipping was enabled for this key
            if stats.clipping_enabled:
                # Store the per-channel expanded tensors
                update_dict["clip_min_used"] = stats.clip_min_val.cpu().numpy()
                update_dict["clip_max_used"] = stats.clip_max_val.cpu().numpy()

            self.input_stats[key].update(update_dict)

        return self.input_stats

    def aggregate_statistics(
        self,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        """Aggregate all statistics.

        Returns:
            image input statistics and task dependent target statistics
        """
        return self.aggregate_image_statistics(), self.aggregate_target_statistics()

    def initialize_running_stats(self) -> None:
        """Initialize running statistics for input data."""
        self.running_stats: dict[str, ImageSatistics] = {}

        for key in self.input_keys:
            batch = next(iter(self.dataloader))
            input_data = batch[key]

            if input_data.dim() == 5:
                # 5D input data (e.g., time series), assume [B, T, C, H, W]
                band_order = self.datamodule.band_order
                if key.removeprefix("image_") in band_order:
                    band_names = band_order[key.removeprefix("image_")]
                else:
                    band_names = band_order
                num_channels = input_data.size(2)

                assert len(band_names) == num_channels, (
                    f"Band names length {len(band_names)} does not match number of channels {num_channels} for key {key}"
                )

                if band_names and len(band_names) == num_channels:
                    self.input_stats[key]["band_names"] = band_names

                shape = (num_channels,)
                dims = [0, 1, 3, 4]

            elif input_data.dim() == 4:
                # assume [B, C, H, W]
                band_order = self.datamodule.band_order
                if key.removeprefix("image_") in band_order:
                    band_names = band_order[key.removeprefix("image_")]
                else:
                    band_names = band_order

                num_channels = input_data.size(1)

                assert len(band_names) == num_channels, (
                    f"Band names length {len(band_names)} does not match number of channels {num_channels} for key {key}"
                )

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

            self.running_stats[key] = ImageSatistics(
                shape,
                dims,
                bins=self.bins,
                range_vals=self.range_vals[key],
                clip_min_val=self.clip_min_vals[key]
                if self.clip_min_vals is not None
                else None,
                clip_max_val=self.clip_max_vals[key]
                if self.clip_max_vals is not None
                else None,
            ).to(self.device)

    def compute_statistics(self) -> dict[str, dict[str, Any]]:
        """Compute statistics for input data using ImageSatistics.

        Returns:
            dictionary with input statistics for each input key
        """
        i = 0
        for batch in tqdm(self.dataloader, desc="Computing dataset statistics"):
            self.compute_batch_image_statistics(batch)
            self.compute_batch_target_statistics(batch[self.target_key])
            i += 1

        return self.aggregate_statistics()


class ClassificationStatistics(DatasetStatistics):
    """Compute statistics for a classification dataset."""

    def __init__(
        self,
        datamodule: LightningDataModule,
        bins: int = 500,
        range_vals: tuple[float, float] = (0.0, 1.0),
        clip_min_vals: dict[str, float] | None = None,
        clip_max_vals: dict[str, float] | None = None,
        input_keys: list[str] = ["image"],
        target_key: str = "label",
        multilabel: bool = False,
        device: str = "cpu",
        save_dir: str | None = None,
        **kwargs,
    ):
        """Initialize classification statistics computer.

        Args:
            datamodule
            num_classes: Number of classes
            input_keys: Keys for input data in batch dict, can compute statistics for multi-modal inputs
            target_key: Key for target data in batch dict, assume only single target
            multilabel: Whether the classification is multilabel
            device: Device for computation
            save_dir: Directory to save statistics
            **kwargs: Additional task-specific arguments
        """
        super().__init__(
            datamodule=datamodule,
            bins=bins,
            range_vals=range_vals,
            clip_min_vals=clip_min_vals,
            clip_max_vals=clip_max_vals,
            input_keys=input_keys,
            target_key=target_key,
            device=device,
            save_dir=save_dir,
            **kwargs,
        )
        self.num_classes = datamodule.num_classes
        self.class_counts = torch.zeros(self.num_classes, device=self.device)
        self.total_samples = 0

        if self.multi_label:
            self.label_co_occurrence = torch.zeros(
                (self.num_classes, self.num_classes), device=self.device
            )
            self.samples_per_class_count = torch.zeros(
                self.num_classes + 1, device=self.device
            )

    def compute_batch_target_statistics(
        self, targets: Tensor
    ) -> dict[str, dict[str, Any]]:
        """Compute classification statistics for target data.

        Args:
            targets: Target data tensor
        """
        targets = targets.to(self.device)
        batch_size = targets.shape[0]

        if self.multi_label:
            if targets.dim() == 2 and targets.shape[1] == self.num_classes:
                self.class_counts += targets.sum(dim=0)

                labels_per_sample = targets.sum(dim=1).long()
                for count in labels_per_sample:
                    if count <= self.num_classes:
                        self.samples_per_class_count[count] += 1

                for i in range(batch_size):
                    sample_labels = targets[i]
                    active_indices = torch.where(sample_labels == 1)[0]
                    for idx1 in active_indices:
                        for idx2 in active_indices:
                            self.label_co_occurrence[idx1, idx2] += 1
            else:
                raise ValueError(
                    f"Multi-label targets should have shape [batch_size, {self.num_classes}] but got {targets.shape}"
                )
        else:
            for c in range(self.num_classes):
                class_count = (targets == c).sum().item()
                self.class_counts[c] += class_count

        self.total_samples += batch_size

    def aggregate_target_statistics(self) -> dict[str, dict[str, Any]]:
        """Aggregate classification target statistics."""
        class_frequencies = self.class_counts.float() / self.total_samples

        self.target_stats = {
            "class_counts": self.class_counts.cpu().numpy(),
            "total_samples": self.total_samples,
            "num_classes": self.num_classes,
            "class_frequencies": class_frequencies.cpu().numpy(),
            "multi_label": self.multi_label,
            "class_names": self.datamodule.class_names,
        }

        if self.multi_label:
            self.target_stats.update(
                {
                    "labels_per_sample": (self.class_counts.sum() / self.total_samples)
                    .cpu()
                    .item(),
                    "samples_per_class_count": self.samples_per_class_count.cpu().numpy(),
                    "label_co_occurrence": self.label_co_occurrence.cpu().numpy(),
                    "samples_with_no_labels": self.samples_per_class_count[0]
                    .cpu()
                    .item(),
                }
            )

            co_occurrence = self.label_co_occurrence.cpu().numpy()
            diag_vals = co_occurrence.diagonal()
            conditional_probs = np.zeros_like(co_occurrence, dtype=float)
            for i in range(self.num_classes):
                if diag_vals[i] > 0:
                    conditional_probs[i] = co_occurrence[i] / diag_vals[i]

            self.target_stats["label_conditional_probabilities"] = conditional_probs

        return self.target_stats


class SegmentationStatistics(DatasetStatistics):
    """Compute statistics for a segmentation dataset."""

    def __init__(
        self,
        datamodule: LightningDataModule,
        bins: int = 500,
        range_vals: tuple[float, float] = (0.0, 1.0),
        clip_min_vals: dict[str, float] | None = None,
        clip_max_vals: dict[str, float] | None = None,
        input_keys: list[str] = ["image"],
        target_key: str = "mask",
        device: str = "cpu",
        save_dir: str | None = None,
        **kwargs,
    ):
        """Initialize segmentation statistics computer.

        Args:
            datamodule:
            input_keys: Keys for input data in batch dict
            target_key: Key for target data in batch dict (typically 'mask')
            device: Device for computation
            save_dir: Directory to save statistics
            **kwargs: Additional task-specific arguments
        """
        super().__init__(
            datamodule=datamodule,
            bins=bins,
            range_vals=range_vals,
            clip_min_vals=clip_min_vals,
            clip_max_vals=clip_max_vals,
            input_keys=input_keys,
            target_key=target_key,
            device=device,
            save_dir=save_dir,
            **kwargs,
        )
        self.num_classes = datamodule.num_classes
        self.pixel_counts = torch.zeros(self.num_classes, device=self.device)
        self.class_presence = torch.zeros(self.num_classes, device=self.device)
        self.total_pixels = 0
        self.total_images = 0

        self.class_cooccurrence = torch.zeros(
            (self.num_classes, self.num_classes), device=self.device
        )
        self.total_pixels = 0
        self.total_images = 0

    def compute_batch_target_statistics(
        self, targets: Tensor
    ) -> dict[str, dict[str, Any]]:
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

            class_present = torch.zeros(
                self.num_classes, dtype=torch.bool, device=self.device
            )

            for c in range(self.num_classes):
                class_pixels = (mask == c).sum().item()
                self.pixel_counts[c] += class_pixels

                if class_pixels > 0:
                    self.class_presence[c] += 1
                    class_present[c] = True

            present_indices = torch.where(class_present)[0]
            for idx1 in present_indices:
                for idx2 in present_indices:
                    self.class_cooccurrence[idx1, idx2] += 1

            self.total_pixels += mask.numel()
            self.total_images += 1

    def aggregate_target_statistics(self) -> dict[str, dict[str, Any]]:
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

        class_cooccurrence_ratio = (
            self.class_cooccurrence.float() / self.total_images
            if self.total_images > 0
            else self.class_cooccurrence.float()
        )

        self.target_stats = {
            "pixel_counts": self.pixel_counts.cpu().numpy(),
            "pixel_distribution": pixel_distribution.cpu().numpy(),
            "class_presence_counts": self.class_presence.cpu().numpy(),
            "class_presence_ratio": class_presence_ratio.cpu().numpy(),
            "class_cooccurrence": self.class_cooccurrence.cpu().numpy(),
            "class_cooccurrence_ratio": class_cooccurrence_ratio.cpu().numpy(),
            "total_pixels": self.total_pixels,
            "total_images": self.total_images,
            "num_classes": self.num_classes,
            "class_names": self.datamodule.class_names,
        }

        return self.target_stats


class PxRegressionStatistics(DatasetStatistics):
    """Compute statistics for a pixel regression dataset."""

    def __init__(
        self,
        datamodule: LightningDataModule,
        bins: int = 500,
        range_vals: tuple[float, float] = (0.0, 1.0),
        clip_min_vals: dict[str, float] | None = None,
        clip_max_vals: dict[str, float] | None = None,
        target_range_vals: tuple[float, float] = (0.0, 1.0),
        input_keys: list[str] = ["image"],
        target_key: str = "label",
        device: str = "cpu",
        save_dir: str | None = None,
        **kwargs,
    ):
        """Initialize pixel regression statistics computer.

        Args:
            datamodule
            input_keys: Keys for input data in batch dict
            target_key: Key for target data in batch dict
            bins: Number of bins for histogram
            range_vals: Range for histogram
            target_range_vals: Range for target histogram
            device: Device for computation
            save_dir: Directory to save statistics
            **kwargs: Additional task-specific arguments
        """
        super().__init__(
            datamodule=datamodule,
            bins=bins,
            range_vals=range_vals,
            clip_min_vals=clip_min_vals,
            clip_max_vals=clip_max_vals,
            input_keys=input_keys,
            target_key=target_key,
            device=device,
            save_dir=save_dir,
            **kwargs,
        )
        self.target_range_vals = target_range_vals
        self.target_stats = ImageSatistics(
            shape=(1,),
            dims=[0, 2, 3],
            bins=self.bins,
            range_vals=self.target_range_vals,
            compute_quantiles=True,
        ).to(self.device)

    def compute_batch_target_statistics(
        self, targets: Tensor
    ) -> dict[str, dict[str, Any]]:
        """Compute pixel regression statistics for target data.

        Args:
            targets: Target data tensor
        """
        targets = targets.to(self.device)
        self.target_stats.update(targets)

    def aggregate_target_statistics(self) -> dict[str, dict[str, Any]]:
        """Aggregate pixelwise regression target statistics."""
        self.target_stats = {
            "mean": self.target_stats.mean.cpu().numpy(),
            "std": self.target_stats.std.cpu().numpy(),
            "var": self.target_stats.var.cpu().numpy(),
            "min": self.target_stats.min.cpu().numpy(),
            "max": self.target_stats.max.cpu().numpy(),
            "count": self.target_stats.count.cpu().item(),
            "histograms": self.target_stats.hist.cpu().numpy(),
            "histogram_bins": torch.linspace(
                self.target_stats.range_min,
                self.target_stats.range_max,
                self.target_stats.bins + 1,
            )
            .cpu()
            .numpy(),
            "pct_02": self.target_stats.pct_02.cpu().numpy(),
            "pct_98": self.target_stats.pct_98.cpu().numpy(),
        }
        return self.target_stats


class ObjectDetectionStatistics(DatasetStatistics):
    """Compute statistics for an object detection dataset."""

    def __init__(
        self,
        datamodule: LightningDataModule,
        bins: int = 500,
        range_vals: tuple[float, float] = (0.0, 1.0),
        clip_min_vals: dict[str, float] | None = None,
        clip_max_vals: dict[str, float] | None = None,
        input_keys: list[str] = ["image"],
        target_key: str = "boxes",
        device: str = "cpu",
        save_dir: str | None = None,
        **kwargs,
    ):
        """Initialize object detection statistics computer.

        Args:
            datamodule
            input_keys: Keys for input data in batch dict
            target_key: Key for target data in batch dict
            bins: Number of bins for histogram
            range_vals: Range for histogram
            device: Device for computation
            save_dir: Directory to save statistics
            **kwargs: Additional task-specific arguments
        """
        super().__init__(
            datamodule=datamodule,
            bins=bins,
            range_vals=range_vals,
            clip_min_vals=clip_min_vals,
            clip_max_vals=clip_max_vals,
            input_keys=input_keys,
            target_key=target_key,
            device=device,
            save_dir=save_dir,
            **kwargs,
        )

        self.num_classes = datamodule.num_classes
        self.class_counts = torch.zeros(self.num_classes, device=self.device)
        self.total_samples = 0
        self.total_boxes = 0
        self.box_counts = torch.zeros(self.num_classes, device=self.device)
        self.box_area = torch.zeros(self.num_classes, device=self.device)
        self.box_aspect_ratio = torch.zeros(self.num_classes, device=self.device)
        self.box_width = torch.zeros(self.num_classes, device=self.device)
        self.box_height = torch.zeros(self.num_classes, device=self.device)
        self.box_width_counts = torch.zeros(self.num_classes, device=self.device)
        self.box_height_counts = torch.zeros(self.num_classes, device=self.device)
        self.box_area_counts = torch.zeros(self.num_classes, device=self.device)
        self.box_aspect_ratio_counts = torch.zeros(self.num_classes, device=self.device)

    def compute_batch_target_statistics(
        self, bboxes: list[Tensor], labels: list[Tensor]
    ) -> None:
        """Compute Object detection target statistics."""
        batch_size = len(bboxes)
        assert len(bboxes) == len(labels)

        for i in range(batch_size):
            boxes = bboxes[i]
            labels_i = labels[i]

            for j in range(len(boxes)):
                box = boxes[j]
                label = labels_i[j]

                if label >= self.num_classes:
                    continue

                self.total_boxes += 1
                self.box_counts[label] += 1
                self.class_counts[label] += 1

                x_min, y_min, x_max, y_max = box
                width = x_max - x_min
                height = y_max - y_min
                area = width * height

                self.box_area[label] += area
                self.box_width[label] += width
                self.box_height[label] += height
                self.box_width_counts[label] += 1
                self.box_height_counts[label] += 1
                self.box_area_counts[label] += 1

                aspect_ratio = width / height if height > 0 else 0.0
                self.box_aspect_ratio[label] += aspect_ratio
                self.box_aspect_ratio_counts[label] += 1

    def aggregate_target_statistics(self) -> dict[str, dict[str, Any]]:
        """Aggregate object detection target statistics."""
        box_area = self.box_area / self.box_area_counts
        box_width = self.box_width / self.box_width_counts
        box_height = self.box_height / self.box_height_counts
        box_aspect_ratio = self.box_aspect_ratio / self.box_aspect_ratio_counts

        class_frequencies = (
            self.class_counts.float() / self.total_samples
            if self.total_samples > 0
            else self.class_counts.float()
        )

        self.target_stats = {
            "class_counts": self.class_counts.cpu().numpy(),
            "total_boxes": self.total_boxes,
            "total_samples": self.total_samples,
            "num_classes": self.num_classes,
            "box_area": box_area.cpu().numpy(),
            "box_width": box_width.cpu().numpy(),
            "box_height": box_height.cpu().numpy(),
            "box_aspect_ratio": box_aspect_ratio.cpu().numpy(),
            "class_frequencies": class_frequencies.cpu().numpy(),
        }

        return self.target_stats

    def compute_statistics(self) -> dict[str, dict[str, Any]]:
        """Compute statistics for input data using ImageSatistics.

        Returns:
            dictionary with input statistics for each input key
        """
        for batch in tqdm(self.dataloader, desc="Computing dataset statistics"):
            self.compute_batch_image_statistics(batch)
            self.compute_batch_target_statistics(batch["bbox_xyxy"], batch["label"])

        return self.aggregate_statistics()
