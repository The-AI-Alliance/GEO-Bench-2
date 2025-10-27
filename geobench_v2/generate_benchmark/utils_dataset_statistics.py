# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Utilities for computing and storing input and target statistics."""

import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch import Tensor
from tqdm.auto import tqdm


# Using Caleb Robinson's implementation: https://gist.github.com/calebrob6/1ef1e64bd62b1274adf2c6f91e20d215
class ImageStatistics(torch.nn.Module):
    """Compute image statistics for a batch of images."""

    valid_normalization_modes = ("none", "clip_only", "clip_rescale", "satmae")

    def __init__(
        self,
        shape: tuple[int],
        dims: list[int],
        bins: int = 100,
        range_vals: tuple[float, float] = (0, 100),
        compute_quantiles: bool = False,
        clip_min_val: Tensor | None = None,
        clip_max_val: Tensor | None = None,
        normalization_mode: str = "none",
    ):
        """Initializes the ImageStatistics method with support for multiple normalization schemes.

        Args:
            shape: The shape of resulting mean and variance.
            dims: The dimensions to calculate stats over.
            bins: Number of bins for histogram
            range_vals: Range for histogram
            compute_quantiles: Whether to compute 2nd and 98th percentiles
            clip_min_val: Minimum values for clipping
            clip_max_val: Maximum values for clipping
            normalization_mode: Type of normalization to apply for the second stage stats
                - "none": No normalization - just compute raw statistics on original data
                - "clip_only": Apply min/max clipping before computing second stage statistics
                - "clip_rescale": Clip to min/max then rescale to [0,1] range
                - "satmae": Shift negatives, clip to mean±2std, scale to [0,1] range
        """
        super().__init__()

        assert normalization_mode in self.valid_normalization_modes, (
            f"Invalid normalization mode '{normalization_mode}'. "
            f"Valid options are: {self.valid_normalization_modes}"
        )

        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("min", torch.full(shape, float("inf")))
        self.register_buffer("max", torch.full(shape, float("-inf")))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("std", torch.ones(shape))
        self.register_buffer("count", torch.zeros(1))

        self.register_buffer("shift_offsets", torch.zeros(shape))

        self.normalization_mode = normalization_mode

        if normalization_mode in ["clip_only", "clip_rescale", "satmae"]:
            self.register_buffer("norm_mean", torch.zeros(shape))
            self.register_buffer("norm_var", torch.ones(shape))
            self.register_buffer("norm_std", torch.ones(shape))
            self.register_buffer("norm_count", torch.zeros(1))

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
            self.register_buffer("clip_min_val", None)
            self.register_buffer("clip_max_val", None)
            self.clipping_enabled = False

    @torch.no_grad()
    def update(self, x: Tensor) -> None:
        """Update the statistics with a new batch of inputs.

        Computes both raw statistics and normalized statistics if enabled.
        """
        with torch.no_grad():
            self._update_raw_stats(x)

            if self.normalization_mode == "none":
                normalized_x = None

            elif self.normalization_mode == "clip_only":
                normalized_x = torch.clamp(
                    x, min=self.clip_min_val, max=self.clip_max_val
                )

            elif self.normalization_mode == "clip_rescale":
                x_clipped = torch.clamp(x, min=self.clip_min_val, max=self.clip_max_val)
                normalized_x = self._apply_clip_rescale(x_clipped)

            elif self.normalization_mode == "satmae":
                if self.count > 0:
                    normalized_x = self._apply_satmae_normalization(x)
                else:
                    normalized_x = None

            if normalized_x is not None and self.normalization_mode != "none":
                self._update_normalized_stats(normalized_x)

    @torch.no_grad()
    def _update_raw_stats(self, x: Tensor) -> None:
        """Update the raw statistics (before normalization)."""
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
        self.var = (M2_a + M2_b + delta**2 * self.count * batch_count / (n_ab)) / (n_ab)
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

        if self.normalization_mode == "satmae":
            negative_channels = self.min < 0
            if negative_channels.any():
                self.shift_offsets[negative_channels] = -self.min[negative_channels]

        all_dims = set(range(x.ndim))
        dims_set = set(self.dims)
        channel_dims = list(all_dims - dims_set)
        if len(channel_dims) != 1:
            raise ValueError("Could not determine unique channel dimension from dims.")

        channel_dim = channel_dims[0]
        channels = self.hist.shape[0]

        for i in range(channels):
            channel_data = x.select(dim=channel_dim, index=i).flatten()
            hist_channel = torch.histc(
                channel_data, bins=self.bins, min=self.range_min, max=self.range_max
            )
            self.hist[i] += hist_channel

            if self.compute_quantiles:
                self.pct_02[i] = torch.quantile(channel_data, 0.02)
                self.pct_98[i] = torch.quantile(channel_data, 0.98)

    @torch.no_grad()
    def _apply_clip_rescale(self, x: Tensor) -> Tensor:
        """Simple rescaling to [0,1] by shifting to non-negative range and dividing by max value.

        This approach:
        1. Shifts all values to be non-negative (if min_val < 0)
        2. Divides by the max value to normalize to [0,1] range

        This is more direct than full min/max normalization when we just want a simple positive rescaling.
        """
        all_dims = set(range(x.ndim))
        dims_set = set(self.dims)
        channel_dim = list(all_dims - dims_set)[0]

        broadcast_shape = [1] * x.ndim
        broadcast_shape[channel_dim] = (
            self.clip_min_val.shape[0] if self.clip_min_val.ndim > 0 else 1
        )

        min_val = self.clip_min_val.view(broadcast_shape)
        max_val = self.clip_max_val.view(broadcast_shape)

        shifted_x = x.clone()
        negative_ranges = min_val < 0

        if negative_ranges.any():
            shift_values = torch.zeros_like(min_val)
            shift_values[negative_ranges] = -min_val[negative_ranges]

            shifted_x = x + shift_values

            shifted_max = max_val + shift_values
        else:
            shifted_max = max_val

        normalized = shifted_x / shifted_max

        normalized = torch.clamp(normalized, min=0.0, max=1.0)

        return normalized

    @torch.no_grad()
    def _apply_satmae_normalization(self, x: Tensor) -> Tensor:
        """Apply SatMAE-style normalization: shift negatives, min/max norm by mean±2std, clip to [0,1].

        The normalization procedure:
        1. Shift negative values to make all values non-negative
        2. Calculate the new clamping range based on the offset-adjusted mean and std
        3. Clamp values to the adjusted range mean±2std
        4. Rescale to [0,1]
        """
        all_dims = set(range(x.ndim))
        dims_set = set(self.dims)
        channel_dim = list(all_dims - dims_set)[0]

        broadcast_shape = [1] * x.ndim
        broadcast_shape[channel_dim] = self.mean.shape[0]

        shift_offsets = self.shift_offsets.view(broadcast_shape)
        mean = self.mean.view(broadcast_shape)
        std = self.std.view(broadcast_shape)

        normalized = x + shift_offsets

        adjusted_mean = mean + shift_offsets

        channel_min = adjusted_mean - 2 * std
        channel_max = adjusted_mean + 2 * std

        normalized = (normalized - channel_min) / (channel_max - channel_min)
        normalized = torch.clamp(normalized, min=0.0, max=1.0)

        return normalized

    @torch.no_grad()
    def _update_normalized_stats(self, x: Tensor) -> None:
        """Update statistics after normalization."""
        batch_mean = torch.mean(x, dim=self.dims)
        batch_var = torch.var(x, dim=self.dims)
        batch_count = torch.tensor(x.shape[self.dims[0]], dtype=torch.float)

        n_ab = self.norm_count + batch_count
        m_a = self.norm_mean * self.norm_count
        m_b = batch_mean * batch_count
        M2_a = self.norm_var * self.norm_count
        M2_b = batch_var * batch_count

        delta = batch_mean - self.norm_mean

        self.norm_mean = (m_a + m_b) / (n_ab)
        self.norm_var = (
            M2_a + M2_b + delta**2 * self.norm_count * batch_count / (n_ab)
        ) / (n_ab)
        self.norm_count += batch_count
        self.norm_std = torch.sqrt(self.norm_var + 1e-8)

    def forward(self, x: Tensor) -> Tensor:
        """Update the statistics with a new batch of inputs and return the inputs."""
        with torch.no_grad():
            self.update(x)
        return x

    def extra_repr(self) -> str:
        """Return a string representation of the ImageStatistics object."""
        base_repr = (
            f"ImageStatistics(mean={self.mean}, var={self.var}, std={self.std}, "
            f"min={self.min}, max={self.max}, count={self.count}, bins={self.bins}, "
            f"dims={self.dims}, clip_min_val={self.clip_min_val}, clip_max_val={self.clip_max_val}, "
            f"normalization_mode={self.normalization_mode})"
        )

        if hasattr(self, "norm_mean"):
            base_repr += (
                f"\nNormalized stats: norm_mean={self.norm_mean}, norm_var={self.norm_var}, "
                f"norm_std={self.norm_std}, norm_count={self.norm_count})"
            )

        return base_repr


class DatasetStatistics(ABC):
    """Base class for computing dataset statistics."""

    def __init__(
        self,
        datamodule: LightningDataModule,
        bins: int = 100,
        range_vals: dict[str, tuple[float, float]] | tuple[float, float] = (0.0, 1.0),
        clip_min_vals: dict[str, float] | None = None,
        clip_max_vals: dict[str, float] | None = None,
        normalization_mode: str = "none",
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
            normalization_mode: Type of normalization to apply for the second stage stats
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
        self.normalization_mode = normalization_mode

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
        """Compute statistics for input data using ImageStatistics.

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
        """Aggregate image input statistics, including two-stage normalization if enabled."""
        for key in self.running_stats:
            stats = self.running_stats[key]
            update_dict = {
                "normalization_mode": self.normalization_mode,
                "mean": stats.mean.cpu().numpy(),
                "std": stats.std.cpu().numpy(),
                "var": stats.var.cpu().numpy(),
                "min": stats.min.cpu().numpy(),
                "max": stats.max.cpu().numpy(),
                "count": stats.count.cpu().item(),
                "histograms": stats.hist.cpu().numpy(),
                "histogram_bins": torch.linspace(
                    stats.range_min, stats.range_max, stats.bins + 1
                )
                .cpu()
                .numpy(),
                "pct_02": stats.pct_02.cpu().numpy()
                if hasattr(stats, "pct_02") and stats.pct_02 is not None
                else None,
                "pct_98": stats.pct_98.cpu().numpy()
                if hasattr(stats, "pct_98") and stats.pct_98 is not None
                else None,
                "shift_offsets": stats.shift_offsets.cpu().numpy()
                if hasattr(stats, "shift_offsets")
                else None,
            }

            if hasattr(stats, "norm_mean"):
                update_dict.update(
                    {
                        "norm_mean": stats.norm_mean.cpu().numpy(),
                        "norm_std": stats.norm_std.cpu().numpy(),
                        "norm_var": stats.norm_var.cpu().numpy(),
                    }
                )

            if stats.clipping_enabled:
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
        self.running_stats: dict[str, ImageStatistics] = {}

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

            if isinstance(self.range_vals, (dict, DictConfig)):
                if key not in self.range_vals:
                    raise KeyError(
                        f"range_vals provided as dict but missing key '{key}'. "
                        f"Available keys: {list(self.range_vals.keys())}"
                    )
                range_vals_key = self.range_vals[key]
            else:
                range_vals_key = self.range_vals  # same tuple for all inputs

            if self.clip_min_vals is not None:
                if isinstance(self.clip_min_vals, (dict, DictConfig)):
                    clip_min_val = self.clip_min_vals.get(key, None)
                else:
                    # single scalar / sequence applied to all (rare)
                    clip_min_val = self.clip_min_vals
            else:
                clip_min_val = None

            if self.clip_max_vals is not None:
                if isinstance(self.clip_max_vals, (dict, DictConfig)):
                    clip_max_val = self.clip_max_vals.get(key, None)
                else:
                    clip_max_val = self.clip_max_vals
            else:
                clip_max_val = None
            self.running_stats[key] = ImageStatistics(
                shape,
                dims,
                bins=self.bins,
                range_vals=range_vals_key,
                clip_min_val=clip_min_val,
                clip_max_val=clip_max_val,
                normalization_mode=self.normalization_mode,
                compute_quantiles=True,
            ).to(self.device)

    def compute_statistics(self) -> dict[str, dict[str, Any]]:
        """Compute statistics for input data using ImageStatistics.

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
        bins: int = 100,
        range_vals: tuple[float, float] = (0.0, 1.0),
        clip_min_vals: dict[str, float] | None = None,
        clip_max_vals: dict[str, float] | None = None,
        normalization_mode: str = "none",
        input_keys: list[str] = ["image"],
        target_key: str = "label",
        multi_label: bool = False,
        device: str = "cpu",
        save_dir: str | None = None,
        **kwargs,
    ):
        """Initialize classification statistics computer.

        Args:
            datamodule: lightning datamodule
            bins: Number of bins for histogram
            range_vals: Range for histogram
            clip_min_vals: Minimum values for clipping per input_key
            clip_max_vals: Maximum values for clipping per input_key
            normalization_mode: Type of normalization to apply for the second stage stats
            input_keys: Keys for input data in batch dict, can compute statistics for multi-modal inputs
            target_key: Key for target data in batch dict, assume only single target
            multi_label: Whether the classification is multilabel
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
            normalization_mode=normalization_mode,
            input_keys=input_keys,
            target_key=target_key,
            device=device,
            save_dir=save_dir,
            **kwargs,
        )
        self.num_classes = datamodule.num_classes
        self.class_counts = torch.zeros(self.num_classes, device=self.device)
        self.total_samples = 0
        self.multi_label = multi_label

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
        bins: int = 100,
        range_vals: tuple[float, float] = (0.0, 1.0),
        clip_min_vals: dict[str, float] | None = None,
        clip_max_vals: dict[str, float] | None = None,
        normalization_mode: str = "none",
        input_keys: list[str] = ["image"],
        target_key: str = "mask",
        device: str = "cpu",
        save_dir: str | None = None,
        **kwargs,
    ):
        """Initialize segmentation statistics computer.

        Args:
            datamodule: lightning datamodule
            bins: Number of bins for histogram
            range_vals: Range for histogram
            clip_min_vals: Minimum values for clipping per input_key
            clip_max_vals: Maximum values for clipping per input_key
            normalization_mode: Type of normalization to apply for the second stage stats
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
            normalization_mode=normalization_mode,
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
        bins: int = 100,
        range_vals: tuple[float, float] = (0.0, 1.0),
        clip_min_vals: dict[str, float] | None = None,
        clip_max_vals: dict[str, float] | None = None,
        normalization_mode: str = "none",
        target_range_vals: tuple[float, float] = (0.0, 1.0),
        input_keys: list[str] = ["image"],
        target_key: str = "label",
        device: str = "cpu",
        save_dir: str | None = None,
        **kwargs,
    ):
        """Initialize pixel regression statistics computer.

        Args:
            datamodule: lightning datamodule
            bins: Number of bins for histogram
            range_vals: Range for histogram
            clip_min_vals: Minimum values for clipping per input_key
            clip_max_vals: Maximum values for clipping per input_key
            normalization_mode: Type of normalization to apply for the second stage stats
            input_keys: Keys for input data in batch dict
            target_key: Key for target data in batch dict
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
            normalization_mode=normalization_mode,
            input_keys=input_keys,
            target_key=target_key,
            device=device,
            save_dir=save_dir,
            **kwargs,
        )
        self.target_range_vals = target_range_vals
        self.target_stats = ImageStatistics(
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
        bins: int = 100,
        range_vals: tuple[float, float] = (0.0, 1.0),
        clip_min_vals: dict[str, float] | None = None,
        clip_max_vals: dict[str, float] | None = None,
        normalization_mode: str = "none",
        input_keys: list[str] = ["image"],
        target_key: str = "boxes",
        device: str = "cpu",
        save_dir: str | None = None,
        **kwargs,
    ):
        """Initialize object detection statistics computer.

        Args:
            datamodule: lightning datamodule
            bins: Number of bins for histogram
            range_vals: Range for histogram
            clip_min_vals: Minimum values for clipping per input_key
            clip_max_vals: Maximum values for clipping per input_key
            normalization_mode: Type of normalization to apply for the second stage stats
            input_keys: Keys for input data in batch dict
            target_key: Key for target data in batch dict
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
            normalization_mode=normalization_mode,
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

        # replace nan in case of non present category or 0 as background
        box_area = torch.nan_to_num(box_area, nan=0)
        box_width = torch.nan_to_num(box_width, nan=0)
        box_height = torch.nan_to_num(box_height, nan=0)
        box_aspect_ratio = torch.nan_to_num(box_aspect_ratio, nan=0)

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
        """Compute statistics for input data using ImageStatistics.

        Returns:
            dictionary with input statistics for each input key
        """
        for batch in tqdm(self.dataloader, desc="Computing dataset statistics"):
            self.compute_batch_image_statistics(batch)
            self.compute_batch_target_statistics(batch["bbox_xyxy"], batch["label"])

        return self.aggregate_statistics()
