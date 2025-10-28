# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Normalization Modules."""

import json
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from torch import Tensor


def _load_stats_from_path_or_dict(
    stats_source: str | Path,
) -> dict[str, dict[str, float]]:
    """Load statistics from a path to a JSON file.

    Returns a processed stats dictionary with keys like 'means', 'stds',
    and optional 'norm_mean', 'norm_std', 'shift_offsets', 'clip_min', 'clip_max'.
    """
    path = Path(stats_source)
    if not path.exists():
        raise FileNotFoundError(f"Statistics file not found: {path}")

    with open(path) as f:
        stats_dict: dict[str, Any] = json.load(f)

    processed_stats: dict[str, dict[str, float]] = {"means": {}, "stds": {}}

    for modality_key, modality_stats in stats_dict["input_stats"].items():
        band_names: list[str] = modality_stats["band_names"]
        means: list[float] = modality_stats["mean"]
        stds: list[float] = modality_stats["std"]
        for i, band_name in enumerate(band_names):
            processed_stats["means"][band_name] = float(means[i])
            processed_stats["stds"][band_name] = float(stds[i])

            for stat_key in ["norm_mean", "norm_std", "shift_offsets"]:
                if stat_key in modality_stats:
                    if stat_key not in processed_stats:
                        processed_stats[stat_key] = {}
                    processed_stats[stat_key][band_name] = float(
                        modality_stats[stat_key][i]
                    )
        if "clip_min_used" in modality_stats:
            if "clip_min" not in processed_stats:
                processed_stats["clip_min"] = {}
            processed_stats["clip_min"][modality_key] = float(
                modality_stats["clip_min_used"]
            )  # type: ignore[index]

        if "clip_max_used" in modality_stats:
            if "clip_max" not in processed_stats:
                processed_stats["clip_max"] = {}
            processed_stats["clip_max"][modality_key] = float(
                modality_stats["clip_max_used"]
            )  # type: ignore[index]

    return processed_stats


class DataNormalizer(nn.Module, ABC):
    """Base Class for Data Normalization."""

    def __init__(
        self,
        stats: dict[str, dict[str, float]] | Path,
        band_order: list[str | float] | dict[str, list[str | float]],
        image_keys: Sequence[str] | None = None,
    ) -> None:
        """Initialize normalizer.

        Args:
            stats: dictionary containing mean and std for each band, or path to a JSON file
            band_order: Either a sequence of bands or dict mapping modalities to sequences
            image_keys: Keys in the data dictionary to normalize (default: ["image"])
        """
        super().__init__()
        if isinstance(stats, (str, Path)):
            stats_dict = _load_stats_from_path_or_dict(stats)
        else:
            stats_dict = stats

        self.stats: dict[str, dict[str, float]] = stats_dict
        self.band_order: list[str | float] | dict[str, list[str | float]] = band_order
        self.image_keys = image_keys or ["image"]

        self._validate_required_stats()

        self.means: dict[str, Tensor] = {}
        self.stds: dict[str, Tensor] = {}
        self.is_fill_value: dict[str, Tensor] = {}

        self._initialize_statistics()

    def _initialize_statistics(self) -> None:
        """Initialize statistics based on band_order.

        This method populates the normalizer's statistics (means, stds, is_fill_value)
        based on the band_order and stats provided during initialization.

        Subclasses should override this to set additional statistics they need.
        """
        if isinstance(self.band_order, dict):
            for modality, bands in self.band_order.items():
                means, stds, is_fill = self._get_band_stats(bands)

                base_key = f"image_{modality}"
                self.means[base_key] = means
                self.stds[base_key] = stds
                self.is_fill_value[base_key] = is_fill

                self._process_additional_keys(base_key, means, stds, is_fill)

                self._set_additional_stats_for_key(
                    base_key, bands, means, stds, is_fill
                )
        else:
            means, stds, is_fill = self._get_band_stats(self.band_order)

            for key in self.image_keys:
                self.means[key] = means
                self.stds[key] = stds
                self.is_fill_value[key] = is_fill

                self._set_additional_stats_for_key(
                    key, self.band_order, means, stds, is_fill
                )

    def _process_additional_keys(
        self, base_key: str, means: Tensor, stds: Tensor, is_fill: Tensor
    ) -> None:
        """Process additional image keys for multi-modal data."""
        if len(self.image_keys) > 1 and self.image_keys != ["image"]:
            for key in self.image_keys:
                if key == "image":
                    continue

                modality = base_key.split("_")[1]
                modality_key = f"{key}_{modality}"

                self.means[modality_key] = means
                self.stds[modality_key] = stds
                self.is_fill_value[modality_key] = is_fill

                bands_map = cast(dict[str, list[str | float]], self.band_order)
                self._set_additional_stats_for_key(
                    modality_key, bands_map[modality], means, stds, is_fill
                )

    def _set_additional_stats_for_key(
        self,
        key: str,
        bands: Sequence[str | float],
        means: Tensor,
        stds: Tensor,
        is_fill: Tensor,
    ) -> None:
        """Set additional statistics for a specific key.

        Subclasses should override this method to set their additional statistics.
        """
        pass

    def _get_band_stats(
        self, bands: Sequence[str | float]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Extract mean, std tensors and a boolean mask identifying fill value channels."""
        means, stds, is_fill = [], [], []
        for band in bands:
            if isinstance(band, (int | float)):
                means.append(0.0)
                stds.append(1.0)
                is_fill.append(True)
            else:
                if band not in self.stats.get(
                    "means", {}
                ) or band not in self.stats.get("stds", {}):
                    raise ValueError(
                        f"Band '{band}' not found in normalization statistics (means/stds)."
                    )
                means.append(self.stats["means"][band])
                stds.append(self.stats["stds"][band])
                is_fill.append(False)
        return torch.tensor(means), torch.tensor(stds), torch.tensor(is_fill)

    def _reshape_and_expand(
        self, tensor_to_reshape: Tensor, target_tensor: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        """Reshape 1D stat/mask tensor and optionally expand boolean masks for broadcasting."""
        orig_dim = target_tensor.dim()
        if orig_dim == 3:  # [C, H, W]
            reshaped = tensor_to_reshape.view(-1, 1, 1)
        elif orig_dim == 4:  # [T, C, H, W]
            reshaped = tensor_to_reshape.view(1, -1, 1, 1)
        elif orig_dim == 5:  # [B, T, C, H, W]
            reshaped = tensor_to_reshape.view(1, 1, -1, 1, 1)
        else:
            raise ValueError(
                f"Expected target tensor with 3, 4, or 5 dimensions, got {orig_dim}"
            )

        expanded = None
        if tensor_to_reshape.dtype == torch.bool:
            expanded = reshaped.expand_as(target_tensor)

        return reshaped, expanded

    @abstractmethod
    def forward(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Normalize input tensors."""
        pass

    @abstractmethod
    def unnormalize(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Unnormalize input tensors."""
        pass

    @abstractmethod
    def _get_required_stats(self) -> dict[str, str]:
        """Return a dictionary of required statistics and their descriptions.

        Returns:
            Dictionary mapping stat keys to descriptions of their purpose
        """
        return {
            "means": "Mean values for each band",
            "stds": "Standard deviation values for each band",
        }

    def _validate_required_stats(self) -> None:
        """Validate that all required statistics are available in the stats dictionary.

        Raises:
            ValueError: If a required statistic is missing
        """
        for stat_key, description in self._get_required_stats().items():
            if stat_key not in self.stats:
                raise ValueError(
                    f"Missing required statistic: {stat_key} ({description})"
                )

    def __repr__(self) -> str:
        """Return string representation."""
        lines = [f"{self.__class__.__name__}("]
        for key in sorted(self.means.keys()):
            lines.append(f"\n  {key}:")
            n_channels = len(self.means[key])

            for i in range(n_channels):
                if self.is_fill_value[key][i]:
                    lines.append(f"    Channel {i}: Fill Value (no normalization)")
                else:
                    lines.append(self._format_channel_stats(key, i))

        return "\n".join(lines) + "\n)"

    def _format_channel_stats(self, key: str, channel_idx: int) -> str:
        """Format statistics for a specific channel.

        Subclasses should override this method to customize the representation.
        """
        mean = self.means[key][channel_idx].item()
        std = self.stds[key][channel_idx].item()
        return f"    Channel {channel_idx}: mean={mean:.4f}, std={std:.4f}"


class ClipZScoreNormalizer(DataNormalizer):
    """Normalization module applying sequential optional clipping and z-score normalization.

    Applies normalization per channel based on band configuration:
    1. If 'clip_min' and 'clip_max' are defined for a band in stats:
        a. Clips values to [clip_min, clip_max]. Bands without defined limits are not clipped.
    2. Applies standard z-score normalization: (value - mean) / std to the (potentially clipped) values.
    3. Fill value bands (numeric values in band_order) are ignored and passed through unchanged.

    Handles both single tensor input (if band_order is a list) and dictionary input
    (if band_order is a dict mapping modalities to lists).
    """

    def __init__(
        self,
        stats: dict[str, dict[str, float]] | Path,
        band_order: list[str | float] | dict[str, list[str | float]],
        image_keys: Sequence[str] | None = None,
    ) -> None:
        """Initialize normalizer applying clip then z-score."""
        self.clip_mins: dict[str, Tensor] = {}
        self.clip_maxs: dict[str, Tensor] = {}

        self.rescale_shifts: dict[str, Tensor] = {}
        self.rescale_scales: dict[str, Tensor] = {}

        self.norm_means: dict[str, Tensor] = {}
        self.norm_stds: dict[str, Tensor] = {}

        super().__init__(stats, band_order, image_keys)

    def _compute_rescale_stats(
        self, clip_means: float, clip_stds: float, shifts: float, scales: float
    ) -> tuple[float, float]:
        """Compute rescale norm statistics from clip statistics."""
        rescale_means = (clip_means + shifts) / scales
        rescale_stds = clip_stds / scales
        return rescale_means, rescale_stds

    def _set_additional_stats_for_key(
        self,
        key: str,
        bands: Sequence[str | float],
        means: Tensor,
        stds: Tensor,
        is_fill: Tensor,
    ) -> None:
        """Set clip min/max values and normalization parameters for this key."""
        clip_min, clip_max = self._get_clip_values(key, bands)
        self.clip_mins[key] = clip_min
        self.clip_maxs[key] = clip_max

        shifts = torch.zeros_like(clip_min)
        neg_values = clip_min < 0
        if neg_values.any():
            shifts[neg_values] = -clip_min[neg_values]

        scales = (clip_max + shifts) - (clip_min + shifts).clamp(min=0)
        scales = scales.clamp(min=1e-6)

        self.rescale_shifts[key] = shifts
        self.rescale_scales[key] = scales

        norm_means = []
        norm_stds = []

        for i, band in enumerate(bands):
            if isinstance(band, (int, float)):
                norm_means.append(0.0)
                norm_stds.append(1.0)
            else:
                if "norm_mean" in self.stats and band in self.stats["norm_mean"]:
                    mean = float(self.stats["norm_mean"][band])
                    std = float(self.stats["norm_std"][band])
                    norm_means.append(mean)
                    norm_stds.append(std)
                else:
                    mean_f = float(self.stats["means"][band])
                    std_f = float(self.stats["stds"][band])
                    shift = float(self.rescale_shifts[key][i].item())
                    scale = float(self.rescale_scales[key][i].item())
                    norm_mean, norm_std = self._compute_rescale_stats(
                        mean_f, std_f, shift, scale
                    )
                    norm_means.append(norm_mean)
                    norm_stds.append(norm_std)

        self.norm_means[key] = torch.tensor(norm_means)
        self.norm_stds[key] = torch.tensor(norm_stds)

    def _get_clip_values(
        self, key: str, bands: Sequence[str | float]
    ) -> tuple[Tensor, Tensor]:
        """Extract clip min/max tensors with infinity as default."""
        if "clip_min" in self.stats:
            clip_min = self.stats["clip_min"].get(key, float("-inf"))
        else:
            clip_min = float("-inf")

        if "clip_max" in self.stats:
            clip_max = self.stats["clip_max"].get(key, float("inf"))
        else:
            clip_max = float("inf")

        # Apply to each band (fill values use infinity)
        clip_mins = [
            clip_min if not isinstance(band, (int | float)) else float("-inf")
            for band in bands
        ]
        clip_maxs = [
            clip_max if not isinstance(band, (int | float)) else float("inf")
            for band in bands
        ]

        return torch.tensor(clip_mins).float(), torch.tensor(clip_maxs).float()

    def forward(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Apply normalization based on mode.

        Args:
            data: Dictionary of input tensors

        Returns:
            Dictionary of normalized tensors
        """
        result = {}
        for key, tensor in data.items():
            if key not in self.means:
                result[key] = tensor
                continue

            normalized = tensor.clone()
            clip_min = self.clip_mins[key]
            clip_max = self.clip_maxs[key]
            shifts = self.rescale_shifts[key]
            scales = self.rescale_scales[key]
            norm_mean = self.norm_means[key]
            norm_std = self.norm_stds[key]
            is_fill = self.is_fill_value[key]

            clip_min_reshaped, _ = self._reshape_and_expand(clip_min, tensor)
            clip_max_reshaped, _ = self._reshape_and_expand(clip_max, tensor)
            shifts_reshaped, _ = self._reshape_and_expand(shifts, tensor)
            scales_reshaped, _ = self._reshape_and_expand(scales, tensor)
            mean_reshaped, _ = self._reshape_and_expand(norm_mean, tensor)
            std_reshaped, _ = self._reshape_and_expand(norm_std, tensor)
            _, is_fill_expanded = self._reshape_and_expand(is_fill, tensor)

            clipped = torch.clamp(tensor, min=clip_min_reshaped, max=clip_max_reshaped)

            shifted = clipped + shifts_reshaped

            rescaled = shifted / scales_reshaped
            rescaled = torch.clamp(rescaled, min=0.0, max=1.0)

            z_score = (rescaled - mean_reshaped) / (std_reshaped + 1e-6)
            normalized = torch.where(is_fill_expanded, normalized, z_score)

            result[key] = normalized

        return result

    def unnormalize(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Reverse the normalization based on mode.

        Args:
            data: Dictionary of normalized tensors

        Returns:
            Dictionary of unnormalized tensors
        """
        result = {}
        for key, tensor in data.items():
            if key not in self.means:
                result[key] = tensor
                continue

            unnormalized = tensor.clone()
            is_fill = self.is_fill_value[key]
            _, is_fill_expanded = self._reshape_and_expand(is_fill, tensor)

            norm_mean = self.norm_means[key]
            norm_std = self.norm_stds[key]
            shifts = self.rescale_shifts[key]
            scales = self.rescale_scales[key]

            mean_reshaped, _ = self._reshape_and_expand(norm_mean, tensor)
            std_reshaped, _ = self._reshape_and_expand(norm_std, tensor)
            shifts_reshaped, _ = self._reshape_and_expand(shifts, tensor)
            scales_reshaped, _ = self._reshape_and_expand(scales, tensor)

            unscaled_01 = tensor * (std_reshaped + 1e-6) + mean_reshaped
            unscaled_01 = torch.clamp(unscaled_01, min=0.0, max=1.0)

            unscaled = unscaled_01 * scales_reshaped
            original = unscaled - shifts_reshaped

            unnormalized = torch.where(is_fill_expanded, unnormalized, original)

            result[key] = unnormalized

        return result

    def _get_required_stats(self) -> dict[str, str]:
        """Return required statistics for Clip-Z-Score normalization."""
        return {
            "means": "Mean values for each band",
            "stds": "Standard deviation values for each band",
            "clip_min": "Optional minimum clipping values by modality",
            "clip_max": "Optional maximum clipping values by modality",
        }

    def _format_channel_stats(self, key: str, channel_idx: int) -> str:
        """Format clip and statistics info for a specific channel."""
        mean = self.means[key][channel_idx].item()
        std = self.stds[key][channel_idx].item()

        clip_min = self.clip_mins[key][channel_idx].item()
        clip_max = self.clip_maxs[key][channel_idx].item()
        shift = self.rescale_shifts[key][channel_idx].item()
        scale = self.rescale_scales[key][channel_idx].item()
        norm_mean = self.norm_means[key][channel_idx].item()
        norm_std = self.norm_stds[key][channel_idx].item()

        clip_info = ""
        if clip_min > float("-inf") or clip_max < float("inf"):
            clip_info = f", clip=[{clip_min:.4f}, {clip_max:.4f}]"

        shift_info = f", shift={shift:.4f}" if shift > 0 else ""
        scale_info = f", scale={scale:.4f}"
        norm_info = f", norm_mean={norm_mean:.4f}, norm_std={norm_std:.4f}"

        return f"    Channel {channel_idx}: mean={mean:.4f}, std={std:.4f}{clip_info}{shift_info}{scale_info}{norm_info}"


class ZScoreNormalizer(DataNormalizer):
    """Normalization module applying standard z-score normalization.

    This normalizer performs standard z-score normalization by subtracting the mean
    and dividing by the standard deviation for each band:
        normalized = (x - mean) / std

    Fill value bands (numeric values in band_order) are preserved unchanged.
    """

    def __init__(
        self,
        stats: dict[str, dict[str, float]] | Path,
        band_order: list[str | float] | dict[str, list[str | float]],
        image_keys: Sequence[str] | None = None,
    ) -> None:
        """Initialize z-score normalizer.

        Args:
            stats: Statistics dictionary or path to JSON file
            band_order: Sequence or dict of band names/fill values
            image_keys: Keys to normalize in data dict
        """
        super().__init__(stats, band_order, image_keys)

    def forward(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Apply z-score normalization to input tensors."""
        result = {}
        for key, tensor in data.items():
            if key not in self.means:
                result[key] = tensor
                continue

            is_fill = self.is_fill_value[key]
            _, is_fill_expanded = self._reshape_and_expand(is_fill, tensor)
            normalized = tensor.clone()

            mean, std = self.means[key], self.stds[key]
            mean_r, _ = self._reshape_and_expand(mean, tensor)
            std_r, _ = self._reshape_and_expand(std, tensor)

            z_score = (tensor - mean_r) / (std_r + 1e-6)
            normalized = torch.where(is_fill_expanded, normalized, z_score)

            result[key] = normalized

        return result

    def unnormalize(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Reverse the z-score normalization process."""
        result = {}
        for key, tensor in data.items():
            if key not in self.means:
                result[key] = tensor
                continue

            is_fill = self.is_fill_value[key]
            _, is_fill_expanded = self._reshape_and_expand(is_fill, tensor)
            unnormalized = tensor.clone()

            mean, std = self.means[key], self.stds[key]
            mean_r, _ = self._reshape_and_expand(mean, tensor)
            std_r, _ = self._reshape_and_expand(std, tensor)

            original = tensor * (std_r + 1e-6) + mean_r
            unnormalized = torch.where(is_fill_expanded, unnormalized, original)

            result[key] = unnormalized

        return result

    def _get_required_stats(self) -> dict[str, str]:
        """Return required statistics for Z-Score normalization."""
        return {
            "means": "Mean values for each band",
            "stds": "Standard deviation values for each band",
        }


class RescaleNormalizer(DataNormalizer):
    """Normalization module applying clipping and rescaling.

    This normalizer performs the following operations:
    1. Clip values to a predefined min/max range
    2. Shift clipped values to ensure non-negative range
    3. Rescale to [0,1] range by dividing by the range size
    4. Optionally adjust to different output ranges

    Fill value bands (numeric values in band_order) are preserved unchanged.
    """

    valid_output_ranges = ["zero_one", "zero_255", "neg_one_one"]

    def __init__(
        self,
        stats: dict[str, dict[str, float]] | Path,
        band_order: list[str | float] | dict[str, list[str | float]],
        image_keys: Sequence[str] | None = None,
        output_range: str = "zero_one",
    ) -> None:
        """Initialize rescale normalizer with configurable output range.

        Args:
            stats: Statistics dictionary or path to JSON file
            band_order: Sequence or dict of band names/fill values
            image_keys: Keys to normalize in data dict
            output_range: Target range ("zero_one", "zero_255", "neg_one_one")
        """
        assert output_range in self.valid_output_ranges, (
            f"output_range must be one of {self.valid_output_ranges}, got {output_range}"
        )

        self.output_range = output_range
        self.clip_mins: dict[str, Tensor] = {}
        self.clip_maxs: dict[str, Tensor] = {}
        self.shifts: dict[str, Tensor] = {}
        self.scales: dict[str, Tensor] = {}

        if output_range == "zero_255":
            self.range_scale, self.range_shift = 255.0, 0.0
        elif output_range == "neg_one_one":
            self.range_scale, self.range_shift = 2.0, -1.0
        else:
            self.range_scale, self.range_shift = 1.0, 0.0

        super().__init__(stats, band_order, image_keys)

    def _set_additional_stats_for_key(
        self,
        key: str,
        bands: Sequence[str | float],
        means: Tensor,
        stds: Tensor,
        is_fill: Tensor,
    ) -> None:
        """Set clip min/max values and compute shifts and scales for rescaling."""
        clip_min, clip_max = self._get_clip_values(key, bands)
        self.clip_mins[key] = clip_min
        self.clip_maxs[key] = clip_max

        shifts = torch.zeros_like(clip_min)
        neg_values = clip_min < 0
        if neg_values.any():
            shifts[neg_values] = -clip_min[neg_values]

        scales = (clip_max + shifts) - (clip_min + shifts).clamp(min=0)
        scales = scales.clamp(min=1e-6)

        self.shifts[key] = shifts
        self.scales[key] = scales

    def _get_clip_values(
        self, key: str, bands: Sequence[str | float]
    ) -> tuple[Tensor, Tensor]:
        """Extract clip min/max tensors with infinity as default."""
        if "clip_min" in self.stats:
            clip_min = self.stats["clip_min"].get(key, float("-inf"))
        else:
            clip_min = float("-inf")

        if "clip_max" in self.stats:
            clip_max = self.stats["clip_max"].get(key, float("inf"))
        else:
            clip_max = float("inf")

        clip_mins = [
            clip_min if not isinstance(band, (int | float)) else float("-inf")
            for band in bands
        ]
        clip_maxs = [
            clip_max if not isinstance(band, (int | float)) else float("inf")
            for band in bands
        ]

        return torch.tensor(clip_mins).float(), torch.tensor(clip_maxs).float()

    def forward(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Apply clipping and rescaling to input tensors."""
        result = {}
        for key, tensor in data.items():
            if key not in self.means:
                result[key] = tensor
                continue

            is_fill = self.is_fill_value[key]
            _, is_fill_expanded = self._reshape_and_expand(is_fill, tensor)
            normalized = tensor.clone()

            clip_min, clip_max = self.clip_mins[key], self.clip_maxs[key]
            shifts, scales = self.shifts[key], self.scales[key]

            clip_min_r, _ = self._reshape_and_expand(clip_min, tensor)
            clip_max_r, _ = self._reshape_and_expand(clip_max, tensor)
            shifts_r, _ = self._reshape_and_expand(shifts, tensor)
            scales_r, _ = self._reshape_and_expand(scales, tensor)

            clipped = torch.clamp(tensor, min=clip_min_r, max=clip_max_r)

            shifted = clipped + shifts_r

            rescaled = shifted / scales_r

            if self.output_range != "zero_one":
                rescaled = rescaled * self.range_scale + self.range_shift

            normalized = torch.where(is_fill_expanded, normalized, rescaled)

            result[key] = normalized

        return result

    def unnormalize(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Reverse the clipping and rescaling process."""
        result = {}
        for key, tensor in data.items():
            if key not in self.means:
                result[key] = tensor
                continue

            is_fill = self.is_fill_value[key]
            _, is_fill_expanded = self._reshape_and_expand(is_fill, tensor)
            unnormalized = tensor.clone()

            shifts, scales = self.shifts[key], self.scales[key]

            shifts_r, _ = self._reshape_and_expand(shifts, tensor)
            scales_r, _ = self._reshape_and_expand(scales, tensor)

            denormalized = tensor

            if self.output_range != "zero_one":
                denormalized = (denormalized - self.range_shift) / self.range_scale

            denormalized = denormalized * scales_r

            denormalized = denormalized - shifts_r

            unnormalized = torch.where(is_fill_expanded, unnormalized, denormalized)

            result[key] = unnormalized

        return result

    def _get_required_stats(self) -> dict[str, str]:
        """Return required statistics for Rescale normalization."""
        return {
            "means": "Mean values for each band",
            "stds": "Standard deviation values for each band",
            "clip_min": "Optional minimum clipping values by modality",
            "clip_max": "Optional maximum clipping values by modality",
        }

    def _format_channel_stats(self, key: str, channel_idx: int) -> str:
        """Format statistics for string representation."""
        mean = self.means[key][channel_idx].item()
        std = self.stds[key][channel_idx].item()

        parts = [f"Channel {channel_idx}: mean={mean:.4f}, std={std:.4f}"]

        if key in self.clip_mins:
            clip_min = self.clip_mins[key][channel_idx].item()
            clip_max = self.clip_maxs[key][channel_idx].item()
            if clip_min > float("-inf") or clip_max < float("inf"):
                parts.append(f"clip=[{clip_min:.4f}, {clip_max:.4f}]")

        if key in self.shifts:
            shift = self.shifts[key][channel_idx].item()
            scale = self.scales[key][channel_idx].item()
            if shift != 0:
                parts.append(f"shift={shift:.4f}")
            parts.append(f"scale={scale:.4f}")

        return " " + ", ".join(parts)


class SatMAENormalizer(DataNormalizer):
    """Normalization module for satellite imagery with SatMAE-style normalization.

    Several papers have cited SatMAE for this normalization procedure:
    - https://github.com/sustainlab-group/SatMAE/blob/e31c11fa1bef6f9a9aa3eb49e8637c8b8952ba5e/util/datasets.py#L358

    They mention that the normalization is inspired from SeCO:
    - https://github.com/ServiceNow/seasonal-contrast/blob/8285173ec205b64bc3e53b880344dd6c3f79fa7a/datasets/bigearthnet_dataset.py#L111

    This normalization:
    1. For bands with negative min values: shifts data to non-negative range first
    2. Clips values to [mean - 2*std, mean + 2*std] (after shifting if needed)
    3. Rescales to target range: [0, 1], [0, 255], or [-1, 1]
    4. Preserves fill values unchanged
    5. Optionally applies ImageNet-style normalization to the [0,1] range data
    """

    valid_ranges = ["zero_one", "zero_255", "neg_one_one"]

    def __init__(
        self,
        stats: dict[str, dict[str, float]] | Path,
        band_order: list[str | float] | dict[str, list[str | float]],
        image_keys: Sequence[str] | None = None,
        output_range: str = "zero_one",
        apply_second_stage: bool = False,
    ) -> None:
        """Initialize SatMAE normalizer with configurable output range and second stage.

        Args:
            stats: Statistics dictionary or path to JSON file
            band_order: Sequence or dict of band names/fill values
            image_keys: Keys to normalize in data dict
            output_range: Target range ("zero_one", "zero_255", "neg_one_one")
            apply_second_stage: Whether to apply ImageNet-style normalization after [0,1] scaling
        """
        if output_range not in self.valid_ranges:
            raise ValueError(f"output_range must be one of {self.valid_ranges}")

        self.output_range = output_range
        self.apply_second_stage = apply_second_stage

        if output_range == "zero_255":
            self.scale_factor, self.shift_factor = 255.0, 0.0
        elif output_range == "neg_one_one":
            self.scale_factor, self.shift_factor = 2.0, -1.0
        else:  # zero_one
            self.scale_factor, self.shift_factor = 1.0, 0.0

        self.raw_min_values: dict[str, Tensor] = {}
        self.raw_max_values: dict[str, Tensor] = {}
        self.offsets: dict[str, Tensor] = {}
        self.min_values: dict[str, Tensor] = {}
        self.max_values: dict[str, Tensor] = {}
        self.norm_means: dict[str, Tensor] = {}
        self.norm_stds: dict[str, Tensor] = {}

        super().__init__(stats, band_order, image_keys)

    def _set_additional_stats_for_key(
        self,
        key: str,
        bands: Sequence[str | float],
        means: Tensor,
        stds: Tensor,
        is_fill: Tensor,
    ) -> None:
        """Set normalization parameters for both stages."""
        raw_min_values = means - 2 * stds
        raw_max_values = means + 2 * stds

        offsets = torch.zeros_like(raw_min_values)

        shift_offsets_for_bands = []
        for i, band in enumerate(bands):
            if isinstance(band, (str | float)):
                if isinstance(band, str) and band in self.stats.get(
                    "shift_offsets", {}
                ):
                    shift_offsets_for_bands.append(self.stats["shift_offsets"][band])
                else:
                    shift_offsets_for_bands.append(
                        -raw_min_values[i].item() if raw_min_values[i] < 0 else 0.0
                    )

        if len(shift_offsets_for_bands) == len(bands):
            offsets = torch.tensor(shift_offsets_for_bands)
        else:
            neg_mask = raw_min_values < 0
            if neg_mask.any():
                offsets[neg_mask] = -raw_min_values[neg_mask]

        self.raw_min_values[key] = raw_min_values
        self.raw_max_values[key] = raw_max_values
        self.offsets[key] = offsets
        self.min_values[key] = raw_min_values + offsets
        self.max_values[key] = raw_max_values + offsets

        if self.apply_second_stage:
            self.norm_means[key], self.norm_stds[key] = self._get_normalized_stats(
                bands
            )

    def _get_normalized_stats(
        self, bands: Sequence[str | float]
    ) -> tuple[Tensor, Tensor]:
        """Extract normalized mean and std values for bands."""
        norm_means, norm_stds = [], []

        for band in bands:
            if isinstance(band, (int | float)):
                norm_means.append(0.0)
                norm_stds.append(1.0)
            else:
                norm_means.append(self.stats.get("norm_mean", {}).get(band, 0.0))
                norm_stds.append(self.stats.get("norm_std", {}).get(band, 1.0))

        return torch.tensor(norm_means), torch.tensor(norm_stds)

    def forward(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Apply SatMAE normalization to input tensors."""
        result = {}
        for key, tensor in data.items():
            if key not in self.min_values:
                result[key] = tensor
                continue

            normalized = self._normalize_first_stage(tensor, key)

            if self.apply_second_stage and key in self.norm_means:
                normalized = self._normalize_second_stage(normalized, key)

            result[key] = normalized

        return result

    def _normalize_first_stage(self, tensor: Tensor, key: str) -> Tensor:
        """First stage: shift, clip to mean±2std, rescale to target range."""
        min_val, _ = self._reshape_and_expand(self.min_values[key], tensor)
        max_val, _ = self._reshape_and_expand(self.max_values[key], tensor)
        offsets, _ = self._reshape_and_expand(self.offsets[key], tensor)
        _, is_fill_mask = self._reshape_and_expand(self.is_fill_value[key], tensor)

        result = tensor.clone()

        shifted = tensor + offsets

        normalized = (shifted - min_val) / (max_val - min_val + 1e-6)
        normalized = torch.clamp(normalized, 0, 1)

        if not (self.apply_second_stage and self.output_range == "zero_one"):
            if self.output_range != "zero_one":
                normalized = normalized * self.scale_factor + self.shift_factor

        return torch.where(is_fill_mask, result, normalized)

    def _normalize_second_stage(self, tensor: Tensor, key: str) -> Tensor:
        """Second stage: apply ImageNet-style normalization to [0,1] data."""
        mean, _ = self._reshape_and_expand(self.norm_means[key], tensor)
        std, _ = self._reshape_and_expand(self.norm_stds[key], tensor)
        _, is_fill_mask = self._reshape_and_expand(self.is_fill_value[key], tensor)

        normalized = (tensor - mean) / (std + 1e-6)

        return torch.where(is_fill_mask, tensor.clone(), normalized)

    def unnormalize(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Revert SatMAE normalization to recover original values."""
        result = {}
        for key, tensor in data.items():
            if key not in self.min_values:
                result[key] = tensor
                continue

            if self.apply_second_stage and key in self.norm_means:
                tensor = self._denormalize_second_stage(tensor, key)

            result[key] = self._denormalize_first_stage(tensor, key)

        return result

    def _denormalize_second_stage(self, tensor: Tensor, key: str) -> Tensor:
        """Undo ImageNet-style normalization."""
        mean, _ = self._reshape_and_expand(self.norm_means[key], tensor)
        std, _ = self._reshape_and_expand(self.norm_stds[key], tensor)
        _, is_fill_mask = self._reshape_and_expand(self.is_fill_value[key], tensor)

        denormalized = tensor * (std + 1e-6) + mean

        return torch.where(is_fill_mask, tensor.clone(), denormalized)

    def _denormalize_first_stage(self, tensor: Tensor, key: str) -> Tensor:
        """Undo first stage normalization (scaling, clipping, shifting)."""
        min_val, _ = self._reshape_and_expand(self.min_values[key], tensor)
        max_val, _ = self._reshape_and_expand(self.max_values[key], tensor)
        offsets, _ = self._reshape_and_expand(self.offsets[key], tensor)
        _, is_fill_mask = self._reshape_and_expand(self.is_fill_value[key], tensor)

        result = tensor.clone()

        temp = tensor
        if not self.apply_second_stage and self.output_range != "zero_one":
            temp = (
                (temp - self.shift_factor) / self.scale_factor
                if self.shift_factor != 0
                else temp / self.scale_factor
            )

        denormalized = temp * (max_val - min_val) + min_val

        denormalized = denormalized - offsets

        return torch.where(is_fill_mask, result, denormalized)

    def _get_required_stats(self) -> dict[str, str]:
        """Return required statistics for SatMAE normalization."""
        required = {
            "means": "Mean values for each band",
            "stds": "Standard deviation values for each band",
        }

        if self.apply_second_stage:
            required.update(
                {
                    "norm_mean": "Mean values for second-stage normalization",
                    "norm_std": "Standard deviation values for second-stage normalization",
                }
            )

        return required

    def _format_channel_stats(self, key: str, channel_idx: int) -> str:
        """Format statistics for string representation with normalization details."""
        mean = self.means[key][channel_idx].item()
        std = self.stds[key][channel_idx].item()

        stats = [f"Channel {channel_idx}: mean={mean:.4f}, std={std:.4f}"]

        if key in self.min_values:
            min_val = self.min_values[key][channel_idx].item()
            max_val = self.max_values[key][channel_idx].item()
            stats.append(f"range=[{min_val:.4f}, {max_val:.4f}]")

            offset = self.offsets[key][channel_idx].item()
            if offset > 0:
                stats.append(f"offset={offset:.4f}")

        if self.apply_second_stage and key in self.norm_means:
            norm_mean = self.norm_means[key][channel_idx].item()
            norm_std = self.norm_stds[key][channel_idx].item()
            stats.append(f"norm_mean={norm_mean:.4f}, norm_std={norm_std:.4f}")

        return "  " + ", ".join(stats)


class ClipOnlyNormalizer(DataNormalizer):
    """Normalization module that only applies clipping to input data.

    This normalizer performs minimal preprocessing by simply clipping values to a
    predefined min/max range for each band. Unlike other normalizers, it doesn't
    rescale or transform the data further.

    Fill value bands (numeric values in band_order) are preserved unchanged.
    """

    def __init__(
        self,
        stats: dict[str, dict[str, float]] | Path,
        band_order: list[str | float] | dict[str, list[str | float]],
        image_keys: Sequence[str] | None = None,
    ) -> None:
        """Initialize clip-only normalizer.

        Args:
            stats: Statistics dictionary or path to JSON file containing clip_min/clip_max values
            band_order: Sequence or dict of band names/fill values
            image_keys: Keys to normalize in data dict
        """
        self.clip_mins: dict[str, Tensor] = {}
        self.clip_maxs: dict[str, Tensor] = {}
        super().__init__(stats, band_order, image_keys)

    def _set_additional_stats_for_key(
        self,
        key: str,
        bands: Sequence[str | float],
        means: Tensor,
        stds: Tensor,
        is_fill: Tensor,
    ) -> None:
        """Set clip min/max values for this key."""
        clip_min, clip_max = self._get_clip_values(key, bands)
        self.clip_mins[key] = clip_min
        self.clip_maxs[key] = clip_max

    def _get_clip_values(
        self, key: str, bands: Sequence[str | float]
    ) -> tuple[Tensor, Tensor]:
        """Extract clip min/max tensors with infinity as default."""
        # Check for modality-specific clip values
        if "clip_min" in self.stats:
            clip_min = self.stats["clip_min"].get(key, float("-inf"))
        else:
            clip_min = float("-inf")

        if "clip_max" in self.stats:
            clip_max = self.stats["clip_max"].get(key, float("inf"))
        else:
            clip_max = float("inf")

        # Apply to each band (fill values use infinity)
        clip_mins = [
            clip_min if not isinstance(band, (int | float)) else float("-inf")
            for band in bands
        ]
        clip_maxs = [
            clip_max if not isinstance(band, (int | float)) else float("inf")
            for band in bands
        ]

        return torch.tensor(clip_mins).float(), torch.tensor(clip_maxs).float()

    def forward(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Apply clipping to input tensors."""
        result = {}
        for key, tensor in data.items():
            if key not in self.means:
                result[key] = tensor
                continue

            is_fill = self.is_fill_value[key]
            _, is_fill_expanded = self._reshape_and_expand(is_fill, tensor)
            normalized = tensor.clone()

            # Only proceed with clipping if we have clip values for this key
            if key in self.clip_mins:
                clip_min, clip_max = self.clip_mins[key], self.clip_maxs[key]

                clip_min_r, _ = self._reshape_and_expand(clip_min, tensor)
                clip_max_r, _ = self._reshape_and_expand(clip_max, tensor)

                # Apply clipping
                clipped = torch.clamp(tensor, min=clip_min_r, max=clip_max_r)
                normalized = torch.where(is_fill_expanded, normalized, clipped)

            result[key] = normalized

        return result

    def unnormalize(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Passthrough since clipping cannot be undone."""
        return data.copy()

    def _get_required_stats(self) -> dict[str, str]:
        """Return required statistics for Clip-only normalization."""
        return {
            "means": "Mean values for each band",
            "stds": "Standard deviation values for each band",
            "clip_min": "Optional minimum clipping values by modality",
            "clip_max": "Optional maximum clipping values by modality",
        }

    def _format_channel_stats(self, key: str, channel_idx: int) -> str:
        """Format channel statistics for string representation."""
        mean = self.means[key][channel_idx].item()
        std = self.stds[key][channel_idx].item()

        parts = [f"Channel {channel_idx}: mean={mean:.4f}, std={std:.4f}"]

        if key in self.clip_mins:
            clip_min = self.clip_mins[key][channel_idx].item()
            clip_max = self.clip_maxs[key][channel_idx].item()

            if clip_min > float("-inf"):
                parts.append(f"clip_min={clip_min:.4f}")
            if clip_max < float("inf"):
                parts.append(f"clip_max={clip_max:.4f}")

        return "    " + ", ".join(parts)


class MultiModalNormalizer(ZScoreNormalizer):
    """Deprecated. Use :class:`ZScoreNormalizer` instead.

    This class is kept for backward compatibility. It will be removed in a future release.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize MultiModalNormalizer and issue deprecation warning.

        Args:
            *args: Positional arguments for ZScoreNormalizer
            **kwargs: Keyword arguments for ZScoreNormalizer
        """
        warnings.warn(
            "MultiModalNormalizer is deprecated. Please use ZScoreNormalizer instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
