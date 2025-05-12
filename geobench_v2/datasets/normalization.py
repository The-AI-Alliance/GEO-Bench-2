# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Normalization Modules."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from pathlib import Path
import json


def _load_stats_from_path_or_dict(stats_path: str):
    """Load statistics from a path to a JSON file or use provided dict directly."""
    stats_path = Path(stats_path)
    if not stats_path.exists():
        raise FileNotFoundError(f"Statistics file not found: {stats_path}")

    with open(stats_path, "r") as f:
        stats_dict = json.load(f)

    processed_stats = {"means": {}, "stds": {}}

    for modality_key, modality_stats in stats_dict["input_stats"].items():
        for i, band_name in enumerate(modality_stats["band_names"]):
            processed_stats["means"][band_name] = modality_stats["mean"][i]
            processed_stats["stds"][band_name] = modality_stats["std"][i]

            for stat_key in [
                "norm_mean",
                "norm_std",
                "pct_02",
                "pct_98",
                "shift_offsets",
            ]:
                if stat_key in modality_stats:
                    if stat_key not in processed_stats:
                        processed_stats[stat_key] = {}
                    processed_stats[stat_key][band_name] = modality_stats[stat_key][i]

        if "clip_min_used" in modality_stats:
            processed_stats["clip_min"] = modality_stats["clip_min_used"]

        if "clip_max_used" in modality_stats:
            processed_stats["clip_max"] = modality_stats["clip_max_used"]

    return processed_stats


class DataNormalizer(nn.Module, ABC):
    """Base Class for Data Normalization."""

    def __init__(
        self,
        stats: dict[str, dict[str, float]] | str,
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
        if isinstance(stats, str | Path):
            stats = _load_stats_from_path_or_dict(stats)

        self.stats = stats
        self.band_order = band_order
        self.image_keys = image_keys or ["image"]

        self.means = {}
        self.stds = {}
        self.is_fill_value = {}

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

                modality_key = f"{key}_{base_key.split('_')[1]}"

                self.means[modality_key] = means
                self.stds[modality_key] = stds
                self.is_fill_value[modality_key] = is_fill

                self._set_additional_stats_for_key(
                    modality_key,
                    self.band_order[base_key.split("_")[1]],
                    means,
                    stds,
                    is_fill,
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
            if isinstance(band, (int, float)):
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

    valid_processing_modes = ["none", "clip_only", "clip_rescale"]

    def __init__(
        self,
        stats: dict[str, dict[str, float]] | str,
        band_order: list[str | float] | dict[str, list[str | float]],
        image_keys: Sequence[str] | None = None,
        processing_mode: str = "none",
    ) -> None:
        """Initialize normalizer applying clip then z-score."""

        assert processing_mode in self.valid_processing_modes, (
            f"processing_mode must be one of {self.valid_processing_modes}, got {processing_mode}"
        )

        self.processing_mode = processing_mode

        self.clip_mins = {}
        self.clip_maxs = {}

        self.rescale_shifts = {}
        self.rescale_scales = {}

        # For using normalized statistics when clipping is applied
        self.norm_means = {}
        self.norm_stds = {}

        super().__init__(stats, band_order, image_keys)

    def _set_additional_stats_for_key(
        self,
        key: str,
        bands: Sequence[str | float],
        means: Tensor,
        stds: Tensor,
        is_fill: Tensor,
    ) -> None:
        """Set clip min/max values and normalization parameters for this key."""
        # Get clip values for each band
        clip_min, clip_max = self._get_clip_values(bands)
        self.clip_mins[key] = clip_min
        self.clip_maxs[key] = clip_max

        # For "clip_rescale" mode, compute shift and scale factors
        if self.processing_mode == "clip_rescale":
            # Calculate shifts for bands with negative min values
            shifts = torch.zeros_like(clip_min)
            neg_values = clip_min < 0
            if neg_values.any():
                shifts[neg_values] = -clip_min[neg_values]

            # Calculate scales for rescaling to [0,1]
            scales = (clip_max + shifts) - (clip_min + shifts).clamp(min=0)
            scales = scales.clamp(min=1e-6)  # Avoid division by zero

            self.rescale_shifts[key] = shifts
            self.rescale_scales[key] = scales

        # Get normalized statistics for "clip_only" and "clip_rescale" modes
        if self.processing_mode in ["clip_only", "clip_rescale"]:
            norm_means = []
            norm_stds = []

            for i, band in enumerate(bands):
                if isinstance(band, (int, float)):
                    # Fill values use neutral normalization
                    norm_means.append(0.0)
                    norm_stds.append(1.0)
                else:
                    # Try to get normalized stats from the stats dictionary
                    if "norm_mean" in self.stats and band in self.stats["norm_mean"]:
                        norm_means.append(self.stats["norm_mean"][band])
                        norm_stds.append(self.stats["norm_std"][band])
                    else:
                        # Fallback to raw statistics if normalized not available
                        norm_means.append(means[i].item())
                        norm_stds.append(stds[i].item())

            self.norm_means[key] = torch.tensor(norm_means)
            self.norm_stds[key] = torch.tensor(norm_stds)

    def _get_clip_values(self, bands: Sequence[str | float]) -> tuple[Tensor, Tensor]:
        """Extract clip min/max tensors. Uses +/- infinity if clipping is not defined."""
        clip_mins, clip_maxs = [], []
        # has_clip_min_stats = "clip_min" in self.stats
        # has_clip_max_stats = "clip_max" in self.stats
        clip_min_val = self.stats.get("clip_min", float("-inf"))
        clip_max_val = self.stats.get("clip_max", float("inf"))

        # import pdb
        # pdb.set_trace()
        for band in bands:
            # if isinstance(band, (int, float)):
            #     clip_mins.append(float("-inf"))
            #     clip_maxs.append(float("inf"))
            # else:
            clip_mins.append(clip_min_val)
            clip_maxs.append(clip_max_val)

        return torch.tensor(clip_mins), torch.tensor(clip_maxs)

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

            if self.processing_mode == "none":
                # Apply regular z-score normalization without clipping
                mean = self.means[key]
                std = self.stds[key]
                is_fill = self.is_fill_value[key]

                mean_reshaped, _ = self._reshape_and_expand(mean, tensor)
                std_reshaped, _ = self._reshape_and_expand(std, tensor)
                _, is_fill_expanded = self._reshape_and_expand(is_fill, tensor)

                z_score = (tensor - mean_reshaped) / (std_reshaped + 1e-6)
                normalized = torch.where(is_fill_expanded, normalized, z_score)

            elif self.processing_mode == "clip_only":
                # First clip, then apply z-score using norm_mean/norm_std
                clip_min = self.clip_mins[key]
                clip_max = self.clip_maxs[key]
                norm_mean = self.norm_means[key]
                norm_std = self.norm_stds[key]
                is_fill = self.is_fill_value[key]

                clip_min_reshaped, _ = self._reshape_and_expand(clip_min, tensor)
                clip_max_reshaped, _ = self._reshape_and_expand(clip_max, tensor)
                mean_reshaped, _ = self._reshape_and_expand(norm_mean, tensor)
                std_reshaped, _ = self._reshape_and_expand(norm_std, tensor)
                _, is_fill_expanded = self._reshape_and_expand(is_fill, tensor)

                # Clip values
                clipped = torch.clamp(
                    tensor, min=clip_min_reshaped, max=clip_max_reshaped
                )

                # Apply z-score normalization to clipped values
                z_score = (clipped - mean_reshaped) / (std_reshaped + 1e-6)
                normalized = torch.where(is_fill_expanded, normalized, z_score)

            elif self.processing_mode == "clip_rescale":
                # Clip, rescale to [0,1], then apply z-score
                clip_min = self.clip_mins[key]
                clip_max = self.clip_maxs[key]
                shifts = self.rescale_shifts[key]
                scales = self.rescale_scales[key]
                # these are wrong
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

                # Step 1: Clip values
                clipped = torch.clamp(
                    tensor, min=clip_min_reshaped, max=clip_max_reshaped
                )

                # Step 2: Shift to non-negative
                shifted = clipped + shifts_reshaped

                # Step 3: Rescale to [0,1]
                rescaled = shifted / scales_reshaped
                rescaled = torch.clamp(rescaled, min=0.0, max=1.0)

                # Step 4: Apply z-score normalization
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

            if self.processing_mode == "none":
                # Reverse standard z-score normalization
                mean = self.means[key]
                std = self.stds[key]

                mean_reshaped, _ = self._reshape_and_expand(mean, tensor)
                std_reshaped, _ = self._reshape_and_expand(std, tensor)

                original = tensor * (std_reshaped + 1e-6) + mean_reshaped
                unnormalized = torch.where(is_fill_expanded, unnormalized, original)

            elif self.processing_mode == "clip_only":
                # Reverse z-score normalization of clipped values
                norm_mean = self.norm_means[key]
                norm_std = self.norm_stds[key]

                mean_reshaped, _ = self._reshape_and_expand(norm_mean, tensor)
                std_reshaped, _ = self._reshape_and_expand(norm_std, tensor)

                unscaled = tensor * (std_reshaped + 1e-6) + mean_reshaped
                unnormalized = torch.where(is_fill_expanded, unnormalized, unscaled)

            elif self.processing_mode == "clip_rescale":
                # Reverse the full transformation: z-score -> rescale -> shift -> original
                norm_mean = self.norm_means[key]
                norm_std = self.norm_stds[key]
                shifts = self.rescale_shifts[key]
                scales = self.rescale_scales[key]

                mean_reshaped, _ = self._reshape_and_expand(norm_mean, tensor)
                std_reshaped, _ = self._reshape_and_expand(norm_std, tensor)
                shifts_reshaped, _ = self._reshape_and_expand(shifts, tensor)
                scales_reshaped, _ = self._reshape_and_expand(scales, tensor)

                # Step 1: Reverse z-score
                unscaled_01 = tensor * (std_reshaped + 1e-6) + mean_reshaped
                unscaled_01 = torch.clamp(unscaled_01, min=0.0, max=1.0)

                # Step 2: Reverse [0,1] rescaling
                unscaled = unscaled_01 * scales_reshaped

                # Step 3: Reverse shift
                original = unscaled - shifts_reshaped

                unnormalized = torch.where(is_fill_expanded, unnormalized, original)

            result[key] = unnormalized

        return result

    def unnormalize(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Unnormalize input tensors by reversing the z-score normalization."""
        result = {}
        for key, tensor in data.items():
            if key not in self.means:
                result[key] = tensor
                continue

            mean, std = self.means[key], self.stds[key]
            is_fill = self.is_fill_value[key]

            mean_r, _ = self._reshape_and_expand(mean, tensor)
            std_r, _ = self._reshape_and_expand(std, tensor)
            _, is_fill_e = self._reshape_and_expand(is_fill, tensor)

            unnormalized_tensor = tensor.clone()

            un_z_score_vals = tensor * (std_r + 1e-6) + mean_r

            unnormalized_tensor = torch.where(
                ~is_fill_e, un_z_score_vals, unnormalized_tensor
            )

            result[key] = unnormalized_tensor

        return result

    def _format_channel_stats(self, key: str, channel_idx: int) -> str:
        """Format clip and statistics info for a specific channel."""
        mean = self.means[key][channel_idx].item()
        std = self.stds[key][channel_idx].item()
        clip_min = self.clip_mins[key][channel_idx].item()
        clip_max = self.clip_maxs[key][channel_idx].item()

        clip_info = ""
        if clip_min > float("-inf") or clip_max < float("inf"):
            clip_info = f", clipping: [{clip_min:.4f}, {clip_max:.4f}]"

        return f"    Channel {channel_idx}: mean={mean:.4f}, std={std:.4f}{clip_info}"


class SatMAENormalizer(DataNormalizer):
    """Normalization module for satellite imagery with SatMAE-style normalization.

    Several papers have cited SatMAE for this normalization procedure:
    - https://github.com/sustainlab-group/SatMAE/blob/e31c11fa1bef6f9a9aa3eb49e8637c8b8952ba5e/util/datasets.py#L358

    They mention that the normalization is inspired from SeCO:
    - https://github.com/ServiceNow/seasonal-contrast/blob/8285173ec205b64bc3e53b880344dd6c3f79fa7a/datasets/bigearthnet_dataset.py#L111

    This normalization:
    1. For bands with negative min values: shifts data to non-negative range first
    2. Clips values to [mean - 2*std, mean + 2*std] (after shifting if needed), these are the raw min/max and mean/std values
    3. Rescales to target range: [0, 1], [0, 255], or [-1, 1]
    4. Preserves fill values unchanged
    5. Optionally, once the data is in the target range of [0, 1], apply mean/std z-score normalization which is then akin
       ImageNet-style normalization.
    """

    valid_ranges = ["zero_one", "zero_255", "neg_one_one"]

    def __init__(
        self,
        stats: dict[str, dict[str, float]] | str,
        band_order: list[str | float] | dict[str, list[str | float]],
        image_keys: Sequence[str] | None = None,
        output_range: str = "zero_one",
        apply_second_stage: bool = False,
    ) -> None:
        """Initialize enhanced two-stage SatMAE normalizer.

        Args:
            stats: Statistics including both raw and normalized values
            band_order: Band order configuration
            image_keys: Keys to normalize
            output_range: Target output range
            apply_second_stage: Whether to apply ImageNet-style normalization
                               after first stage [0,1] normalization
        """

        if output_range not in self.valid_ranges:
            raise AssertionError(
                f"output_range must be one of {self.valid_ranges}, got {output_range}"
            )

        self.output_range = output_range
        self.apply_second_stage = apply_second_stage

        if output_range == "zero_255":
            self.scale_factor = 255.0
            self.shift_factor = 0.0
        elif output_range == "neg_one_one":
            self.scale_factor = 2.0
            self.shift_factor = -1.0
        else:
            self.scale_factor = 1.0
            self.shift_factor = 0.0

        self.raw_min_values = {}
        self.raw_max_values = {}
        self.offsets = {}
        self.min_values = {}
        self.max_values = {}

        self.norm_means = {}
        self.norm_stds = {}

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
            if isinstance(band, (str, float)):
                if isinstance(band, str) and band in self.stats.get(
                    "shift_offsets", {}
                ):
                    shift_offsets_for_bands.append(self.stats["shift_offsets"][band])
                else:
                    if raw_min_values[i] < 0:
                        shift_offsets_for_bands.append(-raw_min_values[i].item())
                    else:
                        shift_offsets_for_bands.append(0.0)

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
            norm_means = []
            norm_stds = []

            for band in bands:
                if isinstance(band, (int, float)):
                    norm_means.append(0.0)
                    norm_stds.append(1.0)
                else:
                    if (
                        band in self.stats.get("norm_mean", {})
                        or band in self.stats["norm_mean"]
                    ):
                        norm_means.append(self.stats["norm_mean"][band])
                        norm_stds.append(self.stats["norm_std"][band])
                    else:
                        norm_means.append(0.0)
                        norm_stds.append(1.0)

            self.norm_means[key] = torch.tensor(norm_means)
            self.norm_stds[key] = torch.tensor(norm_stds)

    def forward(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Apply two-stage SatMAE normalization to input tensors."""
        result = {}
        for key, tensor in data.items():
            if key in self.min_values:
                normalized_tensor = self._normalize_first_stage(
                    tensor,
                    self.min_values[key],
                    self.max_values[key],
                    self.offsets[key],
                    self.is_fill_value[key],
                )

                if self.apply_second_stage and key in self.norm_means:
                    normalized_tensor = self._normalize_second_stage(
                        normalized_tensor,
                        self.norm_means[key],
                        self.norm_stds[key],
                        self.is_fill_value[key],
                    )

                result[key] = normalized_tensor
            else:
                result[key] = tensor

        return result

    def _normalize_first_stage(
        self,
        tensor: Tensor,
        min_value: Tensor,
        max_value: Tensor,
        offsets: Tensor,
        is_fill: Tensor,
    ) -> Tensor:
        """First stage: shift, clip to mean±2std, scale to target range."""
        min_reshaped, _ = self._reshape_and_expand(min_value, tensor)
        max_reshaped, _ = self._reshape_and_expand(max_value, tensor)
        offsets_reshaped, _ = self._reshape_and_expand(offsets, tensor)
        _, fill_mask_expanded = self._reshape_and_expand(is_fill, tensor)

        normalized = tensor.clone()

        shifted_tensor = tensor + offsets_reshaped

        temp_normalized = (shifted_tensor - min_reshaped) / (
            max_reshaped - min_reshaped + 1e-6
        )

        temp_normalized = torch.clamp(temp_normalized, 0, 1)

        if not (self.apply_second_stage and self.output_range == "zero_one"):
            if self.output_range != "zero_one":
                temp_normalized = (
                    temp_normalized * self.scale_factor + self.shift_factor
                )

        normalized = torch.where(fill_mask_expanded, normalized, temp_normalized)

        return normalized

    def _normalize_second_stage(
        self, tensor: Tensor, norm_mean: Tensor, norm_std: Tensor, is_fill: Tensor
    ) -> Tensor:
        """Second stage: apply ImageNet-style normalization to [0,1] data."""
        mean_reshaped, _ = self._reshape_and_expand(norm_mean, tensor)
        std_reshaped, _ = self._reshape_and_expand(norm_std, tensor)
        _, fill_mask_expanded = self._reshape_and_expand(is_fill, tensor)

        normalized = tensor.clone()
        imagenet_style = (tensor - mean_reshaped) / (std_reshaped + 1e-6)

        normalized = torch.where(fill_mask_expanded, normalized, imagenet_style)

        return normalized

    def unnormalize(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Revert two-stage SatMAE normalization."""
        result = {}
        for key, tensor in data.items():
            if key in self.min_values:
                if self.apply_second_stage and key in self.norm_means:
                    tensor = self._denormalize_second_stage(
                        tensor,
                        self.norm_means[key],
                        self.norm_stds[key],
                        self.is_fill_value[key],
                    )

                result[key] = self._denormalize_first_stage(
                    tensor,
                    self.min_values[key],
                    self.max_values[key],
                    self.offsets[key],
                    self.is_fill_value[key],
                )
            else:
                result[key] = tensor

        return result

    def _denormalize_second_stage(
        self, tensor: Tensor, norm_mean: Tensor, norm_std: Tensor, is_fill: Tensor
    ) -> Tensor:
        """Undo the ImageNet-style normalization of second stage."""
        mean_reshaped, _ = self._reshape_and_expand(norm_mean, tensor)
        std_reshaped, _ = self._reshape_and_expand(norm_std, tensor)
        _, fill_mask_expanded = self._reshape_and_expand(is_fill, tensor)

        denormalized = tensor.clone()

        temp = tensor.clone()
        if self.output_range != "zero_one":
            if self.shift_factor != 0:
                temp = (temp - self.shift_factor) / self.scale_factor
            else:
                temp = temp / self.scale_factor

        original_range = temp * (std_reshaped + 1e-6) + mean_reshaped

        denormalized = torch.where(fill_mask_expanded, denormalized, original_range)

        return denormalized

    def _denormalize_first_stage(
        self,
        tensor: Tensor,
        min_value: Tensor,
        max_value: Tensor,
        offsets: Tensor,
        is_fill: Tensor,
    ) -> Tensor:
        """Undo the first stage normalization (scaling, shifting)."""
        min_reshaped, _ = self._reshape_and_expand(min_value, tensor)
        max_reshaped, _ = self._reshape_and_expand(max_value, tensor)
        offsets_reshaped, _ = self._reshape_and_expand(offsets, tensor)
        _, fill_mask_expanded = self._reshape_and_expand(is_fill, tensor)

        denormalized = tensor.clone()

        temp_denormalized = tensor.clone()
        if not self.apply_second_stage and self.output_range != "zero_one":
            if self.shift_factor != 0:
                temp_denormalized = (
                    temp_denormalized - self.shift_factor
                ) / self.scale_factor
            else:
                temp_denormalized = temp_denormalized / self.scale_factor

        temp_denormalized = (
            temp_denormalized * (max_reshaped - min_reshaped) + min_reshaped
        )
        temp_denormalized = temp_denormalized - offsets_reshaped

        denormalized = torch.where(fill_mask_expanded, denormalized, temp_denormalized)

        return denormalized

    def _format_channel_stats(self, key: str, channel_idx: int) -> str:
        """Format statistics for a specific channel with normalization details."""
        mean = self.means[key][channel_idx].item()
        std = self.stds[key][channel_idx].item()
        offset = self.offsets[key][channel_idx].item() if key in self.offsets else 0.0
        min_val = (
            self.min_values[key][channel_idx].item() if key in self.min_values else 0.0
        )
        max_val = (
            self.max_values[key][channel_idx].item() if key in self.max_values else 0.0
        )

        offset_info = f", offset: {offset:.4f}" if offset > 0 else ""
        norm_info = ""
        if self.apply_second_stage and key in self.norm_means:
            norm_mean = self.norm_means[key][channel_idx].item()
            norm_std = self.norm_stds[key][channel_idx].item()
            norm_info = f", norm_mean: {norm_mean:.4f}, norm_std: {norm_std:.4f}"

        return (
            f"    Channel {channel_idx}: mean={mean:.4f}, std={std:.4f}, "
            f"range: [{min_val:.4f}, {max_val:.4f}]{offset_info}{norm_info}"
        )


class SimpleRescaleNormalizer(DataNormalizer):
    """Normalization module applying simple rescaling to [0,1] range with optional second stage.

    This normalizer performs a two-stage process:
    1. First stage:
       - Applies optional clipping based on min/max values in stats
       - For channels with negative min values, shifts to a non-negative range
       - Divides by the adjusted max value to scale to [0,1]
    2. Optional second stage (when apply_second_stage=True):
       - Applies z-score normalization to the [0,1] data using normalized statistics
       - Can rescale to different output ranges

    Fill value bands are passed through unchanged in all operations.
    """

    valid_ranges = ["zero_one", "zero_255", "neg_one_one"]

    def __init__(
        self,
        stats: dict[str, dict[str, float]] | str,
        band_order: list[str | float] | dict[str, list[str | float]],
        image_keys: Sequence[str] | None = None,
        output_range: str = "zero_one",
        apply_second_stage: bool = False,
    ) -> None:
        """Initialize normalizer with optional two-stage normalization.

        Args:
            stats: Statistics including both raw and normalized values
            band_order: Band order configuration
            image_keys: Keys to normalize
            output_range: Target output range ("zero_one", "zero_255", "neg_one_one")
            apply_second_stage: Whether to apply z-score normalization after [0,1] rescaling
        """
        if output_range not in self.valid_ranges:
            raise AssertionError(
                f"output_range must be one of {self.valid_ranges}, got {output_range}"
            )

        self.output_range = output_range
        self.apply_second_stage = apply_second_stage

        if output_range == "zero_255":
            self.scale_factor = 255.0
            self.shift_factor = 0.0
        elif output_range == "neg_one_one":
            self.scale_factor = 2.0
            self.shift_factor = -1.0
        else:
            self.scale_factor = 1.0
            self.shift_factor = 0.0

        self.clip_mins = {}
        self.clip_maxs = {}
        self.shift_values = {}
        self.adjusted_maxs = {}

        self.norm_means = {}
        self.norm_stds = {}

        super().__init__(stats, band_order, image_keys)

    def _set_additional_stats_for_key(
        self,
        key: str,
        bands: Sequence[str | float],
        means: Tensor,
        stds: Tensor,
        is_fill: Tensor,
    ) -> None:
        """Set statistics for both normalization stages."""
        clip_min, clip_max = self._get_clip_values(bands)
        self.clip_mins[key] = clip_min
        self.clip_maxs[key] = clip_max

        shift_values = torch.zeros_like(clip_min)
        negative_ranges = clip_min < 0
        if negative_ranges.any():
            shift_values[negative_ranges] = -clip_min[negative_ranges]

        self.shift_values[key] = shift_values
        self.adjusted_maxs[key] = clip_max + shift_values

        if (
            self.apply_second_stage
            and key in self.stats
            and "norm_mean" in self.stats[key]
        ):
            self.norm_means[key] = torch.tensor(self.stats[key]["norm_mean"])
            self.norm_stds[key] = torch.tensor(self.stats[key]["norm_std"])

    def forward(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Apply one or two-stage normalization to input tensors.

        Args:
            data: Dictionary mapping keys to tensors.

        Returns:
            Dictionary with normalized tensors under the same keys.
        """
        result = {}
        for key, tensor in data.items():
            if key not in self.means:
                result[key] = tensor
                continue

            normalized_tensor = self._normalize_first_stage(
                tensor,
                self.clip_mins[key],
                self.clip_maxs[key],
                self.shift_values[key],
                self.adjusted_maxs[key],
                self.is_fill_value[key],
            )

            if self.apply_second_stage and key in self.norm_means:
                normalized_tensor = self._normalize_second_stage(
                    normalized_tensor,
                    self.norm_means[key],
                    self.norm_stds[key],
                    self.is_fill_value[key],
                )

            result[key] = normalized_tensor

        return result

    def _normalize_first_stage(
        self,
        tensor: Tensor,
        clip_min: Tensor,
        clip_max: Tensor,
        shift_values: Tensor,
        adjusted_max: Tensor,
        is_fill: Tensor,
    ) -> Tensor:
        """First stage: clip, shift negative values, scale to [0,1]."""
        clip_min_r, _ = self._reshape_and_expand(clip_min, tensor)
        clip_max_r, _ = self._reshape_and_expand(clip_max, tensor)
        shift_r, _ = self._reshape_and_expand(shift_values, tensor)
        adj_max_r, _ = self._reshape_and_expand(adjusted_max, tensor)
        _, is_fill_e = self._reshape_and_expand(is_fill, tensor)

        normalized_tensor = tensor.clone()

        clipped = torch.clamp(tensor, min=clip_min_r, max=clip_max_r)

        shifted = clipped + shift_r

        rescaled = shifted / (adj_max_r + 1e-6)

        rescaled = torch.clamp(rescaled, min=0.0, max=1.0)

        normalized_tensor = torch.where(~is_fill_e, rescaled, normalized_tensor)

        return normalized_tensor

    def _normalize_second_stage(
        self, tensor: Tensor, norm_mean: Tensor, norm_std: Tensor, is_fill: Tensor
    ) -> Tensor:
        """Second stage: apply z-score normalization to [0,1] data."""
        mean_reshaped, _ = self._reshape_and_expand(norm_mean, tensor)
        std_reshaped, _ = self._reshape_and_expand(norm_std, tensor)
        _, fill_mask_expanded = self._reshape_and_expand(is_fill, tensor)

        normalized = tensor.clone()
        z_score_norm = (tensor - mean_reshaped) / (std_reshaped + 1e-6)

        if self.output_range != "zero_one":
            z_score_norm = z_score_norm * self.scale_factor + self.shift_factor

        normalized = torch.where(fill_mask_expanded, normalized, z_score_norm)

        return normalized

    def unnormalize(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Unnormalize input tensors by reversing one or two-stage normalization."""
        result = {}
        for key, tensor in data.items():
            if key not in self.means:
                result[key] = tensor
                continue

            unnormalized_tensor = tensor

            if self.apply_second_stage and key in self.norm_means:
                unnormalized_tensor = self._denormalize_second_stage(
                    unnormalized_tensor,
                    self.norm_means[key],
                    self.norm_stds[key],
                    self.is_fill_value[key],
                )

            result[key] = self._denormalize_first_stage(
                unnormalized_tensor,
                self.shift_values[key],
                self.adjusted_maxs[key],
                self.is_fill_value[key],
            )

        return result

    def _denormalize_second_stage(
        self, tensor: Tensor, norm_mean: Tensor, norm_std: Tensor, is_fill: Tensor
    ) -> Tensor:
        """Undo the z-score normalization of second stage."""
        mean_reshaped, _ = self._reshape_and_expand(norm_mean, tensor)
        std_reshaped, _ = self._reshape_and_expand(norm_std, tensor)
        _, fill_mask_expanded = self._reshape_and_expand(is_fill, tensor)

        denormalized = tensor.clone()

        temp = tensor.clone()
        if self.output_range != "zero_one":
            if self.shift_factor != 0:
                temp = (temp - self.shift_factor) / self.scale_factor
            else:
                temp = temp / self.scale_factor
        original_range = temp * (std_reshaped + 1e-6) + mean_reshaped

        denormalized = torch.where(fill_mask_expanded, denormalized, original_range)

        return denormalized

    def _denormalize_first_stage(
        self,
        tensor: Tensor,
        shift_values: Tensor,
        adjusted_max: Tensor,
        is_fill: Tensor,
    ) -> Tensor:
        """Undo the first stage normalization (scaling, shifting)."""

        shift_r, _ = self._reshape_and_expand(shift_values, tensor)
        adj_max_r, _ = self._reshape_and_expand(adjusted_max, tensor)
        _, is_fill_e = self._reshape_and_expand(is_fill, tensor)

        unnormalized_tensor = tensor.clone()

        unscaled = tensor * (adj_max_r + 1e-6)

        unshifted = unscaled - shift_r

        unnormalized_tensor = torch.where(~is_fill_e, unshifted, unnormalized_tensor)

        return unnormalized_tensor

    def _format_channel_stats(self, key: str, channel_idx: int) -> str:
        """Format statistics for this channel with both normalization stages."""
        mean = self.means[key][channel_idx].item()
        std = self.stds[key][channel_idx].item()
        clip_min = self.clip_mins[key][channel_idx].item()
        clip_max = self.clip_maxs[key][channel_idx].item()
        shift = self.shift_values[key][channel_idx].item()

        clip_info = ""
        if clip_min > float("-inf") or clip_max < float("inf"):
            clip_info = f", clipping: [{clip_min:.4f}, {clip_max:.4f}]"

        shift_info = f", shift: {shift:.4f}" if shift > 0 else ""

        norm_info = ""
        if self.apply_second_stage and key in self.norm_means:
            norm_mean = self.norm_means[key][channel_idx].item()
            norm_std = self.norm_stds[key][channel_idx].item()
            norm_info = f", norm_mean: {norm_mean:.4f}, norm_std: {norm_std:.4f}"

        return (
            f"    Channel {channel_idx}: mean={mean:.4f}, std={std:.4f}"
            f"{clip_info}{shift_info}{norm_info}"
        )
