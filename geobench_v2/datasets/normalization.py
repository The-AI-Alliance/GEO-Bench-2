# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Normalization Modules."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
import torch.nn as nn
from torch import Tensor


class DataNormalizer(nn.Module, ABC):
    """Base Class for Data Normalization."""

    def __init__(
        self,
        stats: dict[str, dict[str, float]],
        band_order: list[str | float] | dict[str, list[str | float]],
        image_keys: Sequence[str] | None = None,
    ) -> None:
        """Initialize normalizer.

        Args:
            stats: dictionary containing mean and std for each band
            band_order: Either a sequence of bands or dict mapping modalities to sequences
            image_keys: Keys in the data dictionary to normalize (default: ["image"])
        """
        super().__init__()
        self.stats = stats
        self.band_order = band_order
        self.image_keys = image_keys or ["image"]

        # Common attributes to be populated by subclasses
        self.means = {}
        self.stds = {}
        self.is_fill_value = {}

        # Initialize statistics
        self._initialize_statistics()

    def _initialize_statistics(self) -> None:
        """Initialize statistics based on band_order.

        This method populates the normalizer's statistics (means, stds, is_fill_value)
        based on the band_order and stats provided during initialization.

        Subclasses should override this to set additional statistics they need.
        """
        if isinstance(self.band_order, dict):
            # Multi-modal with different bands for each modality
            for modality, bands in self.band_order.items():
                # Get stats for this modality
                means, stds, is_fill = self._get_band_stats(bands)

                # Basic multi-modal (image_modality)
                base_key = f"image_{modality}"
                self.means[base_key] = means
                self.stds[base_key] = stds
                self.is_fill_value[base_key] = is_fill

                # Process additional image keys if provided
                self._process_additional_keys(base_key, means, stds, is_fill)

                # Allow subclasses to set additional statistics for this modality
                self._set_additional_stats_for_key(
                    base_key, bands, means, stds, is_fill
                )
        else:
            # Single band order for one or multiple image keys
            means, stds, is_fill = self._get_band_stats(self.band_order)

            for key in self.image_keys:
                self.means[key] = means
                self.stds[key] = stds
                self.is_fill_value[key] = is_fill

                # Allow subclasses to set additional statistics for this key
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
                    continue  # Already handled

                # Create modality+timestamp key (e.g. image_pre_s1)
                modality_key = f"{key}_{base_key.split('_')[1]}"

                # Add normalized stats
                self.means[modality_key] = means
                self.stds[modality_key] = stds
                self.is_fill_value[modality_key] = is_fill

                # Allow subclasses to set additional statistics for this key
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

            # Display stats for each channel/band
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
        stats: dict[str, dict[str, float]],
        band_order: list[str | float] | dict[str, list[str | float]],
        image_keys: Sequence[str] | None = None,
    ) -> None:
        """Initialize normalizer applying clip then z-score."""
        # Additional attributes needed for this specific normalizer
        self.clip_mins = {}
        self.clip_maxs = {}

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
        clip_min, clip_max = self._get_clip_values(bands)
        self.clip_mins[key] = clip_min
        self.clip_maxs[key] = clip_max

    def _get_clip_values(self, bands: Sequence[str | float]) -> tuple[Tensor, Tensor]:
        """Extract clip min/max tensors. Uses +/- infinity if clipping is not defined."""
        clip_mins, clip_maxs = [], []
        has_clip_min_stats = "clip_min" in self.stats
        has_clip_max_stats = "clip_max" in self.stats
        clip_min_dict = self.stats.get("clip_min", {})
        clip_max_dict = self.stats.get("clip_max", {})

        for band in bands:
            if isinstance(band, (int, float)):
                # No clipping for fill values (use +/- inf)
                clip_mins.append(float("-inf"))
                clip_maxs.append(float("inf"))
            elif (
                has_clip_min_stats
                and has_clip_max_stats
                and band in clip_min_dict
                and band in clip_max_dict
            ):
                # Clipping is defined for this specific band
                clip_mins.append(clip_min_dict[band])
                clip_maxs.append(clip_max_dict[band])
            else:
                # No clipping defined for this band (use +/- inf)
                clip_mins.append(float("-inf"))
                clip_maxs.append(float("inf"))

        # Return only clip_mins and clip_maxs
        return torch.tensor(clip_mins), torch.tensor(clip_maxs)

    def forward(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Normalize input tensors by applying clipping (optional) then z-score.

        Iterates through the input dictionary. For keys matching the expected pattern
        (e.g., "image", "image_modality"), applies normalization channel-wise:
        - Clips channels if `clip_min`/`clip_max` were provided in stats.
        - Applies z-score normalization to the (potentially clipped) result.
        - Fill value channels are ignored.
        Keys not matching the pattern are passed through unchanged.

        Args:
            data: Dictionary mapping keys (e.g., "image", "image_s1") to tensors.

        Returns:
            Dictionary with normalized tensors under the same keys.
        """
        result = {}
        for key, tensor in data.items():
            if key not in self.means:
                result[key] = tensor
                continue

            # Retrieve stats for the current key
            mean, std = self.means[key], self.stds[key]
            clip_min, clip_max = self.clip_mins[key], self.clip_maxs[key]
            is_fill = self.is_fill_value[key]

            # Reshape stats and expand fill mask
            mean_r, _ = self._reshape_and_expand(mean, tensor)
            std_r, _ = self._reshape_and_expand(std, tensor)
            clip_min_r, _ = self._reshape_and_expand(clip_min, tensor)
            clip_max_r, _ = self._reshape_and_expand(clip_max, tensor)
            _, is_fill_e = self._reshape_and_expand(is_fill, tensor)

            normalized_tensor = tensor.clone()

            # 1. Apply clipping (uses +/- inf for non-clipped bands)
            clipped_tensor = torch.clamp(tensor, min=clip_min_r, max=clip_max_r)

            # 2. Apply z-score to the (potentially) clipped tensor
            z_score_clipped_vals = (clipped_tensor - mean_r) / (std_r + 1e-6)

            # 3. Apply the result only where it's NOT a fill value
            normalized_tensor = torch.where(
                ~is_fill_e, z_score_clipped_vals, normalized_tensor
            )

            result[key] = normalized_tensor

        return result

    def unnormalize(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Unnormalize input tensors by reversing the z-score normalization."""
        result = {}
        for key, tensor in data.items():
            if key not in self.means:
                result[key] = tensor
                continue

            # Retrieve stats needed for inverse z-score
            mean, std = self.means[key], self.stds[key]
            is_fill = self.is_fill_value[key]

            # Reshape stats and expand fill mask
            mean_r, _ = self._reshape_and_expand(mean, tensor)
            std_r, _ = self._reshape_and_expand(std, tensor)
            _, is_fill_e = self._reshape_and_expand(is_fill, tensor)

            # Initialize result tensor
            unnormalized_tensor = tensor.clone()

            # 1. Apply inverse z-score
            un_z_score_vals = tensor * (std_r + 1e-6) + mean_r

            # 2. Apply the result only where it's NOT a fill value
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
        stats: dict[str, dict[str, float]],
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

        # Configure scaling factors based on output range
        if output_range == "zero_255":
            self.scale_factor = 255.0
            self.shift_factor = 0.0
        elif output_range == "neg_one_one":
            self.scale_factor = 2.0
            self.shift_factor = -1.0
        else:  # "zero_one"
            self.scale_factor = 1.0
            self.shift_factor = 0.0

        # Store attributes for both normalization stages
        self.raw_min_values = {}  # mean - 2*std
        self.raw_max_values = {}  # mean + 2*std
        self.offsets = {}  # For shifting negative values
        self.min_values = {}  # After applying offsets
        self.max_values = {}  # After applying offsets

        # For second stage (normalized stats)
        self.norm_means = {}  # mean of data in [0,1] range
        self.norm_stds = {}  # std of data in [0,1] range

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
        # First stage parameters
        raw_min_values = means - 2 * stds
        raw_max_values = means + 2 * stds

        # Get offsets from stats if available, or calculate them
        if key in self.stats and "shift_offsets" in self.stats[key]:
            offsets = torch.tensor(self.stats[key]["shift_offsets"])
        else:
            # Calculate offsets for channels with negative values
            offsets = torch.zeros_like(raw_min_values)
            neg_mask = raw_min_values < 0
            if neg_mask.any():
                offsets[neg_mask] = -raw_min_values[neg_mask]

        # Store first stage parameters
        self.raw_min_values[key] = raw_min_values
        self.raw_max_values[key] = raw_max_values
        self.offsets[key] = offsets
        self.min_values[key] = raw_min_values + offsets
        self.max_values[key] = raw_max_values + offsets

        # Second stage parameters (if available and requested)
        if (
            self.apply_second_stage
            and key in self.stats
            and "norm_mean" in self.stats[key]
        ):
            self.norm_means[key] = torch.tensor(self.stats[key]["norm_mean"])
            self.norm_stds[key] = torch.tensor(self.stats[key]["norm_std"])

    def forward(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Apply two-stage SatMAE normalization to input tensors."""
        result = {}
        for key, tensor in data.items():
            if key in self.min_values:
                # Apply first stage normalization
                normalized_tensor = self._normalize_first_stage(
                    tensor,
                    self.min_values[key],
                    self.max_values[key],
                    self.offsets[key],
                    self.is_fill_value[key],
                )

                # Apply second stage if requested and stats are available
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
        # Reshape for broadcasting
        min_reshaped, _ = self._reshape_and_expand(min_value, tensor)
        max_reshaped, _ = self._reshape_and_expand(max_value, tensor)
        offsets_reshaped, _ = self._reshape_and_expand(offsets, tensor)
        _, fill_mask_expanded = self._reshape_and_expand(is_fill, tensor)

        # Start with original values
        normalized = tensor.clone()

        # Step 1: Apply offset to shift negative values to non-negative range
        shifted_tensor = tensor + offsets_reshaped

        # Step 2: Apply min-max normalization on the shifted data
        temp_normalized = (shifted_tensor - min_reshaped) / (
            max_reshaped - min_reshaped + 1e-6
        )

        # Step 3: Clamp to [0,1]
        temp_normalized = torch.clamp(temp_normalized, 0, 1)

        # Step 4: Scale to target range if requested and not using second stage
        if not (self.apply_second_stage and self.output_range == "zero_one"):
            if self.output_range != "zero_one":
                temp_normalized = (
                    temp_normalized * self.scale_factor + self.shift_factor
                )

        # Apply normalized values only where it's NOT a fill value
        normalized = torch.where(fill_mask_expanded, normalized, temp_normalized)

        return normalized

    def _normalize_second_stage(
        self, tensor: Tensor, norm_mean: Tensor, norm_std: Tensor, is_fill: Tensor
    ) -> Tensor:
        """Second stage: apply ImageNet-style normalization to [0,1] data."""
        # Reshape for broadcasting
        mean_reshaped, _ = self._reshape_and_expand(norm_mean, tensor)
        std_reshaped, _ = self._reshape_and_expand(norm_std, tensor)
        _, fill_mask_expanded = self._reshape_and_expand(is_fill, tensor)

        # Apply z-score normalization
        normalized = tensor.clone()
        imagenet_style = (tensor - mean_reshaped) / (std_reshaped + 1e-6)

        # Scale to target range if needed
        if self.output_range != "zero_one":
            imagenet_style = imagenet_style * self.scale_factor + self.shift_factor

        # Apply only to non-fill values
        normalized = torch.where(fill_mask_expanded, normalized, imagenet_style)

        return normalized

    def unnormalize(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Revert two-stage SatMAE normalization."""
        result = {}
        for key, tensor in data.items():
            if key in self.min_values:
                # First undo second stage if it was applied
                if self.apply_second_stage and key in self.norm_means:
                    tensor = self._denormalize_second_stage(
                        tensor,
                        self.norm_means[key],
                        self.norm_stds[key],
                        self.is_fill_value[key],
                    )

                # Then undo first stage
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
        # Reshape for broadcasting
        mean_reshaped, _ = self._reshape_and_expand(norm_mean, tensor)
        std_reshaped, _ = self._reshape_and_expand(norm_std, tensor)
        _, fill_mask_expanded = self._reshape_and_expand(is_fill, tensor)

        # Start with original values
        denormalized = tensor.clone()

        # Undo scaling to target range if applied
        temp = tensor.clone()
        if self.output_range != "zero_one":
            if self.shift_factor != 0:
                temp = (temp - self.shift_factor) / self.scale_factor
            else:
                temp = temp / self.scale_factor

        # Reverse z-score normalization
        original_range = temp * (std_reshaped + 1e-6) + mean_reshaped

        # Apply only to non-fill values
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
        # Reshape for broadcasting
        min_reshaped, _ = self._reshape_and_expand(min_value, tensor)
        max_reshaped, _ = self._reshape_and_expand(max_value, tensor)
        offsets_reshaped, _ = self._reshape_and_expand(offsets, tensor)
        _, fill_mask_expanded = self._reshape_and_expand(is_fill, tensor)

        # Start with original values
        denormalized = tensor.clone()

        # Step 1: Revert output range scaling if not already handled in second stage
        temp_denormalized = tensor.clone()
        if not self.apply_second_stage and self.output_range != "zero_one":
            if self.shift_factor != 0:
                temp_denormalized = (
                    temp_denormalized - self.shift_factor
                ) / self.scale_factor
            else:
                temp_denormalized = temp_denormalized / self.scale_factor

        # Step 2: Revert min-max normalization
        temp_denormalized = (
            temp_denormalized * (max_reshaped - min_reshaped) + min_reshaped
        )

        # Step 3: Remove the offset we applied during normalization
        temp_denormalized = temp_denormalized - offsets_reshaped

        # Apply the denormalized values only where it's NOT a fill value
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
        stats: dict[str, dict[str, float]],
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

        # Configure two-stage normalization parameters
        self.output_range = output_range
        self.apply_second_stage = apply_second_stage

        # Configure scaling factors based on output range
        if output_range == "zero_255":
            self.scale_factor = 255.0
            self.shift_factor = 0.0
        elif output_range == "neg_one_one":
            self.scale_factor = 2.0
            self.shift_factor = -1.0
        else:  # "zero_one"
            self.scale_factor = 1.0
            self.shift_factor = 0.0

        # First stage normalization parameters
        self.clip_mins = {}
        self.clip_maxs = {}
        self.shift_values = {}
        self.adjusted_maxs = {}

        # Second stage normalization parameters
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
        # First stage: clipping and rescaling parameters
        clip_min, clip_max = self._get_clip_values(bands)
        self.clip_mins[key] = clip_min
        self.clip_maxs[key] = clip_max

        # Calculate shift values for negative ranges
        shift_values = torch.zeros_like(clip_min)
        negative_ranges = clip_min < 0
        if negative_ranges.any():
            shift_values[negative_ranges] = -clip_min[negative_ranges]

        self.shift_values[key] = shift_values
        self.adjusted_maxs[key] = clip_max + shift_values

        # Second stage: normalized mean/std for [0,1] data (if available)
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

            # Apply first stage normalization
            normalized_tensor = self._normalize_first_stage(
                tensor,
                self.clip_mins[key],
                self.clip_maxs[key],
                self.shift_values[key],
                self.adjusted_maxs[key],
                self.is_fill_value[key],
            )

            # Apply second stage if requested and stats are available
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
        # Reshape for broadcasting
        clip_min_r, _ = self._reshape_and_expand(clip_min, tensor)
        clip_max_r, _ = self._reshape_and_expand(clip_max, tensor)
        shift_r, _ = self._reshape_and_expand(shift_values, tensor)
        adj_max_r, _ = self._reshape_and_expand(adjusted_max, tensor)
        _, is_fill_e = self._reshape_and_expand(is_fill, tensor)

        # Initialize with original tensor
        normalized_tensor = tensor.clone()

        # Apply normalization where not a fill value
        # 1. Apply clipping
        clipped = torch.clamp(tensor, min=clip_min_r, max=clip_max_r)

        # 2. Apply shift for negative values
        shifted = clipped + shift_r

        # 3. Scale to [0,1] range
        rescaled = shifted / (adj_max_r + 1e-6)

        # 4. Ensure values stay in [0,1] range
        rescaled = torch.clamp(rescaled, min=0.0, max=1.0)

        # Apply only to non-fill values
        normalized_tensor = torch.where(~is_fill_e, rescaled, normalized_tensor)

        return normalized_tensor

    def _normalize_second_stage(
        self, tensor: Tensor, norm_mean: Tensor, norm_std: Tensor, is_fill: Tensor
    ) -> Tensor:
        """Second stage: apply z-score normalization to [0,1] data."""
        # Reshape for broadcasting
        mean_reshaped, _ = self._reshape_and_expand(norm_mean, tensor)
        std_reshaped, _ = self._reshape_and_expand(norm_std, tensor)
        _, fill_mask_expanded = self._reshape_and_expand(is_fill, tensor)

        # Apply z-score normalization
        normalized = tensor.clone()
        z_score_norm = (tensor - mean_reshaped) / (std_reshaped + 1e-6)

        # Scale to target range if needed
        if self.output_range != "zero_one":
            z_score_norm = z_score_norm * self.scale_factor + self.shift_factor

        # Apply only to non-fill values
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

            # First undo second stage if it was applied
            if self.apply_second_stage and key in self.norm_means:
                unnormalized_tensor = self._denormalize_second_stage(
                    unnormalized_tensor,
                    self.norm_means[key],
                    self.norm_stds[key],
                    self.is_fill_value[key],
                )

            # Then undo first stage
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
        # Reshape for broadcasting
        mean_reshaped, _ = self._reshape_and_expand(norm_mean, tensor)
        std_reshaped, _ = self._reshape_and_expand(norm_std, tensor)
        _, fill_mask_expanded = self._reshape_and_expand(is_fill, tensor)

        # Start with original values
        denormalized = tensor.clone()

        # Undo scaling to target range if applied
        temp = tensor.clone()
        if self.output_range != "zero_one":
            if self.shift_factor != 0:
                temp = (temp - self.shift_factor) / self.scale_factor
            else:
                temp = temp / self.scale_factor

        # Reverse z-score normalization
        original_range = temp * (std_reshaped + 1e-6) + mean_reshaped

        # Apply only to non-fill values
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
        # Reshape for broadcasting
        shift_r, _ = self._reshape_and_expand(shift_values, tensor)
        adj_max_r, _ = self._reshape_and_expand(adjusted_max, tensor)
        _, is_fill_e = self._reshape_and_expand(is_fill, tensor)

        # Initialize result
        unnormalized_tensor = tensor.clone()

        # Undo normalization
        # 1. Undo scaling
        unscaled = tensor * (adj_max_r + 1e-6)

        # 2. Undo shift
        unshifted = unscaled - shift_r

        # Apply only to non-fill values
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
