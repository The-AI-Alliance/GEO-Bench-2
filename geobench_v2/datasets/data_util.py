# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Data Processing Utility Mixin."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from .sensor_util import ModalityConfig, MultiModalConfig


class DataUtilsMixin(ABC):
    """Mixin for datasets that require band manipulation and normalization."""

    @property
    def normalization_stats(self) -> dict[str, dict[str, dict[str, float]]]:
        """Per-modality normalization statistics."""
        pass

    def get_source_order(self) -> list[str] | dict[str, list[str]]:
        """Derive source order from dataset configuration."""
        if isinstance(self.dataset_band_config, MultiModalConfig):
            return {
                mod: config.default_order
                for mod, config in self.dataset_band_config.modalities.items()
            }
        return self.dataset_band_config.default_order

    def resolve_band_order(
        self,
        band_order: Sequence[str | float]
        | dict[str, Sequence[str | float]]
        | None = None,
    ) -> list[str | float] | dict[str, list[str | float]]:
        """Resolve band names to canonical names using modality configurations."""
        if band_order is None:
            return self.dataset_band_config.default_order

        if isinstance(band_order, dict):
            resolved = {}
            for modality, bands in band_order.items():
                if modality not in self.dataset_band_config.modalities:
                    raise ValueError(f"Unknown modality: {modality}")

                resolved_bands = []
                mod_config = self.dataset_band_config.modalities[modality]

                for band_spec in bands:
                    if isinstance(band_spec, (int, float)):
                        resolved_bands.append(band_spec)
                        continue

                    # Check if band exists in this modality's configuration
                    resolved_band = self._resolve_in_config(band_spec, mod_config)
                    if resolved_band is None:
                        raise ValueError(
                            f"Could not resolve band {band_spec} for modality {modality}\n"
                            f"Available bands: {self._format_available_bands()}"
                        )
                    resolved_bands.append(resolved_band)

                resolved[modality] = resolved_bands
            return resolved
        else:
            # Handle sequence input
            resolved_bands = []
            for band_spec in band_order:
                if isinstance(band_spec, (int, float)):
                    resolved_bands.append(band_spec)
                    continue
                # First check if it's already a canonical name
                if band_spec in self.dataset_band_config.band_to_modality:
                    resolved_bands.append(band_spec)
                    continue

                # Search through modalities for matching aliases
                resolved = None
                for mod, config in self.dataset_band_config.modalities.items():
                    resolved_band = self._resolve_in_config(band_spec, config)
                    if (
                        resolved_band is not None
                        and resolved_band in self.dataset_band_config.band_to_modality
                    ):
                        resolved = resolved_band
                        break

                if resolved is None:
                    raise ValueError(
                        f"Could not resolve band {band_spec}\n"
                        f"Available bands: {self._format_available_bands()}"
                    )
                resolved_bands.append(resolved)

            return resolved_bands

    @staticmethod
    def _resolve_in_config(band: str, config: ModalityConfig) -> str | None:
        """Resolve band name within a specific modality configuration.

        Args:
            band: Band name to resolve
            config: Modality configuration to search in

        Returns:
            Canonical band name if found, None otherwise
        """
        return config.resolve_band(band)

    def rearrange_bands(
        self,
        data: Tensor | dict[str, Tensor],
        target_order: list[str | float] | dict[str, list[str | float]],
    ) -> dict[str, Tensor]:
        """Rearrange bands using dataset configuration."""
        if not isinstance(self.dataset_band_config, MultiModalConfig):
            # Handle single modality case
            if isinstance(data, dict) or isinstance(target_order, dict):
                raise ValueError(
                    "Single modality config requires tensor input and list target_order"
                )
            return self._rearrange_bands_single_modality(
                data, self.dataset_band_config.default_order, target_order
            )

        # Multi-modal case
        if not isinstance(data, dict):
            raise ValueError("Multi-modal config requires dict input data")

        # Case 1: dict target order -> return dict of tensors
        if isinstance(target_order, dict):
            return self._rearrange_multimodal_to_dict(data, target_order)

        # Case 2: list target order -> return single tensor
        return self._rearrange_multimodal_to_tensor(data, target_order)

    def _rearrange_bands_single_modality(
        self, data: Tensor, source_order: list[str], target_order: list[str | float]
    ) -> dict[str, Tensor]:
        """Rearrange bands for single modality."""
        output_channels = []
        source_lookup = {band: idx for idx, band in enumerate(source_order)}

        for band_spec in target_order:
            if isinstance(band_spec, (int, float)):
                # Handle fill values
                shape = list(data.shape)
                if len(shape) == 4:  # timeseries of [T, C, H, W]
                    shape[1] = 1
                else:  # assume [C, H, W]
                    shape[0] = 1
                channel = torch.full(shape, float(band_spec))
            else:
                if band_spec not in source_lookup:
                    raise ValueError(
                        f"Band {band_spec} not found in source order.\n"
                        f"Available bands: {', '.join(source_lookup.keys())}"
                    )
                idx = source_lookup[band_spec]
                if len(data.shape) == 4:
                    # timeseries of [T, C, H, W]
                    channel = data[:, idx : idx + 1]
                else:  # assume [C, H, W]
                    channel = data[idx : idx + 1]
            output_channels.append(channel)

        shape = data.shape
        if len(shape) == 4:  # handle time series case
            return {"image": torch.cat(output_channels, dim=1)}
        else:  # assume [C, H, W]
            return {"image": torch.cat(output_channels, dim=0)}

    def _rearrange_multimodal_to_tensor(
        self, data: dict[str, Tensor], target_order: list[str | float]
    ) -> dict[str, Tensor]:
        """Create single tensor from multi-modal data."""
        resolved_order = self.resolve_band_order(target_order)
        output_channels = []

        for band_spec in resolved_order:
            if isinstance(band_spec, (int, float)):
                shape = list(next(iter(data.values())).shape)
                if len(shape) == 4:  # timeseries of [T, C, H, W]
                    shape[1] = 1
                else:  # assume [C, H, W]
                    shape[0] = 1
                channel = torch.full(shape, float(band_spec))
            else:
                # Use band_to_modality mapping to find source
                modality = self.dataset_band_config.band_to_modality[band_spec]
                source_data = data[modality]

                # Find index in source modality's order
                mod_config = self.dataset_band_config.modalities[modality]
                if band_spec not in mod_config.default_order:
                    raise ValueError(
                        f"Band {band_spec} not found in {modality} default order.\n"
                        f"Available bands: {', '.join(mod_config.default_order)}"
                    )
                idx = mod_config.default_order.index(band_spec)
                if len(source_data.shape) == 4:  # timeseries of [T, C, H, W]
                    channel = source_data[:, idx : idx + 1]
                else:  # assume [C, H, W]
                    channel = source_data[idx : idx + 1]

            output_channels.append(channel)

        # TODO Assuming that each time-series has the same length
        shape = list(next(iter(data.values())).shape)
        if len(shape) == 4:  # handle time series case
            return {"image": torch.cat(output_channels, dim=1)}
        else:  # assume [C, H, W]
            return {"image": torch.cat(output_channels, dim=0)}

    def _rearrange_multimodal_to_dict(
        self, data: dict[str, Tensor], target_order: dict[str, list[str | float]]
    ) -> dict[str, Tensor]:
        """Create dict of tensors from multi-modal data."""
        output = {}

        for modality, bands in target_order.items():
            if modality not in self.dataset_band_config.modalities:
                raise ValueError(f"Unknown modality: {modality}")

            if not bands:  # Check for empty sequence
                raise ValueError(
                    f"Empty band sequence provided for modality {modality}. "
                    "Each modality must specify at least one band or fill value."
                )

            resolved = self.resolve_band_order(bands)
            source_data = data[modality]
            mod_config = self.dataset_band_config.modalities[modality]

            channels = []
            for band in resolved:
                if isinstance(band, (int, float)):
                    shape = list(source_data.shape)
                    if len(shape) == 4:  # timeseries of [T, C, H, W]
                        shape[1] = 1
                    else:  # assume [C, H, W]
                        shape[0] = 1
                    channel = torch.full(shape, float(band))
                else:
                    try:
                        idx = mod_config.default_order.index(band)
                    except ValueError:
                        raise ValueError(
                            f"Band {band} not found in {modality} default order.\n"
                            f"Available bands: {', '.join(mod_config.default_order)}"
                        )
                    if len(source_data.shape) == 4:  # timeseries of [T, C, H, W]
                        channel = source_data[:, idx : idx + 1]
                    else:  # assume [C, H, W]
                        channel = source_data[idx : idx + 1]
                channels.append(channel)

            shape = source_data.shape
            if len(shape) == 4:  # handle time series case
                output[f"image_{modality}"] = torch.cat(channels, dim=1)
            else:  # assume [C, H, W]
                output[f"image_{modality}"] = torch.cat(channels, dim=0)

        return output

    def _format_available_bands(self) -> str:
        """Format help string showing available bands for this dataset."""
        lines = ["Available bands:"]
        if isinstance(self.dataset_band_config, MultiModalConfig):
            for mod, config in self.dataset_band_config.modalities.items():
                lines.append(f"\n{mod}:")
                for name, band in config.bands.items():
                    aliases = ", ".join(band.aliases)
                    lines.append(f"  - {name} ({band.canonical_name}): {aliases}")
        else:
            for name, band in self.dataset_band_config.bands.items():
                aliases = ", ".join(band.aliases)
                lines.append(f"  - {name} ({band.canonical_name}): {aliases}")
        return "\n".join(lines)


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

    This approach maintains signal differences in bands with negative values
    without requiring any additional configuration parameters.
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
