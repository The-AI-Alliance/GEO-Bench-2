# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Data Processing Utility Mixin."""

from abc import ABC, abstractmethod
from typing import Union, Sequence, Optional
import torch
import torch.nn as nn
from torch import Tensor
import kornia.augmentation as K
from .sensor_util import ModalityConfig, MultiModalConfig, DatasetBandRegistry


class DataUtilsMixin(ABC):
    """Mixin for datasets that require band manipulation and normalization."""

    @property
    def normalization_stats(self) -> dict[str, dict[str, dict[str, float]]]:
        """Per-modality normalization statistics."""
        pass

    def get_source_order(self) -> Union[list[str], dict[str, list[str]]]:
        """Derive source order from dataset configuration."""
        if isinstance(self.dataset_band_config, MultiModalConfig):
            return {
                mod: config.default_order
                for mod, config in self.dataset_band_config.modalities.items()
            }
        return self.dataset_band_config.default_order

    def resolve_band_order(
        self,
        band_order: Optional[
            Union[Sequence[Union[str, float]], dict[str, Sequence[Union[str, float]]]]
        ] = None,
    ) -> Union[list[Union[str, float]], dict[str, list[Union[str, float]]]]:
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
    def _resolve_in_config(band: str, config: ModalityConfig) -> Optional[str]:
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
        data: Union[Tensor, dict[str, Tensor]],
        target_order: Union[
            list[Union[str, float]], dict[str, list[Union[str, float]]]
        ],
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
        self,
        data: Tensor,
        source_order: list[str],
        target_order: list[Union[str, float]],
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
                channel = data[idx : idx + 1]
            output_channels.append(channel)

        shape = data.shape
        if len(shape) == 4:  # handle time series case
            return {"image": torch.cat(output_channels, dim=1)}
        else:  # assume [C, H, W]
            return {"image": torch.cat(output_channels, dim=0)}

    def _rearrange_multimodal_to_tensor(
        self, data: dict[str, Tensor], target_order: list[Union[str, float]]
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
        self, data: dict[str, Tensor], target_order: dict[str, list[Union[str, float]]]
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
                    idx = mod_config.default_order.index(band)
                    if len(source_data.shape) == 4:  # timeseries of [T, C, H, W]
                        channel = source_data[:, idx : idx + 1]
                    else:  # assume [C, H, W]
                        channel = source_data[idx : idx + 1]
                channels.append(channel)

            shape = list(next(iter(data.values())).shape)
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


class MultiModalNormalizer(nn.Module):
    """Normalization module for single or multi-modal data."""

    def __init__(
        self,
        stats: dict[str, dict[str, float]],
        band_order: Union[list[Union[str, float]], dict[str, list[Union[str, float]]]],
    ) -> None:
        """Initialize normalizer.

        Args:
            stats: dictionary containing mean and std for each band
            band_order: Either a sequence of bands or dict mapping modalities to sequences
        """
        super().__init__()
        self.stats = stats
        self.normalizers = self._setup_normalizers(band_order)

    def _setup_normalizers(
        self,
        band_order: Union[list[Union[str, float]], dict[str, list[Union[str, float]]]],
    ) -> dict[str, K.Normalize]:
        """Set up normalization transforms."""
        if isinstance(band_order, dict):
            # Multi-modal case
            normalizers = {}
            for modality, bands in band_order.items():
                means, stds = [], []
                for band in bands:
                    if isinstance(band, (int, float)):
                        means.append(0.0)
                        stds.append(1.0)
                    else:
                        means.append(self.stats["means"][band])
                        stds.append(self.stats["stds"][band])
                normalizers[f"image_{modality}"] = K.Normalize(
                    torch.tensor(means), torch.tensor(stds), keepdim=True
                )
            return normalizers
        else:
            # Single tensor case
            means, stds = [], []
            for band in band_order:
                if isinstance(band, (int, float)):
                    means.append(0.0)
                    stds.append(1.0)
                else:
                    means.append(self.stats["means"][band])
                    stds.append(self.stats["stds"][band])
            return {
                "image": K.Normalize(
                    torch.tensor(means), torch.tensor(stds), keepdim=True
                )
            }

    def forward(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Normalize input tensors.

        Args:
            data: dictionary mapping keys to tensors
                 For single modality: {"image": tensor}
                 For multi-modal: {"image_s1": tensor1, "image_s2": tensor2}

        Returns:
            dictionary with normalized tensors using same keys
        """
        return {key: self.normalizers[key](tensor) for key, tensor in data.items()}
