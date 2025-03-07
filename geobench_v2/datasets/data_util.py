# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Data Processing Utility Mixin."""

from abc import ABC, abstractmethod
from typing import Dict, List, Union, Sequence, Optional
import torch
from torch import Tensor
import kornia.augmentation as K
from .sensor_util import ModalityConfig, MultiModalConfig, DatasetBandRegistry


class DataUtilsMixin(ABC):
    """Mixin for datasets that require band manipulation and normalization."""

    @property
    def normalization_stats(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Per-modality normalization statistics."""
        pass

    def get_source_order(self) -> Union[List[str], Dict[str, List[str]]]:
        """Derive source order from dataset configuration."""
        if isinstance(self.dataset_band_config, MultiModalConfig):
            return {
                mod: config.default_order
                for mod, config in self.dataset_band_config.modalities.items()
            }
        return self.dataset_band_config.default_order

    def resolve_band_order(
        self, band_order: Optional[Sequence[Union[str, float]]] = None
    ) -> List[Union[str, float]]:
        """Resolve band names to canonical names."""
        if band_order is None:
            return self.dataset_band_config.default_order

        resolved_bands = []
        for band_spec in band_order:
            if isinstance(band_spec, (int, float)):
                resolved_bands.append(band_spec)
                continue

            # First try exact match in band_to_modality mapping
            if band_spec in self.dataset_band_config.band_to_modality:
                resolved_bands.append(band_spec)  # Already canonical
                continue

            # Try resolving through aliases in each modality
            resolved = None
            for mod, config in self.dataset_band_config.modalities.items():
                try:
                    canon = self._resolve_in_config(band_spec, config)
                    if canon in self.dataset_band_config.band_to_modality:
                        resolved = canon
                        break
                except ValueError:
                    continue

            if resolved is None:
                raise ValueError(
                    f"Could not resolve band {band_spec}\n"
                    f"Available bands: {self._format_available_bands()}"
                )
            resolved_bands.append(resolved)

        return resolved_bands

    def rearrange_bands(
        self,
        data: Union[Tensor, Dict[str, Tensor]],
        target_order: Union[
            List[Union[str, float]], Dict[str, List[Union[str, float]]]
        ],
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """Rearrange bands using dataset configuration."""
        if not isinstance(self.dataset_band_config, MultiModalConfig):
            # Handle single modality case
            if isinstance(data, dict) or isinstance(target_order, dict):
                raise ValueError(
                    "Single modality config requires tensor input and list target_order"
                )
            return self._rearrange_single_modality(
                data, self.dataset_band_config.default_order, target_order
            )

        # Multi-modal case
        if not isinstance(data, dict):
            raise ValueError("Multi-modal config requires dict input data")

        # Case 1: Dict target order -> return dict of tensors
        if isinstance(target_order, dict):
            return self._rearrange_multimodal_to_dict(data, target_order)

        # Case 2: List target order -> return single tensor
        return self._rearrange_multimodal_to_tensor(data, target_order)

    def _rearrange_multimodal_to_tensor(
        self, data: Dict[str, Tensor], target_order: List[Union[str, float]]
    ) -> Tensor:
        """Create single tensor from multi-modal data."""
        resolved_order = self.resolve_band_order(target_order)
        output_channels = []

        for band_spec in resolved_order:
            if isinstance(band_spec, (int, float)):
                shape = list(next(iter(data.values())).shape)
                shape[0] = 1
                channel = torch.full(
                    shape, float(band_spec), device=next(iter(data.values())).device
                )
            else:
                # Use band_to_modality mapping to find source
                modality = self.dataset_band_config.band_to_modality[band_spec]
                source_data = data[modality]

                # Find index in source modality's order
                mod_config = self.dataset_band_config.modalities[modality]
                try:
                    idx = mod_config.default_order.index(band_spec)
                except ValueError:
                    raise ValueError(
                        f"Band {band_spec} not found in {modality} default order"
                    )

                channel = source_data[idx : idx + 1]

            output_channels.append(channel)

        return torch.cat(output_channels, dim=0)

    def _rearrange_multimodal_to_dict(
        self, data: Dict[str, Tensor], target_order: Dict[str, List[Union[str, float]]]
    ) -> Dict[str, Tensor]:
        """Create dict of tensors from multi-modal data."""
        output = {}

        for modality, bands in target_order.items():
            if modality not in self.dataset_band_config.modalities:
                raise ValueError(f"Unknown modality: {modality}")

            resolved = self.resolve_band_order(bands)
            source_data = data[modality]
            mod_config = self.dataset_band_config.modalities[modality]

            channels = []
            for band in resolved:
                if isinstance(band, (int, float)):
                    shape = list(source_data.shape)
                    shape[0] = 1
                    channel = torch.full(shape, float(band), device=source_data.device)
                else:
                    idx = mod_config.default_order.index(band)
                    channel = source_data[idx : idx + 1]
                channels.append(channel)

            output[f"image_{modality}"] = torch.cat(channels, dim=0)

        return output

    @staticmethod
    def _resolve_in_config(band: str, config: ModalityConfig) -> str:
        """Resolve band name within a specific modality configuration."""
        for canon, band_config in config.bands.items():
            if band == canon or band in band_config.aliases:
                return canon
        raise ValueError(f"Band {band} not found in configuration")

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

    def set_normalization_stats(self, band_order: List[Union[str, float]]) -> None:
        """Set up normalization transform for the specified band order."""
        means, stds = [], []

        for band_spec in band_order:
            if isinstance(band_spec, (int, float)):
                means.append(0.0)
                stds.append(1.0)
            else:
                # Band names are already resolved, direct lookup
                means.append(self.normalization_stats["means"][band_spec])
                stds.append(self.normalization_stats["stds"][band_spec])

        self.normalizer = K.Normalize(
            torch.tensor(means), torch.tensor(stds), keepdim=True
        )
