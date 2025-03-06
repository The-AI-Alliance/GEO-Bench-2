# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Data Processing Utility Mixin."""

from abc import ABC, abstractmethod
from typing import Dict, List, Union, Sequence, Optional
import torch
from torch import Tensor
import kornia.augmentation as K
from .sensor_util import BandRegistry, ModalityConfig, MultiModalConfig, DatasetBandRegistry



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
    ) -> list[Union[str, float]]:
        """Resolve band names to canonical names using dataset configuration.
        
        Args:
            band_order: Requested band order. If None, uses dataset default order.
                Can include fill values as numbers.
        
        Returns:
            List of canonical band names or fill values.
        """
        if band_order is None:
            return self.dataset_band_config.default_order
            
        resolved_bands = []
        for band_spec in band_order:
            if isinstance(band_spec, (int, float)):
                resolved_bands.append(band_spec)
                continue
                
            # Handle modality-prefixed bands (e.g. "s2_B02")
            if "_" in band_spec and isinstance(self.dataset_band_config, MultiModalConfig):
                mod, band = band_spec.split("_", 1)
                if mod not in self.dataset_band_config.modalities:
                    raise ValueError(f"Unknown modality prefix: {mod}")
                config = self.dataset_band_config.modalities[mod]
                resolved = self._resolve_in_config(band, config)
                resolved_bands.append(f"{mod}_{resolved}")
                continue
            
            # Multi-modal case
            if isinstance(self.dataset_band_config, MultiModalConfig):
                resolved = None
                for mod, config in self.dataset_band_config.modalities.items():
                    try:
                        band_name = self._resolve_in_config(band_spec, config)
                        resolved = f"{mod}_{band_name}"
                        break
                    except ValueError:
                        continue
                        
                if resolved is None:
                    raise ValueError(
                        f"Could not resolve band {band_spec} in any modality\n"
                        f"Available bands: {self._format_available_bands()}"
                    )
                resolved_bands.append(resolved)
                
            # Single modality case
            else:
                try:
                    resolved = self._resolve_in_config(band_spec, self.dataset_band_config)
                    resolved_bands.append(resolved)
                except ValueError:
                    raise ValueError(
                        f"Could not resolve band {band_spec}\n"
                        f"Available bands: {self._format_available_bands()}"
                    )
            
        return resolved_bands

    
    def rearrange_bands(
        self,
        data: Union[Tensor, Dict[str, Tensor]],
        target_order: Union[List[Union[str, float]], Dict[str, List[Union[str, float]]]],
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """Rearrange bands using dataset configuration.

        Args:
            data: Input tensor (single modality) or dict of tensors (multi-modal)
            target_order: Desired band order (resolved canonical names or fill values)
                If List: Returns single tensor with bands from any modality
                If Dict: Returns dict of tensors with modality-specific ordering

        Returns:
            Single tensor or dict of tensors with bands arranged in target order
        """
        source_order = self.get_source_order()

        # Single modality case
        if isinstance(data, Tensor):
            if isinstance(self.dataset_band_config, MultiModalConfig):
                raise ValueError("Single tensor input not allowed for multi-modal config")
            if isinstance(target_order, dict):
                raise ValueError("Dict target order not allowed for single modality")
            return self._rearrange_bands_single_modality(data, source_order, target_order)

        # Multi-modal case
        if isinstance(data, dict):
            if not isinstance(self.dataset_band_config, MultiModalConfig):
                raise ValueError("Dict input requires multi-modal config")
            if isinstance(target_order, dict):
                return self._rearrange_bands_multimodal_to_dict(
                    data, source_order, target_order
                )
            return self._rearrange_bands_multimodal_to_tensor(
                data, source_order, target_order
            )

        raise ValueError("Data must be either Tensor or Dict[str, Tensor]")

    def _rearrange_bands_single_modality(
        self,
        data: Tensor,
        source_order: List[str],
        target_order: List[Union[str, float]],
    ) -> Tensor:
        """Rearrange bands for single modality."""
        output_channels = []
        source_lookup = {band: idx for idx, band in enumerate(source_order)}

        for band_spec in target_order:
            if isinstance(band_spec, (int, float)):
                # Handle fill values
                shape = list(data.shape)
                shape[0] = 1
                channel = torch.full(shape, float(band_spec), device=data.device)
            else:
                try:
                    idx = source_lookup[band_spec]
                    channel = data[idx:idx+1]
                except KeyError:
                    raise ValueError(f"Band {band_spec} not found in source order")
            output_channels.append(channel)

        return torch.cat(output_channels, dim=0)

    def _rearrange_bands_multimodal_to_dict(
        self,
        data: Dict[str, Tensor],
        source_order: Dict[str, List[str]],
        target_order: Dict[str, List[Union[str, float]]],
    ) -> Dict[str, Tensor]:
        """Rearrange bands keeping modalities separate."""
        output = {}
        
        for modality, bands in target_order.items():
            if modality not in data:
                raise ValueError(f"Missing data for modality: {modality}")
            if modality not in source_order:
                raise ValueError(f"Missing source order for modality: {modality}")
                
            output[modality] = self._rearrange_bands_single_modality(
                data[modality], 
                source_order[modality],
                bands
            )
            
        return output

    def _rearrange_bands_multimodal_to_tensor(
        self,
        data: Dict[str, Tensor],
        source_order: Dict[str, List[str]],
        target_order: List[Union[str, float]],
    ) -> Tensor:
        """Rearrange multi-modal bands into single tensor."""
        output_channels = []
        
        # Create modality lookups
        lookups = {
            mod: {band: idx for idx, band in enumerate(bands)}
            for mod, bands in source_order.items()
        }

        for band_spec in target_order:
            if isinstance(band_spec, (int, float)):
                # Handle fill values
                shape = list(next(iter(data.values())).shape)
                shape[0] = 1
                channel = torch.full(shape, float(band_spec), device=next(iter(data.values())).device)
            else:
                # Use mapping from dataset config
                try:
                    modality = self.dataset_band_config.band_to_modality[band_spec]
                    band = band_spec
                except KeyError:
                    raise ValueError(
                        f"Band {band_spec} not found in band_to_modality mapping\n"
                        f"Available mappings: {self.dataset_cband_onfig.band_to_modality}"
                    )
                
                if modality not in data:
                    raise ValueError(f"Missing data for modality: {modality}")
                if modality not in lookups:
                    raise ValueError(f"Unknown modality: {modality}")
                if band not in lookups[modality]:
                    raise ValueError(f"Band {band} not found in modality {modality}")
                
                idx = lookups[modality][band]
                channel = data[modality][idx:idx+1]
                
            output_channels.append(channel)

        return torch.cat(output_channels, dim=0)

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
