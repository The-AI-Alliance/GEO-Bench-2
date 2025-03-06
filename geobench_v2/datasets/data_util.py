from abc import ABC, abstractmethod
from typing import Dict, List, Union
import torch
from torch import Tensor
import kornia.augmentation as K


class DataUtilsMixin(ABC):
    """Mixin for datasets that require band manipulation and normalization."""

    @property
    def normalization_stats(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Per-modality normalization statistics."""
        pass

    def resolve_bands(
        self, band_order: Sequence[Union[str, float]]
    ) -> list[Union[str, float]]:
        """Resolve band names to canonical names."""
        resolved_bands = []
        for band_spec in band_order:
            if isinstance(band_spec, (int, float)):
                resolved_bands.append(band_spec)
            else:
                resolved_bands.append(
                    BandRegistry.resolve_band(band_spec, self.sensor_type)
                )
        return resolved_bands

    def rearrange_bands(
        self,
        data: Union[Tensor, Dict[str, Tensor]],
        source_order: Union[List[str], Dict[str, List[str]]],
        target_order: List[Union[str, float]],
    ) -> Tensor:
        """Rearrange bands using already resolved band names.

        Args:
            data: Input tensor or dict of tensors
            source_order: Current band order (using canonical names)
            target_order: Desired band order (already resolved to canonical names or fill values)
        """
        if isinstance(data, dict):
            return self._rearrange_multimodal(data, source_order, target_order)
        return self._rearrange_single_modality(data, source_order, target_order)

    def _rearrange_single_modality(
        self,
        data: Tensor,
        source_order: List[str],
        target_order: List[Union[str, float]],
    ) -> Tensor:
        """Rearrange bands for single modality data."""
        output_channels = []
        device = data.device

        # Create lookup from canonical names to indices
        source_lookup = {name: idx for idx, name in enumerate(source_order)}

        for band_spec in target_order:
            if isinstance(band_spec, (int, float)):
                # Handle fill values
                shape = list(data.shape)
                shape[0] = 1  # Assuming CHW format
                channel = torch.full(shape, float(band_spec), device=device)
            else:
                # Band names are already resolved, direct lookup
                source_idx = source_lookup[band_spec]
                channel = data[[source_idx], ...]

            output_channels.append(channel)

        return torch.cat(output_channels, dim=0)

    def _rearrange_multimodal(
        self,
        data: Dict[str, Tensor],
        source_order: Dict[str, List[str]],
        target_order: List[Union[str, float]],
    ) -> Tensor:
        """Rearrange bands for multimodal data."""
        output_channels = []
        device = next(iter(data.values())).device

        for band_spec in target_order:
            if isinstance(band_spec, (int, float)):
                shape = list(next(iter(data.values())).shape)
                shape[0] = 1  # Assuming CHW format
                channel = torch.full(shape, float(band_spec), device=device)
            else:
                # Extract modality prefix and band name
                modality, _ = band_spec.split("_", 1)
                source_idx = source_order[modality].index(band_spec)
                channel = data[modality][[source_idx], ...]

            output_channels.append(channel)

        return torch.cat(output_channels, dim=0)

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
