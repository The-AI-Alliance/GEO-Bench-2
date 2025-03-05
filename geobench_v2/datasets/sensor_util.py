# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Utility functions for handling satellite imagery datasets."""

from dataclasses import dataclass
from typing import Dict, List, Union, Optional, TypeVar
from enum import Enum
import torch
from torch import Tensor


class SatelliteType(Enum):
    """Supported satellite types and modalities."""

    SENTINEL1 = "s1"
    SENTINEL2 = "s2"
    # LANDSAT8 = "l8"
    # MODIS = "modis"
    RGB = "rgb"
    RGBN = "rgbn"
    # GRAYSCALE = "gray"
    # MULTIMODAL = "multimodal"


@dataclass
class BandConfig:
    """Configuration for a single band."""

    canonical_name: str
    aliases: List[str]
    wavelength: Optional[float] = None
    resolution: Optional[int] = None  # spatial resolution in meters


@dataclass
class ModalityConfig:
    """Configuration for a satellite/sensor modality."""

    bands: Dict[str, BandConfig]
    default_order: List[str]  # Default band order for this modality
    native_resolution: Optional[int] = None  # Native resolution in meters


class BandRegistry:
    """Global registry of band configurations for different satellites/modalities."""

    # Standard RGB configuration
    RGB = ModalityConfig(
        bands={
            "r": BandConfig("red", ["r", "red", "R", "B04", "B4"], wavelength=0.665),
            "g": BandConfig(
                "green", ["g", "green", "G", "B03", "B3"], wavelength=0.560
            ),
            "b": BandConfig("blue", ["b", "blue", "B", "B02", "B2"], wavelength=0.490),
        },
        default_order=["r", "g", "b"],
    )

    # RGBN configuration (RGB + NIR)
    # extend the RGB configuration with a NIR band
    RGBN = ModalityConfig(
        bands={
            **RGB.bands,
            "n": BandConfig("nir", ["nir", "NIR", "B08", "B8"], wavelength=0.842),
        },
        default_order=["r", "g", "b", "nir"],
    )

    # Sentinel-2 configuration
    SENTINEL2 = ModalityConfig(
        bands={
            "B01": BandConfig(
                "coastal",
                ["coastal", "coastal_aerosol"],
                wavelength=0.443,
                resolution=60,
            ),
            "B02": BandConfig("blue", ["blue", "b"], wavelength=0.490, resolution=10),
            "B03": BandConfig("green", ["green", "g"], wavelength=0.560, resolution=10),
            "B04": BandConfig("red", ["red", "r"], wavelength=0.665, resolution=10),
            "B05": BandConfig(
                "vegetation_red_edge_1", ["re1"], wavelength=0.705, resolution=20
            ),
            "B06": BandConfig(
                "vegetation_red_edge_2", ["re2"], wavelength=0.740, resolution=20
            ),
            "B07": BandConfig(
                "vegetation_red_edge_3", ["re3"], wavelength=0.783, resolution=20
            ),
            "B08": BandConfig(
                "nir", ["near_infrared"], wavelength=0.842, resolution=10
            ),
            "B8A": BandConfig(
                "vegetation_red_edge_4", ["re4"], wavelength=0.865, resolution=20
            ),
            "B09": BandConfig("water_vapor", ["wv"], wavelength=0.945, resolution=60),
            "B11": BandConfig(
                "swir1", ["short_wave_infrared_1"], wavelength=1.610, resolution=20
            ),
            "B12": BandConfig(
                "swir2", ["short_wave_infrared_2"], wavelength=2.190, resolution=20
            ),
        },
        default_order=["B04", "B03", "B02"],  # RGB default
        native_resolution=10,
    )

    # Sentinel-1 configuration
    SENTINEL1 = ModalityConfig(
        bands={
            "VV": BandConfig("vv", ["co_pol"], resolution=10),
            "VH": BandConfig("vh", ["cross_pol"], resolution=10),
        },
        default_order=["VV", "VH"],
        native_resolution=10,
    )

    @classmethod
    def get_modality_config(cls, modality: Union[str, SatelliteType]) -> ModalityConfig:
        """Get configuration for a specific modality."""
        if isinstance(modality, str):
            modality = SatelliteType(modality)

        return getattr(cls, modality.name)

    @classmethod
    def resolve_band(
        cls,
        band_spec: Union[str, float],
        modality: Optional[Union[str, SatelliteType]] = None,
    ) -> Union[tuple[str, str], float]:
        """Resolve band specification to (modality, band_name) or fill value."""
        if isinstance(band_spec, (int, float)):
            return float(band_spec)

        # Handle modality-prefixed bands (e.g., "s2_B02")
        if "_" in band_spec:
            mod, band = band_spec.split("_", 1)
            config = cls.get_modality_config(mod)
            # return mod, cls._resolve_in_config(band, config)
            return cls._resolve_in_config(band, config)
        # Try in specified modality
        if modality:
            config = cls.get_modality_config(modality)
            # return modality, cls._resolve_in_config(band_spec, config)
            return cls._resolve_in_config(band_spec, config)

        # Try all modalities
        for mod in SatelliteType:
            try:
                config = cls.get_modality_config(mod)
                # return mod.value, cls._resolve_in_config(band_spec, config)
                return cls._resolve_in_config(band_spec, config)
            except ValueError:
                continue

        import pdb

        pdb.set_trace()

        raise ValueError(f"Unknown band: {band_spec}\n\n{cls.format_help()}")

    @staticmethod
    def _resolve_in_config(band: str, config: ModalityConfig) -> str:
        """Resolve band name within a specific configuration."""
        for canon, band_config in config.bands.items():
            if band == canon or band in band_config.aliases:
                return canon
        raise ValueError(f"Band {band} not found in configuration")

    @classmethod
    def format_help(cls) -> str:
        """Format help string showing all available bands."""
        lines = ["Available bands by modality:"]
        for modality in SatelliteType:
            try:
                config = cls.get_modality_config(modality)
                lines.append(f"\n{modality.value}:")
                for name, band in config.bands.items():
                    aliases = ", ".join(band.aliases)
                    lines.append(f"  - {name} ({band.canonical_name}): {aliases}")
            except AttributeError:
                continue
        return "\n".join(lines)


def get_wavelengths(
    band_order: List[str], satellite_type: SatelliteType
) -> List[float]:
    """Get center wavelengths in micrometers for given bands.

    Args:
        band_order: List of band names
        satellite_type: Satellite type for wavelength lookup

    Returns:
        List of wavelengths in micrometers. Returns None for non-optical bands (e.g., SAR).

    References:
        Sentinel-2: https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/resolutions/spectral
        Landsat-8: https://www.usgs.gov/landsat-missions/landsat-8
        MODIS: https://modis.gsfc.nasa.gov/about/specifications.php
    """
    # Wavelengths in micrometers (μm)
    WAVELENGTHS = {
        SatelliteType.SENTINEL2: {
            "B01": 0.443,  # Coastal aerosol (60m)
            "B02": 0.490,  # Blue (10m)
            "B03": 0.560,  # Green (10m)
            "B04": 0.665,  # Red (10m)
            "B05": 0.705,  # Vegetation Red Edge (20m)
            "B06": 0.740,  # Vegetation Red Edge (20m)
            "B07": 0.783,  # Vegetation Red Edge (20m)
            "B08": 0.842,  # NIR (10m)
            "B8A": 0.865,  # Narrow NIR (20m)
            "B09": 0.945,  # Water vapor (60m)
            "B11": 1.610,  # SWIR 1 (20m)
            "B12": 2.190,  # SWIR 2 (20m)
        },
        SatelliteType.LANDSAT8: {
            "B1": 0.443,  # Coastal aerosol
            "B2": 0.482,  # Blue
            "B3": 0.562,  # Green
            "B4": 0.655,  # Red
            "B5": 0.865,  # NIR
            "B6": 1.610,  # SWIR 1
            "B7": 2.200,  # SWIR 2
            "B8": 0.590,  # Panchromatic
            "B9": 1.375,  # Cirrus
            "B10": 10.9,  # Thermal Infrared 1
            "B11": 12.0,  # Thermal Infrared 2
        },
        SatelliteType.MODIS: {
            "B1": 0.645,  # Red
            "B2": 0.858,  # NIR
            "B3": 0.469,  # Blue
            "B4": 0.555,  # Green
            "B5": 1.240,  # NIR
            "B6": 1.640,  # SWIR 1
            "B7": 2.130,  # SWIR 2
            # MODIS has more bands (36 total), adding most commonly used ones
        },
        SatelliteType.RGB: {
            "r": 0.665,  # Red
            "g": 0.560,  # Green
            "b": 0.490,  # Blue
        },
        # Sentinel-1 uses C-band SAR (wavelength ~5.6cm)
        SatelliteType.SENTINEL1: {
            "VV": 0.056,  # Vertical transmit, Vertical receive
            "VH": 0.056,  # Vertical transmit, Horizontal receive
        },
    }

    band_order = normalize_band_order(band_order, satellite_type)
    wavelengths = WAVELENGTHS.get(satellite_type, {})

    if not wavelengths:
        raise ValueError(f"No wavelength information available for {satellite_type}")

    try:
        return [wavelengths[band] for band in band_order]
    except KeyError as e:
        raise KeyError(f"No wavelength information for band {e} in {satellite_type}")
