"""Utility functions for handling satellite imagery datasets."""

from dataclasses import dataclass
from typing import Dict, List, Union, Optional, Sequence
from enum import Enum
import torch
from torch import Tensor


@dataclass
class BandConfig:
    """Configuration for a single band."""

    canonical_name: str
    aliases: List[str]
    wavelength: Optional[float] = None
    resolution: Optional[int] = None  # spatial resolution in meters

    def matches_alias(self, name: str) -> bool:
        """Check if name matches canonical name or aliases."""
        return name == self.canonical_name or name in self.aliases


@dataclass
class ModalityConfig:
    """Configuration for a satellite/sensor modality."""

    bands: Dict[str, BandConfig]
    default_order: List[str]  # Default band order for this modality
    native_resolution: Optional[int] = None  # Native resolution in meters

    # Add band_to_modality mapping for consistency with MultiModalConfig
    @property
    def band_to_modality(self) -> Dict[str, str]:
        """Maps band names to their modality. For single modality, all bands map to same modality."""
        return {band: "self" for band in self.bands.keys()}

    @property
    def modalities(self) -> Dict[str, "ModalityConfig"]:
        """For consistency with MultiModalConfig interface."""
        return {"self": self}

    def resolve_band(self, band_spec: str) -> str:
        """Resolve band name to canonical name within this modality."""
        for canon, band_config in self.bands.items():
            if band_spec == canon or band_spec in band_config.aliases:
                return canon
        raise ValueError(f"Band {band_spec} not found in configuration")


@dataclass
class MultiModalConfig:
    """Configuration for multi-modal datasets combining multiple sensors."""

    modalities: Dict[str, ModalityConfig]
    default_order: List[str]  # Default band order across all modalities
    band_to_modality: Dict[str, str]  # Maps band names to their modality


class SensorType(Enum):
    """Supported sensor types."""

    SENTINEL1 = "s1"
    SENTINEL2 = "s2"
    RGB = "rgb"
    RGBN = "rgbn"
    GRAYSCALE = "gray"
    LANDSAT8 = "l8"
    MODIS = "modis"


class SensorBandRegistry:
    """Registry of sensor-specific band configurations."""

    GRAYSCALE = ModalityConfig(
        bands={"gray": BandConfig("gray", ["gray"], wavelength=None)},
        default_order=["gray"],
    )

    RGB = ModalityConfig(
        bands={
            "r": BandConfig("red", ["r", "red", "RED"], wavelength=0.665),
            "g": BandConfig("green", ["g", "green", "GREEN"], wavelength=0.560),
            "b": BandConfig("blue", ["b", "blue", "BLUE"], wavelength=0.490),
        },
        default_order=["r", "g", "b"],
    )

    RGBN = ModalityConfig(
        bands={
            **RGB.bands,
            "nir": BandConfig("nir", ["nir", "NIR", "near_infrared"], wavelength=0.842),
        },
        default_order=["r", "g", "b", "nir"],
    )

    SENTINEL2 = ModalityConfig(
        bands={
            "B01": BandConfig(
                "coastal", ["coastal_aerosol", "b01"], wavelength=0.443, resolution=60
            ),
            "B02": BandConfig("blue", ["b02", "blue"], wavelength=0.490, resolution=10),
            "B03": BandConfig(
                "green", ["b03", "green"], wavelength=0.560, resolution=10
            ),
            "B04": BandConfig("red", ["b04", "red"], wavelength=0.665, resolution=10),
            "B05": BandConfig(
                "vegetation_red_edge_1", ["re1", "b05"], wavelength=0.705, resolution=20
            ),
            "B06": BandConfig(
                "vegetation_red_edge_2", ["re2", "b06"], wavelength=0.740, resolution=20
            ),
            "B07": BandConfig(
                "vegetation_red_edge_3", ["re3", "b07"], wavelength=0.783, resolution=20
            ),
            "B08": BandConfig(
                "nir", ["near_infrared", "b08"], wavelength=0.842, resolution=10
            ),
            "B8A": BandConfig(
                "vegetation_red_edge_4", ["re4", "b8a"], wavelength=0.865, resolution=20
            ),
            "B09": BandConfig(
                "water_vapor", ["wv", "b09"], wavelength=0.945, resolution=60
            ),
            "B11": BandConfig(
                "swir1",
                ["short_wave_infrared_1", "b11"],
                wavelength=1.610,
                resolution=20,
            ),
            "B12": BandConfig(
                "swir2",
                ["short_wave_infrared_2", "b12"],
                wavelength=2.190,
                resolution=20,
            ),
        },
        default_order=[
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B09",
            "B11",
            "B12",
        ],
        native_resolution=10,
    )

    SENTINEL1 = ModalityConfig(
        bands={
            "VV": BandConfig("vv", ["co_pol"], wavelength=0.056, resolution=10),
            "VH": BandConfig("vh", ["cross_pol"], wavelength=0.056, resolution=10),
        },
        default_order=["VV", "VH"],
        native_resolution=10,
    )

    @classmethod
    def get_modality_config(cls, modality: Union[str, SensorType]) -> ModalityConfig:
        """Get configuration for a specific modality."""
        if isinstance(modality, str):
            modality = SensorType(modality)
        return getattr(cls, modality.name)


class DatasetBandRegistry:
    """Registry of dataset-specific band configurations."""

    BENV2 = MultiModalConfig(
        modalities={
            "s2": SensorBandRegistry.SENTINEL2,
            "s1": SensorBandRegistry.SENTINEL1,
        },
        default_order=[
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B09",
            "B11",
            "B12",
            "VV",
            "VH",
        ],
        band_to_modality={
            "B01": "s2",
            "B02": "s2",
            "B03": "s2",
            "B04": "s2",
            "B05": "s2",
            "B06": "s2",
            "B07": "s2",
            "B08": "s2",
            "B8A": "s2",
            "B09": "s2",
            "B11": "s2",
            "B12": "s2",
            "VV": "s1",
            "VH": "s1",
        },
    )

    PASTIS = MultiModalConfig(
        modalities={
            "s2": SensorBandRegistry.SENTINEL2,
            "s1": SensorBandRegistry.SENTINEL1,
        },
        default_order=[
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B11",
            "B12",
            "VV_asc",
            "VH_asc",
            "VV/VH_asc",
            "VV_desc",
            "VH_desc",
            "VV/VH_desc",
        ],
        band_to_modality={
            "B02": "s2",
            "B03": "s2",
            "B04": "s2",
            "B05": "s2",
            "B06": "s2",
            "B07": "s2",
            "B08": "s2",
            "B8A": "s2",
            "B11": "s2",
            "B12": "s2",
            "VV_asc": "s1",
            "VH_asc": "s1",
            "VV/VH_asc": "s1",
            "VV_desc": "s1",
            "VH_desc": "s1",
            "VV/VH_desc": "s1",
        },
    )

    CAFFE = ModalityConfig(
        bands=SensorBandRegistry.GRAYSCALE.bands, default_order=["gray"]
    )

    EVERWATCH = ModalityConfig(
        bands=SensorBandRegistry.RGB.bands, default_order=["r", "g", "b"]
    )

    FOTW = ModalityConfig(
        bands=SensorBandRegistry.RGBN.bands, default_order=["r", "g", "b", "nir"]
    )

    RESISC45 = ModalityConfig(
        bands=SensorBandRegistry.RGB.bands, default_order=["r", "g", "b"]
    )

    SPACENET6 = ModalityConfig(
        bands=SensorBandRegistry.RGBN.bands, default_order=["r", "g", "b", "nir"]
    )

    @classmethod
    def get_dataset_config(
        cls, dataset_name: str
    ) -> Union[ModalityConfig, MultiModalConfig]:
        """Get configuration for a specific dataset."""
        return getattr(cls, dataset_name.upper())


def get_wavelengths(band_order: Sequence[str], sensor_type: SensorType) -> List[float]:
    """Get wavelengths in micrometers for given bands."""
    config = SensorBandRegistry.get_modality_config(sensor_type)
    wavelengths = []

    for band in band_order:
        if band not in config.bands:
            raise ValueError(
                f"Band {band} not found in {sensor_type.value} configuration"
            )
        wavelength = config.bands[band].wavelength
        if wavelength is None:
            raise ValueError(f"No wavelength information for band {band}")
        wavelengths.append(wavelength)

    return wavelengths
