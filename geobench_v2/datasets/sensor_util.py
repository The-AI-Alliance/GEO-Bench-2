# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

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

    def resolve_band(self, band_spec: str) -> Optional[str]:
        """Resolve band name to canonical name within this modality.

        Args:
            band_spec: Band name or alias to resolve

        Returns:
            Canonical band name if found, None otherwise
        """
        for canon, band_config in self.bands.items():
            if band_spec == canon or band_spec in band_config.aliases:
                return canon
        return None  # Return None instead of raising ValueError


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
            "B10": BandConfig(
                "cirrus", ["cirrus", "b10"], wavelength=1.375, resolution=60
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
            "B10",
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

    # https://modis.gsfc.nasa.gov/about/specifications.php
    # https://modis.gsfc.nasa.gov/about/specifications.php
    MODIS = ModalityConfig(
        bands={
            # Land/Cloud/Aerosols Boundaries, MODIS bands 1-2 are 250m resolution
            "M01": BandConfig(
                "red", ["band1", "red"], wavelength=0.645, resolution=250
            ),
            "M02": BandConfig(
                "nir", ["band2", "near_infrared"], wavelength=0.8585, resolution=250
            ),
            # Land/Cloud/Aerosols Properties, MODIS bands 3-7 are 500m resolution
            "M03": BandConfig(
                "blue", ["band3", "blue"], wavelength=0.469, resolution=500
            ),
            "M04": BandConfig(
                "green", ["band4", "green"], wavelength=0.555, resolution=500
            ),
            "M05": BandConfig(
                "swir",
                ["band5", "short_wave_infrared"],
                wavelength=1.24,
                resolution=500,
            ),
            "M06": BandConfig(
                "swir2",
                ["band6", "short_wave_infrared_2"],
                wavelength=1.64,
                resolution=500,
            ),
            "M07": BandConfig(
                "swir3",
                ["band7", "short_wave_infrared_3"],
                wavelength=2.13,
                resolution=500,
            ),
            # Ocean Color/Phytoplankton/Biogeochemistry, MODIS bands 8-36 are 1000m resolution
            "M08": BandConfig(
                "deep_blue", ["band8", "deep_blue"], wavelength=0.4125, resolution=1000
            ),
            "M09": BandConfig(
                "blue_2", ["band9", "blue_2"], wavelength=0.443, resolution=1000
            ),
            "M10": BandConfig(
                "blue_3", ["band10", "blue_3"], wavelength=0.488, resolution=1000
            ),
            "M11": BandConfig(
                "green_2", ["band11", "green_2"], wavelength=0.531, resolution=1000
            ),
            "M12": BandConfig(
                "green_3", ["band12", "green_3"], wavelength=0.551, resolution=1000
            ),
            "M13": BandConfig(
                "red_2", ["band13", "red_2"], wavelength=0.667, resolution=1000
            ),
            "M14": BandConfig(
                "red_3", ["band14", "red_3"], wavelength=0.678, resolution=1000
            ),
            "M15": BandConfig(
                "red_edge", ["band15", "red_edge"], wavelength=0.748, resolution=1000
            ),
            "M16": BandConfig(
                "nir_2",
                ["band16", "near_infrared_2"],
                wavelength=0.8695,
                resolution=1000,
            ),
            # Atmospheric Water Vapor
            "M17": BandConfig(
                "water_vapor_1",
                ["band17", "water_vapor_1"],
                wavelength=0.905,
                resolution=1000,
            ),
            "M18": BandConfig(
                "water_vapor_2",
                ["band18", "water_vapor_2"],
                wavelength=0.936,
                resolution=1000,
            ),
            "M19": BandConfig(
                "water_vapor_3",
                ["band19", "water_vapor_3"],
                wavelength=0.94,
                resolution=1000,
            ),
        },
        default_order=[
            "M01",
            "M02",
            "M03",
            "M04",
            "M05",
            "M06",
            "M07",
            "M08",
            "M09",
            "M10",
            "M11",
            "M12",
            "M13",
            "M14",
            "M15",
            "M16",
            "M17",
            "M18",
            "M19",
        ],
        native_resolution=500,
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
            "s2": ModalityConfig(
                bands={
                    k: v
                    for k, v in SensorBandRegistry.SENTINEL2.bands.items()
                    if k
                    in [
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
                    ]
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
                ],
                native_resolution=10,
            ),
            "s1_asc": ModalityConfig(
                bands={
                    "VV_asc": BandConfig(
                        "vv_ascending",
                        ["co_pol_asc", "vv_asc"],
                        wavelength=0.056,
                        resolution=10,
                    ),
                    "VH_asc": BandConfig(
                        "vh_ascending",
                        ["cross_pol_asc", "vh_asc"],
                        wavelength=0.056,
                        resolution=10,
                    ),
                    "VV/VH_asc": BandConfig(
                        "ratio_ascending",
                        ["ratio_asc", "vv/vh_asc"],
                        wavelength=0.056,
                        resolution=10,
                    ),
                },
                default_order=["VV_asc", "VH_asc", "VV/VH_asc"],
                native_resolution=10,
            ),
            "s1_desc": ModalityConfig(
                bands={
                    "VV_desc": BandConfig(
                        "vv_descending",
                        ["co_pol_desc", "vv_desc"],
                        wavelength=0.056,
                        resolution=10,
                    ),
                    "VH_desc": BandConfig(
                        "vh_descending",
                        ["cross_pol_desc", "vv_desc"],
                        wavelength=0.056,
                        resolution=10,
                    ),
                    "VV/VH_desc": BandConfig(
                        "ratio_descending",
                        ["ratio_desc", "vv/vh_desc"],
                        wavelength=0.056,
                        resolution=10,
                    ),
                },
                default_order=["VV_desc", "VH_desc", "VV/VH_desc"],
                native_resolution=10,
            ),
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
            *["VV_asc", "VH_asc", "VV/VH_asc"],  # S1 ascending bands
            *["VV_desc", "VH_desc", "VV/VH_desc"],  # S1 descending bands
        ],
        band_to_modality={
            **{
                k: "s2"
                for k in [
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
                ]
            },
            **{band: "s1_asc" for band in ["VV_asc", "VH_asc", "VV/VH_asc"]},
            **{band: "s1_desc" for band in ["VV_desc", "VH_desc", "VV/VH_desc"]},
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

    # spacenet 6 is multimodal with rgbn and sar intensity bands (HH, HV,VH, and VV)
    SPACENET6 = MultiModalConfig(
        modalities={
            "rgbn": SensorBandRegistry.RGBN,
            "sar": ModalityConfig(
                bands={
                    "hh": BandConfig("hh", ["HH"], wavelength=0.056),
                    "hv": BandConfig("hv", ["HV"], wavelength=0.056),
                    "vv": BandConfig("vv", ["VV"], wavelength=0.056),
                    "vh": BandConfig("vh", ["VH"], wavelength=0.056),
                },
                default_order=["hh", "hv", "vv", "vh"],
            ),
        },
        default_order=["r", "g", "b", "nir", "hh", "hv", "vv", "vh"],
        band_to_modality={
            "r": "rgbn",
            "g": "rgbn",
            "b": "rgbn",
            "nir": "rgbn",
            "hh": "sar",
            "hv": "sar",
            "vv": "sar",
            "vh": "sar",
        },
    )
    
    SPACENET7 = ModalityConfig(
        bands=SensorBandRegistry.RGBN.bands, default_order=["r", "g", "b", "nir"]
    )

    SPACENET8 = ModalityConfig(
        bands=SensorBandRegistry.RGB.bands, default_order=["r", "g", "b"]
    )

    # flair 2 has rgbn and elevation bands
    FLAIR2 = ModalityConfig(
        bands={
            **SensorBandRegistry.RGBN.bands,
            "elevation": BandConfig("elevation", ["elevation"], wavelength=None),
        },
        default_order=["r", "g", "b", "nir", "elevation"],
    )

    # CLOUDSEN12 has cloudsen12-l1c Sentinel2 data is actually just a single ModalityConfig
    CLOUDSEN12 = ModalityConfig(
        bands={**SensorBandRegistry.SENTINEL2.bands},
        # band_to_modality={**SensorBandRegistry.SENTINEL2.band_to_modality},
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
    )

    FLOGA = MultiModalConfig(
        modalities={
            "s2": ModalityConfig(
                bands={
                    k: v
                    for k, v in SensorBandRegistry.SENTINEL2.bands.items()
                    if k in ["B02", "B03", "B04", "B08"] and v.resolution == 10
                },
                default_order=["B02", "B03", "B04", "B08"],
                native_resolution=10,
            ),
            "modis": ModalityConfig(
                bands={
                    k: v
                    for k, v in SensorBandRegistry.MODIS.bands.items()
                    if k in ["M01", "M02", "M03", "M04", "M05", "M06", "M07"]
                },
                default_order=["M01", "M02", "M03", "M04", "M05", "M06", "M07"],
                native_resolution=500,
            ),
        },
        default_order=[
            "B02",
            "B03",
            "B04",
            "B08",
            "M01",
            "M02",
            "M03",
            "M04",
            "M05",
            "M06",
            "M07",
        ],
        band_to_modality={
            "B02": "s2",
            "B03": "s2",
            "B04": "s2",
            "B08": "s2",
            "M01": "modis",
            "M02": "modis",
            "M03": "modis",
            "M04": "modis",
            "M05": "modis",
            "M06": "modis",
            "M07": "modis",
        },
    )

    KURO_SIWO = MultiModalConfig(
        modalities={
            "sar": ModalityConfig(
                bands={
                    "vv": BandConfig("vv", ["VV"], wavelength=0.056),
                    "vh": BandConfig("vh", ["VH"], wavelength=0.056),
                },
                default_order=["vv", "vh"],
            ),
            "dem": ModalityConfig(
                bands={"dem": BandConfig("dem", ["elevation", "dem"], wavelength=None)},
                default_order=["dem"],
            ),
        },
        default_order=["vv", "vh", "dem"],
        band_to_modality={"vv": "sar", "vh": "sar", "dem": "dem"},
    )

    BIOMASSTERS = MultiModalConfig(
        modalities={
            "s1": ModalityConfig(
                bands={
                    "VV_asc": BandConfig(
                        "vv_ascending",
                        ["co_pol_asc", "vv_asc"],
                        wavelength=0.056,
                        resolution=10,
                    ),
                    "VH_asc": BandConfig(
                        "vh_ascending",
                        ["cross_pol_asc", "vh_asc"],
                        wavelength=0.056,
                        resolution=10,
                    ),
                    "VV_desc": BandConfig(
                        "vv_descending",
                        ["co_pol_desc", "vv_desc"],
                        wavelength=0.056,
                        resolution=10,
                    ),
                    "VH_desc": BandConfig(
                        "vh_descending",
                        ["cross_pol_desc", "vv_desc"],
                        wavelength=0.056,
                        resolution=10,
                    ),
                },
                default_order=["VV_asc", "VH_asc", "VV_desc", "VH_desc"],
                native_resolution=10,
            ),
            "s2": ModalityConfig(
                bands={
                    k: v
                    for k, v in SensorBandRegistry.SENTINEL2.bands.items()
                    if k not in ["B01", "B09", "B10"]
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
                ],
                native_resolution=10,
            ),
        },
        default_order={
            "s1": {"VV_asc", "VH_asc", "VV_desc", "VH_desc"},
            "s2": {
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
            },
        },
        band_to_modality={
            "VV_asc": "s1",
            "VH_asc": "s1",
            "VV_desc": "s1",
            "VH_desc": "s1",
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
        },
    )

    TREESATAI = MultiModalConfig(
        modalities={
            "aerial": ModalityConfig(
                bands={
                    "nir": BandConfig("nir", ["IR", "NIR"], wavelength=None),
                    "g": BandConfig("g", ["G", "green", "GREEN"], wavelength=None),
                    "b": BandConfig("b", ["B", "blue", "BLUE"], wavelength=None),
                    "r": BandConfig("r", ["R", "red", "RED"], wavelength=None),
                },
                default_order=["nir", "g", "b", "r"],
            ),
            "s1": ModalityConfig(
                bands={
                    "vv": BandConfig("vv", ["VV", "vv"], wavelength=None),
                    "vh": BandConfig("vh", ["VH", "vh"], wavelength=None),
                    "vv/vh": BandConfig("vv/vh", ["VV/VH", "vv/vh"], wavelength=None),
                },
                default_order=["vv", "vh", "vv/vh"],
            ),
            "s2": ModalityConfig(
                bands={
                    band: config
                    for band, config in SensorBandRegistry.SENTINEL2.bands.items()
                    if band
                    in [
                        "B02",
                        "B03",
                        "B04",
                        "B08",
                        "B05",
                        "B06",
                        "B07",
                        "B8A",
                        "B11",
                        "B12",
                        "B01",
                        "B09",
                    ]
                },
                default_order=[
                    "B02",
                    "B03",
                    "B04",
                    "B08",
                    "B05",
                    "B06",
                    "B07",
                    "B8A",
                    "B11",
                    "B12",
                    "B01",
                    "B09",
                ],
            ),
        },
        default_order=[
            "nir",
            "g",
            "b",
            "r",
            "vv",
            "vh",
            "vv/vh",
            "B02",
            "B03",
            "B04",
            "B08",
            "B05",
            "B06",
            "B07",
            "B8A",
            "B11",
            "B12",
            "B01",
            "B09",
        ],
        band_to_modality={
            "nir": "aerial",
            "g": "aerial",
            "b": "aerial",
            "r": "aerial",
            "vv": "s1",
            "vh": "s1",
            "vv/vh": "s1",
            "B02": "s2",
            "B03": "s2",
            "B04": "s2",
            "B08": "s2",
            "B05": "s2",
            "B06": "s2",
            "B07": "s2",
            "B8A": "s2",
            "B11": "s2",
            "B12": "s2",
            "B01": "s2",
            "B09": "s2",
        },
    )

    MADOS = ModalityConfig(
        bands={
            band: config
            for band, config in SensorBandRegistry.SENTINEL2.bands.items()
            if band not in ["B09", "B10"]
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
            "B11",
            "B12",
        ],
        native_resolution=10,
    )

    # has rgbn planet imagery, 8 sentinel 1 bands and 12 sentinel 2 bands
    DYNAMICEARTHNET = MultiModalConfig(
        modalities={
            "planet": ModalityConfig(
                bands={
                    "r": BandConfig("red", ["r", "red", "RED"], wavelength=0.665),
                    "g": BandConfig("green", ["g", "green", "GREEN"], wavelength=0.560),
                    "b": BandConfig("blue", ["b", "blue", "BLUE"], wavelength=0.490),
                    "nir": BandConfig(
                        "nir", ["nir", "NIR", "near_infrared"], wavelength=0.842
                    ),
                },
                # the nativ order in the dataset is
                # https://github.com/aysim/dynnet/blob/1e7d90294b54f52744ae2b35db10b4d0a48d093d/data/utae_dynamicen.py#L105
                # order of bands is BGRN,
                default_order=["b", "g", "r", "nir"],
            ),
            # except B9
            "s2": ModalityConfig(
                bands={
                    band: config
                    for band, config in SensorBandRegistry.SENTINEL2.bands.items()
                    if band != "B09"
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
                    "B10",
                    "B11",
                    "B12",
                ],
            ),
            # TODO wait for inof
            # "s1": ModalityConfig(
            #     bands={
            #         "VV": BandConfig("vv", ["co_pol"], wavelength=0.056, resolution=10),
            #         "VH": BandConfig("vh", ["cross_pol"], wavelength=0.056, resolution=10),
            #     },
            #     default_order=["VV", "VH"],
            #     native_resolution=10,
            # ),
        },
        default_order=[
            "r",
            "g",
            "b",
            "nir",
            "B01",
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
        ],
        band_to_modality={
            "r": "planet",
            "g": "planet",
            "b": "planet",
            "nir": "planet",
            **{
                band: "s2"
                for band in [
                    "B01",
                    "B02",
                    "B03",
                    "B04",
                    "B05",
                    "B06",
                    "B07",
                    "B08",
                    "B8A",
                    "B10",
                    "B11",
                    "B12",
                ]
            },
        },
    )

    SEN4AGRINET = ModalityConfig(
        bands={**SensorBandRegistry.SENTINEL2.bands},
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
            "B10",
            "B11",
            "B12",
        ],
        native_resolution=10,
    )

    DOTAV2 = ModalityConfig(
        bands=SensorBandRegistry.RGB.bands, default_order=["r", "g", "b"]
    )

    MMFLOOD = MultiModalConfig(
        modalities={
            "s1": ModalityConfig(
                bands={
                    "vv": BandConfig("vv", ["VV"], wavelength=0.056),
                    "vh": BandConfig("vh", ["VH"], wavelength=0.056),
                },
                default_order=["vv", "vh"],
            ),
            "dem": ModalityConfig(
                bands={"dem": BandConfig("dem", ["elevation", "dem"], wavelength=None)},
                default_order=["dem"],
            ),
            "hydro": ModalityConfig(
                bands={
                    "hydro": BandConfig(
                        "hydro", ["hydro_layer", "HYDRO"], wavelength=None
                    )
                },
                default_order=["hydro"],
            ),
        },
        default_order=["vv", "vh", "dem", "hydro"],
        band_to_modality={"vv": "s1", "vh": "s1", "dem": "dem", "hydro": "hydro"},
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
