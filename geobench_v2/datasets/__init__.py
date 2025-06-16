# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""GeoBench Datasets."""

from .benv2 import GeoBenchBENV2
from .biomassters import GeoBenchBioMassters
from .bright import GeoBenchBRIGHT
from .caffe import GeoBenchCaFFe
from .cloudsen12 import GeoBenchCloudSen12
from .dotav2 import GeoBenchDOTAV2
from .dynamic_earthnet import GeoBenchDynamicEarthNet
from .everwatch import GeoBenchEverWatch
from .flair2 import GeoBenchFLAIR2
from .fotw import GeoBenchFieldsOfTheWorld
from .kuro_siwo import GeoBenchKuroSiwo
from .m4sar import GeoBenchM4SAR
from .mmflood import GeoBenchMMFlood
from .pastis import GeoBenchPASTIS
from .qfabric import GeoBenchQFabric
from .spacenet2 import GeoBenchSpaceNet2
from .spacenet6 import GeoBenchSpaceNet6
from .spacenet7 import GeoBenchSpaceNet7
from .spacenet8 import GeoBenchSpaceNet8
from .treesatai import GeoBenchTreeSatAI
from .wind_turbine import GeoBenchWindTurbine
from .burn_scars import GeoBenchBurnScars

__all__ = (
    "GeoBenchCaFFe",
    "GeoBenchFieldsOfTheWorld",
    "GeoBenchPASTIS",
    "GeoBenchSpaceNet2",
    "GeoBenchSpaceNet6",
    "GeoBenchSpaceNet7",
    "GeoBenchSpaceNet8",
    "GeoBenchBENV2",
    "GeoBenchEverWatch",
    "GeoBenchDOTAV2",
    "GeoBenchFLAIR2",
    "GeoBenchCloudSen12",
    "GeoBenchKuroSiwo",
    "GeoBenchTreeSatAI",
    "GeoBenchBioMassters",
    "GeoBenchM4SAR",
    "GeoBenchDynamicEarthNet",
    "GeoBenchMMFlood",
    "GeoBenchBRIGHT",
    "GeoBenchWindTurbine",
    "GeoBenchQFabric",
    "GeoBenchBurnScars"
)
