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
from .floga import GeoBenchFLOGA
from .fotw import GeoBenchFieldsOfTheWorld
from .kuro_siwo import GeoBenchKuroSiwo
from .mados import GeoBenchMADOS
from .mmflood import GeoBenchMMFlood
from .pastis import GeoBenchPASTIS
from .qfabric import GeoBenchQFabric
from .resisc45 import GeoBenchRESISC45
from .sen4agrinet import GeoBenchSen4AgriNet
from .spacenet2 import GeoBenchSpaceNet2
from .spacenet6 import GeoBenchSpaceNet6
from .spacenet7 import GeoBenchSpaceNet7
from .spacenet8 import GeoBenchSpaceNet8
from .treesatai import GeoBenchTreeSatAI
from .wind_turbine import GeoBenchWindTurbine

__all__ = (
    "GeoBenchCaFFe",
    "GeoBenchFieldsOfTheWorld",
    "GeoBenchPASTIS",
    "GeoBenchRESISC45",
    "GeoBenchSpaceNet2",
    "GeoBenchSpaceNet6",
    "GeoBenchSpaceNet7",
    "GeoBenchSpaceNet8",
    "GeoBenchBENV2",
    "GeoBenchEverWatch",
    "GeoBenchDOTAV2",
    "GeoBenchFlair2",
    "GeoBenchCloudSen12",
    "GeoBenchFLOGA",
    "GeoBenchKuroSiwo",
    "GeoBenchTreeSatAI",
    "GeoBenchBioMassters",
    "GeoBenchMADOS",
    "GeoBenchDynamicEarthNet",
    "GeoBenchSen4AgriNet",
    "GeoBenchMMFlood",
    "GeoBenchBRIGHT",
    "GeoBenchWindTurbine",
    "GeoBenchQFabric",
)
