# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""GeoBench Datasets."""

from .benv2 import GeoBenchBENV2
from .biomassters import GeoBenchBioMassters
from .burn_scars import GeoBenchBurnScars
from .caffe import GeoBenchCaFFe
from .cloudsen12 import GeoBenchCloudSen12
from .dynamic_earthnet import GeoBenchDynamicEarthNet
from .everwatch import GeoBenchEverWatch
from .flair2 import GeoBenchFLAIR2
from .forestnet import GeoBenchForestnet
from .fotw import GeoBenchFieldsOfTheWorld
from .kuro_siwo import GeoBenchKuroSiwo
from .nzcattle import GeoBenchNZCattle
from .pastis import GeoBenchPASTIS
from .so2sat import GeoBenchSo2Sat
from .spacenet2 import GeoBenchSpaceNet2
from .spacenet6 import GeoBenchSpaceNet6
from .spacenet7 import GeoBenchSpaceNet7
from .spacenet8 import GeoBenchSpaceNet8
from .substation import GeoBenchSubstation
from .treesatai import GeoBenchTreeSatAI

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
    "GeoBenchFLAIR2",
    "GeoBenchCloudSen12",
    "GeoBenchKuroSiwo",
    "GeoBenchTreeSatAI",
    "GeoBenchBioMassters",
    "GeoBenchDynamicEarthNet",
    "GeoBenchBurnScars",
    "GeoBenchNZCattle",
    "GeoBenchSubstation",
    "GeoBenchSo2Sat",
    "GeoBenchForestnet",
)
