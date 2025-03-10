# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""GeoBench Datasets."""

from .caffe import GeoBenchCaFFe
from .fotw import GeoBenchFieldsOfTheWorld
from .pastis import GeoBenchPASTIS
from .resisc45 import GeoBenchRESISC45
from .spacenet6 import GeoBenchSpaceNet6
from .benv2 import GeoBenchBENV2
from .everwatch import GeoBenchEverWatch
from .flair2 import GeoBenchFLAIR2

__all__ = (
    "GeoBenchCaFFe",
    "GeoBenchFieldsOfTheWorld",
    "GeoBenchPASTIS",
    "GeoBenchRESISC45",
    "GeoBenchSpaceNet6",
    "GeoBenchBENV2",
    "GeoBenchEverWatch",
    "GeoBenchFlair2",
)
