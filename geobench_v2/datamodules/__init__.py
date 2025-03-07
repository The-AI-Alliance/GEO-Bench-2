# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""GeoBench DataModules."""

from .caffe import GeoBenchCaFFeDataModule
from .fotw import GeoBenchFieldsOfTheWorldDataModule
from .pastis import GeoBenchPASTISDataModule
from .resisc45 import GeoBenchRESISC45DataModule
from .spacenet6 import GeoBenchSpaceNet6DataModule
from .benv2 import GeoBenchBENV2DataModule
from .everwatch import GeoBenchEverWatchDataModule
from .flair2 import GeoBenchFLAIR2DataModule

__all__ = (
    "GeoBenchCaFFeDataModule",
    "GeoBenchFieldsOfTheWorldDataModule",
    "GeoBenchPASTISDataModule",
    "GeoBenchRESISC45DataModule",
    "GeoBenchSpaceNet6DataModule",
    "GeoBenchBENV2DataModule",
    "GeoBenchEverWatchDataModule",
    "GeoBenchFLAIR2DataModule",
)
