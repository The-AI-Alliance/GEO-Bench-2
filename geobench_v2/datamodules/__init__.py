# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""GeoBench DataModules."""

from .caffe import GeoBenchCaFFeDataModule
from .fotw import GeoBenchFieldsOfTheWorldDataModule
from .pastis import GeoBenchPASTISDataModule
from .resisc45 import GeoBenchRESISC45DataModule
from .spacenet6 import GeoBenchSpaceNet6DataModule
from .spacenet8 import GeoBenchSpaceNet8DataModule
from .benv2 import GeoBenchBENV2DataModule
from .everwatch import GeoBenchEverWatchDataModule
from .flair2 import GeoBenchFLAIR2DataModule
from .cloudsen12 import GeoBenchCloudSen12DataModule
from .floga import GeoBenchFLOGADataModule
from .kuro_siwo import GeoBenchKuroSiwoDataModule
from .treesatai import GeoBenchTreeSatAIDataModule
from .mados import GeoBenchMADOSDataModule
from .biomassters import GeoBenchBioMasstersDataModule
from .dynamic_earthnet import GeoBenchDynamicEarthNetDataModule


from .base import (GeoBenchClassificationDataModule, GeoBenchSegmentationDataModule, 
                    GeoBenchObjectDetectionDataModule, GeoBenchDataModule)

__all__ = (
    "GeoBenchCaFFeDataModule",
    "GeoBenchFieldsOfTheWorldDataModule",
    "GeoBenchPASTISDataModule",
    "GeoBenchRESISC45DataModule",
    "GeoBenchSpaceNet6DataModule",
    "GeoBenchSpaceNet8DataModule",
    "GeoBenchBENV2DataModule",
    "GeoBenchEverWatchDataModule",
    "GeoBenchFLAIR2DataModule",
    "GeoBenchCloudSen12DataModule",
    "GeoBenchClassificationDataModule",
    "GeoBenchSegmentationDataModule",
    "GeoBenchObjectDetectionDataModule",
    "GeoBenchDataModule"
    "GeoBenchFLOGADataModule",
    "GeoBenchKuroSiwoDataModule",
    "GeoBenchTreeSatAIDataModule",
    "GeoBenchBioMasstersDataModule",
    "GeoBenchMADOSDataModule",
    "GeoBenchDynamicEarthNetDataModule",
)
