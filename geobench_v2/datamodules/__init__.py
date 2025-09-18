# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""GeoBench DataModules."""

from .base import (
    GeoBenchClassificationDataModule,
    GeoBenchDataModule,
    GeoBenchObjectDetectionDataModule,
    GeoBenchSegmentationDataModule,
)
from .benv2 import GeoBenchBENV2DataModule
from .biomassters import GeoBenchBioMasstersDataModule
from .burn_scars import GeoBenchBurnScarsDataModule
from .caffe import GeoBenchCaFFeDataModule
from .cloudsen12 import GeoBenchCloudSen12DataModule
from .dotav2 import GeoBenchDOTAV2DataModule
from .dynamic_earthnet import GeoBenchDynamicEarthNetDataModule
from .everwatch import GeoBenchEverWatchDataModule
from .flair2 import GeoBenchFLAIR2DataModule
from .fotw import GeoBenchFieldsOfTheWorldDataModule
from .kuro_siwo import GeoBenchKuroSiwoDataModule
from .mmflood import GeoBenchMMFloodDataModule
from .nzcattle import GeoBenchNZCattleDataModule
from .pastis import GeoBenchPASTISDataModule
from .spacenet2 import GeoBenchSpaceNet2DataModule
from .spacenet6 import GeoBenchSpaceNet6DataModule
from .spacenet7 import GeoBenchSpaceNet7DataModule
from .spacenet8 import GeoBenchSpaceNet8DataModule
from .substation import GeoBenchSubstationDataModule
from .treesatai import GeoBenchTreeSatAIDataModule
from .wind_turbine import GeoBenchWindTurbineDataModule

__all__ = (
    "GeoBenchDataModule",
    "GeoBenchCaFFeDataModule",
    "GeoBenchFieldsOfTheWorldDataModule",
    "GeoBenchPASTISDataModule",
    "GeoBenchSpaceNet2DataModule",
    "GeoBenchSpaceNet6DataModule",
    "GeoBenchSpaceNet7DataModule",
    "GeoBenchSpaceNet8DataModule",
    "GeoBenchBENV2DataModule",
    "GeoBenchEverWatchDataModule",
    "GeoBenchDOTAV2DataModule",
    "GeoBenchFLAIR2DataModule",
    "GeoBenchCloudSen12DataModule",
    "GeoBenchClassificationDataModule",
    "GeoBenchSegmentationDataModule",
    "GeoBenchObjectDetectionDataModule",
    "GeoBenchKuroSiwoDataModule",
    "GeoBenchTreeSatAIDataModule",
    "GeoBenchBioMasstersDataModule",
    "GeoBenchDynamicEarthNetDataModule",
    "GeoBenchMMFloodDataModule",
    "GeoBenchWindTurbineDataModule",
    "GeoBenchNZCattleDataModule",
    "GeoBenchSubstationDataModule",
    "GeoBenchBurnScarsDataModule",
)
