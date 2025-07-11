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
from .bright import GeoBenchBRIGHTDataModule
from .caffe import GeoBenchCaFFeDataModule
from .cloudsen12 import GeoBenchCloudSen12DataModule
from .dotav2 import GeoBenchDOTAV2DataModule
from .dynamic_earthnet import GeoBenchDynamicEarthNetDataModule
from .everwatch import GeoBenchEverWatchDataModule
from .flair2 import GeoBenchFLAIR2DataModule
from .fotw import GeoBenchFieldsOfTheWorldDataModule
from .kuro_siwo import GeoBenchKuroSiwoDataModule
from .m4sar import GeoBenchM4SARDataModule
from .mmflood import GeoBenchMMFloodDataModule
from .pastis import GeoBenchPASTISDataModule
from .qfabric import GeoBenchQFabricDataModule
from .spacenet2 import GeoBenchSpaceNet2DataModule
from .spacenet6 import GeoBenchSpaceNet6DataModule
from .spacenet7 import GeoBenchSpaceNet7DataModule
from .spacenet8 import GeoBenchSpaceNet8DataModule
from .treesatai import GeoBenchTreeSatAIDataModule
from .wind_turbine import GeoBenchWindTurbineDataModule
from .nzcattle import GeoBenchNZCattleDataModule

__all__ = (
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
    "GeoBenchDataModuleGeoBenchFLOGADataModule",
    "GeoBenchKuroSiwoDataModule",
    "GeoBenchTreeSatAIDataModule",
    "GeoBenchBioMasstersDataModule",
    "GeoBenchDynamicEarthNetDataModule",
    "GeoBenchMMFloodDataModule",
    "GeoBenchM4SARDataModule",
    "GeoBenchBRIGHTDataModule",
    "GeoBenchQFabricDataModule",
    "GeoBenchWindTurbineDataModule",
    "GeoBenchNZCattleDataModule",
     "GeoBenchSubstationDataModule",
     "GeoBenchPASTISPanopticDataModule",
)
