"""Unit tests for GeoBench datamodules."""

from pathlib import Path

import pytest
import torch

from geobench_v2.datamodules import (
    GeoBenchBENV2DataModule,
    GeoBenchBioMasstersDataModule,
    GeoBenchCaFFeDataModule,
    GeoBenchCloudSen12DataModule,
    GeoBenchDataModule,
    GeoBenchFieldsOfTheWorldDataModule,
    GeoBenchFLAIR2DataModule,
    GeoBenchKuroSiwoDataModule,
    GeoBenchMADOSDataModule,
    GeoBenchMMFloodDataModule,
    GeoBenchPASTISDataModule,
    GeoBenchSpaceNet2DataModule,
    GeoBenchSpaceNet6DataModule,
    GeoBenchSpaceNet7DataModule,
    GeoBenchSpaceNet8DataModule,
    GeoBenchTreeSatAIDataModule,
)

ALL_DATAMODULES = [
    GeoBenchFieldsOfTheWorldDataModule,
    GeoBenchSpaceNet2DataModule,
    GeoBenchSpaceNet6DataModule,
    GeoBenchSpaceNet7DataModule,
    GeoBenchSpaceNet8DataModule,
    GeoBenchBENV2DataModule,
    GeoBenchFLAIR2DataModule,
    GeoBenchCloudSen12DataModule,
    GeoBenchTreeSatAIDataModule,
    GeoBenchMADOSDataModule,
    GeoBenchBioMasstersDataModule,
    GeoBenchMMFloodDataModule,
    GeoBenchKuroSiwoDataModule,
    GeoBenchPASTISDataModule,
    # WIP
    # GeoBenchCaFFeDataModule,
    # GeoBenchBRIGHTDataModule,
    # GeoBenchDynamicEarthNetDataModule,
    # GeoBenchWindTurbineDataModule,
    # GeoBenchEverWatchDataModule,
    # GeoBenchDOTAV2DataModule,
    # GeoBenchSen4AgriNetDataModule,
    # GeoBenchFLOGADataModule,
    # GeoBenchQFabricDataModule,
    # GeoBenchRESISC45DataModule,
]

dataset_path_mapping = {
    # "GeoBenchCaFFeDataModule": "caffe",
    "GeoBenchFieldsOfTheWorldDataModule": "fotw",
    "GeoBenchPASTISDataModule": "pastis",
    "GeoBenchSpaceNet2DataModule": "spacenet2",
    "GeoBenchSpaceNet6DataModule": "spacenet6",
    "GeoBenchSpaceNet7DataModule": "spacenet7",
    "GeoBenchSpaceNet8DataModule": "spacenet8",
    "GeoBenchBENV2DataModule": "benv2",
    "GeoBenchEverWatchDataModule": "everwatch",
    "GeoBenchDOTAV2DataModule": "dota_v2",
    "GeoBenchFLAIR2DataModule": "flair2",
    "GeoBenchCloudSen12DataModule": "cloudsen12",
    "GeoBenchKuroSiwoDataModule": "kuro_siwo",
    "GeoBenchTreeSatAIDataModule": "treesatai",
    "GeoBenchMADOSDataModule": "mados",
    "GeoBenchBioMasstersDataModule": "biomassters",
    "GeoBenchDynamicEarthNetDataModule": "dynamic_earthnet",
    "GeoBenchSen4AgriNetDataModule": "sen4agri",
    "GeoBenchMMFloodDataModule": "mmflood",
    "GeoBenchBRIGHTDataModule": "bright",
    "GeoBenchWindTurbineDataModule": "wind_turbine",
    # "GeoBenchRESISC45DataModule": "resisc45",
    # "GeoBenchFLOGADataModule": "floga",
    # "GeoBenchQFabricDataModule": "qfabric",
}


class TestDataModules:
    """Test class for GeoBench datamodules."""

    @pytest.fixture(scope="session")
    def data_root(self) -> Path:
        """Get the root directory for test datasets."""
        return Path("/mnt/rg_climate_benchmark/data/final_geobenchV2")

    def get_dataset_path(
        self, datamodule_class: type[GeoBenchDataModule], data_root: Path
    ) -> Path:
        """Get path for specific dataset."""
        class_name = datamodule_class.__name__
        return (
            data_root
            # / class_name.replace("GeoBench", "").replace("DataModule", "").lower()
            / dataset_path_mapping[class_name]
        )

    def setup_datamodule(
        self, datamodule_class: type[GeoBenchDataModule], dataset_path: Path
    ):
        """Initialize a datamodule with test params."""
        dm = datamodule_class(
            batch_size=4, eval_batch_size=2, num_workers=0, root=str(dataset_path)
        )
        dm.setup("fit")
        dm.setup("test")
        return dm

    @pytest.mark.parametrize("datamodule_class", ALL_DATAMODULES)
    def test_train_dataloader(self, datamodule_class, data_root):
        """Test train dataloader has data and can load a batch."""
        dataset_path = self.get_dataset_path(datamodule_class, data_root)

        datamodule = self.setup_datamodule(datamodule_class, dataset_path)

        train_loader = datamodule.train_dataloader()
        assert len(train_loader) > 0, "Train dataloader is empty"

        batch = next(iter(train_loader))
        assert batch is not None, "Failed to load batch from train dataloader"

        if isinstance(batch, dict):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    assert value.shape[0] > 0, (
                        f"Batch dimension for '{key}' is incorrect: {value.shape}"
                    )

    @pytest.mark.parametrize("datamodule_class", ALL_DATAMODULES)
    def test_val_dataloader(self, datamodule_class, data_root):
        """Test validation dataloader has data and can load a batch."""
        dataset_path = self.get_dataset_path(datamodule_class, data_root)

        datamodule = self.setup_datamodule(datamodule_class, dataset_path)

        val_loader = datamodule.val_dataloader()
        assert len(val_loader) > 0, "Validation dataloader is empty"

        batch = next(iter(val_loader))
        assert batch is not None, "Failed to load batch from validation dataloader"

    @pytest.mark.parametrize("datamodule_class", ALL_DATAMODULES)
    def test_test_dataloader(self, datamodule_class, data_root):
        """Test test dataloader has data and can load a batch."""
        dataset_path = self.get_dataset_path(datamodule_class, data_root)

        datamodule = self.setup_datamodule(datamodule_class, dataset_path)

        test_loader = datamodule.test_dataloader()
        assert len(test_loader) > 0, "Test dataloader is empty"

        batch = next(iter(test_loader))
        assert batch is not None, "Failed to load batch from test dataloader"
