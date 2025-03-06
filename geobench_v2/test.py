from datamodules import (
    GeoBenchFieldsOfTheWorldDataModule,
    GeoBenchBENV2DataModule,
    # GeoBenchCaFFeDataModule,
    # GeoBenchRESISC45DataModule,
    # GeoBenchSpaceNet6DataModule,
    # GeoBenchPASTISDataModule,
    # GeoBenchEverWatchDataModule
)


def check_batch_for_each_split(dm):
    dm.setup("fit")
    train_batch = next(iter(dm.train_dataloader()))
    try:
        assert train_batch["image"].shape[0] == dm.batch_size
        assert train_batch["image"].shape[1] == len(dm.band_order)
        # assert train_batch["image"].shape[2] == dm.img_size
    except AssertionError:
        import pdb; pdb.set_trace()
        print(0)

    # val_batch = next(iter(dm.val_dataloader()))
    # assert val_batch["image"].shape[0] == dm.eval_batch_size
    # assert val_batch["image"].shape[1] == len(dm.band_order)
    # # assert val_batch["image"].shape[2] == dm.img_size

    # test_batch = next(iter(dm.test_dataloader()))
    # assert test_batch["image"].shape[0] == dm.eval_batch_size
    # assert test_batch["image"].shape[1] == len(dm.band_order)
    # assert test_batch["image"].shape[2] == dm.img_size


dm = GeoBenchFieldsOfTheWorldDataModule(
    img_size=256,
    batch_size=32,
    eval_batch_size=64,
    num_workers=0,
    pin_memory=False,
    band_order=["r", 1.4, "green", "green", "blue", "nir"],
    root="/mnt/rg_climate_benchmark/data/datasets_segmentation/FieldsOfTheWorld",
)

check_batch_for_each_split(dm)

dm = GeoBenchBENV2DataModule(
    img_size=256,
    batch_size=32,
    eval_batch_size=64,
    num_workers=0,
    pin_memory=False,
    band_order=["VV", "B01", "B02", 1.5, "B03"],
    root="/mnt/rg_climate_benchmark/data/datasets_classification/benv2",
)

check_batch_for_each_split(dm)

# dm = GeoBenchCaFFeDataModule(
#     img_size=256,
#     batch_size=32,
#     eval_batch_size=64,
#     num_workers=0,
#     band_order=["gray", 0.0, "gray"],
#     pin_memory=False,
#     root="/mnt/rg_climate_benchmark/data/datasets_segmentation/Caffe",
# )

# check_batch_for_each_split(dm)

# dm = GeoBenchRESISC45DataModule(
#     img_size=256,
#     batch_size=32,
#     eval_batch_size=64,
#     num_workers=0,
#     pin_memory=False,
#     band_order=["red", "green", "blue", 0, "green"],
#     root="/mnt/rg_climate_benchmark/data/datasets_classification/resisc45",
# )

# check_batch_for_each_split(dm)

# dm = GeoBenchPASTISDataModule(
#     img_size=256,
#     batch_size=32,
#     eval_batch_size=64,
#     num_workers=0,
#     pin_memory=False,
#     band_order=["r", "g", "b", "nir", "nir"],
#     root="/mnt/rg_climate_benchmark/data/datasets_segmentation/pastis_r",
# )

# check_batch_for_each_split(dm)

# dm = GeoBenchEverWatchDataModule(
#     img_size=256,
#     batch_size=32,
#     eval_batch_size=64,
#     num_workers=0,
#     pin_memory=False,
#     band_order=["red", "green", "blue", 0, "green"],
#     root="/mnt/rg_climate_benchmark/data/datasets_object_detection/everwatch",
# )

# check_batch_for_each_split(dm)
