from datamodules import GeoBenchFieldsOfTheWorldDataModule, GeoBenchBENV2DataModule


# dm = GeoBenchFieldsOfTheWorldDataModule(
#     img_size=256,
#     batch_size=32,
#     eval_batch_size=64,
#     num_workers=0,
#     pin_memory=False,
#     band_order=["r", 1.4, "green", "green", "blue", "n"],
#     root="/mnt/rg_climate_benchmark/data/datasets_segmentation/FieldsOfTheWorld",
# )

dm = GeoBenchBENV2DataModule(
    img_size=256,
    batch_size=32,
    eval_batch_size=64,
    num_workers=0,
    pin_memory=False,
    root="/mnt/rg_climate_benchmark/data/datasets_classification/benv2",
)

dm.setup("fit")

train_dataset = dm.train_dataset

sample = train_dataset[0]

# out = next(iter(dm.train_dataloader()))

import pdb

pdb.set_trace()

print(0)
sa
