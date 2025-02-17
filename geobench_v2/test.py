from datamodules import GeoBenchFieldsOfTheWorldDataModule


dm = GeoBenchFieldsOfTheWorldDataModule(
    img_size=256,
    batch_size=32,
    eval_batch_size=64,
    num_workers=0,
    pin_memory=False,
    root="/mnt/rg_climate_benchmark/data/datasets_segmentation/FieldsOfTheWorld",
)

dm.setup("fit")

out = next(iter(dm.train_dataloader()))

import pdb

pdb.set_trace()

print(0)
