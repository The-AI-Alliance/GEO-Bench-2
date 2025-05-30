
from collections.abc import Callable
from typing import Any, ClassVar

from torch import Tensor
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import patches
from functools import partial

from torchgeo.datamodules import NonGeoDataModule

import albumentations as A
from albumentations.pytorch import transforms as T
import torchvision.transforms as orig_transforms

from torch.utils.data import DataLoader
from collections.abc import Sequence
import torch
from torch import nn
import numpy as np
from nzcattle_dataset import GeoBenchNZCattleObjectDetection


def collate_fn_detection(batch):
    new_batch = {
        "image": [item["image"] for item in batch],
        "boxes": [item["boxes"] for item in batch],
        "labels": [item["labels"] for item in batch],
    }

    return new_batch


def get_transform(train, image_size=512):
    transforms = []
    transforms.append(A.Resize(height=image_size, width=image_size))
    if train:
        transforms.append(A.D4(p=1))
    transforms.append(T.ToTensorV2())
    return A.Compose(transforms, bbox_params=A.BboxParams(format="pascal_voc", label_fields=['labels']), is_check_shapes=False)


def apply_transforms(sample, transforms):

    sample['image'] = torch.stack(tuple(sample["image"]))
    sample['image'] = sample['image'].permute(1, 2, 0) if len(sample['image'].shape) == 3 else sample['image'].permute(0, 2, 3, 1)
    sample['image'] = np.array(sample['image'].cpu())
    sample["boxes"] = np.array(sample["boxes"].cpu())
    sample["labels"] = np.array(sample["labels"].cpu())
    transformed = transforms(image=sample['image'],
                             bboxes=sample["boxes"],
                             labels=sample["labels"])
    transformed['boxes'] = torch.tensor(transformed['bboxes'], dtype=torch.float32)
    transformed['labels'] = torch.tensor(transformed['labels'], dtype=torch.int64)
    del transformed['bboxes']

    return transformed


class Normalize(Callable):
    def __init__(self, means, stds, max_pixel_value=None):
        super().__init__()
        self.means = means
        self.stds = stds
        self.max_pixel_value = max_pixel_value

    def __call__(self, batch):

        batch['image']=torch.stack(tuple(batch["image"]))
        image = batch["image"]/self.max_pixel_value if self.max_pixel_value is not None else batch["image"]
        if len(image.shape) == 5:
            means = torch.tensor(self.means, device=image.device).view(1, -1, 1, 1, 1)
            stds = torch.tensor(self.stds, device=image.device).view(1, -1, 1, 1, 1)
        elif len(image.shape) == 4:
            means = torch.tensor(self.means, device=image.device).view(1, -1, 1, 1)
            stds = torch.tensor(self.stds, device=image.device).view(1, -1, 1, 1)
        else:
            msg = f"Expected batch to have 5 or 4 dimensions, but got {len(image.shape)}"
            raise Exception(msg)
        batch["image"] = (image - means) / stds
        # pdb.set_trace()
        return batch


class IdentityTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class GeoBenchNZCattleDataModule(NonGeoDataModule):

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        bands: Sequence[str] = ("RED", "GREEN", "BLUE"),
        transform: A.Compose | None = None,
        partition: str = "default",
        use_metadata: bool = False,
        download: bool = False,
        checksum: bool = False,
        batch_size: int = 4,
        num_workers: int = 0,
        image_size=512,
        collate_fn = None,
        *args,
        **kwargs):

        super().__init__(GeoBenchNZCattleDataModule,
                         data_root = data_root,
                         split = split,
                         bands = bands,
                         transform = transform,
                         partition = partition,
                         use_metadata = use_metadata,
                         download = download,
                         checksum = checksum,
                         batch_size = batch_size,
                         num_workers = num_workers,
                         image_size=image_size,
                         collate_fn = collate_fn,
                         *args,
                         **kwargs)

        self.train_transform = partial(apply_transforms,transforms=get_transform(True, image_size))
        self.val_transform = partial(apply_transforms,transforms=get_transform(False, image_size))
        self.test_transform = partial(apply_transforms,transforms=get_transform(False, image_size))

        self.aug = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), max_pixel_value=255)
        self.bands = bands
        self.partition = partition
        self.use_metadata = use_metadata
        self.data_root = data_root
        self.split = split
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn_detection if collate_fn is None else collate_fn
        self.download = download
        self.checksum = checksum

    def setup(self, stage: str) -> None:

        if stage in ["fit"]:
            self.train_dataset = GeoBenchNZCattleObjectDetection(
                data_root = self.data_root,
                split = self.split,
                bands = self.bands,
                transform = self.train_transform,
                partition = self.partition,
                use_metadata = self.use_metadata,
                download = self.download
            )            
        if stage in ["fit", "validate"]:
            self.val_dataset = GeoBenchNZCattleObjectDetection(
                data_root = self.data_root,
                split = self.split,
                bands = self.bands,
                transform = self.val_transform,
                partition = self.partition,
                use_metadata = self.use_metadata,
                download = self.download
            )       
        if stage in ["test"]:
            self.test_dataset = GeoBenchNZCattleObjectDetection(
                data_root = self.data_root,
                split = self.split,
                bands = self.bands,
                transform = self.test_transform,
                partition = self.partition,
                use_metadata = self.use_metadata,
                download = self.download
            )       

    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders.

        Args:
            split: Either 'train', 'val', 'test', or 'predict'.

        Returns:
            A collection of data loaders specifying samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset or sampler, or if the dataset or sampler has length 0.
        """
        dataset = self._valid_attribute(f"{split}_dataset", "dataset")
        batch_size = self.batch_size

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )


