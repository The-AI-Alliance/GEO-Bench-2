# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""CaFFe DataMdule."""

from collections.abc import Callable
from typing import Any
import kornia.augmentation as K

from geobench_v2.datasets.caffe import GeoBenchCaFFe

from .base import GeoBenchSegmentationDataModule


class GeoBenchCaFFeDataModule(GeoBenchSegmentationDataModule):
    """GeoBench CaFFe Data Module."""

    # https://github.com/microsoft/torchgeo/blob/68e0cfebcd18edb6605008eeeaba96388e63eca7/torchgeo/datamodules/caffe.py#L22
    band_means = {"gray": 0.5517}
    band_stds = {"gray": 11.8478}

    def __init__(
        self,
        img_size: int,
        batch_size: int = 32,
        eval_batch_size: int = 64,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        train_augmentations: nn.Module | None = None,
        eval_augmentations: nn.Module | None = None,
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize GeoBench CaFFe dataset module.

        Args:
            img_size: Image size
            batch_size: Batch size during
            eval_batch_size: Evaluation batch size
            num_workers: Number of workers
            collate_fn: Collate function
            train_augmentations: Transforms/Augmentations to apply during training, they will be applied
                at the sample level and should include normalization. See :method:`define_augmentations`
                for the default transformation.
            eval_augmentations: Transforms/Augmentations to apply during evaluation, they will be applied
                at the sample level and should include normalization. See :method:`define_augmentations`
                for the default transformation.
            pin_memory: Pin memory
            **kwargs: Additional keyword arguments for the dataset class
        """
        super().__init__(
            dataset_class=GeoBenchCaFFe,
            img_size=img_size,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            train_augmentations=train_augmentations,
            eval_augmentations=eval_augmentations,
            pin_memory=pin_memory,
            **kwargs,
        )

    def setup(self, stage: str | None = None) -> None:
        """Setup data for train, val, test.

        Args:
            stage: One of 'fit', 'validate', 'test', or 'predict'.
        """
        norm_transform = K.AugmentationSequential(
            K.Normalize(self.mean, self.std, keepdim=True),
            data_keys=["image", "mask"],
        )
        self.train_dataset = self.dataset_class(split="train", transforms=norm_transform, **self.kwargs)
        self.val_dataset = self.dataset_class(split="val", transforms=norm_transform, **self.kwargs)
        self.test_dataset = self.dataset_class(split="test", transforms=norm_transform, **self.kwargs)

    def visualize_geolocation_distribution(self) -> None:
        """Visualize the geolocation distribution of the dataset."""
        pass
