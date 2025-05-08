# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""DataModule utils."""

import kornia.augmentation as K
import torch
import torch.nn as nn


class TimeSeriesResize(nn.Module):
    """Resize a dictionary of both time-series and single time step images."""

    def __init__(self, img_size: int):
        """Initialize the TimeSeriesResize module.

        Args:
            img_size (int): The target image size for resizing.
        """
        super().__init__()
        self.img_size = img_size

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Resize the images in the batch.

        Args:
            batch (dict[str, torch.Tensor]): The input batch containing images.

        Returns:
            dict[str, torch.Tensor]: The resized images in the batch.
        """
        for key in batch.keys():
            if key.startswith("image_"):
                if len(batch[key].shape) == 4:  # Time series
                    batch[key] = K.Resize((self.img_size, self.img_size), keepdim=True)(
                        batch[key]
                    )
                elif len(batch[key].shape) == 3:  # Single time step
                    batch[key] = K.Resize((self.img_size, self.img_size), keepdim=True)(
                        batch[key]
                    )
                else:
                    raise ValueError(f"Unsupported shape for {key}: {batch[key].shape}")
            elif key.startswith("mask"):
                batch[key] = K.Resize((self.img_size, self.img_size), keepdim=True)(
                    batch[key].float()
                ).long()
        return batch
