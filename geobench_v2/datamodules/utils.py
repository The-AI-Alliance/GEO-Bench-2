# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""DataModule utils."""

import gc

import einops
import kornia.augmentation as K
import torch
import torch.nn as nn


class MultiModalSegmentationAugmentation(nn.Module):
    """Apply augmentations to multi-modal segmentation datasets."""

    def __init__(self, transforms) -> None:
        """Initialize the TimeSeriesResize module.

        Args:
            transforms: transfroms to be applied
        """
        super().__init__()
        self.transforms = transforms

    @torch.no_grad()  # disable gradients for efficiency
    def forward(self, batch) -> dict:
        """Forward method to apply augmentations to multi-modal segmentation datasets."""
        # unpack dict of input images
        nested_keys = {}
        dims = []
        original_keys = list(batch.keys())
        for key in original_keys:
            if "image" in key:
                if isinstance(batch[key], dict):
                    subkey_names = []
                    subkey_data = []
                    for subkey in batch[key]:
                        dims.append(batch[key][subkey].dim())
                        subkey_names.append(subkey)
                        subkey_data.append(batch[key][subkey])
                    # concat subkeys on C dimension
                    batch[key] = torch.concat(subkey_data, dim=1)
                    nested_keys[key] = {}
                    nested_keys[key]["subkey_names"] = subkey_names
                    nested_keys[key]["channel_length"] = [
                        t.shape[-3] if t.dim() == 4 else t.shape[-4]
                        for t in subkey_data
                    ]
                    nested_keys[key]["channel_start"] = torch.tensor(
                        [0] + nested_keys[key]["channel_length"]
                    ).cumsum(0)
                    del subkey_data
                    del subkey_names
                    gc.collect()
                else:
                    dims.append(batch[key].dim())
        if len(set(dims)) != 1:
            raise ValueError("Inputs have different dimensions")

        if dims[0] == 5:  # BxCxTxHxW
            # if image keys have 5 dimensions, expand mask dim
            if len(batch["mask"].shape) != 3:
                raise ValueError("Mask does not contain the expected dimensions")
            for key in batch:
                if ("image" in key) and (len(batch[key].shape) == 5):
                    B, C, T, H, W = batch[key].shape
                    batch["mask"] = einops.repeat(
                        batch["mask"], "b h w -> b C T h w", C=C, T=T
                    )
                    break
            if len(batch["mask"].shape) != 5:
                raise ValueError("Mask does not contain the expected dimensions")

        # TODO: unpack metadata as well
        batch_in = {}
        batch_in["image"] = batch["image"]
        batch_in["mask"] = batch["mask"]
        del batch
        gc.collect()

        batch_in = self.transforms(batch_in)

        # undo expansion
        if dims[0] == 5:  # BxCxTxHxW
            batch_in["mask"] = batch_in["mask"][:, 0, 0, :, :]

        # repack dict of input images
        for key in nested_keys:
            subkeys = nested_keys[key]["subkey_names"]
            channel_start = nested_keys[key]["channel_start"]
            channel_length = nested_keys[key]["channel_length"]
            if dims[0] == 5:
                batch_in[key] = {
                    mod: batch_in[key][..., start : start + length, :, :, :]
                    for mod, start, length in zip(
                        subkeys, channel_start, channel_length
                    )
                }
            else:
                batch_in[key] = {
                    mod: batch_in[key][..., start : start + length, :, :]
                    for mod, start, length in zip(
                        subkeys, channel_start, channel_length
                    )
                }

        # force contiguous
        for key in batch_in:
            if isinstance(batch_in[key], dict):
                for subkey in batch_in[key]:
                    batch_in[key][subkey] = batch_in[key][subkey].contiguous()
            else:
                batch_in[key] = batch_in[key].contiguous()
        return batch_in


class MultiModalClassificationAugmentation(nn.Module):
    """Apply augmentations to multi-modal classification datasets."""

    def __init__(self, transforms) -> None:
        """Initialize the TimeSeriesResize module.

        Args:
            transforms: transfroms to be applied
        """
        super().__init__()
        self.transforms = transforms

    @torch.no_grad()  # disable gradients for efficiency
    def forward(self, batch) -> dict:
        """Forward method to apply augmentations to multi-temporal classification dataset."""
        # unpack dict of input images
        nested_keys = {}
        original_keys = list(batch.keys())
        for key in original_keys:
            if isinstance(batch[key], dict):
                subkeys = []
                for subkey in batch[key]:
                    batch[f"{key}_{subkey}"] = batch[key][subkey]
                    subkeys.append(subkey)
                nested_keys[key] = subkeys
                del batch[key]
            else:
                batch[key] = batch[key]
        batch_out = self.transforms(batch)
        # repack dict of input images
        for key in nested_keys:
            subkeys = nested_keys[key]
            batch_out[key] = {}
            for subkey in subkeys:
                batch_out[key][subkey] = batch_out[f"{key}_{subkey}"]
                del batch_out[f"{key}_{subkey}"]
        return batch_out


class MultiTemporalSegmentationAugmentation(nn.Module):
    """Apply augmentations to multi-temporal segmentation datasets."""

    def __init__(self, transforms) -> None:
        """Initialize the TimeSeriesResize module.

        Args:
            transforms: transfroms to be applied
        """
        super().__init__()
        self.transforms = transforms

    @torch.no_grad()  # disable gradients for efficiency
    def forward(self, batch) -> dict:
        """Forward method to apply augmentations to multi-temporal segmentation datasets."""
        if len(batch["mask"].shape) != 3:
            raise ValueError("Mask does not contain the expected dimensions")
        for key in batch:
            if ("image" in key) and (len(batch[key].shape) == 5):
                B, C, T, H, W = batch[key].shape
                batch["mask"] = einops.repeat(
                    batch["mask"], "b h w -> b C T h w", C=C, T=T
                )
                break
        if len(batch["mask"].shape) != 5:
            raise ValueError("Mask does not contain the expected dimensions")
        batch_out = self.transforms(batch)  # for image, mask == BxCXTxHxW

        batch_out["mask"] = batch_out["mask"][:, 0, 0, :, :]
        for key in batch_out:
            batch_out[key] = batch_out[key].contiguous()
        return batch_out


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
