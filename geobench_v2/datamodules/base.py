# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Base DataModules."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, Literal

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import kornia.augmentation as K
import pandas as pd
import torch.nn as nn
from lightning import LightningDataModule
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from einops import rearrange
from torchgeo.datasets.utils import percentile_normalization

# TODO come up with an expected metadata file scheme
# with common names etc. so a standardization
# - datamodules have functions to create nice visualizations of data distribution etc
# - datasets return an id that can be used to link back to all metadata available
# - datasets return lat/lon, if available time, and wavelength information
# - show how to allow for more elaborate analysis of predictions etc.


class GeoBenchDataModule(LightningDataModule, ABC):
    """GeoBench DataModule."""

    has_extra_test_samples = False

    def __init__(
        self,
        dataset_class: Dataset,
        img_size: int,
        band_order: Sequence[float | str],
        batch_size: int = 32,
        eval_batch_size: int = 64,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        train_augmentations: Callable | None | str = "default",
        eval_augmentations: Callable | None | str = "default",
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize GeoBench DataModule.

        Args:
            dataset_class: Dataset class to use in the DataModule
            img_size: Desired image input size for the model
            band_order: band order of the image sample to be returned
            batch_size: Batch size during training
            eval_batch_size: Batch size during evaluation, can usually be larger than batch_size,
                to speed up evaluation.
            num_workers: Number of workers for dataloaders
            collate_fn: Collate function that can reformat samples to the needs of the model.
            train_augmentations: Transforms/Augmentations to apply during training, they will be applied
                at the sample level and should *not* include normalization, normalization happens on the dataset level for each
                sample, while geometric and color augmentations will be applied on a batch of data
            eval_augmentations: Transforms/Augmentations to apply during evaluation, they will be applied
                at the sample level and should *not* include normalization, normalization happens on the dataset level for each
                sample, while geometric and color augme]ntations will be applied on a batch of data
            pin_memory: whether to pin memory in dataloaders
            **kwargs: Additional keyword arguments passed to ``dataset_class``
        """
        super().__init__()
        if isinstance(train_augmentations, str):
            assert train_augmentations == "default", (
                "Please provide one of the follow for eval_augmentations: Callable or None or 'default'"
            )
        if isinstance(eval_augmentations, str):
            assert eval_augmentations == "default", (
                "Please provide one of the follow for eval_augmentations: Callable or None or 'default'"
            )

        self.dataset_class = dataset_class
        self.img_size = img_size
        self.band_order = band_order
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.kwargs = kwargs
        self.train_augmentations = train_augmentations
        self.eval_augmentations = eval_augmentations

        self.define_augmentations()

    def prepare_data(self) -> None:
        """Download and prepare data, only for distributed setup."""
        if self.kwargs.get("download", False):
            self.dataset_class(**self.kwargs)

    def setup(self, stage: str | None = None) -> None:
        """Setup data for train, val, test.

        Args:
            stage: One of 'fit', 'validate', 'test', or 'predict'.
        """
        self.train_transform, self.val_transform, self.test_transform = (
            self.setup_image_size_transforms()
        )

        if stage in ["fit"]:
            self.train_dataset = self.dataset_class(
                split="train",
                band_order=self.band_order,
                transforms=self.train_transform,
                **self.kwargs,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(
                split="validation",
                band_order=self.band_order,
                transforms=self.val_transform,
                **self.kwargs,
            )
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(
                split="test",
                band_order=self.band_order,
                transforms=self.test_transform,
                **self.kwargs,
            )

        self.dataset_band_config = self.train_dataset.dataset_band_config

        self.data_normalizer = self.train_dataset.data_normalizer

        self.band_order = self.train_dataset.band_order

        if hasattr(self.train_dataset, "num_classes"):
            self.num_classes = self.train_dataset.num_classes
            self.class_names = self.train_dataset.classes

    @abstractmethod
    def setup_image_size_transforms(self) -> tuple[nn.Module, nn.Module, nn.Module]:
        """Setup image resizing transforms for train, val, and test.

        Image resizing and normalization happens on the dataset level on individual data samples.
        """
        pass

    @abstractmethod
    def load_metadata(self) -> pd.DataFrame:
        """Load metadata file.

        Returns:
            pandas DataFrame with metadata.
        """
        pass

    @abstractmethod
    def visualize_batch(
        self, split: str = "train"
    ) -> tuple[plt.Figure, dict[str, Tensor]]:
        """Visualize a batch of data.

        Args:
            split: One of 'train', 'validation', 'test'.

        Returns:
            The matplotlib figure and the batch of data.
        """
        pass

    @abstractmethod
    def define_augmentations(self) -> None:
        """Define augmentations for the dataset and task, applied on a batch of data.

        Augmentations will be applied in `on_after_batch_transfer` in the LightningDataModule.
        """
        pass

    def visualize_geospatial_distribution(
        self,
        split_column: str = "tortilla:data_split",
        buffer_degrees: float = 5.0,
        sample_fraction: float | None = None,
        scale: Literal["10m", "50m", "110m"] = "50m",
        alpha: float = 0.5,
        s: float = 0.5,
    ) -> plt.Figure:
        """Visualize the geospatial distribution of dataset samples on a map.

        Creates a plot showing the geographic locations of samples, colored by dataset split
        (train, validation, test). This helps to understand the spatial distribution
        and potential geographic biases in the dataset.

        Args:
            split_column: Column name in the metadata DataFrame that indicates the dataset split.
            buffer_degrees: Buffer around the data extent in degrees.
            sample_fraction: Optional fraction of samples to plot (0.0-1.0) for performance with large datasets.
            scale: Scale of cartopy features (e.g., '10m', '50m', '110m').
            alpha: Transparency of plotted points.
            s: Size of plotted points.
        """
        if not hasattr(self, "data_df") or self.data_df is None:
            self.load_metadata()

        data_df = self.data_df.copy()

        if "lat" not in data_df.columns or "lon" not in data_df.columns:
            if "latitude" in data_df.columns and "longitude" in data_df.columns:
                data_df.rename(
                    columns={"latitude": "lat", "longitude": "lon"}, inplace=True
                )
            else:
                raise ValueError(
                    "Metadata is missing required latitude and longitude information"
                )

        # Optional sub-sampling for performance
        if sample_fraction is not None and 0.0 < sample_fraction < 1.0:
            sample_n = max(1, int(len(data_df) * sample_fraction))
            data_df = data_df.sample(n=sample_n, random_state=0)

        dataset_name = self.__class__.__name__.replace("DataModule", "")

        min_lon = data_df["lon"].min() - buffer_degrees
        max_lon = data_df["lon"].max() + buffer_degrees
        min_lat = data_df["lat"].min() - buffer_degrees
        max_lat = data_df["lat"].max() + buffer_degrees

        min_lon = max(-180, min_lon)
        max_lon = min(180, max_lon)
        min_lat = max(-90, min_lat)
        max_lat = min(90, max_lat)

        fig = plt.figure(figsize=(12, 10))

        lon_extent = max_lon - min_lon
        lat_extent = max_lat - min_lat

        if lon_extent > 180:
            projection = ccrs.Robinson()
        else:
            central_lon = (min_lon + max_lon) / 2
            central_lat = (min_lat + max_lat) / 2

            if lat_extent > 60:
                projection = ccrs.AlbersEqualArea(
                    central_longitude=central_lon, central_latitude=central_lat
                )
            else:
                projection = ccrs.LambertConformal(
                    central_longitude=central_lon, central_latitude=central_lat
                )

        ax = plt.axes(projection=projection)
        ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.LAND.with_scale(scale), facecolor="lightgray")
        ax.add_feature(cfeature.OCEAN.with_scale(scale), facecolor="lightblue")
        ax.add_feature(cfeature.COASTLINE.with_scale(scale), linewidth=0.8)
        ax.add_feature(cfeature.BORDERS.with_scale(scale), linewidth=0.8, linestyle=":")

        if max_lon - min_lon < 90:
            ax.add_feature(cfeature.RIVERS, linewidth=0.2, alpha=0.5)
            ax.add_feature(cfeature.LAKES, facecolor="lightblue", alpha=0.5)

        splits = data_df[split_column].unique()

        split_colors = {
            "train": "blue",
            "val": "green",
            "validation": "green",
            "test": "red",
        }

        legend_elements = []

        for split in splits:
            split_data = data_df[data_df[split_column] == split]
            if len(split_data) > 0:
                color = split_colors.get(split, "gray")
                ax.scatter(
                    split_data["lon"],
                    split_data["lat"],
                    transform=ccrs.PlateCarree(),
                    c=color,
                    s=s,
                    alpha=alpha,
                    label=split,
                )
                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=color,
                        markersize=8,
                        label=f"{split} (n={len(split_data)})",
                    )
                )

        ax.legend(handles=legend_elements, loc="lower right", title="Dataset Splits")
        title = f"Geographic Distribution of {dataset_name} Samples by Split"

        gl = ax.gridlines(
            draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
        )
        gl.top_labels = False
        gl.right_labels = False

        plt.title(title, fontsize=14)

        return fig

    # @abstractmethod
    # def visualize_target_distribution(self) -> None:
    #     """Visualize the target distribution of the dataset."""
    #     # for single vector targets this should be easy, but how to make this easier for pixel-wise targets, also store in metadata?
    #     pass

    def train_dataloader(self) -> DataLoader:
        """Return train dataloader.

        Returns:
            Train Dataloader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader.

        Returns:
            Validation Dataloader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader.

        Returns:
            Test Dataloader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    def extra_test_dataloader(self) -> DataLoader:
        """Return extra test dataloader.

        Returns:
            Test Dataloader with kept out samples from train partition
        """
        if self.has_extra_test_samples:
            extra_dataset = self.dataset_class(
                split="extra_test",
                band_order=self.band_order,
                transforms=self.val_transform,
                **self.kwargs,
            )
            return DataLoader(
                extra_dataset,
                batch_size=self.eval_batch_size,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
                pin_memory=self.pin_memory,
                shuffle=False,
                drop_last=False,
            )
        else:
            raise RuntimeError("This dataset does not have extra test samples")

    def on_after_batch_transfer(
        self, batch: dict[str, Tensor], dataloader_idx: int
    ) -> dict[str, Tensor]:
        """Apply batch augmentations to the batch after it is transferred to the device.

        Args:
            batch: A batch of data that needs to be altered or augmented.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A batch of data.
        """
        if self.trainer:
            if self.trainer.training:
                split = "train"
            else:
                split = "eval"

            aug = self._valid_attribute(f"{split}_augmentations")

            batch = aug(batch)

        return batch

    def _valid_attribute(self, args) -> Any:
        """Find a valid attribute with length > 0.

        Args:
            args: One or more names of attributes to check (string or sequence of strings).

        Returns:
            The first valid attribute found.

        Raises:
            RuntimeError: If no attribute is defined, or has length 0.
        """
        names = args if isinstance(args, (list, tuple)) else [args]
        for name in names:
            obj = getattr(self, name, None)

            if obj is None:
                continue

            if not obj:
                msg = f"{self.__class__.__name__}.{name} has length 0."
                print(msg)
                raise RuntimeError

            return obj

        msg = f"{self.__class__.__name__}.setup must define one of {names}."
        print(msg)
        raise RuntimeError


class GeoBenchClassificationDataModule(GeoBenchDataModule):
    """GeoBench Classification DataModule.

    By default, will yield a batch of images and their corresponding labels as
    a dictionary with keys 'image' and 'label'.
    """

    def __init__(
        self,
        dataset_class: Dataset,
        img_size: int,
        band_order: Sequence[float | str],
        batch_size: int = 32,
        eval_batch_size: int = 64,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        train_augmentations: Callable | None | str = "default",
        eval_augmentations: Callable | None | str = "default",
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize GeoBench Classification DataModule.

        Args:
            dataset_class: Dataset class to use in the DataModule
            img_size: Desired image input size for the model
            band_order: band order of the image sample to be returned
            batch_size: Batch size during training
            eval_batch_size: Batch size during evaluation, can usually be larger than batch_size,
                to speed up evaluation.
            num_workers: Number of workers for dataloaders
            collate_fn: Collate function that can reformat samples to the needs of the model.
            train_augmentations: Transforms/Augmentations to apply during training, they will be applied
                at the sample level and should include normalization. See :method:`define_augmentations`
                for the default transformation.
            eval_augmentations: Transforms/Augmentations to apply during evaluation, they will be applied
                at the sample level and should include normalization. See :method:`define_augmentations`
                for the default transformation.
            pin_memory: whether to pin memory in dataloaders
            **kwargs: Additional keyword arguments passed to ``dataset_class``
        """
        super().__init__(
            dataset_class=dataset_class,
            img_size=img_size,
            band_order=band_order,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            train_augmentations=train_augmentations,
            eval_augmentations=eval_augmentations,
            pin_memory=pin_memory,
            **kwargs,
        )

    def define_augmentations(self) -> None:
        """Define augmentations for the dataset and task, that are applied on a batch of data.

        Augmentations will be applied in `on_after_batch_transfer` in the LightningDataModule.
        """
        if self.train_augmentations == "default":
            self.train_augmentations = K.AugmentationSequential(
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.5),
                data_keys=None,
                keepdim=True,
            )
        elif self.train_augmentations is None:
            self.train_augmentations = nn.Identity()

        if (self.eval_augmentations == "default") or (self.eval_augmentations is None):
            self.eval_augmentations = nn.Identity()

    def setup_image_size_transforms(self) -> tuple[nn.Module, nn.Module, nn.Module]:
        """Setup image resizing transforms for train, val, test.

        Image resizing and normalization happens on dataset level on individual data samples.
        """
        return (
            K.AugmentationSequential(
                K.Resize(size=(self.img_size, self.img_size), keepdim=True),
                data_keys=None,
            ),
            K.AugmentationSequential(
                K.Resize(size=(self.img_size, self.img_size), keepdim=True),
                data_keys=None,
            ),
            K.AugmentationSequential(
                K.Resize(size=(self.img_size, self.img_size), keepdim=True),
                data_keys=None,
            ),
        )

    def load_metadata(self) -> pd.DataFrame:
        """Load metadata file.

        Returns:
            pandas DataFrame with metadata.
        """
        pass

    def visualize_batch(
        self, split: str = "train"
    ) -> tuple[plt.Figure, dict[str, Tensor]]:
        """Visualize a batch of data.

        Args:
            split: One of 'train', 'validation', 'test'

        Returns:
            The matplotlib figure and the batch of data
        """
        pass

    def visualize_geolocation_distribution(self) -> None:
        """Visualize the geolocation distribution of the dataset."""
        pass


class GeoBenchSegmentationDataModule(GeoBenchDataModule):
    """GeoBench Segmentation DataModule.

    By default, will yield a batch of images and their corresponding masks as
    a dictionary with keys 'image' and 'mask'.
    """

    def __init__(
        self,
        dataset_class: Dataset,
        img_size: int,
        band_order: Sequence[float | str],
        batch_size: int = 32,
        eval_batch_size: int = 64,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        train_augmentations: Callable | None | str = "default",
        eval_augmentations: Callable | None | str = "default",
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize GeoBench Segmentation DataModule.

        Args:
            dataset_class: Dataset class to use in the DataModule
            img_size: Desired image input size for the model
            band_order: band order of the image sample to be returned
            batch_size: Batch size during training
            eval_batch_size: Batch size during evaluation, can usually be larger than batch_size,
                to speed up evaluation.
            num_workers: Number of workers for dataloaders
            collate_fn: Collate function that can reformat samples to the needs of the model.
            train_augmentations: Transforms/Augmentations to apply during training, they will be applied
                at the sample level and should include normalization. See :method:`define_augmentations`
                for the default transformation.
            eval_augmentations: Transforms/Augmentations to apply during evaluation, they will be applied
                at the sample level and should include normalization. See :method:`define_augmentations`
                for the default transformation.
            pin_memory: whether to pin memory in dataloaders
            **kwargs: Additional keyword arguments passed to ``dataset_class``
        """
        super().__init__(
            dataset_class=dataset_class,
            img_size=img_size,
            band_order=band_order,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            train_augmentations=train_augmentations,
            eval_augmentations=eval_augmentations,
            pin_memory=pin_memory,
            **kwargs,
        )

    def define_augmentations(self) -> None:
        """Define augmentations for the dataset and task, that are applied on a batch of data.

        Augmentations will be applied in `on_after_batch_transfer` in the LightningDataModule.
        """
        if self.train_augmentations == "default":
            self.train_augmentations = K.AugmentationSequential(
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.5),
                # data_keys=["image", "mask"],
                data_keys=None,
                keepdim=True,
            )
        elif self.train_augmentations is None:
            self.train_augmentations = nn.Identity()

        if (self.eval_augmentations == "default") or (self.eval_augmentations is None):
            self.eval_augmentations = nn.Identity()

    def setup_image_size_transforms(self) -> tuple[nn.Module, nn.Module, nn.Module]:
        """Setup image resizing transforms for train, val, test.

        Image resizing and normalization happens on dataset level on individual data samples.
        """
        return (
            K.AugmentationSequential(
                K.Resize(size=(self.img_size, self.img_size), keepdim=True),
                data_keys=None,
            ),
            K.AugmentationSequential(
                K.Resize(size=(self.img_size, self.img_size), keepdim=True),
                data_keys=None,
            ),
            K.AugmentationSequential(
                K.Resize(size=(self.img_size, self.img_size), keepdim=True),
                data_keys=None,
            ),
        )

    def load_metadata(self) -> pd.DataFrame:
        """Load metadata file.

        Returns:
            pandas DataFrame with metadata.
        """
        pass

    def visualize_batch(
        self, split: str = "train"
    ) -> tuple[plt.Figure, dict[str, Tensor]]:
        """Visualize a batch of data.

        Args:
            split: One of 'train', 'validation', 'test'

        Returns:
            The matplotlib figure and the batch of data
        """
        pass

    def visualize_geolocation_distribution(self) -> None:
        """Visualize the geolocation distribution of the dataset."""
        pass


class GeoBenchObjectDetectionDataModule(GeoBenchDataModule):
    """GeoBench Object Detection DataModule.

    By default, will yield a batch of images and their corresponding bounding boxes and labels as
    a dictionary with keys 'image', 'boxes_xyxy', and 'labels'.
    """

    def __init__(
        self,
        dataset_class: Dataset,
        img_size: int,
        band_order: Sequence[float | str],
        batch_size: int = 32,
        eval_batch_size: int = 64,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        train_augmentations: Callable | None | str = "default",
        eval_augmentations: Callable | None | str = "default",
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize GeoBench Object Detection DataModule.

        Args:
            dataset_class: Dataset class to use in the DataModule
            img_size: Desired image input size for the model
            band_order: band order of the image sample to be returned
            batch_size: Batch size during training
            eval_batch_size: Batch size during evaluation, can usually be larger than batch_size,
                to speed up evaluation.
            num_workers: Number of workers for dataloaders
            collate_fn: Collate function that can reformat samples to the needs of the model.
            train_augmentations: Transforms/Augmentations to apply during training, they will be applied
                at the sample level and should include normalization. See :method:`define_augmentations`
                for the default transformation.
            eval_augmentations: Transforms/Augmentations to apply during evaluation, they will be applied
                at the sample level and should include normalization. See :method:`define_augmentations`
                for the default transformation.
            pin_memory: whether to pin memory in dataloaders
            **kwargs: Additional keyword arguments passed to ``dataset_class``
        """
        super().__init__(
            dataset_class=dataset_class,
            img_size=img_size,
            band_order=band_order,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            train_augmentations=train_augmentations,
            eval_augmentations=eval_augmentations,
            pin_memory=pin_memory,
            **kwargs,
        )

    def define_augmentations(self) -> None:
        """Define augmentations for the dataset and task, that are applied on a batch of data.

        Augmentations will be applied in `on_after_batch_transfer` in the LightningDataModule.
        """
        if self.train_augmentations == "default":
            self.train_augmentations = K.AugmentationSequential(
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.5),
                data_keys=None,
                keepdim=True,
            )
        elif self.train_augmentations is None:
            self.train_augmentations = nn.Identity()

        if (self.eval_augmentations == "default") or (self.eval_augmentations is None):
            self.eval_augmentations = nn.Identity()

    def setup_image_size_transforms(self) -> tuple[nn.Module, nn.Module, nn.Module]:
        """Setup image resizing transforms for train, val, test.

        Image resizing and normalization happens on dataset level on individual data samples.
        """
        return (
            K.AugmentationSequential(
                K.Resize(size=(self.img_size, self.img_size), keepdim=True),
                data_keys=None,
            ),
            K.AugmentationSequential(
                K.Resize(size=(self.img_size, self.img_size), keepdim=True),
                data_keys=None,
            ),
            K.AugmentationSequential(
                K.Resize(size=(self.img_size, self.img_size), keepdim=True),
                data_keys=None,
            ),
        )

    def load_metadata(self) -> pd.DataFrame:
        """Load metadata file.

        Returns:
            pandas DataFrame with metadata.
        """
        pass

    def visualize_batch(
        self, split: str = "train"
    ) -> tuple[plt.Figure, dict[str, Tensor]]:
        """Visualize a batch of data.

        Args:
            split: One of 'train', 'validation', 'test'

        Returns:
            The matplotlib figure and the batch of data
        """
        pass

    def visualize_geolocation_distribution(self) -> None:
        """Visualize the geolocation distribution of the dataset."""
        pass
