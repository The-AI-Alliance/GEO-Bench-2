import json
import re
from collections.abc import Sequence
from pathlib import Path

import albumentations as A
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from .base import GeoBenchBaseDataset
import skimage
from terratorch.datasets.utils import (
    default_transform,
    validate_bands,
)
import os
from collections.abc import Callable
from typing import Any, ClassVar

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patches
from matplotlib.figure import Figure
from torch import Tensor
import h5py
import pdb
from torchgeo.datasets.utils import download_and_extract_archive
import pycocotools.coco
import pycocotools
import requests

from .sensor_util import DatasetBandRegistry


def convert_coco_poly_to_mask(
    segmentations: list[int], height: int, width: int
) -> Tensor:
    """Convert coco polygons to mask tensor.

    Args:
        segmentations (List[int]): polygon coordinates
        height (int): image height
        width (int): image width

    Returns:
        Tensor: Mask tensor

    Raises:
        DependencyNotFoundError: If pycocotools is not installed.
    """
    masks = []
    for polygons in segmentations:
        rles = pycocotools.mask.frPyObjects(polygons, height, width)
        mask = pycocotools.mask.decode(rles)
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    masks_tensor = torch.stack(masks, dim=0)
    return masks_tensor


class ConvertCocoAnnotations:
    """Callable for converting the boxes, masks and labels into tensors.

    This is a modified version of ConvertCocoPolysToMask() from torchvision found in
    https://github.com/pytorch/vision/blob/v0.14.0/references/detection/coco_utils.py
    """

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Converts MS COCO fields (boxes, masks & labels) from list of ints to tensors.

        Args:
            sample: Sample

        Returns:
            Processed sample
        """
        # pdb.set_trace()
        image = sample['image']
        h, w = image.size()[-2:]        
        target = sample['label']

        image_id = target['image_id']
        image_id = torch.tensor([image_id])

        anno = target['annotations']

        anno = [obj for obj in anno if obj['iscrowd'] == 0]

        bboxes = [obj['bbox'] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(bboxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        categories = [obj['category_id'] for obj in anno]
        classes = torch.tensor(categories, dtype=torch.int64)

        segmentations = [obj['segmentation'] for obj in anno]

        masks = convert_coco_poly_to_mask(segmentations, h, w)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {'boxes': boxes, 'labels': classes, 'image_id': image_id}
        if masks.nelement() > 0:
            masks = masks[keep]
            target['masks'] = masks

        # for conversion to coco api
        area = torch.tensor([obj['area'] for obj in anno])
        iscrowd = torch.tensor([obj['iscrowd'] for obj in anno])
        target['area'] = area
        target['iscrowd'] = iscrowd
        return {'image': image, 'label': target}



class GeoBenchNZCattle(GeoBenchBaseDataset):
    """ "GeoBenchNZCattle dataset. Object detection version of the segmentation nz-cattle dataset in GEO-Bench v1."""
    
    url = ""
    paths = ["geobench_nzcattle.tortilla"]
    sha256strsumsumsumsumsum: Sequence[str] = []

    dataset_band_config = DatasetBandRegistry.EVERWATCH
    band_default_order = ("red", "green", "blue")

    normalization_stats = {
        "means": {"red": 0.0, "green": 0.0, "blue": 0.0},
        "stds": {"red": 255.0, "green": 255.0, "blue": 255.0},
    }

    classes = ('cattle')

    num_classes = len(classes)

    partition = "default"
    partition_file_template = "{partition}_partition.json"


    all_band_names = ("RED", "GREEN", "BLUE")

    rgb_bands = ("RED", "GREEN", "BLUE")

    BAND_SETS = {"all": all_band_names, "rgb": rgb_bands}

    splits = {"train": "train", "val": "valid", "test": "test"}

    data_dir = "m-nz-cattle"
    

    def __init__(
        self,
        root: str,
        split: str = "train",
        bands: Sequence[str] = BAND_SETS["all"],
        transforms: nn.Module | None = None,
        partition: str = "default",
        use_metadata: bool = False,
        download: bool = False
    ) -> None:
        """Initialize the dataset.

        Args:
            root (str): Path to the data root directory.
            split (str): One of 'train', 'val', or 'test'.
            bands (Sequence[str]): Bands to be used. Defaults to all bands.
            transform (nn.Module | None): Transform to be applied.
            partition (str): Partition name for the dataset splits. Defaults to 'default'.
            use_metadata (bool): Whether to return metadata info (time and location).
        """
        super().__init__()

        if split not in self.splits:
            msg = f"Incorrect split '{split}', please choose one of {list(self.splits.keys())}."
            raise ValueError(msg)
        split_name = self.splits[split]
        self.split = split

        validate_bands(bands, self.all_band_names)
        self.bands = bands
        self.band_indices = [self.all_band_names.index(b) for b in bands]

        self.use_metadata = use_metadata

        self.root = Path(root)
        self.data_directory = self.root / self.data_dir

        partition_file = self.data_directory / self.partition_file_template.format(partition=partition)

        if download:

            self._download()

        with open(partition_file) as file:
            partitions = json.load(file)

        if split_name not in partitions:
            msg = f"Split '{split_name}' not found."
            raise ValueError(msg)

        self.image_files = [self.data_directory / f"{filename}.hdf5" for filename in partitions[split_name]]

        self.transform = transform if transform else default_transform
        self.annotation_file = f"{root}/m-nz-cattle/annotations.json"
        self.coco = pycocotools.coco.COCO(self.annotation_file)
        self.coco_convert = ConvertCocoAnnotations()


    def __len__(self) -> int:
        return len(self.image_files)
    
    
    def _load_target(self, id_):
        """Load the annotations for a single image.

        Args:
            id_: unique ID of the image

        Returns:
            the annotations
        """

        annot = []
        ann_ids = self.coco.getAnnIds(imgIds=id_)
        annot = self.coco.loadAnns(ann_ids)

        target = dict(image_id=id_, annotations=annot)

        return target

    
    def __getitem__(self, id_: int) -> dict[str, torch.Tensor]:

        file_name = self.image_files[id_]

        with h5py.File(file_name, "r") as h5file:
            keys = sorted(h5file.keys())

            data_keys = [key for key in keys if "label" not in key]

            temporal_coords = self._get_date(data_keys[0])

            bands = [np.array(h5file[key]) for key in data_keys]
            image = np.stack(bands, axis=-1)

        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        image = image.float()

        id_ = [img_id for img_id, img_info in self.coco.imgs.items() if img_info['file_name'] == str(file_name)][0]

        sample: dict[str, Any] = {'image': image,
                                'label': self._load_target(id_)}
        
        if sample['label']['annotations']:
            sample = self.coco_convert(sample)
            sample['class'] = sample['label']['labels']
            sample['boxes'] = sample['label']['boxes']
            sample['labels'] = sample.pop('class')

        if self.transform:
            sample = self.transform(sample)
 
        if self.use_metadata:
            location_coords = self._get_coords(file_name)
            temporal_coords = temporal_coords
            sample["location_coords"] = location_coords
            sample["temporal_coords"] = temporal_coords

        return sample

    def _get_coords(self, file_name: str) -> torch.Tensor:
        """Extract spatial coordinates from the file name."""
        match = re.search(r"_(\-?\d+\.\d+),(\-?\d+\.\d+)", file_name)
        if match:
            longitude, latitude = map(float, match.groups())

        return torch.tensor([latitude, longitude], dtype=torch.float32)

    def _get_date(self, band_name: str) -> torch.Tensor:
        date_str = band_name.split("_")[-1]
        date = pd.to_datetime(date_str, format="%Y-%m-%d")

        return torch.tensor([[date.year, date.dayofyear - 1]], dtype=torch.float32)

    def plot(self,
             sample: dict[str, Tensor],
             show_titles: bool = True,
             suptitle: str | None = None,
             box_alpha: float = 0.7,
             confidence_score = 0.5) -> Figure:

        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            suptitle: optional string to use as a suptitle
            show_titles: flag indicating whether to show titles above each panel
            box_alpha: alpha value of box
            confidence_score: 

        Returns:
            a matplotlib Figure with the rendered sample

        Raises:
            AssertionError: if ``show_feats`` argument is invalid
            DependencyNotFoundError: If plotting masks and scikit-image is not installed.

        """


        categories = ( 'background', 'cattle')

        image = sample['image'].permute(1, 2, 0).numpy()
        
        if image.mean() > 1:
            image = image/255
        
        image = np.clip(image, 0, 1)

        boxes = sample['boxes'].cpu().numpy() if 'boxes' in sample else []
        labels = sample['labels'].cpu().numpy() if 'labels' in sample else []

        n_gt = len(boxes)
        print(n_gt)
        ncols = 1
        show_predictions = 'prediction_labels' in sample

        if show_predictions:
            show_pred_boxes = False
            prediction_labels = sample['prediction_labels'].numpy()
            prediction_scores = sample['prediction_scores'].numpy()
            if 'prediction_boxes' in sample:
                prediction_boxes = sample['prediction_boxes'].numpy()
                show_pred_boxes = True

            n_pred = len(prediction_labels)
            ncols += 1

        # Display image
        fig, axs = plt.subplots(ncols=ncols, squeeze=False, figsize=(ncols * 10, 13))
        axs[0, 0].imshow(image)
        axs[0, 0].axis('off')

        cm = plt.get_cmap('gist_rainbow')
        for i in range(n_gt):
            class_num = labels[i]
            color = cm(class_num / len(categories))

            # Add bounding boxes
            x1, y1, x2, y2 = boxes[i]
            r = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                alpha=box_alpha,
                linestyle='dashed',
                edgecolor=color,
                facecolor='none',
            )
            axs[0, 0].add_patch(r)

            # Add labels
            label = categories[class_num]
            caption = label
            axs[0, 0].text(
                x1, y1 - 8, caption, color='white', size=11, backgroundcolor='none'
            )

            if show_titles:
                axs[0, 0].set_title('Ground Truth')

        if show_predictions:
            axs[0, 1].imshow(image)
            axs[0, 1].axis('off')
            for i in range(n_pred):
                score = prediction_scores[i]
                if score < confidence_score:
                    continue

                class_num = prediction_labels[i]
                color = cm(class_num / len(categories))

                if show_pred_boxes:
                    # Add bounding boxes
                    x1, y1, x2, y2 = prediction_boxes[i]
                    r = patches.Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        linewidth=2,
                        alpha=box_alpha,
                        linestyle='dashed',
                        edgecolor=color,
                        facecolor='none',
                    )
                    axs[0, 1].add_patch(r)

                    # Add labels
                    label = categories[class_num]
                    caption = f'{label} {score:.3f}'
                    axs[0, 1].text(
                        x1,
                        y1 - 8,
                        caption,
                        color='white',
                        size=11,
                        backgroundcolor='none',
                    )

            if show_titles:
                axs[0, 1].set_title('Prediction')

        if suptitle is not None:
            plt.suptitle(suptitle)

        plt.tight_layout()

        return fig
    
    def _download(self) -> None:
        """Download the dataset and extract it."""

        if not os.path.exists(self.root / self.data_dir):
        # Download splits
            download_and_extract_archive("https://ibm.box.com/shared/static/m3pd94zqp3wwon8hftdyhc9p5hzyl5dz.zip", 
                                        download_root=self.root,
                                        extract_root=self.root,
                                        filename="m-nz-cattle.zip",
                                        remove_finished=True)
                

 