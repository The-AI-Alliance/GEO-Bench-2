
import argparse
import os

import numpy as np
import pandas as pd

import json
from tqdm import tqdm
from shapely.geometry import shape

import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
from tqdm import tqdm
import h5py


import os
import json
import glob
import rasterio
import tacoreader
import tacotoolbox
import pdb
import json

from geobench_v2.generate_benchmark.utils import create_unittest_subset

def generate_metadata_df(root_dir: str) -> pd.DataFrame:

    with open(root_dir + 'annotations.json', 'r') as f:
        d = json.load(f)

    
    # load images meta
    images_df = pd.DataFrame(d['images'])
    images_df = images_df.rename(columns={"id": "image_id"})

    # load splits meta
    splits_df = pd.read_csv(root_dir + 'substation_meta_splits_full.csv')
    splits_df = splits_df.drop(columns=['id', 'index_right'])
    splits_df = splits_df.rename(columns={"image": "file_name"})

    # join splits and images
    images_df = pd.merge(images_df, splits_df, on=['file_name', 'lat', 'lon'])

    # load annotations
    annotations_df = pd.DataFrame(d['annotations'])

    metadata_df = pd.merge(images_df, annotations_df, on='image_id')

    metadata_df['file_name'] = [root_dir.replace('Substation/', '') + x for x in metadata_df['file_name'].values]
    
    return metadata_df

def generate_random_subsample(metadata_df, n_splits = [7000, 1500, 1500]):

    splits = ['train', 'val', 'test']

    metadata_sub_df = pd.DataFrame() 

    for n, split in zip(n_splits, n_splits):

        tmp = metadata_df[metadata_df['split'].values == split]
        tmp = tmp.sample(n=n, random_state=123)
        metadata_sub_df = pd.concat([metadata_sub_df, tmp], axis=0)
    
    return metadata_sub_df

def download(root):

    """
    Download the substation dataset
    """

    # To be implemented. Use the download method from https://github.com/IBM/terratorch/blob/main/terratorch/datasets/substation.py
    return 



def create_tortilla(metadata_df, save_dir, tortilla_name):
    """Create a tortilla version of an object detection dataset.

    Args:
        annotations_df: DataFrame with annotations including image_path, label, bbox coordinates
        save_dir: Directory to save the tortilla files
        tortilla_name: Name of the final tortilla file
    """
    
    tortilla_dir = os.path.join(save_dir, "tortilla")
    os.makedirs(tortilla_dir, exist_ok=True)

    unique_images = metadata_df["file_name"].unique()


    for idx, img_name in enumerate(tqdm(unique_images, desc="Creating tortillas")):

        ###### image conversion
        image_data = np.load(image_path)['arr_0']

        img_annotations = metadata_df[metadata_df["file_name"] == img_name]





        geotiff_path = os.path.join(save_dir, img_name)

        # Create annotations dictionary in COCO-like format
        boxes = []
        for _, ann in img_annotations.iterrows():
            boxes.append(
                {
                    "category_id": ann["category_id"],
                    "bbox": ann["bbox"],
                    "bbox_mode": "xywh",
                    "mask": ann["segmentation"] 
                }
            )

        with rasterio.open(geotiff_path) as src:
            profile = src.profile
            height, width = profile["height"], profile["width"]
            crs = "EPSG:" + str(profile["crs"].to_epsg()) if profile["crs"] else None
            transform = profile["transform"].to_gdal() if profile["transform"] else None

        first_row = img_annotations.iloc[0]
        split = first_row["split"]
        if split == "val": split = "validation"
        lon = first_row["lon"] if not pd.isna(first_row["lon"]) else None
        lat = first_row["lat"] if not pd.isna(first_row["lat"]) else None

        annotations_file = os.path.join(
            tortilla_dir, f"{os.path.splitext(img_name.split('/')[-1])[0]}_annotations.HDF5"
        )

        with h5py.File(annotations_file, 'w') as f:
            # Store the entire dictionary as a JSON string attribute
            f.attrs['annotation'] = json.dumps({"boxes": boxes, "image_size": (height, width)})

        # create image
        image_sample = tacotoolbox.tortilla.datamodel.Sample(
            id="image",
            path=geotiff_path,
            file_format="GTiff",
            data_split=split,
            stac_data={
                "crs": crs,
                "geotransform": transform,
                "raster_shape": (height, width),
                "time_start": "2020",
            },
            lon=lon,
            lat=lat,
        )

        # Create annotation part
        annotations_sample = tacotoolbox.tortilla.datamodel.Sample(
            id="annotations",
            path=annotations_file,
            file_format="HDF5",
            data_split=split,
        )

        taco_samples = tacotoolbox.tortilla.datamodel.Samples(
            samples=[image_sample, annotations_sample]
        )

        sample_path = os.path.join(tortilla_dir, f"sample_{idx}.tortilla")
        tacotoolbox.tortilla.create(taco_samples, sample_path, quiet=True)


    # Merge all individual tortillas into one dataset
    all_tortilla_files = sorted(glob.glob(os.path.join(tortilla_dir, "*.tortilla")))

    samples = []
    for tortilla_file in tqdm(all_tortilla_files, desc="Building final tortilla"):
        sample_data = tacoreader.load(tortilla_file).iloc[0]

        sample = tacotoolbox.tortilla.datamodel.Sample(
            id=os.path.basename(tortilla_file).split(".")[0],
            path=tortilla_file,
            file_format="TORTILLA",
            stac_data={
                "crs": sample_data.get("stac:crs"),
                "geotransform": sample_data.get("stac:geotransform"),
                "raster_shape": sample_data.get("stac:raster_shape"),
                "time_start": "2016"
            },
            data_split=sample_data["tortilla:data_split"],
            lon=sample_data.get("lon"),
            lat=sample_data.get("lat"),
        )
        samples.append(sample)

    final_samples = tacotoolbox.tortilla.datamodel.Samples(samples=samples)
    final_path = os.path.join(save_dir, tortilla_name)
    pdb.set_trace()
    tacotoolbox.tortilla.create(final_samples, final_path, quiet=False, nworkers=1)


if __name__ == '__main__':
    """Generate nzCattle Object Dection Benchmark."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for nz-Cattle dataset"
    )
    parser.add_argument(
        "--save_dir",
        default="geobenchV2/nzcattle",
        help="Directory to save the subset",
    )

    args = parser.parse_args()

    download(args.root)

    metadata_df = generate_metadata_df(args.root)
    tortilla_name = "geobench_nzcattle.tortilla"

    create_tortilla(metadata_df, args.save_dir, tortilla_name=tortilla_name)

    create_unittest_subset(
        data_dir=args.save_dir,
        tortilla_pattern=tortilla_name,
        test_dir_name="substation",
        n_train_samples=2,
        n_val_samples=1,
        n_test_samples=1,
    )


