
import argparse
import os

import numpy as np
import pandas as pd

import json
from tqdm import tqdm
from PIL import ImageFile
from PIL import Image
import geopandas as gpd
from shapely.geometry import shape

import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
from PIL import Image
from tqdm import tqdm
import multiprocessing
from functools import partial
import h5py

from geobench_v2.generate_benchmark.object_detection_util import (
    convert_pngs_to_geotiffs,
)

from geobench_v2.generate_benchmark.utils import create_subset_from_df


import os
import json
import glob
import rasterio
import tacoreader
import tacotoolbox
import pdb
import glob

def generate_metadata_df(root_dir: str) -> pd.DataFrame:

    # generate df with split info and keys
    splits = glob.glob(root_dir + "splits/*")
    splits_df = [pd.read_csv(split, header=None, names=['image_key']) for split in splits]
    splits = [x.split('/')[-1].replace('.txt', '') for x in splits]
    for i, split in enumerate(splits): 
        splits_df[i]['data_split'] = split
    splits_df = pd.concat(splits_df, axis=0)
    splits_df['data_split'] = [x if x != 'val' else 'validation' for x in splits_df['data_split'].values]

    # assign split to each image
    images = glob.glob(root_dir + "data/*_merged.tif")
    metadata_df = pd.DataFrame({'file_path': images})
    metadata_df['image_key'] = [x.split('S30.')[-1].split('.4')[0] for x in images]
    metadata_df = pd.merge(metadata_df, splits_df, how='left', on='image_key')

    # add labels
    metadata_df['label_file'] = [x.replace('_merge.tif', '.mask.tif') for x in metadata_df['file_path'].values]

    return metadata_df


def create_tortilla(dataset_dir, save_dir, tortilla_name):
    """Create a tortilla version of an object detection dataset.

    Args:
        dataset_dir: Directory containing the GeoTIFF images and labels
        save_dir: Directory to save the tortilla files
        tortilla_name: Name of the final tortilla file
    """
    
    tortilla_dir = os.path.join(save_dir, "tortilla")
    os.makedirs(tortilla_dir, exist_ok=True)

    metadata_df = generate_metadata_df(dataset_dir)

    for idx, geotiff_path in enumerate(tqdm(metadata_df['file_path'].values, desc="Creating tortillas")):

        with rasterio.open(geotiff_path) as src:
            profile = src.profile
            height, width = profile["height"], profile["width"]
            crs = "EPSG:32660"
            transform = profile["transform"].to_gdal() if profile["transform"] else None

        tile = metadata_df['image_key'].values[idx].split('.')[0]
        time = metadata_df['image_key'].values[idx].split('.')[1].split('.')[0]
        
        # create image
        image_sample = tacotoolbox.tortilla.datamodel.Sample(
            id="image",
            path=geotiff_path,
            file_format="GTiff",
            data_split=metadata_df['data_split'].values[idx],
            stac_data={
                "crs": crs,
                "geotransform": transform,
                "raster_shape": (height, width),
                "time_start": time,
            },
            tile=tile
        )

        # Create annotation part
        label_sample = tacotoolbox.tortilla.datamodel.Sample(
            id="label",
            path=metadata_df['label_file'].values[idx],
            file_format="GTiff",
            data_split=metadata_df['data_split'].values[idx],
            stac_data={
                "crs": crs,
                "geotransform": transform,
                "raster_shape": (height, width),
                "time_start": time,
            },
            tile=tile
        )

        taco_samples = tacotoolbox.tortilla.datamodel.Samples(
            samples=[image_sample, label_sample]
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
    tacotoolbox.tortilla.create(final_samples, final_path, quiet=False, nworkers=1)


if __name__ == '__main__':
    """Generate Burn Scars Benchmark."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data", help="Root directory for nz-Cattle dataset"
    )
    parser.add_argument(
        "--save_dir",
        help="Directory to save the subset",
    )

    args = parser.parse_args()

    tortilla_name = "geobench_burn_scars.tortilla"

    create_tortilla(args.root, args.save_dir, tortilla_name=tortilla_name)



