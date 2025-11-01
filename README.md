# GEO-Bench 2 (Still under development, stay tuned for full release)!

![1-earth](https://github.com/The-AI-Alliance/GEO-Bench-2/assets/5478516/738b5aa6-b46d-48bc-bdde-fd71605b9bac)
[![huggingface](https://img.shields.io/badge/Hugging_Face-join-FFD21E?logo=huggingface)](https://hf.co/datasets/aialliance/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Language: Python](https://img.shields.io/badge/language-Python%203.10%2B-green?logo=python&logoColor=green)](https://www.python.org)
[![pypi](https://badge.fury.io/py/geobenchv2.svg)](https://pypi.org/project/geobenchv2)
[![docs](https://readthedocs.org/projects/pip/badge/)](https://geo-bench-2.readthedocs.io/en/latest/)

## Overview

GEO-Bench-2 is a framework for robust evaluation of Geospatial Foundation Models (GeoFMs) which expands on the work of GEO-Bench. It has been carefully curated for evaluation of state-of-the-art model features such as such as multi-modality and multi-temporality.
This library aims to facilitate extensive benchmarking of GeoFMs on the GEo-Bench-2 datasets, including features such band re-ordering, changing normalizations, and more.

For details on the GEO-Bench-2 methodology, please see: #TODO:link to paper

The [Geo-Bench-2-Leaderbaord](https://github.com/The-AI-Alliance/GEO-Bench-2-Leaderboard) tracks the performance of state-of-the-art models on GeoBench-2 datasets. It further acts as a public repository of model performance. We strongly encourage users of the this library to submit experimental results to the leaderboard.

## Installation

For a stable release, install with `pip install geobenchv2==<version>`.

For the the most recent version of the main branch, install with `pip install git+https://https://github.com/The-AI-Alliance/GEO-Bench-2.git`.

To use the library as a developer, install in editable mode with:
```
git clone https://https://github.com/The-AI-Alliance/GEO-Bench-2.git
cd GEO-Bench-2
pip install -e .
```

## Download
To download the datasets individually, please set `download=True` for each dtamodule. All the datasets can also be downloaded directly from this link #TODO

## Documentation

The latest documentation can be found at this [link](https://geo-bench-2.readthedocs.io/en/latest/)

## Generating the Benchmark

The directory `./generate_benchmark` contains a script for each included dataset that has three purposes:

1. Generate a dataset subset that is sufficient for benchmark purposes and minimal in size to reduce disk space
requirements for users.
2. Generate possible partition sizes for experiments across dataset sizes
3. Generate a super tiny dataset version of dummy data that is used for unit testing all implemented functionality

## Downloading the datasets

While each dataset class has automatic download capabilites similar to [TorchGeo](https://github.com/torchgeo/torchgeo), downloads can be sped up by using the Huggingface Command Line Interface. We have therefore provided a bash script named [download_geobenchV2.sh](./download_geobenchV2.sh), which you can use to download all or selected datasets to a specified root.

First, you need to install the Hugging Face CLI: 

```bash
pip install huggingface_hub[cli]
```

Then, you can use the download script as follows:

```bash
./download_geobenchV2.sh
# or
./download_geobenchV2.sh all
```

if you want to download all datasets.

Or only download specific datasets, for example:

```bash
./download_geobenchV2.sh dynamic_earthnet
./download_geobenchV2.sh caffe
./download_geobenchV2.sh spacenet7
```

The default root where the datasets will be downloaded is `./data`, if you like to specify a dedicated download root for a specific or all dataset downloads, preprend the above commands as follows:

```bash
DOWNLOAD_ROOT="path/to/your/desired/location" ./download_geobenchV2.sh spacenet7
```

## License
This code is licensed under the Apache License 2.0. By contributing to this repository, you agree that your contributions will be licensed under the Apache 2.0 License unless otherwise stated.

## Dataset Licenses

All dataset are distributed under open-licenses. For license details please see the respective Huggingface repository of each dataset. A summary of the license files can be found in [this file](./dataset_licenses.md).


## Credits
This project was developed as part of the AI Alliance with involvement from IBM and ServiceNow.


## Citation
Please add the following citation whereevr this library is used: #TODO


