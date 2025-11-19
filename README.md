# GEO-Bench 2

![1-earth](https://raw.githubusercontent.com/The-AI-Alliance/GEO-Bench-2/main/docs/_assets/GEO-Bench-2_banner.jpg)
[![huggingface](https://img.shields.io/badge/Hugging_Face-join-FFD21E?logo=huggingface)](https://huggingface.co/collections/aialliance/geo-bench-2)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Language: Python](https://img.shields.io/badge/language-Python%203.10%2B-green?logo=python&logoColor=green)](https://www.python.org)
[![pypi](https://badge.fury.io/py/geobenchv2.svg)](https://pypi.org/project/GeoBenchV2/)
[![docs](https://readthedocs.org/projects/pip/badge/)](https://geo-bench-2.readthedocs.io/en/latest/)

## Overview

GEO-Bench-2 is a framework for robust evaluation of Geospatial Foundation Models (GeoFMs) which expands on the work of GEO-Bench. It has been carefully curated for evaluation of state-of-the-art model features such as such as multi-modality and multi-temporality.
This library aims to facilitate extensive benchmarking of GeoFMs on the GEO-Bench-2 datasets, including features such band re-ordering, changing normalizations, and more.

For details on the GEO-Bench-2 methodology, please see: #TODO:link to paper

The [Geo-Bench-2-Leaderbaord](https://github.com/The-AI-Alliance/GEO-Bench-2-Leaderboard) tracks the performance of state-of-the-art models on GeoBench-2 datasets. It further acts as a public repository of model performance. We strongly encourage users of the this library to submit experimental results to the leaderboard.

## Installation

For a stable release, install with:

```
pip install GeoBenchV2
```

For the most recent version of the main branch, install with:

```
pip install git+https://github.com/The-AI-Alliance/GEO-Bench-2.git
```

To use the package as a developer, install in editable mode with:
```
git clone https://github.com/The-AI-Alliance/GEO-Bench-2.git
cd GEO-Bench-2
pip install -e .
```

## Documentation

The latest documentation can be found at this [link](https://geo-bench-2.readthedocs.io/en/latest/).

## Downloading the Benchmark

Datasets can be downloaded using the `geobench-download` command-line tool, which is installed automatically with the package.

First, make sure you have installed the GEO-Bench-2 package: 

```bash
pip install geobenchv2
```

To download all datasets to a specified directory:

```bash
geobench-download --root /path/to/data
```

To download only selected datasets:

```bash
geobench-download --root /path/to/data spacenet7 caffe dynamic_earthnet
```

The following datasets are available for download:
- `benv2`, `biomassters`, `burn_scars`, `caffe`, `cloudsen12`
- `dynamic_earthnet`, `everwatch`, `flair2`, `forestnet`, `fotw`
- `kuro_siwo`, `pastis`, `spacenet2`, `spacenet7`, `substation`
- `treesatai`, `wind_turbine`, `so2sat`

**Note:** The `--root` argument is required to ensure you explicitly choose where to store the datasets, as they can be quite large.

## Code License
This code is licensed under the Apache License 2.0. By contributing to this repository, you agree that your contributions will be licensed under the Apache 2.0 License unless otherwise stated.

## Dataset Licenses

All dataset are distributed under open-licenses. For license details please see the respective Huggingface repository of each dataset. A summary of the license files can be found in [this file](./dataset_licenses.md). License information is also contained in each `README.md` file that will be downloaded with the data from Huggingface. For our disclaimer about the licenses and our takedown policy, please see [disclaimer file](./benchmark_license_disclaimer.md).

## How was the Benchmark Generated?

All scripts that were used to generate the GEO-Bench-2 dataset versions are included in the [generate_benchmark](/geobench_v2/generate_benchmark/) directory. These scripts are included for transparency purposes, but leaderboard submissions are solely accepted based on official dataset versions on [HuggingFace](https://huggingface.co/collections/aialliance/geo-bench-2).


## Credits
This project was developed as part of the AI Alliance with involvement from IBM and ServiceNow.


## Citation

If you are using GEO-Bench-2 in your work, please cite:

```bibtex
```


