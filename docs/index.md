# GEO-Bench-2:

The emergence of **Geospatial Foundation Models (GeoFMs)** holds great promise for advancing **Earth Observation (EO)**, enabling more general and scalable solutions for a variety of tasks. However, the rapidly evolving nature of this field has meant that evaluation protocols have been difficult to standardize.

**GEO-Bench-2** is an effort to address this challenge by providing a **comprehensive and community-focused benchmarking framework** tailored to various EO applications. Expanding upon its predecessor, this framework is designed to facilitate consistent, insightful, and fair comparison of GeoFMs.

---

## What GEO-Bench-2 Offers

We aim to simplify and standardize the evaluation process through several key features:

* **Diverse and Permissively Licensed Data:** We include a curated selection of **19 datasets** covering core EO tasks, including **classification, segmentation, regression, object detection, and instance segmentation**, ensuring broad usability.
* **Targeted Evaluation via "Capabilities":** Datasets are grouped into **"capabilities"** based on shared characteristics (e.g., resolution, band usage, temporality). This feature supports flexible benchmarking, allowing users to assess a model's strengths on specific types of EO data.
* **Robust Metrics and Efficiency:** We utilize the **normalized interquartile mean (IQM)** for more robust model comparison and incorporate subsampling strategies to help make large-scale evaluation more efficient.
* **Ease of Use with TerraTorch:** Integration with the TerraTorch open-source toolkit. However, all datasets and datamodules can also be used independently of TerraTorch


## Installation

```bash
pip install geobenchv2
```

## Dataset Overview

| Dataset                | Task                                 | Modalities                                   | Train/Val/Test Samples | # Classes         | License              | Citation                  |
|------------------------|--------------------------------------|----------------------------------------------|-----------------------|-------------------|----------------------|---------------------------|
| BigEarthNetV2          | Multi-label land cover classification| Sentinel-1 SAR (VV,VH) + Sentinel-2 Optical  | 20000 / 4000 / 4000   | 19 (multi-label)  | CDLA-Permissive-1.0  | [Clasen et al. 2025](https://arxiv.org/abs/2407.03653)        |
| TreeSatAI              | Tree species classification          | Sentinel-2 Time Series                       | 20000 / 4000 / 4000   | 13                | CC-BY-4.0            | [Ahlswede et al. 2023](https://essd.copernicus.org/articles/15/681/2023/)      |
| m-so2sat               | Climate zone classification          | Sentinel-2 Optical                           | 19992 / 986 / 986     | 17                | CC-BY-4.0            | [So2Sat LCZ42](https://doi.org/10.1109/MGRS.2020.2964708)      |
| m-forestnet            | Tree species classification          | Landsat 8                                    | 6464 / 989 / 989      | 12                | CC-BY-4.0            | [ForestNet](https://arxiv.org/abs/2011.05479)                  |
| BioMassters            | Biomass regression                   | Sentinel-1 SAR (VV,VH) + Sentinel-2 Optical  | 4000 / 1000 / 2000    | Continuous        | CC-BY-4.0            | [Nascetti et al. 2023](https://openreview.net/pdf?id=hrWsIC4Cmz)      |
| CaFFe                  | Glacier zone segmentation            | Sentinel-1 SAR                               | 4000 / 1000 / 2000    | 4                 | CC-BY-4.0            | [Gourmelon et al. 2022](https://essd.copernicus.org/articles/14/4287/2022/)     |
| CloudSEN12             | Cloud/shadow segmentation            | Sentinel-1 SAR + Sentinel-2 Optical          | 4000 / 1000 / 2000    | 4                 | CC0                  | [Aybar et al. 2022](https://doi.org/10.1038/s41597-022-01878-2) |
| Burn Scars             | Burned area segmentation             | HLS (Landsat 8/9 + Sentinel-2)               | 524 / 160 / 120       | 2 (binary)        | CC-BY-4.0            | [Phillips et al. 2024](https://arxiv.org/abs/2310.18660)      |
| Dynamic EarthNet       | Land cover semantic segmentation     | Sentinel-2 (10 bands) + Planet (4 bands)     | 4000 / 1000 / 2000    | 7                 | CC-BY-4.0            | [Toker et al. 2022](https://arxiv.org/abs/2203.12560)         |
| FLAIR2                 | Land cover semantic segmentation     | Aerial RGB+NIR + DEM                         | 4000 / 1000 / 2000    | 13                | Open License 2.0     | [Garioud et al. 2023](https://arxiv.org/abs/2305.14467)       |
| m-FoTW                 | Field boundary segmentation          | Sentinel-2 Optical                           | 4000 / 1000 / 2000    | 2 (binary)        | CC-BY-SA             | [Kerner et al. 2025](https://arxiv.org/abs/2409.16252)        |
| KuroSiwo               | Flood segmentation                   | Sentinel-1 SAR + DEM + Slope                 | 4000 / 1000 / 2000    | 4                 | MIT                  | [Bountos et al. 2024](https://arxiv.org/abs/2311.12056)       |
| PASTIS (R)             | Crop type + parcel segmentation      | Sentinel-1 (asc/desc) + Sentinel-2 time series| 1455 / 482 / 496      | 19 (18 crops + bg)| CC BY 4.0            | [Garnot et al. 2022](https://arxiv.org/abs/2112.07558)        |
| SpaceNet2              | Building footprint segmentation      | WorldView VHR Optical RGB                    | 4000 / 1000 / 2000    | 2 (binary)        | CC-BY-SA-4.0         | [Van Etten et al. 2018](https://arxiv.org/abs/2102.11958)     |
| SpaceNet7              | Building segmentation/tracking       | Planet RGB time series                       | 3888 / 652 / 1152     | 2 (binary)        | CC-BY-SA-4.0         | [Van Etten et al. 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Van_Etten_The_Multi-Temporal_Urban_Development_SpaceNet_Dataset_CVPR_2021_paper.html)     |
| EverWatch              | Bird object detection                | Aerial RGB                                   | 4429 / 500 / 196      | 9                 | CC0                  | [Garner et al. 2024](https://zenodo.org/records/11165946)        |
| m-nzcattle             | Cattle object detection              | Aerial RGB                                   | 524 / 66 / 65         | 2                 | CC-BY-4.0            | [NZ Cattle](https://zenodo.org/records/5908869)                |
| Substation             | Power substation segmentation        | Sentinel-2 Optical + OSM                     | 4000 / 500 / 500      | 2 (binary)        | CC-BY-4.0 / ODbL 1.0 | [Lindsay et al. 2024](https://arxiv.org/abs/2409.17363)        |
| So2Sat  | Local Climate Zones          | Sentinel-1 SAR (VV,VH) + Sentinel-2 Optical        | 19992/986/986  | 17                | CC-BY-4.0   | [Lacoste et al. 2023](https://arxiv.org/abs/2306.03831)      |


## Geographical Distribution of Datasets

```{figure} _assets/global_coverage_map.png
:alt: Global Sample Distribution
:width: 100%
:align: center

Global Sample Distribution
```

## Geographical Distribution across Continents

```{figure} _assets/global_coverage_bar.png
:alt: Global Sample Distribution Continets
:width: 80%
:align: center

Global Coverage Distribution Continets
```

```{toctree}
:maxdepth: 1

dataset_notebooks/index
normalization/index
api/index
GitHub Repository <https://github.com/The-AI-Alliance/GEO-Bench-2>
```