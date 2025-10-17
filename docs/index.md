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

| Dataset                | Task                                 | Modalities                                   | Train/Val/Test Samples | # Classes         | License        | Citation                  |
|------------------------|--------------------------------------|----------------------------------------------|-----------------------|-------------------|-----------------|---------------------------|
| BigEarthNetV2          | Multi-label land cover classification| Sentinel-1 SAR (VV,VH) + Sentinel-2 Optical  | 20000 / 4000 / 4000   | 19 (multi-label)  | CDLA-Permissive-1.0   | [Clasen et al. 2025](https://arxiv.org/abs/2407.03653)        |
| BioMassters            | Biomass regression                   | Sentinel-1 SAR (VV,VH) + Sentinel-2 Optical  | 4000 / 1000 / 2000    | Continuous        | CC-BY-4.0   | [Nascetti et al. 2023](https://openreview.net/pdf?id=hrWsIC4Cmz)      |
| CaFFe                  | Glacier zone segmentation            | Sentinel-1 SAR (1 ch)                        | 4000 / 1000 / 2000    | 4                 | CC-BY-4.0   | [Gourmelon et al. 2022](https://essd.copernicus.org/articles/14/4287/2022/)     |
| Dynamic EarthNet       | Land cover semantic segmentation     | Sentinel-2 (10 bands) + Planet (4 bands)     | 4000 / 1000 / 2000    | 7                 | CC-BY-4.0   | [Toker et al. 2022](https://arxiv.org/abs/2203.12560)         |
| EverWatch              | Bird object detection                | Aerial RGB                                   | N/A                   | 7                 | CC0 1.0 Universal   | [Garner et al. 2024](https://zenodo.org/records/11165946)        |
| FLAIR2                 | Land cover semantic segmentation     | Aerial RGB+NIR + DEM                         | 4000 / 1000 / 2000    | 13                | OPEN LICENCE 2.0   | [Garioud et al. 2023](https://arxiv.org/abs/2305.14467)       |
| Fields of the World    | Field boundary segmentation          | Multi-temporal Sentinel-2 (10 bands)         | 4000 / 1000 / 2000    | 2 (binary)        | CC-BY-4.0  | [Kerner et al. 2025](https://arxiv.org/abs/2409.16252)        |
| KuroSiwo               | Flood segmentation                   | Sentinel-1 SAR (VV,VH) + DEM + Slope         | 4000 / 1000 / 2000    | 4                 | CC-BY-4.0    | [Bountos et al. 2024](https://arxiv.org/abs/2311.12056)       |
| PASTIS (R)             | Crop type + parcel segmentation      | Sentinel-1 (asc/desc) + Sentinel-2 time series| 1200 / 482 / 496      | 19 (18 crops + bg)| CC BY 4.0   | [Garnot et al. 2022](https://arxiv.org/abs/2112.07558)        |
| SpaceNet2              | Building footprint segmentation      | VHR Optical RGB                              | 4000 / 1000 / 2000    | 2 (binary)        | CC-BY-4.0   | [Van Etten et al. 2018](https://arxiv.org/abs/2102.11958)     |
| SpaceNet7              | Building segmentation/tracking       | Planet RGB time series                       | N/A                   | 2 (binary)        | CC-BY-4.0   | [Van Etten et al. 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Van_Etten_The_Multi-Temporal_Urban_Development_SpaceNet_Dataset_CVPR_2021_paper.html)     |
| TreeSatAI Time Series  | Tree species classification          | Multi-temporal Sentinel-2 (10 bands)         | N/A                   | 13                | CC-BY-4.0   | [Ahlswede et al. 2023](https://essd.copernicus.org/articles/15/681/2023/)      |

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