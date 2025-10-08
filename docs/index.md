# GEO-Bench-2

Welcome to the GeoBenchV2 documentation.

## Installation

```bash
pip install geobenchv2
```

## Dataset Overview

| Dataset                | Task(s)                              | Modalities / Sources                              | Train/Val/Test Samples | # Classes                      | License                  | Citation                                    | GSD / Pixel Size (Patch)                               |
|------------------------|--------------------------------------|---------------------------------------------------|------------------------|--------------------------------|--------------------------|---------------------------------------------|--------------------------------------------------------|
| BigEarthNetV2          | Multi-label land cover classification| Sentinel-1 SAR (VV,VH) + Sentinel-2 Optical       | 20000 / 4000 / 4000    | 19 (multi-label)               | CDLA-Permissive-1.0      | [Clasen et al. 2025](https://arxiv.org/abs/2407.03653) | 10 m; 120×120                                          |
| BurnScars (TBD)        | (TBD – likely segmentation / change) | Sentinel-2 (potential MODIS auxiliary)            | TBD                    | TBD                            | TBD                      | TBD                                         | 10–60 m (multi-scale); patch size TBD                  |
| BioMassters            | Biomass regression                   | Sentinel-1 SAR (VV,VH) + Sentinel-2 Optical       | 4000 / 1000 / 2000     | Continuous                     | CC-BY-4.0                | [Nascetti et al. 2023](https://openreview.net/pdf?id=hrWsIC4Cmz) | 10 m; 256×256                                          |
| CaFFe                  | Glacier zone segmentation            | Sentinel-1 SAR (1 ch)                             | 4000 / 1000 / 2000     | 4                              | CC-BY-4.0                | [Gourmelon et al. 2022](https://essd.copernicus.org/articles/14/4287/2022/) | 6–20 m (sensor dependent); 512×512 (TBD verify)        |
| CloudSen12             | Cloud and shadow segmentation        | Sentinel-2 L1C/L2A                                | 4000 / 1000 / 2000     | 4                              | CC0 1.0 (per source CSV) | TBD                                         | 10 / 20 / 60 m; 512×512                                |
| Dynamic EarthNet       | Land cover semantic segmentation     | Sentinel-2 (10 bands) + Planet (4 bands)          | 4000 / 1000 / 2000     | 7                              | CC-BY-4.0                | [Toker et al. 2022](https://arxiv.org/abs/2203.12560)  | 3 m (Planet), 10 m (S2); 256×256                       |
| EverWatch              | Bird object detection                | Aerial RGB                                        | N/A                    | 7                              | CC0 1.0 Universal        | [Garner et al. 2024](https://zenodo.org/records/11165946) | GSD TBD (aerial); 1500×1500                            |
| FLAIR2                 | Land cover semantic segmentation     | Aerial RGB+NIR + DEM (+ Sentinel-2 in project)    | 4000 / 1000 / 2000     | 13                             | Open Licence 2.0         | [Garioud et al. 2023](https://arxiv.org/abs/2305.14467) | 0.20 m (aerial) to 10 m (S2); 512×512                  |
| Fields of the World    | Field boundary segmentation          | Multi-temporal Sentinel-2 (10 bands)              | 4000 / 1000 / 2000     | 2 (binary)                     | CC-BY-4.0 (some CC-BY-SA region-specific – verify) | [Kerner et al. 2025](https://arxiv.org/abs/2409.16252) | 10 m; 256×256                                          |
| KuroSiwo               | Flood segmentation                   | Sentinel-1 SAR (VV,VH) + DEM + Slope              | 4000 / 1000 / 2000     | 4                              | CC-BY-4.0                | [Bountos et al. 2024](https://arxiv.org/abs/2311.12056) | 10 m (S1) + DEM (res TBD); 224×224                     |
| PASTIS (R)             | Crop type + parcel segmentation      | Sentinel-1 (asc/desc) + Sentinel-2 time series    | 1200 / 482 / 496       | 19 (18 crops + bg)             | CC-BY-4.0                | [Garnot et al. 2022](https://arxiv.org/abs/2112.07558) | 1.5–10 m (multi-sensor); 128×128                       |
| SpaceNet2              | Building footprint segmentation      | VHR Optical RGB                                   | 4000 / 1000 / 2000     | 2 (binary)                     | CC-BY-4.0                | [Van Etten et al. 2018](https://arxiv.org/abs/2102.11958) | ~0.5 m; 900×900 / 450×450 chips                        |
| SpaceNet7              | Building segmentation/tracking       | Planet RGB time series                            | N/A                    | 2 (binary)                     | CC-BY-4.0                | [Van Etten et al. 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Van_Etten_The_Multi-Temporal_Urban_Development_SpaceNet_Dataset_CVPR_2021_paper.html) | 4 m (PlanetScope approx.); 5000×5000 (pre-patch)       |
| Substation (TBD)       | Instance segmentation (power assets) | Sentinel-2 Optical                                | TBD                    | 2 (binary) (TBD verify)        | CC-BY-4.0 (TBD verify)    | TBD                                         | 10 m; 228×228                                          |
| NzCattle (TBD)         | Object detection (cattle)            | Aerial RGB (drone)                                | TBD                    | 2 (cattle vs background)       | CC-BY-4.0 (from source list) | TBD                                     | 0.1 m; 500×500                                         |
| TreeSatAI Time Series  | Tree species classification          | Multi-temporal Sentinel-2 (10 bands)              | 20000 / 4000 / 4000    | 13                             | CC-BY-4.0                | [Ahlswede et al. 2023](https://essd.copernicus.org/articles/15/681/2023/) | 10 m; (Extended variant includes 0.2 m aerial; 304×304) |

## Geographical Distribution of Datasets

```{figure} _static/global_distribution.png
:alt: Global Sample Distribution
:width: 100%
:align: center

Global Sample Distribution
```

## Geographical Distribution across Continents

```{figure} _static/global_coverage_bar.png
:alt: Global Sample Distribution Continets
:width: 100%
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