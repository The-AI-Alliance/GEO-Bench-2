# GEO-Bench-2

Welcome to the GeoBenchV2 documentation.

## Installation

```bash
pip install geobenchv2
```

## Dataset Overview

| Dataset                | Task                                 | Modalities                                   | Train/Val/Test Samples | # Classes         | License        | Citation                  |
|------------------------|--------------------------------------|----------------------------------------------|-----------------------|-------------------|-----------------|---------------------------|
| BigEarthNetV2          | Multi-label land cover classification| Sentinel-1 SAR (VV,VH) + Sentinel-2 Optical  | 20000 / 4000 / 4000   | 19 (multi-label)  | Not specified   | Clasen et al. 2025        |
| BioMassters            | Biomass regression                   | Sentinel-1 SAR (VV,VH) + Sentinel-2 Optical  | 4000 / 1000 / 2000    | Continuous        | Not specified   | Nascetti et al. 2023      |
| CaFFe                  | Glacier zone segmentation            | Sentinel-1 SAR (1 ch)                        | 4000 / 1000 / 2000    | 4                 | Not specified   | Gourmelon et al. 2022     |
| Dynamic EarthNet       | Land cover semantic segmentation     | Sentinel-2 (10 bands) + Planet (4 bands)     | 4000 / 1000 / 2000    | 7                 | Not specified   | Toker et al. 2022         |
| EverWatch              | Bird object detection                | Aerial RGB                                   | N/A                   | 7                 | Not specified   | Garner et al. 2024        |
| FLAIR2                 | Land cover semantic segmentation     | Aerial RGB+NIR + DEM                         | 4000 / 1000 / 2000    | 13                | Not specified   | Garioud et al. 2023       |
| Fields of the World    | Field boundary segmentation          | Multi-temporal Sentinel-2 (10 bands)         | 4000 / 1000 / 2000    | 2 (binary)        | CC-BY (subset)  | Kerner et al. 2025        |
| KuroSiwo               | Flood segmentation                   | Sentinel-1 SAR (VV,VH) + DEM + Slope         | 4000 / 1000 / 2000    | 4                 | Not specified   | Bountos et al. 2024       |
| PASTIS (R)             | Crop type + parcel segmentation      | Sentinel-1 (asc/desc) + Sentinel-2 time series| 1200 / 482 / 496      | 19 (18 crops + bg)| Not specified   | Garnot et al. 2022        |
| SpaceNet2              | Building footprint segmentation      | VHR Optical RGB                              | 4000 / 1000 / 2000    | 2 (binary)        | Not specified   | Van Etten et al. 2018     |
| SpaceNet7              | Building segmentation/tracking       | Planet RGB time series                       | N/A                   | 2 (binary)        | Not specified   | Van Etten et al. 2021     |
| TreeSatAI Time Series  | Tree species classification          | Multi-temporal Sentinel-2 (10 bands)         | N/A                   | 13                | Not specified   | Ahlswede et al. 2023      |



```{toctree}
:maxdepth: 1

dataset_tour/index
dataset_notebooks/index
normalization/index
api/index
GitHub Repository <https://github.com/The-AI-Alliance/GEO-Bench-2>
```