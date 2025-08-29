# GEO-Bench-2

Welcome to the GeoBenchV2 documentation.

## Installation

```bash
pip install geobenchv2
```

## Dataset Overview

| Dataset Name      | Task Type                | Modalities                                      |
|-------------------|-------------------------|-------------------------------------------------|
| BigEarthNetV2     | Multi-label Classification | Sentinel-1 SAR, Sentinel-2 Optical              |
| BioMassters       | Regression (Biomass Est.) | Sentinel-1 SAR, Sentinel-2 Optical              |
| CaFFe             | Segmentation             | Sentinel-1 SAR, Sentinel-2 Optical              |
| CloudSen12        | Segmentation             | Sentinel-2 Optical                              |
| FLAIR2            | Segmentation             | Aerial RGB+NIR, DEM, Sentinel-2 Optical         |
| Fields of The World (FoTW) | Segmentation    | Sentinel-2 Optical, Sentinel-1 SAR              |
| Dynamic EarthNet  | Spatio-temporal Forecasting | Sentinel-1 SAR, Sentinel-2 Optical, PlanetScope|
| EverWatch         | Object Detection         | Drone RGB                                       |
| KuroSiwo          | Segmentation             | Sentinel-1 SAR, DEM, Slope                      |
| M4SAR             | Classification           | Sentinel-1 SAR                                  |
| MMFlood           | Segmentation             | Sentinel-1 SAR, Sentinel-2 Optical              |
| PASTIS            | Segmentation             | Sentinel-1 SAR, Sentinel-2 Optical              |
| SpaceNet2         | Segmentation             | WorldView-3 Optical                             |
| SpaceNet6         | Segmentation             | SAR, WorldView-3 Optical                        |
| SpaceNet7         | Segmentation             | PlanetScope Optical                             |
| SpaceNet8         | Segmentation             | SAR, WorldView-3 Optical                        |
| Substation        | Object Detection         | Aerial RGB                                      |
| TreeSatAI         | Multi-label Classification          | Sentinel-2 Optical, Multi-temporal              |
| Wind Turbine      | Object Detection         | Aerial RGB                                      |
| BRIGHT            | Classification           | Sentinel-2 Optical                              |
| DOTAV2            | Object Detection         | Aerial RGB                                      |
| Burn Scars        | Segmentation             | Sentinel-2 Optical                              |
| NZCattle          | Object Detection         | Aerial RGB                                      |
| QFabric           | Classification           | Sentinel-2 Optical    




```{toctree}
:maxdepth: 1

dataset_tour/index
dataset_notebooks/index
api/index
GitHub Repository <https://github.com/The-AI-Alliance/GEO-Bench-2>
```