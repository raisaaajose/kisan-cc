# Deep Transfer Learning for Smallholder Yield Prediction

This repository contains the implementation of a **Spatio-Temporal Graph Attention Network (GAT)** designed for crop yield prediction in data-sparse regions. The framework utilizes differential spatial gating to enable robust cross-continental transfer learning, specifically from Argentina to Brazil.

This research directly supports **United Nations Sustainable Development Goal 2 (Zero Hunger)** by providing scalable agricultural monitoring tools for smallholder environments where ground-truth data is severely limited.

---

## Data Structure

The experiments are conducted using the [SustainBench Dataset](https://sustainlab-group.github.io/sustainbench/docs/datasets/sdg2/crop_yield.html), a benchmark for monitoring the UN Sustainable Development Goals with machine learning.


The input data consists of multispectral satellite imagery processed into multi-band histograms. Each county is represented by a $32 \times 32 \times 9$ tensor, where the 9 channels correspond to the MODIS spectral bands.

![App Screenshot](data/satellitebands.png)
![App Screenshot](data/satellitebands2.png)

The dataset structure follows a time-series of these histograms representing the crop growing season, coupled with NDVI (Normalized Difference Vegetation Index) values:


---

## Key Features

* **Spatio-Temporal GATv2:** Dynamically weighs signals from neighboring farms to mitigate local noise and high signal-to-noise ratio (SNR) challenges.
* **Differential Gating Mechanism:** Implements a learnable $\alpha$ parameter to balance local spectral features with regional spatial context, preventing over-smoothing.
* **Cross-Continental Transfer:** Recovers predictive power in regions where domestic training is infeasible due to extreme data scarcity.
* **Spectral-Temporal Encoding:** Integrates MODIS multi-band histograms with NDVI time-series and one-hot encoded temporal signals.

---

## Methodology

The architecture processes multispectral satellite imagery through a CNN-based histogram encoder, which is then refined by a spatial graph regularizer.

[Image of Graph Attention Network architecture for geospatial data]

### Spatial Graph Construction
A spatial graph is constructed using a **k-Nearest Neighbors (k-NN)** approach ($k=12$). Edge weights are determined via a Gaussian kernel:

$$W_{ij} = \exp\left(-\frac{d(i, j)^2}{2\sigma^2}\right)$$

### Differential Logic
The model preserves local signals by computing a differential representation ($\Delta_{h}$) between local county features ($h_{local}$) and gated regional signals ($h_{gated}$):

$$h_{combined} = \text{proj}(h_{local}) + (\alpha \cdot \Delta_{h})$$

---

## Results

Evaluation was conducted using soybean yields in Argentina and Brazil from 2005 to 2016.

### Performance Summary (Mean $R^2$)

| Model Configuration | Argentina (Domestic) | Brazil (Domestic) | Transfer (Arg $\to$ Bra) |
| :--- | :---: | :---: | :---: |
| Pixel-only Baseline ($k=0$) | 0.1918 | -0.4086 | 0.0732 |
| **Proposed GAT Model ($k=12$)** | **0.2075** | **-0.2043** | **0.2150** |

---

## Project Structure

```text
project_root/
├── config/
│   └── params.yml      # Hyperparameters and task settings
├── src/
│   ├── models.py      # CustomGAT and Differential Gating logic
│   ├── main.py        # Training loop and transfer logic
│   └── utils.py       # Data loading and graph construction
├── data/              # SustainBench files
├── models_saved/      # Checkpoints (best_gat.pth)
├── requirements.txt
└── Dockerfile