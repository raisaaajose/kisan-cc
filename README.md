# Deep Transfer Learning for Smallholder Yield Prediction

This repository contains the implementation of a **Spatio-Temporal Graph Attention Network (GAT)** designed for crop yield prediction in data-sparse regions. The framework utilizes differential spatial gating to enable robust cross-continental transfer learning, specifically from Argentina to Brazil.

This research directly supports **United Nations Sustainable Development Goal 2 (Zero Hunger)** by providing scalable agricultural monitoring tools for smallholder environments where ground-truth data is severely limited.

---

## Key Features

* **Spatio-Temporal GATv2:** Dynamically weighs signals from neighboring farms to mitigate local noise and high signal-to-noise ratio (SNR) challenges.
* **Differential Gating Mechanism:** Implements a learnable $\alpha$ parameter to balance local spectral features with regional spatial context, preventing over-smoothing.
* **Cross-Continental Transfer:** Recovers predictive power in regions where domestic training is infeasible due to extreme data scarcity (e.g., Brazil's 32 counties vs. Argentina's 135).
* **Spectral-Temporal Encoding:** Integrates MODIS multi-band histograms with NDVI time-series and one-hot encoded temporal signals.

---

## Methodology

The architecture processes multispectral satellite imagery through a CNN-based histogram encoder, which is then refined by a spatial graph regularizer.



### Spatial Graph Construction
A spatial graph is constructed using a **k-Nearest Neighbors (k-NN)** approach ($k=12$). Edge weights are determined via a Gaussian kernel:

$$W_{ij} = \exp\left(-\frac{d(i, j)^2}{2\sigma^2}\right)$$

### Differential Logic
The model preserves local signals by computing a differential representation ($\Delta_{h}$) between local county features ($h_{local}$) and gated regional signals ($h_{gated}$). This prevents the model from converging to indistinguishable regional means:

$$h_{combined} = \text{proj}(h_{local}) + (\alpha \cdot \Delta_{h})$$



---

## Results

Evaluation was conducted using the **SustainBench** dataset, focusing on soybean yields in Argentina and Brazil from 2005 to 2016.

### Performance Summary (Mean $R^2$)

| Model Configuration | Argentina (Domestic) | Brazil (Domestic) | Transfer (Arg $\to$ Bra) |
| :--- | :---: | :---: | :---: |
| Pixel-only Baseline ($k=0$) | 0.1918 | -0.4086 | 0.0732 |
| **Proposed GAT Model ($k=12$)** | **0.2075** | **-0.2043** | **0.2150** |

### Key Findings
* **193% Improvement:** The GAT-enabled transfer task outperformed the pixel-only baseline significantly ($R^2$ 0.2150 vs 0.0732).
* **Spatial Adaptation:** During transfer, the learnable $\alpha$ parameter adapted from 0.05 to 0.31, illustrating an increased reliance on spatial neighborhood signals to compensate for spectral drift between geographies.



---

## Project Structure

```text
project_root/
├── config/
│   └── params.yml      # Hyperparameters and task settings
├── src/
│   ├── models.py      # CustomGAT and Differential Gating logic
│   ├── train.py       # LOYO training loop and transfer logic
│   └── utils.py       # Data loading and graph construction
├── data/              # SustainBench npz files
├── models_saved/      # Checkpoints (best_gat.pth)
├── requirements.txt
└── Dockerfile