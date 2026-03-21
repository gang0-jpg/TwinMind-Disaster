# TwinMind-Disaster

## AI Digital Twin for Terrain-Aware Flood Prediction

TwinMind-Disaster is a prototype system that integrates terrain data, rainfall simulation, and deep learning to predict flood distribution and support observation strategies.

This project extends AI Digital Twin concepts from **prediction to decision support**.

---

## Overview

The system performs the following:

* Predict flood depth from terrain (DEM) and rainfall
* Estimate uncertainty in model predictions
* Use uncertainty to guide sensor placement

This system places sensors where the model is uncertain.

---

## Background

Conventional flood prediction approaches:

* Focus primarily on improving prediction accuracy
* Do not explicitly quantify uncertainty
* Do not connect prediction results to sensor deployment

TwinMind-Disaster addresses these gaps by integrating uncertainty into the decision loop.

---

## Approach

Predict → Uncertainty → Sensor Placement → Observe → Update

This closed loop enables adaptive and efficient observation.

---

## Core Components

| Component           | Role             | Description                                                |
| ------------------- | ---------------- | ---------------------------------------------------------- |
| U-Net               | Spatial modeling | Flood depth prediction from terrain and rainfall           |
| Reservoir Computing | Feature support  | Auxiliary temporal representation for structural stability |
| MC Dropout          | Uncertainty      | Mean and variance estimation via Monte Carlo inference     |
| Sensor Placement    | Decision layer   | Optimize observation points based on uncertainty           |

---

## Results

| Metric            | Improvement                    |
| ----------------- | ------------------------------ |
| MAE               | -16.4%                         |
| IoU               | +25.6%                         |
| Sensor Efficiency | +89.4% (6.68× efficiency gain) |
| Uncertainty       | 3.1× reduction per cycle       |

---

## System Flow

Terrain + Rainfall
↓
U-Net + Reservoir
↓
Flood Prediction
↓
MC Dropout
↓
Uncertainty Map
↓
Sensor Placement
↓
Observation
↓
Update

---

## Visualization & Dashboard

Run locally:

```bash
streamlit run ui/twinmind_dashboard.py
```

The dashboard visualizes:

* DEM terrain
* Flood prediction map
* Uncertainty map (MC Dropout)
* Sensor placement decisions

---

## Implementation

* PyTorch
* Reservoir module
* MC Dropout
* Streamlit
* MLflow

---

## Environment

* NVIDIA A5000 (local inference)
* io.net / io.cloud (H100 GPU infrastructure)

---

## Data Source

Geospatial Information Authority of Japan (GSI)
https://www.gsi.go.jp/kiban/

* Dataset: Numerical Elevation Model (DEM)
* Map reference: GSI Maps

---

## Project Structure

```
TwinMind-Disaster/
├── data/
├── scripts/
├── docs/
├── ui/
├── models/
├── README.md
└── requirements.txt
```

---

## Demo

https://gang0-jpg.github.io/TwinMind-Disaster/

---

## License

MIT License

---

## Author

Zenji Oka
https://github.com/gang0-jpg
