# KuiperHunter: Deep Learning for Low-SNR Moving Object Detection

A deep learning pipeline for detecting faint, slow-moving objects (like Kuiper Belt Objects) in astronomical image sequences. Developed as a mirror of the NRC "injection + detection + evaluation" workflow.

![Demo](demo_result.gif)

## Key Features
- **Realistic Injection Engine**: Generates synthetic movers with subpixel motion, PSF variation, and realistic noise (Poisson/Gaussian). Supports **Real FITS Backgrounds**.
- **Deep Learning Detector**: 3D U-Net (Spatiotemporal) architecture to detect motion below simple single-frame thresholds.
- **Tracking & Evaluation**: Connects frame-wise detections into tracklets and automated robustness analysis (Recall vs Magnitude).

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart

### 1. Run on Synthetic Data (Smoke Test)
Generate random noise images and detect movers.
```bash
python src/demo_pipeline.py --config config/smoke_test.yaml --view
```

### 2. Run on Real Data (Hard Mode)
Downloads a real FITS starfield (Horsehead Nebula sample) and injects faint movers.
```bash
python src/demo_pipeline.py --config config/real_data.yaml --view
```

### 3. Training
Train a fresh model from scratch:
```bash
python src/models/train.py --config config/default_simulation.yaml
```

## Project Structure
- `src/injection`: Data generation (PSF, Backgrounds, Noise).
- `src/models`: Deep Learning (3D U-Net, Focal Loss).
- `src/evaluation`: Tracking (Linear Greedy), Metrics, Robustness Sweeps.
- `notebooks/`: Interactive demos for each milestone.

## License
MIT
