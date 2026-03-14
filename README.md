# KuiperHunter

KuiperHunter is a computer vision pipeline for finding faint, moving celestial objects (like Kuiper Belt Objects) in high-noise astronomical image sequences. Traditional approaches like Shift-and-Stack (`kbmod`) are computationally expensive. This project formulates the problem as a 3D spatio-temporal segmentation task, utilizing a custom **3D Minimal U-Net** to isolate moving signals from deep background noise.

## 3D Minimal U-Net Architecture

The core of our detection engine is a highly optimized, lightweight 3D U-Net designed to run efficiently even on CPUs (~20K parameters).

Instead of treating time as just another 2D channel, the 3D U-Net naturally processes volumetric data `(Time, Height, Width)`. This allows the network to learn rich spatio-temporal features and track moving objects across frames.

- **Encoder:** Uses 3×3×3 convolutions with BatchNorm and ReLU.
- **Pooling (Spatial Only):** The downsampling operations use a `(1, 2, 2)` max-pooling kernel. This intentionally preserves the temporal dimension (`T`) intact while compressing the spatial domain, ensuring we don't lose the temporal evolution of the trajectory.
- **Decoder:** Upsamples the spatial dimensions using trilinear interpolation, concatenates skip connections from the encoder, and resolves the feature maps back to the original `(T, H, W)` shape.
- **Output Head:** A 1×1×1 3D convolution outputs a continuous probability score (regression) for each pixel in the volume, highlighting the presence of a moving object.

## Directory Structure
- `InjectionEngine`: The core pipeline package. Handles synthetic data generation, simulation of moving-source trajectories (with PSF convolution and Poisson noise), model training, and inference.
- `kbmod`: Contains components for Kernel-Based Moving Object Detection (for comparison and baseline).
- `make_hero.py` / `pick_hero.py`: Utilities for generating specific validation edge cases (easy vs. faint positive cases, strict negatives) to benchmark the model's sensitivity limit.

## Installation

The primary package is located in `InjectionEngine`.

```bash
cd InjectionEngine
pip install -e .
```

### Optional Dependencies

```bash
# For development and testing
pip install -e ".[dev]"

# For model training and visual demonstrations
pip install -e ".[model]"
```

## Available Commands

Once installed, use the following CLI tools to run the pipeline:

- `kuiper-build-stack`: Generate a single synthetic image stack with complex noise profiles and an artificial trajectory.
- `kuiper-make-cases`: Sweep parameter spaces to build datasets for training and benchmarking.
- `kuiper-train`: Train the 3D U-Net (or the 2D Baseline).
- `kuiper-infer`: Run full volumetric inference across an image sequence.
