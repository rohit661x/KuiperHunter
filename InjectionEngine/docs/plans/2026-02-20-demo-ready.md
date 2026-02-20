# Demo-Ready Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make KuiperHunter demo-ready with a single `demo/run_demo.py` script that shows injection visualization and UNet3D inference on 3 hero cases, with interactive matplotlib pop-up plots.

**Architecture:** Two-section linear script — Section 1 loads a smoke_case npz and plots background/signal/input; Section 2 loads the trained UNet3D checkpoint and runs inference on easy_positive/hard_positive/negative hero cases, showing 3-panel figures with `plt.show()`. The script is self-contained (does not import from `infer.py`, which hardcodes the Agg backend) and resolves all paths relative to the script's own location.

**Tech Stack:** Python, PyTorch, NumPy, Matplotlib (interactive backend), existing model classes `UNet3DMinimal` from `InjectionEngine/src/model/model.py`

---

## Task 1: Clean git commit of all current work

**Files:**
- Modify (stage): `InjectionEngine/src/model/__init__.py`, `InjectionEngine/src/model/dataset.py`, `InjectionEngine/src/model/infer.py`, `InjectionEngine/src/model/model.py`, `InjectionEngine/src/model/train.py`, `InjectionEngine/tests/test_step4_smoke.py`
- Add (untracked): `InjectionEngine/checkpoints/unet3d_local_best.pt`, `InjectionEngine/checkpoints/unet3d_smoke.pt`, `InjectionEngine/checkpoints/unet3d_smoke_log.json`, `InjectionEngine/demo/unet3d_smoke_pred.png`, `InjectionEngine/demo/cases_demo/`, `InjectionEngine/demo/stacks/`, `make_hero.py`, `pick_hero.py`

**Step 1: Stage all modified and untracked files**

```bash
git add \
  InjectionEngine/src/model/__init__.py \
  InjectionEngine/src/model/dataset.py \
  InjectionEngine/src/model/infer.py \
  InjectionEngine/src/model/model.py \
  InjectionEngine/src/model/train.py \
  InjectionEngine/tests/test_step4_smoke.py \
  InjectionEngine/checkpoints/unet3d_local_best.pt \
  InjectionEngine/checkpoints/unet3d_smoke.pt \
  InjectionEngine/checkpoints/unet3d_smoke_log.json \
  InjectionEngine/demo/unet3d_smoke_pred.png \
  InjectionEngine/demo/cases_demo/ \
  InjectionEngine/demo/stacks/ \
  make_hero.py \
  pick_hero.py
```

**Step 2: Commit**

```bash
git commit -m "feat: demo-ready — Step 4 model updates, trained checkpoint, hero demo cases"
```

Expected: one commit, all files staged cleanly.

---

## Task 2: Write `demo/run_demo.py`

**Files:**
- Create: `InjectionEngine/demo/run_demo.py`

**Step 1: Write the script**

```python
"""
run_demo.py — KuiperHunter end-to-end demo.

Section 1: Injection visualization — loads a smoke case and shows
           Background | Signal | Input (mid-frame) as a pop-up plot.

Section 2: UNet3D inference — loads the trained checkpoint and runs
           on 3 hero cases (easy/hard/negative), showing
           Input | Target | Prediction per case.

Run from repo root:
    python InjectionEngine/demo/run_demo.py

Or from InjectionEngine/demo/:
    python run_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup — make model package importable regardless of CWD
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent          # InjectionEngine/demo/
IE_ROOT    = SCRIPT_DIR.parent                        # InjectionEngine/
SRC_DIR    = IE_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from model.model import UNet3DMinimal  # noqa: E402 (after sys.path setup)

# ---------------------------------------------------------------------------
# Fixed paths (relative to InjectionEngine/)
# ---------------------------------------------------------------------------
SMOKE_CASE  = IE_ROOT / "data" / "smoke_cases" / "case_0000.npz"
CKPT        = IE_ROOT / "checkpoints" / "unet3d_local_best.pt"
HERO_DIR    = IE_ROOT / "demo" / "cases_demo"
HERO_CASES  = [
    ("easy_positive",  HERO_DIR / "easy_positive.npz"),
    ("hard_positive",  HERO_DIR / "hard_positive.npz"),
    ("negative",       HERO_DIR / "negative.npz"),
]


# ---------------------------------------------------------------------------
# Section 1: Injection visualization
# ---------------------------------------------------------------------------
def section1_injection() -> None:
    print("\n[1/2] Injection Visualization")
    print(f"  Loading: {SMOKE_CASE}")

    data = np.load(SMOKE_CASE, allow_pickle=True)
    Y = data["Y"].astype(np.float32)           # (T, H, W) — signal only
    patch_stack = data["patch_stack"].astype(np.float32)  # (T, H, W) — background
    X = (patch_stack + Y)                      # (T, H, W) — combined input

    T = X.shape[0]
    t = T // 2
    sigma = float(data.get("sigma_patch", np.nan))
    flux_peak = float(Y.max())

    print(f"  Frames: {T}  |  Patch size: {X.shape[1]}×{X.shape[2]}")
    print(f"  Flux peak (Y): {flux_peak:.1f}  |  sigma_patch: {sigma:.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Section 1 — Injection Visualization (mid-frame)", fontsize=11)

    axes[0].imshow(patch_stack[t], cmap="gray")
    axes[0].set_title(f"Background (frame {t})")
    axes[0].axis("off")

    axes[1].imshow(Y[t], cmap="hot")
    axes[1].set_title(f"Signal / Y (frame {t})")
    axes[1].axis("off")

    axes[2].imshow(X[t], cmap="gray")
    axes[2].set_title(f"Input / X (frame {t})")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Section 2: UNet3D inference on hero cases
# ---------------------------------------------------------------------------
def _load_model(ckpt_path: Path) -> UNet3DMinimal:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    base_ch = ckpt.get("base_channels", 8)
    model = UNet3DMinimal(base_channels=base_ch)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _infer_case(model: UNet3DMinimal, case_path: Path) -> tuple:
    """Returns (X_orig, Y, Y_hat, mse, sigma) all as numpy arrays/floats."""
    data = np.load(str(case_path), allow_pickle=True)
    Y = data["Y"].astype(np.float32)
    if "X" in data:
        X = data["X"].astype(np.float32)
    else:
        X = (data["patch_stack"] + Y).astype(np.float32)

    sigma = float(data.get("sigma_patch", np.nan))

    # Per-case normalize (same as training)
    mu  = X.mean()
    std = X.std() + 1e-8
    X_norm = (X - mu) / std

    X_tensor = torch.from_numpy(X_norm).unsqueeze(0).unsqueeze(0)  # (1,1,T,H,W)
    Y_tensor = torch.from_numpy(Y).unsqueeze(0).unsqueeze(0)       # (1,1,T,H,W)

    with torch.no_grad():
        Y_hat_tensor = model(X_tensor)

    mse   = torch.mean((Y_tensor - Y_hat_tensor) ** 2).item()
    Y_hat = Y_hat_tensor[0, 0].numpy()   # (T, H, W)
    return X, Y, Y_hat, mse, sigma


def section2_inference() -> None:
    print("\n[2/2] UNet3D Inference on Hero Cases")
    print(f"  Checkpoint: {CKPT}")

    model = _load_model(CKPT)

    for name, case_path in HERO_CASES:
        print(f"\n  [{name}]")
        X, Y, Y_hat, mse, sigma = _infer_case(model, case_path)

        max_y    = float(Y.max())
        max_yhat = float(Y_hat.max())
        snr      = f"{max_y / sigma:.1f}" if max_y > 0 and sigma > 0 else "—"

        print(f"    MSE={mse:.4f}  max(Y)={max_y:.1f}  max(Ŷ)={max_yhat:.1f}  SNR={snr}")

        T = X.shape[0]
        t = T // 2

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(f"Section 2 — {name} (mid-frame, SNR={snr})", fontsize=11)

        axes[0].imshow(X[t], cmap="gray")
        axes[0].set_title(f"Input / X (frame {t})")
        axes[0].axis("off")

        axes[1].imshow(Y[t], cmap="hot")
        axes[1].set_title(f"Target / Y (frame {t})")
        axes[1].axis("off")

        axes[2].imshow(Y_hat[t], cmap="hot")
        axes[2].set_title(f"Prediction / Ŷ (frame {t})")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== KuiperHunter Demo ===")
    section1_injection()
    section2_inference()
    print("\nDemo complete.")
```

**Step 2: Verify the script is syntactically valid**

```bash
python -c "import ast; ast.parse(open('InjectionEngine/demo/run_demo.py').read()); print('Syntax OK')"
```

Expected output: `Syntax OK`

**Step 3: Run the script (requires display)**

```bash
python InjectionEngine/demo/run_demo.py
```

Expected console output:
```
=== KuiperHunter Demo ===

[1/2] Injection Visualization
  Loading: .../InjectionEngine/data/smoke_cases/case_0000.npz
  Frames: 5  |  Patch size: 64×64
  Flux peak (Y): ...  |  sigma_patch: ...

[2/2] UNet3D Inference on Hero Cases
  Checkpoint: .../InjectionEngine/checkpoints/unet3d_local_best.pt

  [easy_positive]
    MSE=...  max(Y)=...  max(Ŷ)=...  SNR=...

  [hard_positive]
    MSE=...  max(Y)=...  max(Ŷ)=...  SNR=...

  [negative]
    MSE=...  max(Y)=...  max(Ŷ)=...  SNR=...

Demo complete.
```

4 pop-up windows open sequentially (close each to advance).

**Step 4: Commit**

```bash
git add InjectionEngine/demo/run_demo.py
git commit -m "feat: add demo/run_demo.py — injection + inference walkthrough"
```

---

## Task 3: Verify existing tests still pass

**Step 1: Run smoke tests**

```bash
cd InjectionEngine && python -m pytest tests/test_step4_smoke.py -v
```

Expected: 8 passed, 0 failed.

**Step 2: Run all tests (optional full check)**

```bash
cd InjectionEngine && python -m pytest tests/ -v --tb=short
```

Expected: ~115 passed, ≤2 skipped.
