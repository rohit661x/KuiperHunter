"""
infer.py – Run inference on a single case.

Supported models:
  baseline2d — Baseline2DNet
  unet3d     — UNet3DMinimal

Run from repo root after pip install -e InjectionEngine/:

    kuiper-infer --model unet3d \\
        --ckpt  InjectionEngine/checkpoints/unet3d_local_best.pt \\
        --case  InjectionEngine/data/smoke_cases/case_0000.npz \\
        --out_png InjectionEngine/demo/unet3d_pred.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from model.model import Baseline2DNet, UNet3DMinimal

MODEL_CHOICES = ("baseline2d", "unet3d")


def infer(ckpt_path: str, case_path: str, out_png: str, model_type: str | None = None) -> None:
    # Step 1: Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Auto-detect model type from checkpoint (fall back to CLI arg or baseline2d)
    ckpt_model_type = ckpt.get("model_type", "baseline2d")
    if model_type is None:
        model_type = ckpt_model_type

    # Step 2: Reconstruct model from checkpoint
    if model_type == "unet3d":
        base_ch = ckpt.get("base_channels", 8)
        model = UNet3DMinimal(base_channels=base_ch)
    else:
        model = Baseline2DNet(n_frames=ckpt["n_frames"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Step 3: Load case from npz
    data = np.load(case_path, allow_pickle=True)
    Y = data["Y"].astype(np.float32)  # (T, H, W)
    if "X" in data:
        X = data["X"].astype(np.float32)
    else:
        X = (data["patch_stack"] + Y).astype(np.float32)

    # Step 4: Prepare tensors based on model type
    if model_type == "unet3d":
        # Per-case normalize X (same as training)
        mu = X.mean()
        std = X.std() + 1e-8
        X_norm = (X - mu) / std
        X_tensor = torch.from_numpy(X_norm).unsqueeze(0).unsqueeze(0)  # (1, 1, T, H, W)
        Y_tensor = torch.from_numpy(Y).unsqueeze(0).unsqueeze(0)      # (1, 1, T, H, W)
    else:
        X_tensor = torch.from_numpy(X).unsqueeze(0)  # (1, T, H, W)
        Y_tensor = torch.from_numpy(Y).unsqueeze(0)  # (1, T, H, W)

    # Step 5: Inference
    with torch.no_grad():
        Y_hat = model(X_tensor)

    # Step 6: Compute and print metrics
    mse = torch.mean((Y_tensor - Y_hat) ** 2).item()
    max_y = Y_tensor.max().item()
    max_y_hat = Y_hat.max().item()
    print(f"MSE(Y, Ŷ)    = {mse:.6f}")
    print(f"max(Y)        = {max_y:.4f}")
    print(f"max(Ŷ)        = {max_y_hat:.4f}")

    # Step 7: Flatten to (T, H, W) arrays for plotting
    if model_type == "unet3d":
        X_plot = X  # original un-normalized for display
        Y_plot = Y_tensor[0, 0].numpy()
        Yh_plot = Y_hat[0, 0].numpy()
    else:
        X_plot = X_tensor[0].numpy()
        Y_plot = Y_tensor[0].numpy()
        Yh_plot = Y_hat[0].numpy()

    T = X_plot.shape[0]
    t = T // 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Panel 0: X input frame
    axes[0].imshow(X_plot[t], cmap="gray")
    axes[0].set_title(f"X (input, frame {t})")

    # Panel 1: Y target frame
    axes[1].imshow(Y_plot[t], cmap="hot")
    axes[1].set_title(f"Y (target, frame {t})")

    # Panel 2: Y_hat predicted frame
    axes[2].imshow(Yh_plot[t], cmap="hot")
    axes[2].set_title(f"Ŷ (predicted, frame {t})")

    plt.suptitle(f"Model: {model_type}", fontsize=10)
    plt.tight_layout()

    # Step 8: Save PNG
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()

    print(f"Saved → {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on a single case.")
    parser.add_argument("--model", choices=MODEL_CHOICES, default=None,
                        help="Model architecture (auto-detected from checkpoint if omitted)")
    parser.add_argument("--ckpt", required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--case", required=True, help="Path to .npz case file")
    parser.add_argument("--out_png", required=True, help="Path to save output PNG")
    args = parser.parse_args()

    infer(ckpt_path=args.ckpt, case_path=args.case, out_png=args.out_png, model_type=args.model)


if __name__ == "__main__":
    main()
