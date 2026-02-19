"""
infer.py – Run inference with Baseline2DNet on a single case.

Run from repo root after pip install -e InjectionEngine/:

    python -m model.infer \\
        --ckpt  InjectionEngine/checkpoints/baseline2d_best.pt \\
        --case  InjectionEngine/data/smoke_cases/case_0000.npz \\
        --out_png InjectionEngine/demo/step4_pred_case0000.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from model.model import Baseline2DNet


def infer(ckpt_path: str, case_path: str, out_png: str) -> None:
    # Step 1: Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Step 2: Reconstruct model from checkpoint
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

    # Convert to float32 tensors and add batch dimension: (1, T, H, W)
    X_tensor = torch.from_numpy(X).unsqueeze(0)  # (1, T, H, W)
    Y_tensor = torch.from_numpy(Y).unsqueeze(0)  # (1, T, H, W)

    # Step 4: Inference
    with torch.no_grad():
        Y_hat = model(X_tensor)

    # Step 5: Compute and print metrics
    mse = torch.mean((Y_tensor - Y_hat) ** 2).item()
    max_y = Y_tensor.max().item()
    max_y_hat = Y_hat.max().item()
    print(f"MSE(Y, Y_hat) = {mse:.6f}")
    print(f"max(Y)        = {max_y:.4f}")
    print(f"max(Y_hat)    = {max_y_hat:.4f}")

    # Step 6: Plot 3 panels for t = T // 2
    T = X_tensor.shape[1]
    t = T // 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Panel 0: X input frame
    axes[0].imshow(X_tensor[0, t].numpy(), cmap="gray")
    axes[0].set_title(f"X (input, frame {t})")

    # Panel 1: Y target frame
    axes[1].imshow(Y_tensor[0, t].numpy(), cmap="hot")
    axes[1].set_title(f"Y (target, frame {t})")

    # Panel 2: Y_hat predicted frame
    axes[2].imshow(Y_hat[0, t].numpy(), cmap="hot")
    axes[2].set_title(f"\u0176 (predicted, frame {t})")

    plt.tight_layout()

    # Step 7: Save PNG
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()

    # Step 8: Print save confirmation
    print(f"Saved \u2192 {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with Baseline2DNet on a single case.")
    parser.add_argument("--ckpt", required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--case", required=True, help="Path to .npz case file")
    parser.add_argument("--out_png", required=True, help="Path to save output PNG")
    args = parser.parse_args()

    infer(ckpt_path=args.ckpt, case_path=args.case, out_png=args.out_png)


if __name__ == "__main__":
    main()
