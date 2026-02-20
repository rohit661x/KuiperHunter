"""
train.py – Train models on injection cases.

Supported models:
  baseline2d — Baseline2DNet (treats T frames as 2D channels)
  unet3d     — UNet3DMinimal (true 3D convs, CPU-friendly)

Run from repo root after pip install -e InjectionEngine/:

    kuiper-train --model unet3d \\
        --train_dir InjectionEngine/data/smoke_cases \\
        --epochs 2 --batch 1 \\
        --out_ckpt InjectionEngine/checkpoints/unet3d_local_best.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.dataset import CaseDataset
from model.model import Baseline2DNet, UNet3DMinimal

MODEL_CHOICES = ("baseline2d", "unet3d")


def train(
    train_dir: str,
    out_ckpt: str,
    model_type: str = "baseline2d",
    val_dir: str | None = None,
    epochs: int = 2,
    batch: int | None = None,
    lr: float = 1e-3,
    base_channels: int = 8,
) -> None:
    # Ensure parent directory for checkpoint exists
    ckpt_path = Path(out_ckpt)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    # Defaults per model type
    if batch is None:
        batch = 1 if model_type == "unet3d" else 4

    # Build datasets
    is_3d = model_type == "unet3d"
    ds_kwargs = dict(
        mode="3d" if is_3d else "2d",
        normalize="per_case" if is_3d else None,
    )
    train_dataset = CaseDataset(train_dir, **ds_kwargs)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch, num_workers=0)

    val_loader = None
    if val_dir is not None:
        val_dataset = CaseDataset(val_dir, **ds_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch, num_workers=0)

    # Infer T from first batch and build model
    first_X, _ = next(iter(train_loader))
    if model_type == "unet3d":
        model = UNet3DMinimal(base_channels=base_channels)
    else:
        T = first_X.shape[1]
        model = Baseline2DNet(n_frames=T)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_type}  params: {n_params:,}")

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_epoch = 1
    final_train_loss = None
    final_val_loss = None

    for e in range(1, epochs + 1):
        # Training loop
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        for X, Y in train_loader:
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, Y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_sum / train_batches
        final_train_loss = avg_train_loss
        print(f"Epoch {e}/{epochs}  train_loss={avg_train_loss:.6f}")

        # Validation loop (optional)
        avg_val_loss = None
        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_batches = 0
            with torch.no_grad():
                for X, Y in val_loader:
                    out = model(X)
                    loss = criterion(out, Y)
                    val_loss_sum += loss.item()
                    val_batches += 1
            avg_val_loss = val_loss_sum / val_batches
            final_val_loss = avg_val_loss
            print(f"Epoch {e}/{epochs}  val_loss={avg_val_loss:.6f}")

        # Track best epoch (use val_loss if available, else train_loss)
        epoch_loss = avg_val_loss if avg_val_loss is not None else avg_train_loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = e
            ckpt_data = {
                "model_state": model.state_dict(),
                "model_type": model_type,
                "epoch": best_epoch,
                "loss": best_loss,
            }
            if model_type == "unet3d":
                ckpt_data["base_channels"] = base_channels
            else:
                ckpt_data["n_frames"] = first_X.shape[1]
            torch.save(ckpt_data, ckpt_path)

    # Save JSON log file
    log_path = ckpt_path.parent / (ckpt_path.stem + "_log.json")
    log_data = {
        "model_type": model_type,
        "epochs": epochs,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "best_epoch": best_epoch,
    }
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

    print(f"Saved checkpoint → {out_ckpt}")
    print(f"Saved log → {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train model on injection cases.")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="baseline2d",
                        help="Model architecture (default: baseline2d)")
    parser.add_argument("--train_dir", required=True, help="Path to directory of case_*.npz training files")
    parser.add_argument("--val_dir", default=None, help="Path to directory of case_*.npz validation files")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs (default: 2)")
    parser.add_argument("--batch", type=int, default=None, help="Batch size (default: 4 for baseline2d, 1 for unet3d)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam optimizer (default: 1e-3)")
    parser.add_argument("--base_channels", type=int, default=8,
                        help="Base channel count for UNet3DMinimal (default: 8)")
    parser.add_argument("--out_ckpt", required=True, help="Path to save the best model checkpoint")
    args = parser.parse_args()

    train(
        train_dir=args.train_dir,
        out_ckpt=args.out_ckpt,
        model_type=args.model,
        val_dir=args.val_dir,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        base_channels=args.base_channels,
    )


if __name__ == "__main__":
    main()

