"""
dataset.py – PyTorch dataset for KBO injection cases.

Each .npz case contains:
    X           : (T, H, W) float32  — patch + injected signal
    Y           : (T, H, W) float32  — injected signal only (label)
    patch_stack : (T, H, W) float32  — background patch (used to reconstruct X if absent)
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class CaseDataset(Dataset):
    """
    Dataset of .npz injection cases.

    Parameters
    ----------
    case_dir : str or Path — directory containing case_*.npz files
    use_X_if_present : bool — if True and 'X' key exists, use it directly;
                              otherwise reconstruct X = patch_stack + Y
    mode : str — "2d" (default, returns (T,H,W)) or "3d" (returns (1,T,H,W))
    normalize : str or None — None (default) or "per_case" to zero-mean / unit-std X
    """

    def __init__(
        self,
        case_dir: str | Path,
        use_X_if_present: bool = True,
        mode: str = "2d",
        normalize: str | None = None,
    ) -> None:
        self.paths = sorted(Path(case_dir).glob("case_*.npz"))
        if not self.paths:
            raise ValueError(f"No case_*.npz files found in {case_dir}")
        self.use_X_if_present = use_X_if_present
        if mode not in ("2d", "3d"):
            raise ValueError(f"mode must be '2d' or '3d', got '{mode}'")
        self.mode = mode
        if normalize is not None and normalize != "per_case":
            raise ValueError(f"normalize must be None or 'per_case', got '{normalize}'")
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (X, Y) float32 tensors.

        Shape depends on *mode*:
          - "2d": (T, H, W)
          - "3d": (1, T, H, W)   ← channel-first for Conv3d
        """
        with np.load(self.paths[idx], allow_pickle=True) as data:
            Y = data["Y"].astype(np.float32)  # (T, H, W)
            if self.use_X_if_present and "X" in data:
                X = data["X"].astype(np.float32)
            else:
                X = (data["patch_stack"] + Y).astype(np.float32)

        # Optional per-case normalization (X only)
        if self.normalize == "per_case":
            mu = X.mean()
            std = X.std() + 1e-8
            X = (X - mu) / std

        X_t = torch.from_numpy(X)
        Y_t = torch.from_numpy(Y)

        # Add channel dimension for 3D mode
        if self.mode == "3d":
            X_t = X_t.unsqueeze(0)  # (1, T, H, W)
            Y_t = Y_t.unsqueeze(0)  # (1, T, H, W)

        return X_t, Y_t
