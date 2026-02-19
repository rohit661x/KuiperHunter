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
    """

    def __init__(self, case_dir: str | Path, use_X_if_present: bool = True) -> None:
        self.paths = sorted(Path(case_dir).glob("case_*.npz"))
        if not self.paths:
            raise ValueError(f"No case_*.npz files found in {case_dir}")
        self.use_X_if_present = use_X_if_present

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = np.load(self.paths[idx], allow_pickle=True)
        Y = data["Y"].astype(np.float32)  # (T, H, W)
        if self.use_X_if_present and "X" in data:
            X = data["X"].astype(np.float32)
        else:
            X = (data["patch_stack"] + Y).astype(np.float32)
        return torch.from_numpy(X), torch.from_numpy(Y)
