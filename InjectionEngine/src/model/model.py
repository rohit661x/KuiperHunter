"""
model.py – 2D baseline model treating time frames as channels.

Input:  (B, T, H, W)
Output: (B, T, H, W)  — predicted injected signal Y_hat (non-negative)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Baseline2DNet(nn.Module):
    """
    Lightweight 2D CNN baseline for KBO signal detection.

    Treats T time frames as 2D channels. Keeps it small for fast CPU runs.

    Architecture:
        Conv2d(T → 32, 3×3, padding=1) → ReLU
        Conv2d(32 → 32, 3×3, padding=1) → ReLU
        Conv2d(32 → T, 3×3, padding=1) → ReLU   ← final ReLU keeps output >= 0

    Parameters
    ----------
    n_frames : int  (default 5) — T, the number of time frames = number of input channels
    """

    def __init__(self, n_frames: int = 5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_frames, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_frames, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),  # clamp output >= 0, consistent with Y >= 0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, H, W) float32

        Returns
        -------
        torch.Tensor (B, T, H, W) float32, non-negative
        """
        return self.net(x)
