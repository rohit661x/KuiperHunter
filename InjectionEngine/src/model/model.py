"""
model.py – Models for KBO signal detection.

Baseline2DNet — 2D CNN treating time frames as channels.
UNet3DMinimal — Lightweight 3D U-Net for volumetric (T, H, W) data.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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


# ---------------------------------------------------------------------------
# 3D U-Net
# ---------------------------------------------------------------------------

def _conv_block_3d(in_ch: int, out_ch: int) -> nn.Sequential:
    """Two 3×3×3 conv layers with BatchNorm + ReLU."""
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet3DMinimal(nn.Module):
    """
    Minimal 3D U-Net for KBO signal detection.

    Designed to run on CPU:
      • base_channels = 8  → ~20 K parameters
      • depth = 2 downsamples (spatial only)
      • pooling kernel (1, 2, 2) → preserves T dimension

    Input : (B, 1, T, H, W) float32
    Output: (B, 1, T, H, W) float32  — no activation on head (regression)

    Parameters
    ----------
    base_channels : int — number of channels after first encoder block (default 8)
    """

    def __init__(self, base_channels: int = 8) -> None:
        super().__init__()
        c = base_channels

        # Encoder
        self.enc1 = _conv_block_3d(1, c)        # → c
        self.enc2 = _conv_block_3d(c, c * 2)    # → 2c

        # Bottleneck
        self.bottleneck = _conv_block_3d(c * 2, c * 4)  # → 4c

        # Spatial-only pooling: keep T intact
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2))

        # Decoder
        self.dec2 = _conv_block_3d(c * 4 + c * 2, c * 2)  # skip + up
        self.dec1 = _conv_block_3d(c * 2 + c, c)           # skip + up

        # 1×1×1 head — regression, no activation
        self.head = nn.Conv3d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, T, H, W) float32

        Returns
        -------
        torch.Tensor (B, 1, T, H, W) float32
        """
        # Encoder
        e1 = self.enc1(x)                 # (B, c,  T, H,   W)
        e2 = self.enc2(self.pool(e1))     # (B, 2c, T, H/2, W/2)

        # Bottleneck
        b = self.bottleneck(self.pool(e2))  # (B, 4c, T, H/4, W/4)

        # Decoder — upsample spatial only, then concat skip
        d2 = F.interpolate(b, size=e2.shape[2:], mode="trilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))  # (B, 2c, T, H/2, W/2)

        d1 = F.interpolate(d2, size=e1.shape[2:], mode="trilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))  # (B, c, T, H, W)

        return self.head(d1)  # (B, 1, T, H, W)

