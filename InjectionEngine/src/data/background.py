"""
background.py – Per-frame background subtraction and noise estimation.

Background: per-frame median (robust to compact sources).
Sigma:      MAD × 1.4826  (= robust estimate of Gaussian sigma).
"""
from __future__ import annotations

import numpy as np


def subtract_background(
    imgs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Subtract per-frame background and estimate per-frame noise.

    Parameters
    ----------
    imgs : float32 array, shape (T, H, W)

    Returns
    -------
    imgs_bgsub  : float32 array, shape (T, H, W)
    sigma_frame : float32 array, shape (T,) — robust sigma per frame
    """
    T = imgs.shape[0]
    imgs_out = np.empty_like(imgs)
    sigma = np.empty(T, dtype=np.float32)

    for t in range(T):
        bg = np.median(imgs[t])
        residual = imgs[t] - bg
        imgs_out[t] = residual
        mad = np.median(np.abs(residual))
        sigma[t] = float(mad * 1.4826)

    return imgs_out, sigma
