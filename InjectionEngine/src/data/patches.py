"""
patches.py – Grid-based patch extraction and per-patch noise estimation.

extract_patch_grid: yields (patch_stack, row_start, col_start) for every
  non-overlapping (or strided) patch in the image stack.

patch_sigma: estimates per-epoch noise in a patch via MAD.
"""
from __future__ import annotations

from typing import Generator
import numpy as np


def extract_patch_grid(
    imgs: np.ndarray,
    patch_size: int,
    stride: int,
) -> Generator[tuple[np.ndarray, int, int], None, None]:
    """
    Yield (patch_stack, row_start, col_start) for all valid grid positions.

    Parameters
    ----------
    imgs       : float32 array, shape (T, H, W)
    patch_size : edge length of the square patch in pixels
    stride     : step size between patch centres

    Yields
    ------
    patch_stack : float32 array, shape (T, patch_size, patch_size) — copy
    row_start   : int — top-left row of this patch in imgs
    col_start   : int — top-left col of this patch in imgs
    """
    T, H, W = imgs.shape
    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            yield imgs[:, r : r + patch_size, c : c + patch_size].copy(), r, c


def patch_sigma(patch_stack: np.ndarray) -> np.ndarray:
    """
    Estimate per-epoch noise in a patch via MAD × 1.4826.

    Parameters
    ----------
    patch_stack : float32 array, shape (T, P, P)

    Returns
    -------
    sigma : float32 array, shape (T,)
    """
    T = patch_stack.shape[0]
    sigma = np.empty(T, dtype=np.float32)
    for t in range(T):
        mad = np.median(np.abs(patch_stack[t]))
        sigma[t] = float(mad * 1.4826)
    return sigma
