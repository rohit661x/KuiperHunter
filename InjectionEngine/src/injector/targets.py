"""
targets.py – Choose where in the patch to place the injection.

A "target" is a spatial starting position (start_x, start_y) for the source
at t=0.  Different strategies let callers control whether injections appear
near the centre, uniformly random, or at a specific location.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Literal


TargetStrategy = Literal["uniform", "center", "grid", "fixed"]


@dataclass
class TargetConfig:
    strategy: TargetStrategy = "uniform"
    # Used only when strategy == "fixed"
    fixed_x: float = 0.0
    fixed_y: float = 0.0
    # Used only when strategy == "grid" (divides patch into n×n cells)
    grid_n: int = 4
    # Fraction of patch width/height to use as margin (0 → no margin)
    margin: float = 0.05


def draw_target(
    patch_shape: tuple[int, int],
    config: TargetConfig,
    rng: np.random.Generator,
    grid_index: int | None = None,
) -> tuple[float, float]:
    """
    Return (start_x, start_y) in pixel coordinates.

    Parameters
    ----------
    patch_shape : (n_rows, n_cols)
    config      : TargetConfig instance.
    rng         : numpy random Generator.
    grid_index  : only used when strategy == "grid"; selects which cell to
                  sample from (wraps modulo grid_n**2).

    Returns
    -------
    (x, y) in pixel coordinates (column, row), both float.
    """
    n_rows, n_cols = patch_shape
    margin_x = config.margin * n_cols
    margin_y = config.margin * n_rows

    if config.strategy == "uniform":
        x = rng.uniform(margin_x, n_cols - margin_x)
        y = rng.uniform(margin_y, n_rows - margin_y)

    elif config.strategy == "center":
        cx, cy = n_cols / 2.0, n_rows / 2.0
        # Gaussian jitter ± 10 % of patch size
        sigma_x = n_cols * 0.10
        sigma_y = n_rows * 0.10
        x = float(np.clip(rng.normal(cx, sigma_x), margin_x, n_cols - margin_x))
        y = float(np.clip(rng.normal(cy, sigma_y), margin_y, n_rows - margin_y))

    elif config.strategy == "grid":
        n = config.grid_n
        if grid_index is None:
            grid_index = int(rng.integers(0, n * n))
        cell = grid_index % (n * n)
        row_cell = cell // n
        col_cell = cell % n
        cell_w = n_cols / n
        cell_h = n_rows / n
        x = rng.uniform(col_cell * cell_w, (col_cell + 1) * cell_w)
        y = rng.uniform(row_cell * cell_h, (row_cell + 1) * cell_h)
        x = float(np.clip(x, margin_x, n_cols - margin_x))
        y = float(np.clip(y, margin_y, n_rows - margin_y))

    elif config.strategy == "fixed":
        x, y = config.fixed_x, config.fixed_y

    else:
        raise ValueError(f"Unknown target strategy: '{config.strategy}'")

    return float(x), float(y)
