"""
trajectory.py – Convert motion parameters into per-frame (x, y) positions.

All positions are in pixel coordinates relative to the top-left corner of the
patch.  Sub-pixel precision is preserved so that render_psf can apply
fractional-pixel shifts correctly.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class Trajectory:
    """Pixel positions for each frame in the stack."""
    xs: np.ndarray   # shape (n_frames,)  float64
    ys: np.ndarray   # shape (n_frames,)  float64

    def __len__(self) -> int:
        return len(self.xs)

    def as_array(self) -> np.ndarray:
        """Return shape (n_frames, 2) array of (x, y) pairs."""
        return np.stack([self.xs, self.ys], axis=1)


def build_trajectory(
    timestamps: np.ndarray,
    start_x: float,
    start_y: float,
    motion_ra: float,
    motion_dec: float,
    plate_scale: float,
) -> Trajectory:
    """
    Compute per-frame pixel positions for a linearly moving source.

    Parameters
    ----------
    timestamps  : (n_frames,) float array – observation times in hours from
                  the first epoch (t=0 at frame 0).
    start_x     : sub-pixel column position at t=0.
    start_y     : sub-pixel row position at t=0.
    motion_ra   : RA proper motion in arcsec / hour (positive = east).
    motion_dec  : Dec proper motion in arcsec / hour (positive = north).
    plate_scale : arcsec / pixel.

    Returns
    -------
    Trajectory with xs, ys in pixel coordinates (may be fractional).
    """
    timestamps = np.asarray(timestamps, dtype=np.float64)
    dt = timestamps - timestamps[0]

    # Convert arcsec/hour → pixels/hour, then multiply by elapsed time
    xs = start_x + (motion_ra / plate_scale) * dt
    ys = start_y - (motion_dec / plate_scale) * dt  # Dec increases upward

    return Trajectory(xs=xs, ys=ys)


def is_in_patch(traj: Trajectory, patch_shape: tuple[int, int]) -> np.ndarray:
    """
    Boolean mask – True for frames where the source centre lies inside the
    patch footprint (with a half-pixel margin so the PSF is at least partly
    visible).

    Parameters
    ----------
    patch_shape : (n_rows, n_cols)
    """
    n_rows, n_cols = patch_shape
    in_x = (traj.xs >= -0.5) & (traj.xs < n_cols + 0.5)
    in_y = (traj.ys >= -0.5) & (traj.ys < n_rows + 0.5)
    return in_x & in_y
