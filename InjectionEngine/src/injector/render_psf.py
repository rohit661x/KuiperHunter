"""
render_psf.py – Render a PSF-convolved point source at a sub-pixel position.

Supports Gaussian PSFs (analytic) and optionally an empirical PSF supplied as
a 2-D array.  The output stamp is the same shape as the input patch frame.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Literal


@dataclass
class PSFParams:
    """Parameters that define the PSF model."""
    # ---- Gaussian model ----
    fwhm_pixels: float = 2.5          # FWHM in pixels
    # ---- Empirical model ----
    kernel: np.ndarray | None = None  # pre-normalised 2-D kernel (odd size)
    # ---- Shared ----
    model: Literal["gaussian", "empirical"] = "gaussian"

    def __post_init__(self):
        if self.model == "empirical" and self.kernel is None:
            raise ValueError("model='empirical' requires a kernel array.")


def _gaussian_kernel(fwhm: float, size: int) -> np.ndarray:
    """Return a normalised 2-D Gaussian kernel of given size (odd integer)."""
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    ax = np.arange(size) - size // 2
    x, y = np.meshgrid(ax, ax)
    k = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return k / k.sum()


def _shift_kernel(kernel: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Apply a sub-pixel shift to a kernel via DFT phase ramp (Fourier shift
    theorem).  dx/dy are fractional pixel offsets.
    """
    ny, nx = kernel.shape
    fx = np.fft.fftfreq(nx)
    fy = np.fft.fftfreq(ny)
    Fx, Fy = np.meshgrid(fx, fy)
    phase = np.exp(-2j * np.pi * (Fx * dx + Fy * dy))
    shifted = np.fft.ifft2(np.fft.fft2(kernel) * phase).real
    return shifted


def render_stamp(
    shape: tuple[int, int],
    x: float,
    y: float,
    flux: float,
    psf_params: PSFParams,
) -> np.ndarray:
    """
    Render a single point source into an array of the given shape.

    Parameters
    ----------
    shape       : (n_rows, n_cols) of the output frame.
    x           : sub-pixel column position of the source centre.
    y           : sub-pixel row position of the source centre.
    flux        : total integrated flux (counts) to inject.
    psf_params  : PSFParams instance.

    Returns
    -------
    stamp : float64 array with shape ``shape``.  Out-of-bounds sources
            contribute where the PSF overlaps; fully outside → zero array.
    """
    n_rows, n_cols = shape
    stamp = np.zeros((n_rows, n_cols), dtype=np.float64)

    # Integer pixel centre
    ix, iy = int(round(x)), int(round(y))

    # Build the base kernel
    if psf_params.model == "gaussian":
        ksize = int(np.ceil(psf_params.fwhm_pixels * 5)) | 1  # odd
        ksize = max(ksize, 7)
        kernel = _gaussian_kernel(psf_params.fwhm_pixels, ksize)
    else:
        kernel = psf_params.kernel.copy()
        ksize = kernel.shape[0]

    # Sub-pixel shift: fractional part of (x, y)
    dx = x - ix
    dy = y - iy
    if abs(dx) > 1e-6 or abs(dy) > 1e-6:
        kernel = _shift_kernel(kernel, dx, dy)
        kernel = np.clip(kernel, 0.0, None)  # Fourier shift can ring; enforce ≥ 0

    kernel = kernel * flux  # scale to requested flux

    # Stamp half-size
    half = ksize // 2

    # Source and destination slices (handles edges / out-of-bounds)
    row_lo = iy - half
    row_hi = iy + half + 1
    col_lo = ix - half
    col_hi = ix + half + 1

    k_row_lo = max(0, -row_lo)
    k_row_hi = ksize - max(0, row_hi - n_rows)
    k_col_lo = max(0, -col_lo)
    k_col_hi = ksize - max(0, col_hi - n_cols)

    s_row_lo = max(0, row_lo)
    s_row_hi = min(n_rows, row_hi)
    s_col_lo = max(0, col_lo)
    s_col_hi = min(n_cols, col_hi)

    if (k_row_hi > k_row_lo) and (k_col_hi > k_col_lo):
        stamp[s_row_lo:s_row_hi, s_col_lo:s_col_hi] += (
            kernel[k_row_lo:k_row_hi, k_col_lo:k_col_hi]
        )

    return stamp


def render_stack(
    patch_shape: tuple[int, int],
    xs: np.ndarray,
    ys: np.ndarray,
    fluxes: np.ndarray,
    psf_params: PSFParams,
) -> np.ndarray:
    """
    Render a moving source into every frame of a stack.

    Parameters
    ----------
    patch_shape : (n_rows, n_cols)
    xs, ys      : per-frame pixel positions, shape (n_frames,)
    fluxes      : per-frame flux values, shape (n_frames,)
    psf_params  : PSFParams instance

    Returns
    -------
    rendered : float64 array, shape (n_frames, n_rows, n_cols)
    """
    n_frames = len(xs)
    rendered = np.zeros((n_frames, *patch_shape), dtype=np.float64)
    for i in range(n_frames):
        rendered[i] = render_stamp(patch_shape, xs[i], ys[i], fluxes[i], psf_params)
    return rendered
