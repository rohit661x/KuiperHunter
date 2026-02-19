"""
psf_estimator.py – Per-epoch PSF FWHM estimation.

Algorithm:
  1. Find the peak pixel in the background-subtracted frame.
  2. Extract a 15×15 stamp centred on it.
  3. Fit a 2D Gaussian using scipy.optimize.curve_fit.
  4. Convert fitted sigma → FWHM = sigma * 2.3548.
  5. If fitting fails (flat frame, crowded field, edge case), return fallback.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


_FWHM_SCALE = 2.3548  # 2 * sqrt(2 * ln 2)
_STAMP_HALF = 7       # stamp is (2*HALF+1) × (2*HALF+1) = 15×15


def _gauss2d_flat(xy, amplitude, x0, y0, sigma_x, sigma_y, offset):
    """Ravelled 2D Gaussian for use with curve_fit."""
    x, y = xy
    return (
        offset
        + amplitude
        * np.exp(
            -(
                (x - x0) ** 2 / (2 * sigma_x ** 2)
                + (y - y0) ** 2 / (2 * sigma_y ** 2)
            )
        )
    )


def _fit_fwhm_stamp(stamp: np.ndarray, fallback: float) -> float:
    """Fit a 2D Gaussian to stamp (H×W) and return FWHM in pixels."""
    H, W = stamp.shape
    if stamp.max() <= 0 or stamp.size < 9:
        return fallback

    rows, cols = np.mgrid[0:H, 0:W]
    y_c, x_c = np.unravel_index(np.argmax(stamp), stamp.shape)
    p0 = [float(stamp.max()), float(x_c), float(y_c), 1.5, 1.5, 0.0]
    bounds = (
        [0.0, 0.0, 0.0, 0.3, 0.3, -np.inf],
        [np.inf, W, H, float(max(H, W)), float(max(H, W)), np.inf],
    )
    try:
        popt, _ = curve_fit(
            _gauss2d_flat,
            (cols.ravel().astype(float), rows.ravel().astype(float)),
            stamp.ravel().astype(float),
            p0=p0,
            bounds=bounds,
            maxfev=2000,
        )
        sigma_avg = (abs(popt[3]) + abs(popt[4])) / 2.0
        return float(sigma_avg * _FWHM_SCALE)
    except Exception:
        return fallback


def estimate_psf_fwhm(
    imgs_bgsub: np.ndarray,
    fallback_fwhm: float = 2.5,
) -> np.ndarray:
    """
    Estimate PSF FWHM (pixels) per epoch.

    Parameters
    ----------
    imgs_bgsub    : float32 array, shape (T, H, W) — background-subtracted
    fallback_fwhm : FWHM returned when fitting fails (default = 2.5 px)

    Returns
    -------
    psf_fwhm : float32 array, shape (T,)
    """
    T, H, W = imgs_bgsub.shape
    fwhm_arr = np.full(T, fallback_fwhm, dtype=np.float32)

    for t in range(T):
        img = imgs_bgsub[t]
        iy, ix = int(np.unravel_index(np.argmax(img), img.shape)[0]), \
                 int(np.unravel_index(np.argmax(img), img.shape)[1])

        r0 = max(0, iy - _STAMP_HALF)
        r1 = min(H, iy + _STAMP_HALF + 1)
        c0 = max(0, ix - _STAMP_HALF)
        c1 = min(W, ix + _STAMP_HALF + 1)

        stamp = img[r0:r1, c0:c1].astype(np.float64)
        fwhm_arr[t] = _fit_fwhm_stamp(stamp, fallback_fwhm)

    return fwhm_arr
