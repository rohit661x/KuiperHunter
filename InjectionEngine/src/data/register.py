"""
register.py – Align a stack of images to a common sidereal frame.

Two-tier approach:
  1. If all WCS fingerprints match epoch-0, return a copy (no-op — typical
     for same-pointing stacks like kbmod test data).
  2. Otherwise: for each epoch > 0, build a pixel-coordinate mapping from
     epoch-N's WCS to epoch-0's WCS and resample with
     scipy.ndimage.map_coordinates (bilinear).

No external 'reproject' package is required.
"""
from __future__ import annotations

import math
import numpy as np


# ---------------------------------------------------------------------------
# WCS fingerprint
# ---------------------------------------------------------------------------

def _wcs_fingerprint(hdr: dict) -> tuple:
    """Return a hashable key summarising the WCS of a header dict."""
    return (
        round(float(hdr.get("CRVAL1", 0.0)), 6),
        round(float(hdr.get("CRVAL2", 0.0)), 6),
        round(float(hdr.get("CRPIX1", 0.0)), 4),
        round(float(hdr.get("CRPIX2", 0.0)), 4),
        round(float(hdr.get("CD1_1",  0.0)), 12),
        round(float(hdr.get("CD1_2",  0.0)), 12),
        round(float(hdr.get("CD2_1",  0.0)), 12),
        round(float(hdr.get("CD2_2",  0.0)), 12),
    )


# ---------------------------------------------------------------------------
# Coordinate mapping helpers
# ---------------------------------------------------------------------------

def _hdr_to_cd(hdr: dict) -> np.ndarray:
    """Extract 2×2 CD matrix from header dict."""
    return np.array([
        [hdr["CD1_1"], hdr["CD1_2"]],
        [hdr["CD2_1"], hdr["CD2_2"]],
    ])


def _pixel_to_world(hdr: dict, rows: np.ndarray, cols: np.ndarray):
    """Convert pixel (col, row) → (RA, Dec) using simple TAN (no SIP)."""
    crpix1 = float(hdr["CRPIX1"])
    crpix2 = float(hdr["CRPIX2"])
    crval1 = float(hdr["CRVAL1"])
    crval2 = float(hdr["CRVAL2"])
    cd = _hdr_to_cd(hdr)

    dx = cols - (crpix1 - 1)  # FITS is 1-indexed
    dy = rows - (crpix2 - 1)

    # Linear TAN approximation (valid for small fields)
    dra  = cd[0, 0] * dx + cd[0, 1] * dy
    ddec = cd[1, 0] * dx + cd[1, 1] * dy

    ra  = crval1 + dra  / math.cos(math.radians(crval2))
    dec = crval2 + ddec
    return ra, dec


def _world_to_pixel(hdr: dict, ra: np.ndarray, dec: np.ndarray):
    """Convert (RA, Dec) → pixel (col, row) using simple TAN (no SIP)."""
    crpix1 = float(hdr["CRPIX1"])
    crpix2 = float(hdr["CRPIX2"])
    crval1 = float(hdr["CRVAL1"])
    crval2 = float(hdr["CRVAL2"])
    cd = _hdr_to_cd(hdr)
    cd_inv = np.linalg.inv(cd)

    dra  = (ra  - crval1) * math.cos(math.radians(crval2))
    ddec = dec - crval2

    dx = cd_inv[0, 0] * dra + cd_inv[0, 1] * ddec
    dy = cd_inv[1, 0] * dra + cd_inv[1, 1] * ddec

    cols = dx + (crpix1 - 1)
    rows = dy + (crpix2 - 1)
    return cols, rows


def _reproject_frame(
    img: np.ndarray,
    src_hdr: dict,
    ref_hdr: dict,
) -> np.ndarray:
    """
    Resample img (taken with src_hdr WCS) onto the ref_hdr pixel grid.
    Uses bilinear interpolation via scipy.ndimage.map_coordinates.
    """
    from scipy.ndimage import map_coordinates

    n_rows, n_cols = img.shape
    # Build a grid of (row, col) coordinates in the *reference* frame
    ref_rows, ref_cols = np.mgrid[0:n_rows, 0:n_cols]
    ref_rows = ref_rows.astype(np.float64)
    ref_cols = ref_cols.astype(np.float64)

    # Map ref pixels → world → src pixels
    ra, dec = _pixel_to_world(ref_hdr, ref_rows, ref_cols)
    src_cols, src_rows = _world_to_pixel(src_hdr, ra, dec)

    coords = np.array([src_rows.ravel(), src_cols.ravel()])
    out = map_coordinates(img.astype(np.float64), coords, order=1,
                          mode="constant", cval=0.0)
    return out.reshape(n_rows, n_cols).astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_to_epoch0(
    imgs: np.ndarray,
    hdrs: list[dict],
) -> np.ndarray:
    """
    Align all frames to the epoch-0 WCS grid.

    Parameters
    ----------
    imgs : float32 array, shape (T, H, W)
    hdrs : list of T header dicts — each must contain CRVAL1/2, CRPIX1/2, CD matrix

    Returns
    -------
    imgs_reg : float32 array, shape (T, H, W) — copy aligned to epoch-0
    """
    ref_fp = _wcs_fingerprint(hdrs[0])

    # Fast path: all WCS identical → same-pointing stack, no reprojection needed
    if all(_wcs_fingerprint(h) == ref_fp for h in hdrs[1:]):
        return imgs.copy()

    # Slow path: reproject each epoch onto epoch-0 grid
    out = np.empty_like(imgs)
    out[0] = imgs[0].copy()
    for t in range(1, len(hdrs)):
        out[t] = _reproject_frame(imgs[t], hdrs[t], hdrs[0])
    return out
