# Step 3: Real Data Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Build a one-stack pipeline that reads 5 kbmod FITS files, aligns/background-subtracts/PSF-estimates them, writes a zarr cache, then generates demo/training injection cases with physically calibrated amplitudes.

**Architecture:** Six pure-function modules (`load_stack → register → background → psf_estimator → patches → build_one_stack`) write a zarr store; `make_demo_cases.py` loads the zarr, extracts patches, and calls `inject()` with per-epoch sigma calibration; `demo.py` gains a `--case` mode to replay saved cases.

**Tech Stack:** `astropy` (FITS + WCS), `zarr` (output store), `scipy.optimize` (PSF fitting), `numpy`/`scipy.stats` (MAD noise), `pytest` (tests with synthetic FITS via `astropy.io.fits`)

---

## Data contract (locked for this plan)

The kbmod test FITS files live at `kbmod/kbmod/data/small/`:

| Field | Value |
|---|---|
| HDU layout | `[0]` PrimaryHDU (MJD in header), `[1]` science float32 64×64 + TAN-SIP WCS, `[2]` variance (zeros), `[3]` mask (all 4s) |
| Already bg-subtracted | Yes (median ≈ 0) — `background.py` still runs for correctness + sigma |
| All same WCS | Yes → `register.py` returns copy (no-op path) |
| Plate scale | 0.2635 arcsec/px from CD matrix |
| T used | First 5 files: `000000.fits` … `000004.fits` |
| Patch size | `--patch 64 --stride 32` (images are 64×64) |

**Setup (run once):**
```bash
mkdir -p data/raw/stack01
cp kbmod/kbmod/data/small/00000{0,1,2,3,4}.fits data/raw/stack01/
```

---

## Task 1: Add dependencies + `src/data/__init__.py`

**Files:**
- Modify: `pyproject.toml`
- Create: `src/data/__init__.py`

**Step 1: Add zarr, astropy, scipy to pyproject.toml**

Edit the `[project]` `dependencies` list:

```toml
[project]
name = "injection-engine"
version = "0.1.0"
description = "Synthetic moving-source injection into astronomical image stacks"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "astropy>=5.0",
    "zarr>=2.16",
    "scipy>=1.9",
]

[project.optional-dependencies]
demo = ["matplotlib>=3.7"]
dev  = ["pytest>=7"]
```

**Step 2: Install**

```bash
pip install zarr astropy scipy
```

Expected: `Successfully installed zarr-...` (zarr, astropy already present but re-installs harmlessly)

**Step 3: Create `src/data/__init__.py`**

```python
"""src/data – real-data loading and preprocessing pipeline."""
```

**Step 4: Verify import**

```bash
python -c "import zarr; import astropy; print('ok')"
```
Expected: `ok`

**Step 5: Commit**

```bash
git add pyproject.toml src/data/__init__.py
git commit -m "feat: add zarr/astropy/scipy deps; create src/data package"
```

---

## Task 2: `src/data/load_stack.py`

**Files:**
- Create: `src/data/load_stack.py`
- Create: `tests/test_data_load_stack.py`

### Step 1: Write failing tests

```python
# tests/test_data_load_stack.py
"""Tests for load_stack.py — uses synthetic in-memory FITS, no real files."""
from pathlib import Path
import numpy as np
import pytest
from astropy.io import fits


def _write_fake_fits(path: Path, mjd: float, shape=(64, 64), seed=0):
    """Write a minimal 4-HDU FITS matching kbmod format."""
    rng = np.random.default_rng(seed)
    sci = rng.normal(0, 5, shape).astype(np.float32)
    variance = np.zeros(shape, dtype=np.float32)
    mask = np.full(shape, 4, dtype=np.float32)

    primary = fits.PrimaryHDU()
    primary.header["MJD"] = mjd

    sci_hdu = fits.ImageHDU(data=sci)
    sci_hdu.header["WCSAXES"] = 2
    sci_hdu.header["CTYPE1"] = "RA---TAN-SIP"
    sci_hdu.header["CTYPE2"] = "DEC--TAN-SIP"
    sci_hdu.header["CRVAL1"] = 200.615
    sci_hdu.header["CRVAL2"] = -7.789
    sci_hdu.header["CRPIX1"] = 1033.9
    sci_hdu.header["CRPIX2"] = 2043.5
    sci_hdu.header["CD1_1"] = -1.14e-07
    sci_hdu.header["CD1_2"] = 7.318e-05
    sci_hdu.header["CD2_1"] = -7.301e-05
    sci_hdu.header["CD2_2"] = -1.28e-07

    hdul = fits.HDUList([primary, sci_hdu, fits.ImageHDU(variance), fits.ImageHDU(mask)])
    hdul.writeto(path, overwrite=True)


class TestLoadFitsStack:
    def test_returns_correct_shapes(self, tmp_path):
        paths = []
        for i in range(3):
            p = tmp_path / f"{i:06d}.fits"
            _write_fake_fits(p, mjd=57130.0 + i * 0.01)
            paths.append(p)

        from src.data.load_stack import load_fits_stack
        imgs, hdrs, mjds = load_fits_stack(paths)

        assert imgs.shape == (3, 64, 64)
        assert imgs.dtype == np.float32
        assert len(hdrs) == 3
        assert mjds.shape == (3,)
        assert mjds.dtype == np.float64

    def test_mjds_correct(self, tmp_path):
        paths = []
        expected_mjds = [57130.0, 57130.5, 57131.0]
        for i, mjd in enumerate(expected_mjds):
            p = tmp_path / f"{i:06d}.fits"
            _write_fake_fits(p, mjd=mjd)
            paths.append(p)

        from src.data.load_stack import load_fits_stack
        _, _, mjds = load_fits_stack(paths)

        np.testing.assert_allclose(mjds, expected_mjds)

    def test_hdr_contains_wcs_keys(self, tmp_path):
        p = tmp_path / "000000.fits"
        _write_fake_fits(p, mjd=57130.0)

        from src.data.load_stack import load_fits_stack
        _, hdrs, _ = load_fits_stack([p])

        assert "CRVAL1" in hdrs[0]
        assert "CD1_2" in hdrs[0]

    def test_science_data_is_hdu1(self, tmp_path):
        """HDU[1] data should be loaded, not HDU[2] or HDU[3]."""
        p = tmp_path / "000000.fits"
        rng = np.random.default_rng(0)
        sci = rng.normal(0, 5, (64, 64)).astype(np.float32)
        primary = fits.PrimaryHDU()
        primary.header["MJD"] = 57130.0
        sci_hdu = fits.ImageHDU(data=sci)
        junk = fits.ImageHDU(data=np.zeros((64, 64), dtype=np.float32))
        fits.HDUList([primary, sci_hdu, junk, junk]).writeto(p, overwrite=True)

        from src.data.load_stack import load_fits_stack
        imgs, _, _ = load_fits_stack([p])

        np.testing.assert_array_equal(imgs[0], sci)
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_data_load_stack.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.data.load_stack'`

**Step 3: Write `src/data/load_stack.py`**

```python
"""
load_stack.py – Read a list of FITS files (kbmod 4-HDU format).

HDU layout expected:
  HDU[0]: PrimaryHDU  — MJD in header['MJD']
  HDU[1]: ImageHDU    — science image (float32) with TAN-SIP WCS
  HDU[2]: ImageHDU    — variance plane (may be zeros)
  HDU[3]: ImageHDU    — mask plane

Returns imgs[t], hdrs[t], mjds[t].
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from astropy.io import fits


def load_fits_stack(
    paths: list[str | Path],
) -> tuple[np.ndarray, list[dict], np.ndarray]:
    """
    Read FITS files and return stacked science images, headers, and MJDs.

    Parameters
    ----------
    paths : list of str or Path
        Ordered list of FITS file paths.  Each must follow the kbmod
        4-HDU convention (PrimaryHDU + science + variance + mask).

    Returns
    -------
    imgs   : float32 array, shape (T, H, W)
    hdrs   : list of T dicts — HDU[1] header key/value pairs (contains WCS)
    mjds   : float64 array, shape (T,)  — MJD timestamps from HDU[0].header
    """
    imgs_list: list[np.ndarray] = []
    hdrs_list: list[dict] = []
    mjds_list: list[float] = []

    for p in paths:
        with fits.open(p) as hdul:
            imgs_list.append(hdul[1].data.astype(np.float32))
            hdrs_list.append(dict(hdul[1].header))
            mjds_list.append(float(hdul[0].header["MJD"]))

    return (
        np.stack(imgs_list, axis=0),
        hdrs_list,
        np.array(mjds_list, dtype=np.float64),
    )
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_data_load_stack.py -v
```
Expected: `4 passed`

**Step 5: Commit**

```bash
git add src/data/load_stack.py tests/test_data_load_stack.py
git commit -m "feat: load_stack.py reads kbmod 4-HDU FITS into numpy arrays"
```

---

## Task 3: `src/data/register.py`

**Files:**
- Create: `src/data/register.py`
- Create: `tests/test_data_register.py`

### Step 1: Write failing tests

```python
# tests/test_data_register.py
"""Tests for register.py — sidereal alignment."""
import numpy as np
import pytest


def _make_identical_hdrs(n=3) -> list[dict]:
    """All epochs share the same WCS (typical for same-pointing stacks)."""
    hdr = {
        "CRVAL1": 200.615, "CRVAL2": -7.789,
        "CRPIX1": 1033.9,  "CRPIX2": 2043.5,
        "CD1_1": -1.14e-07, "CD1_2": 7.318e-05,
        "CD2_1": -7.301e-05, "CD2_2": -1.28e-07,
    }
    return [dict(hdr) for _ in range(n)]


def _make_imgs(T=3, H=64, W=64, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 5, (T, H, W)).astype(np.float32)


class TestRegisterToEpoch0:
    def test_identical_wcs_returns_copy(self):
        from src.data.register import register_to_epoch0
        imgs = _make_imgs()
        hdrs = _make_identical_hdrs(3)
        out = register_to_epoch0(imgs, hdrs)
        np.testing.assert_array_equal(out, imgs)
        assert out is not imgs  # must be a copy

    def test_output_shape_unchanged(self):
        from src.data.register import register_to_epoch0
        imgs = _make_imgs(T=5, H=64, W=64)
        hdrs = _make_identical_hdrs(5)
        out = register_to_epoch0(imgs, hdrs)
        assert out.shape == imgs.shape

    def test_shifted_wcs_shifts_image(self):
        """One epoch shifted by 1 pixel should produce a shifted output."""
        from src.data.register import register_to_epoch0
        import numpy as np
        # Build a simple image with a bright point source at (32, 32)
        T, H, W = 2, 64, 64
        imgs = np.zeros((T, H, W), dtype=np.float32)
        imgs[0, 32, 32] = 100.0
        imgs[1, 32, 32] = 100.0  # same pixel coords before registration

        # Build hdrs: epoch-0 is reference; epoch-1 is shifted by 1 px in x
        base_hdr = {
            "CRVAL1": 200.615, "CRVAL2": -7.789,
            "CRPIX1": 1033.9,  "CRPIX2": 2043.5,
            "CD1_1": -1.14e-07, "CD1_2": 7.318e-05,
            "CD2_1": -7.301e-05, "CD2_2": -1.28e-07,
        }
        shifted_hdr = dict(base_hdr)
        shifted_hdr["CRPIX1"] = base_hdr["CRPIX1"] + 1.0  # shift ref pixel by 1

        hdrs = [base_hdr, shifted_hdr]
        out = register_to_epoch0(imgs, hdrs)
        # Epoch-0 stays unchanged
        assert out[0, 32, 32] == pytest.approx(100.0, abs=1.0)
        # Epoch-1 peak should move by ~1 pixel
        assert out[1].max() > 50.0  # still bright somewhere
        assert out[1, 32, 32] < 100.0  # no longer exactly at (32, 32)
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_data_register.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.data.register'`

**Step 3: Write `src/data/register.py`**

```python
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
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_data_register.py -v
```
Expected: `3 passed`

**Step 5: Commit**

```bash
git add src/data/register.py tests/test_data_register.py
git commit -m "feat: register.py aligns stack to epoch-0 WCS (fast no-op for same-pointing)"
```

---

## Task 4: `src/data/background.py`

**Files:**
- Create: `src/data/background.py`
- Create: `tests/test_data_background.py`

### Step 1: Write failing tests

```python
# tests/test_data_background.py
"""Tests for background.py — per-frame background subtraction and noise estimation."""
import numpy as np
import pytest


class TestSubtractBackground:
    def _make_imgs(self, T=3, H=64, W=64, bg=100.0, sigma=5.0, seed=0):
        rng = np.random.default_rng(seed)
        return (bg + rng.normal(0, sigma, (T, H, W))).astype(np.float32)

    def test_output_shapes(self):
        from src.data.background import subtract_background
        imgs = self._make_imgs()
        imgs_out, sigma = subtract_background(imgs)
        assert imgs_out.shape == imgs.shape
        assert sigma.shape == (3,)

    def test_background_removed(self):
        """After subtraction, per-frame median should be near zero."""
        from src.data.background import subtract_background
        imgs = self._make_imgs(bg=200.0)
        imgs_out, _ = subtract_background(imgs)
        for t in range(imgs_out.shape[0]):
            assert abs(np.median(imgs_out[t])) < 1.0

    def test_sigma_positive(self):
        from src.data.background import subtract_background
        imgs = self._make_imgs()
        _, sigma = subtract_background(imgs)
        assert (sigma > 0).all()

    def test_sigma_estimates_noise(self):
        """Sigma should be within 20% of the true noise std for Gaussian noise."""
        from src.data.background import subtract_background
        true_sigma = 7.5
        imgs = self._make_imgs(sigma=true_sigma, T=1, H=128, W=128)
        _, sigma = subtract_background(imgs)
        assert abs(sigma[0] - true_sigma) / true_sigma < 0.20

    def test_does_not_modify_input(self):
        from src.data.background import subtract_background
        imgs = self._make_imgs()
        orig = imgs.copy()
        subtract_background(imgs)
        np.testing.assert_array_equal(imgs, orig)
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_data_background.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.data.background'`

**Step 3: Write `src/data/background.py`**

```python
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
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_data_background.py -v
```
Expected: `5 passed`

**Step 5: Commit**

```bash
git add src/data/background.py tests/test_data_background.py
git commit -m "feat: background.py per-frame median subtraction and MAD sigma estimation"
```

---

## Task 5: `src/data/psf_estimator.py`

**Files:**
- Create: `src/data/psf_estimator.py`
- Create: `tests/test_data_psf_estimator.py`

### Step 1: Write failing tests

```python
# tests/test_data_psf_estimator.py
"""Tests for psf_estimator.py — per-epoch PSF FWHM estimation."""
import numpy as np
import pytest


def _gauss2d(shape, cx, cy, sigma):
    H, W = shape
    rows, cols = np.mgrid[0:H, 0:W]
    return np.exp(-((cols - cx)**2 + (rows - cy)**2) / (2 * sigma**2)).astype(np.float32)


class TestEstimatePsfFwhm:
    def test_output_shape(self):
        from src.data.psf_estimator import estimate_psf_fwhm
        imgs = np.random.default_rng(0).normal(0, 1, (4, 64, 64)).astype(np.float32)
        fwhm = estimate_psf_fwhm(imgs)
        assert fwhm.shape == (4,)
        assert fwhm.dtype == np.float32

    def test_all_positive(self):
        from src.data.psf_estimator import estimate_psf_fwhm
        imgs = np.random.default_rng(0).normal(0, 1, (3, 64, 64)).astype(np.float32)
        fwhm = estimate_psf_fwhm(imgs)
        assert (fwhm > 0).all()

    def test_recovers_known_fwhm(self):
        """For a clean Gaussian star, returned FWHM should be within 20% of truth."""
        from src.data.psf_estimator import estimate_psf_fwhm
        sigma_true = 2.0
        fwhm_true = sigma_true * 2.3548
        # Plant a Gaussian star at (32, 32) in a noise-free image
        img = (_gauss2d((64, 64), 32, 32, sigma_true) * 100.0)
        imgs = img[np.newaxis]  # T=1
        fwhm = estimate_psf_fwhm(imgs)
        assert abs(fwhm[0] - fwhm_true) / fwhm_true < 0.20

    def test_fallback_on_flat_image(self):
        """All-zero image should return the fallback FWHM, not raise."""
        from src.data.psf_estimator import estimate_psf_fwhm
        imgs = np.zeros((2, 64, 64), dtype=np.float32)
        fallback = 2.5
        fwhm = estimate_psf_fwhm(imgs, fallback_fwhm=fallback)
        np.testing.assert_allclose(fwhm, fallback)
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_data_psf_estimator.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.data.psf_estimator'`

**Step 3: Write `src/data/psf_estimator.py`**

```python
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
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_data_psf_estimator.py -v
```
Expected: `4 passed`

**Step 5: Commit**

```bash
git add src/data/psf_estimator.py tests/test_data_psf_estimator.py
git commit -m "feat: psf_estimator.py Gaussian stamp fit for per-epoch FWHM"
```

---

## Task 6: `src/data/patches.py`

**Files:**
- Create: `src/data/patches.py`
- Create: `tests/test_data_patches.py`

### Step 1: Write failing tests

```python
# tests/test_data_patches.py
"""Tests for patches.py — grid-based patch extraction and per-patch sigma."""
import numpy as np
import pytest


class TestExtractPatchGrid:
    def _make_imgs(self, T=5, H=64, W=64, seed=0):
        return np.random.default_rng(seed).normal(0, 5, (T, H, W)).astype(np.float32)

    def test_yields_correct_shape(self):
        from src.data.patches import extract_patch_grid
        imgs = self._make_imgs(T=5, H=64, W=64)
        patches = list(extract_patch_grid(imgs, patch_size=32, stride=32))
        assert len(patches) > 0
        patch_stack, row_start, col_start = patches[0]
        assert patch_stack.shape == (5, 32, 32)

    def test_grid_covers_image(self):
        """All non-overlapping 32×32 patches in a 64×64 image: expect 4 patches."""
        from src.data.patches import extract_patch_grid
        imgs = self._make_imgs(H=64, W=64)
        patches = list(extract_patch_grid(imgs, patch_size=32, stride=32))
        assert len(patches) == 4  # (64/32) * (64/32)

    def test_row_col_starts_are_correct(self):
        from src.data.patches import extract_patch_grid
        imgs = self._make_imgs(H=64, W=64)
        patches = list(extract_patch_grid(imgs, patch_size=32, stride=32))
        starts = [(r, c) for _, r, c in patches]
        assert (0, 0) in starts
        assert (0, 32) in starts
        assert (32, 0) in starts
        assert (32, 32) in starts

    def test_patch_data_matches_source(self):
        from src.data.patches import extract_patch_grid
        imgs = self._make_imgs(H=64, W=64)
        patches = list(extract_patch_grid(imgs, patch_size=32, stride=32))
        for patch_stack, r, c in patches:
            np.testing.assert_array_equal(
                patch_stack, imgs[:, r:r + 32, c:c + 32]
            )


class TestPatchSigma:
    def test_output_shape(self):
        from src.data.patches import patch_sigma
        patch_stack = np.random.default_rng(0).normal(0, 5, (5, 32, 32)).astype(np.float32)
        sigma = patch_sigma(patch_stack)
        assert sigma.shape == (5,)
        assert sigma.dtype == np.float32

    def test_sigma_positive(self):
        from src.data.patches import patch_sigma
        patch_stack = np.random.default_rng(0).normal(0, 5, (3, 32, 32)).astype(np.float32)
        sigma = patch_sigma(patch_stack)
        assert (sigma > 0).all()

    def test_sigma_estimates_noise(self):
        """Sigma from patch should be within 30% of true noise std."""
        from src.data.patches import patch_sigma
        true_sigma = 8.0
        imgs = np.random.default_rng(7).normal(0, true_sigma, (2, 64, 64)).astype(np.float32)
        sigma = patch_sigma(imgs)
        for s in sigma:
            assert abs(s - true_sigma) / true_sigma < 0.30
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_data_patches.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.data.patches'`

**Step 3: Write `src/data/patches.py`**

```python
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
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_data_patches.py -v
```
Expected: `7 passed`

**Step 5: Commit**

```bash
git add src/data/patches.py tests/test_data_patches.py
git commit -m "feat: patches.py grid extraction and per-patch MAD sigma"
```

---

## Task 7: `src/data/build_one_stack.py` + zarr output

**Files:**
- Create: `src/data/build_one_stack.py`
- Create: `tests/test_data_build_one_stack.py`

The test uses the same `_write_fake_fits` helper from Task 2; copy it into `tests/conftest.py` so it's shared.

### Step 1: Create `tests/conftest.py` with shared helper

```python
# tests/conftest.py
"""Shared pytest fixtures and helpers for src/data tests."""
from pathlib import Path
import numpy as np
from astropy.io import fits


def write_fake_fits(path: Path, mjd: float, shape=(64, 64), seed=0):
    """Write a minimal 4-HDU FITS matching kbmod format."""
    rng = np.random.default_rng(seed)
    sci = rng.normal(0, 5, shape).astype(np.float32)
    variance = np.zeros(shape, dtype=np.float32)
    mask = np.full(shape, 4, dtype=np.float32)

    primary = fits.PrimaryHDU()
    primary.header["MJD"] = mjd

    sci_hdu = fits.ImageHDU(data=sci)
    sci_hdu.header["WCSAXES"] = 2
    sci_hdu.header["CTYPE1"] = "RA---TAN-SIP"
    sci_hdu.header["CTYPE2"] = "DEC--TAN-SIP"
    sci_hdu.header["CRVAL1"] = 200.615
    sci_hdu.header["CRVAL2"] = -7.789
    sci_hdu.header["CRPIX1"] = 1033.9
    sci_hdu.header["CRPIX2"] = 2043.5
    sci_hdu.header["CD1_1"] = -1.14e-07
    sci_hdu.header["CD1_2"] = 7.318e-05
    sci_hdu.header["CD2_1"] = -7.301e-05
    sci_hdu.header["CD2_2"] = -1.28e-07

    fits.HDUList(
        [primary, sci_hdu, fits.ImageHDU(variance), fits.ImageHDU(mask)]
    ).writeto(path, overwrite=True)
```

Also update `tests/test_data_load_stack.py` to import `write_fake_fits` from conftest instead of re-defining it:

```python
# At top of test_data_load_stack.py, replace the local helper with:
from tests.conftest import write_fake_fits
```

### Step 2: Write failing tests

```python
# tests/test_data_build_one_stack.py
"""Tests for build_one_stack.py — end-to-end zarr pipeline."""
import numpy as np
import pytest
import zarr
from tests.conftest import write_fake_fits


class TestBuildOneStack:
    def _make_stack_dir(self, tmp_path, T=5):
        stack_dir = tmp_path / "raw"
        stack_dir.mkdir()
        for i in range(T):
            write_fake_fits(stack_dir / f"{i:06d}.fits", mjd=57130.0 + i * 0.01, seed=i)
        return stack_dir

    def test_creates_zarr(self, tmp_path):
        from src.data.build_one_stack import build_one_stack
        stack_dir = self._make_stack_dir(tmp_path)
        out = tmp_path / "stack.zarr"
        build_one_stack(stack_dir, out, T=5)
        assert out.exists()

    def test_zarr_has_required_datasets(self, tmp_path):
        from src.data.build_one_stack import build_one_stack
        stack_dir = self._make_stack_dir(tmp_path)
        out = tmp_path / "stack.zarr"
        build_one_stack(stack_dir, out, T=5)
        z = zarr.open(str(out), mode="r")
        assert "images" in z
        assert "timestamps" in z
        assert "psf_fwhm" in z

    def test_zarr_images_shape(self, tmp_path):
        from src.data.build_one_stack import build_one_stack
        stack_dir = self._make_stack_dir(tmp_path)
        out = tmp_path / "stack.zarr"
        build_one_stack(stack_dir, out, T=5)
        z = zarr.open(str(out), mode="r")
        assert z["images"].shape == (5, 64, 64)

    def test_zarr_plate_scale_attr(self, tmp_path):
        from src.data.build_one_stack import build_one_stack
        stack_dir = self._make_stack_dir(tmp_path)
        out = tmp_path / "stack.zarr"
        build_one_stack(stack_dir, out, T=5)
        z = zarr.open(str(out), mode="r")
        assert "plate_scale" in z.attrs
        ps = z.attrs["plate_scale"]
        assert 0.1 < ps < 1.0  # reasonable arcsec/px

    def test_plate_scale_override(self, tmp_path):
        from src.data.build_one_stack import build_one_stack
        stack_dir = self._make_stack_dir(tmp_path)
        out = tmp_path / "stack.zarr"
        build_one_stack(stack_dir, out, T=5, plate_scale=0.187)
        z = zarr.open(str(out), mode="r")
        assert z.attrs["plate_scale"] == pytest.approx(0.187)

    def test_too_few_fits_raises(self, tmp_path):
        from src.data.build_one_stack import build_one_stack
        stack_dir = tmp_path / "raw"
        stack_dir.mkdir()
        write_fake_fits(stack_dir / "000000.fits", mjd=57130.0)
        with pytest.raises(ValueError, match="at least"):
            build_one_stack(stack_dir, tmp_path / "stack.zarr", T=5)
```

**Step 3: Run tests to verify they fail**

```bash
python -m pytest tests/test_data_build_one_stack.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.data.build_one_stack'`

**Step 4: Write `src/data/build_one_stack.py`**

```python
"""
build_one_stack.py – CLI + library: load → register → background → PSF → zarr.

Usage::

    python -m src.data.build_one_stack \\
        --stack_dir data/raw/stack01 \\
        --out       data/processed/stack01.zarr \\
        --T 5 --patch 64 --stride 32

Output zarr schema::

    images      : (T, H, W) float32 — aligned, background-subtracted
    timestamps  : (T,)      float64 — MJD per epoch
    psf_fwhm    : (T,)      float32 — Gaussian FWHM per epoch (pixels)
    .attrs['plate_scale'] : float   — arcsec/px
    .attrs['T']           : int
    .attrs['patch_size']  : int     — intended extraction patch (metadata)
    .attrs['stride']      : int
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import zarr

from .load_stack import load_fits_stack
from .register import register_to_epoch0
from .background import subtract_background
from .psf_estimator import estimate_psf_fwhm


# ---------------------------------------------------------------------------
# Plate-scale helper
# ---------------------------------------------------------------------------

def _plate_scale_from_hdr(hdr: dict) -> float:
    """Derive plate scale (arcsec/px) from CD matrix or CDELT."""
    if "CD1_2" in hdr and "CD2_2" in hdr:
        return math.sqrt(float(hdr["CD1_2"]) ** 2 + float(hdr["CD2_2"]) ** 2) * 3600.0
    if "CDELT2" in hdr:
        return abs(float(hdr["CDELT2"])) * 3600.0
    return 0.263  # MegaCam-like fallback


# ---------------------------------------------------------------------------
# Library function
# ---------------------------------------------------------------------------

def build_one_stack(
    stack_dir: str | Path,
    out_path: str | Path,
    T: int = 5,
    patch_size: int = 64,
    stride: int = 32,
    plate_scale: float | None = None,
) -> None:
    """
    Build a zarr stack from T FITS files in stack_dir.

    Parameters
    ----------
    stack_dir   : directory containing *.fits files (sorted alphabetically)
    out_path    : output zarr path (created or overwritten)
    T           : number of epochs to use (uses first T files)
    patch_size  : stored as metadata attribute; used by make_demo_cases.py
    stride      : stored as metadata attribute; used by make_demo_cases.py
    plate_scale : arcsec/px; if None, derived from first frame's CD matrix
    """
    stack_dir = Path(stack_dir)
    fits_paths = sorted(stack_dir.glob("*.fits"))[:T]

    if len(fits_paths) < T:
        raise ValueError(
            f"Expected at least {T} FITS files in {stack_dir}, "
            f"found {len(fits_paths)}."
        )

    # --- Pipeline ---
    imgs, hdrs, mjds = load_fits_stack(fits_paths)
    imgs_reg = register_to_epoch0(imgs, hdrs)
    imgs_bgsub, _sigma_frame = subtract_background(imgs_reg)
    psf_fwhm = estimate_psf_fwhm(imgs_bgsub)

    if plate_scale is None:
        plate_scale = _plate_scale_from_hdr(hdrs[0])

    # --- Write zarr ---
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    store = zarr.open(str(out_path), mode="w")
    store.create_dataset("images",     data=imgs_bgsub, dtype="float32", overwrite=True)
    store.create_dataset("timestamps", data=mjds,       dtype="float64", overwrite=True)
    store.create_dataset("psf_fwhm",   data=psf_fwhm,  dtype="float32", overwrite=True)
    store.attrs["plate_scale"] = float(plate_scale)
    store.attrs["T"]           = T
    store.attrs["patch_size"]  = patch_size
    store.attrs["stride"]      = stride

    # --- Summary ---
    print(f"Written: {out_path}")
    print(f"  images     : {imgs_bgsub.shape}  dtype=float32")
    print(f"  timestamps : {mjds}")
    print(f"  psf_fwhm   : {psf_fwhm}")
    print(f"  plate_scale: {plate_scale:.4f} arcsec/px")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build one aligned image stack and write to zarr."
    )
    parser.add_argument("--stack_dir",   required=True,
                        help="Directory with *.fits files")
    parser.add_argument("--out",         required=True,
                        help="Output zarr path")
    parser.add_argument("--T",           type=int,   default=5,
                        help="Number of epochs (default: 5)")
    parser.add_argument("--patch",       type=int,   default=64,
                        help="Patch size for extraction (default: 64)")
    parser.add_argument("--stride",      type=int,   default=32,
                        help="Stride for patch grid (default: 32)")
    parser.add_argument("--plate_scale", type=float, default=None,
                        help="arcsec/px; if omitted, read from header CD matrix")
    args = parser.parse_args()

    build_one_stack(
        stack_dir=args.stack_dir,
        out_path=args.out,
        T=args.T,
        patch_size=args.patch,
        stride=args.stride,
        plate_scale=args.plate_scale,
    )


if __name__ == "__main__":
    main()
```

**Step 5: Run tests**

```bash
python -m pytest tests/test_data_build_one_stack.py -v
```
Expected: `6 passed`

**Step 6: Smoke-test the CLI on real data**

```bash
mkdir -p data/raw/stack01
cp kbmod/kbmod/data/small/00000{0,1,2,3,4}.fits data/raw/stack01/
python -m src.data.build_one_stack \
  --stack_dir data/raw/stack01 \
  --out data/processed/stack01.zarr \
  --T 5 --patch 64 --stride 32
```

Expected output (approximate):
```
Written: data/processed/stack01.zarr
  images     : (5, 64, 64)  dtype=float32
  timestamps : [57130.199 57130.211 57130.219 57131.199 57131.211]
  psf_fwhm   : [3.5 3.2 3.8 3.1 3.6]   ← values vary
  plate_scale: 0.2635 arcsec/px
```

**Step 7: Commit**

```bash
git add src/data/build_one_stack.py tests/conftest.py tests/test_data_build_one_stack.py
git commit -m "feat: build_one_stack.py load→register→background→PSF→zarr pipeline"
```

---

## Task 8: Upgrade injector — per-epoch sigma calibration (Step 3.6)

**Files:**
- Modify: `src/injector/injector.py` (add `sigma_map` parameter)
- Modify: `tests/test_injection.py` (add sigma_map tests)

The current `flux_peak = snr` (dimensionless). After this task, if a caller passes
`sigma_map` (per-frame noise), fluxes become `snr * sigma_map[t]` (physical ADU amplitude).

### Step 1: Write failing tests

Add to `tests/test_injection.py` (append the new test class — do NOT delete existing tests):

```python
class TestSigmaMapCalibration:
    """Test Step 3.6: per-epoch sigma calibration."""

    def _patch(self, T=5, size=64):
        rng = np.random.default_rng(0)
        return rng.normal(0, 1, (T, size, size))

    def test_sigma_map_scales_flux(self):
        """With sigma_map, injected peak should scale proportionally."""
        from src.injector import inject
        from src.injector.render_psf import PSFParams

        patch = self._patch()
        times = np.arange(5, dtype=float) * 0.5
        psf = PSFParams(fwhm_pixels=2.5)

        sigma_low  = np.ones(5) * 1.0
        sigma_high = np.ones(5) * 10.0

        _, Y_low,  _ = inject(patch, times, 0.263, psf, seed=42, sigma_map=sigma_low)
        _, Y_high, _ = inject(patch, times, 0.263, psf, seed=42, sigma_map=sigma_high)

        ratio = Y_high.sum() / Y_low.sum()
        assert abs(ratio - 10.0) < 0.5  # should be ~10x

    def test_no_sigma_map_backward_compat(self):
        """inject() without sigma_map must still work (backward compat)."""
        from src.injector import inject
        from src.injector.render_psf import PSFParams

        patch = self._patch()
        times = np.arange(5, dtype=float) * 0.5
        psf = PSFParams(fwhm_pixels=2.5)

        X, Y, meta = inject(patch, times, 0.263, psf, seed=42)
        assert Y.sum() > 0

    def test_sigma_map_wrong_length_raises(self):
        from src.injector import inject
        from src.injector.render_psf import PSFParams

        patch = self._patch(T=5)
        times = np.arange(5, dtype=float) * 0.5
        psf = PSFParams(fwhm_pixels=2.5)

        with pytest.raises(ValueError, match="sigma_map"):
            inject(patch, times, 0.263, psf, seed=42, sigma_map=np.ones(3))

    def test_meta_records_sigma_map_used(self):
        """meta should indicate whether sigma calibration was applied."""
        from src.injector import inject
        from src.injector.render_psf import PSFParams

        patch = self._patch()
        times = np.arange(5, dtype=float) * 0.5
        psf = PSFParams(fwhm_pixels=2.5)

        _, _, meta_no  = inject(patch, times, 0.263, psf, seed=42)
        _, _, meta_yes = inject(patch, times, 0.263, psf, seed=42,
                                sigma_map=np.ones(5) * 5.0)

        assert meta_no["sigma_calibrated"] is False
        assert meta_yes["sigma_calibrated"] is True
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_injection.py::TestSigmaMapCalibration -v
```
Expected: `TypeError` or `AssertionError` (sigma_map not accepted yet)

**Step 3: Modify `src/injector/injector.py`**

Add `sigma_map` parameter. Locate the section after `prior` is drawn (around line 83) and replace the `fluxes` assignment:

```python
def inject(
    patch_stack: np.ndarray,
    timestamps: np.ndarray,
    plate_scale: float,
    psf_params: PSFParams,
    sample_type: str = "tno",
    seed: int | None = None,
    *,
    target_config: TargetConfig | None = None,
    sigma_map: np.ndarray | None = None,     # NEW in Step 3.6
) -> tuple[np.ndarray, np.ndarray, dict]:
```

Then replace the line:
```python
    # Constant flux per frame (could be made variable in future)
    fluxes = np.full(n_frames, prior.flux_peak, dtype=np.float64)
```

With:
```python
    # Per-frame flux: scale by per-epoch noise if sigma_map is provided.
    # sigma_map = None  → flux_peak is used as-is (backward compat).
    # sigma_map provided → flux = snr * sigma_map[t]  (physical amplitude).
    if sigma_map is not None:
        sigma_arr = np.asarray(sigma_map, dtype=np.float64)
        if sigma_arr.shape != (n_frames,):
            raise ValueError(
                f"sigma_map must have shape ({n_frames},), got {sigma_arr.shape}."
            )
        fluxes = prior.flux_peak * sigma_arr
    else:
        fluxes = np.full(n_frames, prior.flux_peak, dtype=np.float64)
    sigma_calibrated = sigma_map is not None
```

And in the meta dict, add:
```python
        "sigma_calibrated": sigma_calibrated,
```

**Step 4: Run all tests**

```bash
python -m pytest tests/ -v
```
Expected: all existing tests still pass + 4 new sigma_map tests pass.

**Step 5: Commit**

```bash
git add src/injector/injector.py tests/test_injection.py
git commit -m "feat: inject() accepts sigma_map for per-epoch noise-calibrated amplitude"
```

---

## Task 9: `demo/make_demo_cases.py`

**Files:**
- Create: `demo/make_demo_cases.py`
- Create: `tests/test_make_demo_cases.py`

### Step 1: Write failing tests

```python
# tests/test_make_demo_cases.py
"""Tests for make_demo_cases.py — injection cases from a zarr stack."""
import numpy as np
import pytest
import zarr
from tests.conftest import write_fake_fits


def _write_fake_zarr(tmp_path, T=5, H=64, W=64, seed=0):
    """Create a minimal zarr store matching build_one_stack output."""
    rng = np.random.default_rng(seed)
    imgs = rng.normal(0, 5, (T, H, W)).astype(np.float32)
    mjds = np.array([57130.0 + i * 0.01 for i in range(T)])
    psf_fwhm = np.full(T, 2.5, dtype=np.float32)

    zarr_path = tmp_path / "stack.zarr"
    z = zarr.open(str(zarr_path), mode="w")
    z.create_dataset("images",     data=imgs)
    z.create_dataset("timestamps", data=mjds)
    z.create_dataset("psf_fwhm",   data=psf_fwhm)
    z.attrs["plate_scale"] = 0.263
    z.attrs["T"] = T
    z.attrs["patch_size"] = 32
    z.attrs["stride"] = 32
    return zarr_path


class TestMakeDemoCases:
    def test_creates_npz_files(self, tmp_path):
        from demo.make_demo_cases import make_demo_cases
        zarr_path = _write_fake_zarr(tmp_path)
        out_dir = tmp_path / "cases"
        make_demo_cases(str(zarr_path), str(out_dir), n_cases=2, seed=0)
        npz_files = list(out_dir.glob("*.npz"))
        assert len(npz_files) == 2

    def test_npz_has_required_keys(self, tmp_path):
        from demo.make_demo_cases import make_demo_cases
        zarr_path = _write_fake_zarr(tmp_path)
        out_dir = tmp_path / "cases"
        make_demo_cases(str(zarr_path), str(out_dir), n_cases=1, seed=0)
        npz = np.load(list((out_dir).glob("*.npz"))[0], allow_pickle=True)
        for key in ("patch_stack", "X", "Y", "sigma_patch",
                    "timestamps", "plate_scale"):
            assert key in npz, f"Missing key: {key}"

    def test_x_equals_patch_plus_y(self, tmp_path):
        """X = patch_stack + Y must hold exactly."""
        from demo.make_demo_cases import make_demo_cases
        zarr_path = _write_fake_zarr(tmp_path)
        out_dir = tmp_path / "cases"
        make_demo_cases(str(zarr_path), str(out_dir), n_cases=1, seed=0)
        npz = np.load(list((out_dir).glob("*.npz"))[0], allow_pickle=True)
        np.testing.assert_allclose(
            npz["X"].astype(np.float64),
            npz["patch_stack"].astype(np.float64) + npz["Y"].astype(np.float64),
            atol=1e-5,
        )
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_make_demo_cases.py -v
```
Expected: `ModuleNotFoundError: No module named 'demo.make_demo_cases'`

**Step 3: Write `demo/make_demo_cases.py`**

```python
"""
make_demo_cases.py – Generate injection cases from a zarr stack.

Usage::

    python demo/make_demo_cases.py \\
        --stack data/processed/stack01.zarr \\
        --out   demo/cases \\
        --n_cases 20 --seed 0

Each output .npz contains:
    patch_stack  : (T, P, P) float32  — background-subtracted patch
    X            : (T, P, P) float32  — patch + injected signal
    Y            : (T, P, P) float32  — injected signal only
    sigma_patch  : (T,)      float32  — per-epoch MAD noise in patch
    timestamps   : (T,)      float64  — MJD timestamps
    plate_scale  : float              — arcsec/px
    psf_fwhm     : (T,)      float32  — per-epoch PSF FWHM (pixels)
    meta         : dict               — injection provenance (stored as object)
"""
from __future__ import annotations

import sys
import os
import argparse
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.injector import inject, PSFParams, TargetConfig
from src.data.patches import extract_patch_grid, patch_sigma


def make_demo_cases(
    zarr_path: str,
    out_dir: str,
    n_cases: int = 20,
    seed: int = 0,
    sample_type: str = "tno",
) -> None:
    """
    Load a zarr stack, extract patches, inject signals, save .npz cases.

    Parameters
    ----------
    zarr_path   : path to zarr store produced by build_one_stack
    out_dir     : directory to write .npz files
    n_cases     : number of injection cases to generate
    seed        : base RNG seed (each case gets seed + i)
    sample_type : prior type (default 'tno')
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load zarr
    z = zarr.open(zarr_path, mode="r")
    imgs       = z["images"][:]          # (T, H, W)
    timestamps = z["timestamps"][:]      # (T,) MJD
    psf_fwhm   = z["psf_fwhm"][:]       # (T,)
    plate_scale = float(z.attrs["plate_scale"])
    patch_size  = int(z.attrs.get("patch_size", 64))
    stride      = int(z.attrs.get("stride", 32))

    # Build the patch pool once
    pool = list(extract_patch_grid(imgs, patch_size=patch_size, stride=stride))
    if not pool:
        raise ValueError(
            f"No patches of size {patch_size} fit in image shape {imgs.shape[1:]}. "
            f"Use a smaller --patch or larger images."
        )

    rng = np.random.default_rng(seed)
    n_saved = 0

    for i in range(n_cases):
        # Sample a random patch from the pool
        idx = int(rng.integers(len(pool)))
        patch_stack, row_start, col_start = pool[idx]

        # Per-epoch sigma from this patch
        sigma = patch_sigma(patch_stack)

        # Per-epoch PSF (use average FWHM across T for this case)
        fwhm_mean = float(psf_fwhm.mean())
        psf_params = PSFParams(fwhm_pixels=fwhm_mean)

        # Timestamps in hours relative to first epoch
        t_hours = (timestamps - timestamps[0]) * 24.0

        # Inject with sigma calibration
        X, Y, meta = inject(
            patch_stack=patch_stack,
            timestamps=t_hours,
            plate_scale=plate_scale,
            psf_params=psf_params,
            sample_type=sample_type,
            seed=int(seed + i),
            sigma_map=sigma,
        )

        case_name = f"case_{i:04d}"
        np.savez(
            out_dir / f"{case_name}.npz",
            patch_stack=patch_stack.astype(np.float32),
            X=X.astype(np.float32),
            Y=Y.astype(np.float32),
            sigma_patch=sigma,
            timestamps=timestamps,
            plate_scale=np.float32(plate_scale),
            psf_fwhm=psf_fwhm,
            meta=np.array(meta, dtype=object),
        )
        n_saved += 1

    print(f"Saved {n_saved} cases to {out_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate injection demo cases from a zarr stack."
    )
    parser.add_argument("--stack",    required=True, help="Path to zarr store")
    parser.add_argument("--out",      required=True, help="Output directory")
    parser.add_argument("--n_cases",  type=int,   default=20)
    parser.add_argument("--seed",     type=int,   default=0)
    parser.add_argument("--sample_type", default="tno")
    args = parser.parse_args()

    make_demo_cases(
        zarr_path=args.stack,
        out_dir=args.out,
        n_cases=args.n_cases,
        seed=args.seed,
        sample_type=args.sample_type,
    )


if __name__ == "__main__":
    main()
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_make_demo_cases.py -v
```
Expected: `3 passed`

**Step 5: Commit**

```bash
git add demo/make_demo_cases.py tests/test_make_demo_cases.py
git commit -m "feat: make_demo_cases.py generates injection .npz cases from zarr stack"
```

---

## Task 10: Update `demo/demo.py` to support `--case`

**Files:**
- Modify: `demo/demo.py`

### Step 1: Add `demo_from_case()` and `--case` argument

Replace `demo.py`'s `main()` with:

```python
def demo_from_case(case_path: str) -> None:
    """Replay a saved injection case (.npz) and print a summary."""
    data = np.load(case_path, allow_pickle=True)
    patch_stack = data["patch_stack"]
    X = data["X"]
    Y = data["Y"]
    meta = data["meta"].item()

    print(f"\n=== Case: {os.path.basename(case_path)} ===")
    print(f"  patch shape  : {patch_stack.shape}")
    print(f"  Y peak       : {Y.max():.3f}")
    print(f"  Y total      : {Y.sum():.3f}")
    print(f"  flux_peak    : {meta['flux_peak']:.3f}")
    print(f"  sigma_calib  : {meta.get('sigma_calibrated', False)}")
    print(f"  motion_ra    : {meta['motion_ra_arcsec_per_hour']:.4f} arcsec/hr")
    print(f"  motion_dec   : {meta['motion_dec_arcsec_per_hour']:.4f} arcsec/hr")
    print()


def main():
    parser = argparse.ArgumentParser(description="InjectionEngine demo")
    parser.add_argument("--plot", action="store_true", help="Show matplotlib figures")
    parser.add_argument("--case", default=None,
                        help="Path to a saved .npz case (e.g. demo/cases/case_0000.npz)")
    args = parser.parse_args()

    if args.case:
        demo_from_case(args.case)
        return

    patch, times, X, Y, meta = demo_basic()
    demo_all_types()
    demo_reproducibility()
    demo_target_strategies()
    demo_sanity()

    if args.plot:
        demo_plot(patch, X, Y, meta)

    print("All demo sections completed successfully.")
```

**Step 2: Verify existing demo still works**

```bash
python demo/demo.py
```
Expected: same 6/6 pass output as before.

**Step 3: Commit**

```bash
git add demo/demo.py
git commit -m "feat: demo.py --case mode loads and replays a saved injection case"
```

---

## Task 11: End-to-end verification

**Step 1: Run full test suite**

```bash
python -m pytest tests/ -v
```
Expected: all tests pass (≥ 50 total across all modules).

**Step 2: Run full end-to-end pipeline**

```bash
# Setup data (once)
mkdir -p data/raw/stack01
cp kbmod/kbmod/data/small/00000{0,1,2,3,4}.fits data/raw/stack01/

# Build zarr
python -m src.data.build_one_stack \
  --stack_dir data/raw/stack01 \
  --out data/processed/stack01.zarr \
  --T 5 --patch 64 --stride 32

# Generate demo cases
python demo/make_demo_cases.py \
  --stack data/processed/stack01.zarr \
  --out demo/cases --n_cases 5

# Run a case
python demo/demo.py --case demo/cases/case_0000.npz

# Run original demo (must still work)
python demo/demo.py
```

Expected final output from `demo/demo.py --case demo/cases/case_0000.npz`:
```
=== Case: case_0000.npz ===
  patch shape  : (5, 32, 32)     ← or (5, 64, 64) depending on patch_size
  Y peak       : 4.xxx
  Y total      : ...
  flux_peak    : ...
  sigma_calib  : True
  motion_ra    : -2.xxxx arcsec/hr
  motion_dec   : 0.xxxx arcsec/hr
```

**Step 3: Run priors sanity (Step 2 regression)**

```bash
python -m src.injector.sanity --priors tno --mode kbo --N 10000
```
Expected: `[PASS]` on all 5 checks.

**Step 4: Commit**

```bash
git add -A
git commit -m "chore: Step 3 complete — end-to-end real-data pipeline verified"
```

---

## Notes for implementer

- `data/` is gitignored — never commit FITS or zarr files.
- The kbmod images are synthetic (pre-background-subtracted, all-zero variance, all-4 mask).
  `background.py` still runs but produces near-zero correction and sigma ≈ 3–4 ADU.
- PSF fitting on kbmod synthetic images may return the fallback value (2.5 px) since
  there may not be isolated point sources bright enough to fit cleanly. That is fine for Step 3.
- The user's spec mentions `--patch 128 --stride 128` for MegaCam data. With 64×64 kbmod
  test images, use `--patch 64 --stride 32`. The CLI accepts any values.
- Do NOT add CADC automation, WCS-aware ecliptic direction, dropout_mask, or anything
  not in this plan. Those are Step 4+ concerns.
- `tests/conftest.py` must exist before Task 7 tests run (created in Task 7 Step 1).
  Update `test_data_load_stack.py` to import the helper from there (avoid duplication).
