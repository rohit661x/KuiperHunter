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
