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
    imgs        = z["images"][:]         # (T, H, W)
    timestamps  = z["timestamps"][:]     # (T,) MJD
    psf_fwhm    = z["psf_fwhm"][:]      # (T,)
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

        # PSF — use mean FWHM across epochs
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
    parser.add_argument("--stack",       required=True, help="Path to zarr store")
    parser.add_argument("--out",         required=True, help="Output directory")
    parser.add_argument("--n_cases",     type=int,   default=20)
    parser.add_argument("--seed",        type=int,   default=0)
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
