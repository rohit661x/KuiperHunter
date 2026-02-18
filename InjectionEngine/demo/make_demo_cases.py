"""
make_demo_cases.py – Generate and save a set of reference demo cases to disk.

Run directly::

    python demo/make_demo_cases.py

Outputs one .npz file per case in demo/cases/.
"""

from __future__ import annotations

import sys
import os
import numpy as np

# Allow running from repo root or from demo/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.injector import inject, PSFParams, TargetConfig


# ---------------------------------------------------------------------------
# Synthetic patch factory
# ---------------------------------------------------------------------------

def make_blank_patch(
    n_frames: int = 10,
    n_rows: int = 64,
    n_cols: int = 64,
    sky_level: float = 500.0,
    read_noise: float = 10.0,
    seed: int = 99,
) -> np.ndarray:
    """Return a stack of sky-dominated Gaussian-noise frames."""
    rng = np.random.default_rng(seed)
    shot = rng.poisson(sky_level, size=(n_frames, n_rows, n_cols)).astype(np.float64)
    read = rng.normal(0, read_noise, size=(n_frames, n_rows, n_cols))
    return shot + read


def make_timestamps(n_frames: int = 10, cadence_hours: float = 0.5) -> np.ndarray:
    """Uniformly spaced timestamps starting at 0."""
    return np.arange(n_frames, dtype=np.float64) * cadence_hours


# ---------------------------------------------------------------------------
# Case definitions
# ---------------------------------------------------------------------------

CASES = [
    dict(
        name="tno_slow",
        sample_type="tno",
        seed=0,
        psf_fwhm=2.5,
        plate_scale=0.263,
    ),
    dict(
        name="mba_moderate",
        sample_type="mba",
        seed=1,
        psf_fwhm=2.5,
        plate_scale=0.263,
    ),
    dict(
        name="nea_fast",
        sample_type="nea",
        seed=2,
        psf_fwhm=3.0,
        plate_scale=0.263,
    ),
    dict(
        name="static_star",
        sample_type="static",
        seed=3,
        psf_fwhm=2.5,
        plate_scale=0.263,
    ),
    dict(
        name="tno_center_target",
        sample_type="tno",
        seed=4,
        psf_fwhm=2.5,
        plate_scale=0.263,
        target_strategy="center",
    ),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = os.path.join(os.path.dirname(__file__), "cases")
    os.makedirs(out_dir, exist_ok=True)

    patch_stack = make_blank_patch()
    timestamps = make_timestamps()

    for case in CASES:
        name = case["name"]
        psf = PSFParams(fwhm_pixels=case["psf_fwhm"])
        tc = TargetConfig(strategy=case.get("target_strategy", "uniform"))

        X, Y, meta = inject(
            patch_stack,
            timestamps,
            plate_scale=case["plate_scale"],
            psf_params=psf,
            sample_type=case["sample_type"],
            seed=case["seed"],
            target_config=tc,
        )

        out_path = os.path.join(out_dir, f"{name}.npz")
        np.savez_compressed(
            out_path,
            X=X,
            Y=Y,
            patch_stack=patch_stack,
            timestamps=timestamps,
        )
        print(f"Saved {out_path}")
        print(
            f"  flux_peak={meta['flux_peak']:.1f}  "
            f"motion_ra={meta['motion_ra_arcsec_per_hour']:.3f} '/hr  "
            f"motion_dec={meta['motion_dec_arcsec_per_hour']:.3f} '/hr"
        )

    print(f"\nDone – {len(CASES)} cases written to {out_dir}/")


if __name__ == "__main__":
    main()
