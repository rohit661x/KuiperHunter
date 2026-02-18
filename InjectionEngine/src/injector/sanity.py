"""
sanity.py – Fast, self-contained checks to verify injection correctness.

These are not unit tests (no framework dependency); call run_all() from a
notebook or script to get a quick confidence report.
"""

from __future__ import annotations

import numpy as np
from typing import Callable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rel_err(a: float, b: float) -> float:
    return abs(a - b) / (abs(b) + 1e-30)


def _check(name: str, ok: bool, msg: str = "") -> dict:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f": {msg}" if msg else ""))
    return {"name": name, "ok": ok, "msg": msg}


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_flux_conservation(X, Y, patch_stack, tol: float = 0.01) -> dict:
    """Total injected flux matches sum(Y) within tol (relative)."""
    diff = (X - patch_stack).sum()
    inj = Y.sum()
    ok = _rel_err(diff, inj) < tol
    return _check(
        "flux_conservation",
        ok,
        f"X-patch sum={diff:.4g}, Y sum={inj:.4g}, rel_err={_rel_err(diff, inj):.2e}",
    )


def check_y_nonnegative(Y) -> dict:
    """Injected signal Y should be everywhere ≥ 0."""
    ok = float(Y.min()) >= -1e-9
    return _check("Y_nonneg", ok, f"min(Y)={Y.min():.3e}")


def check_output_shape(X, Y, patch_stack) -> dict:
    """X and Y must have the same shape as patch_stack."""
    ok = X.shape == patch_stack.shape and Y.shape == patch_stack.shape
    return _check(
        "output_shape",
        ok,
        f"patch={patch_stack.shape}, X={X.shape}, Y={Y.shape}",
    )


def check_no_nan(X, Y) -> dict:
    """Neither X nor Y contains NaN or Inf."""
    ok = np.isfinite(X).all() and np.isfinite(Y).all()
    return _check("no_nan_inf", ok,
                  f"X finite={np.isfinite(X).all()}, Y finite={np.isfinite(Y).all()}")


def check_motion_direction(meta) -> dict:
    """
    Source should move in the expected direction across frames.
    Checks sign of x-displacement vs sign of motion_ra.
    """
    xs = np.array(meta["trajectory_x"])
    motion_ra = meta["motion_ra_arcsec_per_hour"]
    if abs(motion_ra) < 1e-6 or len(xs) < 2:
        return _check("motion_direction", True, "skipped (zero or single-frame)")
    dx = xs[-1] - xs[0]
    expected_sign = np.sign(motion_ra)
    ok = np.sign(dx) == expected_sign
    return _check(
        "motion_direction",
        ok,
        f"motion_ra={motion_ra:.3g}, dx={dx:.3f}",
    )


def check_seed_reproducibility(inject_fn: Callable, *args, **kwargs) -> dict:
    """
    Calling inject with the same seed must produce bit-identical results.
    Uses keyword argument 'seed' from kwargs (default 42).
    """
    seed = kwargs.pop("seed", 42)
    X1, Y1, _ = inject_fn(*args, seed=seed, **kwargs)
    X2, Y2, _ = inject_fn(*args, seed=seed, **kwargs)
    ok = np.array_equal(X1, X2) and np.array_equal(Y1, Y2)
    return _check("seed_reproducibility", ok)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all(
    inject_fn: Callable,
    patch_stack: np.ndarray,
    timestamps: np.ndarray,
    plate_scale: float,
    psf_params,
    sample_type: str = "tno",
    seed: int = 0,
    **inject_kwargs,
) -> list[dict]:
    """
    Run all sanity checks and return a list of result dicts.

    Parameters mirror injector.inject; inject_fn should be injector.inject.
    """
    print(f"\n=== InjectionEngine sanity checks (type={sample_type}, seed={seed}) ===")

    X, Y, meta = inject_fn(
        patch_stack, timestamps, plate_scale, psf_params,
        sample_type=sample_type, seed=seed, **inject_kwargs
    )

    results = [
        check_output_shape(X, Y, patch_stack),
        check_no_nan(X, Y),
        check_y_nonnegative(Y),
        check_flux_conservation(X, Y, patch_stack),
        check_motion_direction(meta),
        check_seed_reproducibility(
            inject_fn,
            patch_stack, timestamps, plate_scale, psf_params,
            sample_type=sample_type,
            **inject_kwargs,
        ),
    ]

    n_pass = sum(r["ok"] for r in results)
    print(f"\n  {n_pass}/{len(results)} checks passed.\n")
    return results
