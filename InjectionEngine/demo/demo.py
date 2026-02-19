"""
demo.py – Interactive walkthrough of the InjectionEngine API.

Run::

    python demo/demo.py

Optionally pass --plot to display matplotlib figures (requires matplotlib).
"""

from __future__ import annotations

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.injector import inject, PSFParams, TargetConfig
from src.injector.sanity import run_all


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_patch(n_frames=10, size=64, sky=500, noise=10, seed=7):
    rng = np.random.default_rng(seed)
    return (
        rng.poisson(sky, (n_frames, size, size)).astype(np.float64)
        + rng.normal(0, noise, (n_frames, size, size))
    )


def make_timestamps(n=10, cadence=0.5):
    return np.arange(n, dtype=np.float64) * cadence


# ---------------------------------------------------------------------------
# Demo sections
# ---------------------------------------------------------------------------

def demo_basic():
    print("=" * 60)
    print("1. Basic injection (TNO prior, Gaussian PSF)")
    print("=" * 60)

    patch = make_patch()
    times = make_timestamps()
    psf = PSFParams(fwhm_pixels=2.5)

    X, Y, meta = inject(
        patch_stack=patch,
        timestamps=times,
        plate_scale=0.263,          # arcsec/pixel (DECam-like)
        psf_params=psf,
        sample_type="tno",
        seed=42,
    )

    print(f"  patch shape : {patch.shape}")
    print(f"  X shape     : {X.shape}")
    print(f"  Y shape     : {Y.shape}")
    print(f"  flux_peak   : {meta['flux_peak']:.1f} counts")
    print(f"  motion_ra   : {meta['motion_ra_arcsec_per_hour']:.4f} arcsec/hr")
    print(f"  motion_dec  : {meta['motion_dec_arcsec_per_hour']:.4f} arcsec/hr")
    print(f"  start (x,y) : ({meta['start_x_px']:.2f}, {meta['start_y_px']:.2f}) px")
    print()
    return patch, times, X, Y, meta


def demo_all_types():
    print("=" * 60)
    print("2. All sample types")
    print("=" * 60)

    patch = make_patch()
    times = make_timestamps()
    psf = PSFParams(fwhm_pixels=2.5)

    for st in ("tno", "mba", "nea", "static"):
        X, Y, meta = inject(patch, times, 0.263, psf, sample_type=st, seed=0)
        peak = Y.max()
        total = Y.sum()
        print(
            f"  {st:8s}  Y_peak={peak:8.2f}  Y_total={total:10.1f}  "
            f"motion=({meta['motion_ra_arcsec_per_hour']:+.3f}, "
            f"{meta['motion_dec_arcsec_per_hour']:+.3f}) arcsec/hr"
        )
    print()


def demo_reproducibility():
    print("=" * 60)
    print("3. Seed reproducibility")
    print("=" * 60)

    patch = make_patch()
    times = make_timestamps()
    psf = PSFParams(fwhm_pixels=2.5)

    X1, Y1, m1 = inject(patch, times, 0.263, psf, seed=123)
    X2, Y2, m2 = inject(patch, times, 0.263, psf, seed=123)
    X3, Y3, m3 = inject(patch, times, 0.263, psf, seed=456)

    same = np.array_equal(X1, X2)
    diff = not np.array_equal(X1, X3)
    print(f"  Same seed → identical outputs : {same}")
    print(f"  Diff seed → different outputs : {diff}")
    print()


def demo_target_strategies():
    print("=" * 60)
    print("4. Target placement strategies")
    print("=" * 60)

    patch = make_patch()
    times = make_timestamps()
    psf = PSFParams(fwhm_pixels=2.5)
    n_rows, n_cols = patch.shape[1], patch.shape[2]
    cx, cy = n_cols / 2, n_rows / 2

    for strategy in ("uniform", "center", "grid"):
        tc = TargetConfig(strategy=strategy)
        _, _, meta = inject(patch, times, 0.263, psf, seed=7, target_config=tc)
        sx, sy = meta["start_x_px"], meta["start_y_px"]
        dist = np.hypot(sx - cx, sy - cy)
        print(f"  {strategy:8s}  start=({sx:.1f}, {sy:.1f})  dist_from_centre={dist:.1f} px")
    print()


def demo_sanity():
    print("=" * 60)
    print("5. Sanity checks")
    print("=" * 60)

    patch = make_patch()
    times = make_timestamps()
    psf = PSFParams(fwhm_pixels=2.5)

    run_all(inject, patch, times, 0.263, psf, sample_type="mba", seed=99)


def demo_plot(patch, X, Y, meta):
    """Optional matplotlib visualisation."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots.")
        return

    mid = len(X) // 2
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    im_kw = dict(origin="lower", cmap="gray")

    axes[0].imshow(patch[mid], **im_kw)
    axes[0].set_title(f"Original (frame {mid})")

    axes[1].imshow(Y[mid], origin="lower", cmap="hot")
    axes[1].set_title("Injected signal Y")

    axes[2].imshow(X[mid], **im_kw)
    axes[2].set_title("X = patch + Y")

    # Overlay trajectory
    xs = meta["trajectory_x"]
    ys = meta["trajectory_y"]
    for ax in axes[1:]:
        ax.plot(xs, ys, "c--", lw=0.8, alpha=0.6)
        ax.plot(xs[mid], ys[mid], "cx", ms=8, mew=1.5)

    plt.suptitle(
        f"sample_type={meta['sample_type']}  "
        f"flux_peak={meta['flux_peak']:.0f}  "
        f"seed={meta['seed']}"
    )
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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


if __name__ == "__main__":
    main()
