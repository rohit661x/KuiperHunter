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


# ---------------------------------------------------------------------------
# Prior report (Step 2)
# ---------------------------------------------------------------------------

def report_priors(
    sample_type: str = "tno",
    mode: str = "kbo",
    N: int = 100_000,
    T: int = 5,
    baseline_hours: float = 4.0,
    plate_scale: float = 0.187,
    patch: int = 128,
    out_dir: str | None = None,
    seed: int | None = None,
    plot: bool = True,
) -> dict:
    """
    Draw N prior samples, compute statistics, run pass/fail checks.

    Parameters
    ----------
    sample_type   : prior name (currently "tno" routes to KBO sampler).
    mode          : "kbo" (only mode with hard thresholds).
    N             : number of samples to draw.
    T             : number of frames (for drift_px).
    baseline_hours: total observation baseline (hours).
    plate_scale   : arcsec/px.
    patch         : patch edge length in pixels (for clipping rate).
    out_dir       : directory for saved files; None -> no files written.
    seed          : RNG seed.
    plot          : whether to save matplotlib figures.

    Returns
    -------
    dict with arrays: R_au, mu, phi_offset, snr, drift_px, clipping_rate,
    class_counts, checks.
    """
    import os
    from .kbo_prior import sample_kbo, KBOConfig

    cfg = KBOConfig(
        mode=mode,
        plate_scale=plate_scale,
        baseline_hours=baseline_hours,
        T=T,
    )
    rng = np.random.default_rng(seed)

    R_arr, mu_arr, phi_arr, snr_arr = [], [], [], []
    drift_arr = []
    class_counts: dict[str, int] = {}

    for _ in range(N):
        s = sample_kbo(rng, cfg)
        R_arr.append(s.R_au)
        mu_arr.append(s.mu_arcsec_hr)
        phi_arr.append(s.phi_offset_deg)
        snr_arr.append(s.snr)
        # Total drift over the observation baseline (in pixels)
        total_drift = np.hypot(
            s.vx_px_per_frame * (T - 1),
            s.vy_px_per_frame * (T - 1),
        )
        drift_arr.append(total_drift)
        class_counts[s.population_class] = class_counts.get(s.population_class, 0) + 1

    R_arr   = np.array(R_arr)
    mu_arr  = np.array(mu_arr)
    phi_arr = np.array(phi_arr)
    snr_arr = np.array(snr_arr)
    drift_arr = np.array(drift_arr)

    clipping_rate = float((drift_arr > patch).mean())

    # --- Pass/fail checks (KBO mode) ---
    checks: dict[str, bool] = {}
    if mode == "kbo":
        checks["mu_max_le_4.5"]      = float(mu_arr.max())  <= 4.5
        checks["mu_mean_2.5_to_3.5"] = 2.5 <= float(mu_arr.mean()) <= 3.5
        checks["phi_p95_le_35deg"]   = float(np.percentile(np.abs(phi_arr), 95)) <= 35.0
        checks["R_mean_40_to_46"]    = 40.0 <= float(R_arr.mean()) <= 46.0
        checks["clipping_le_5pct"]   = clipping_rate <= 0.05

    # --- Console output ---
    print(f"\n=== Prior report: sample_type={sample_type}  mode={mode}  N={N:,} ===\n")

    print("Population class counts:")
    for cls, cnt in sorted(class_counts.items()):
        print(f"  {cls:20s}: {cnt:7d}  ({cnt/N*100:.1f}%)")

    def _stats(name, arr, fmt=".2f"):
        fmt_s = f"{{:{fmt}}}"
        print(f"\n{name:12s}: mean={fmt_s.format(arr.mean())}  "
              f"std={fmt_s.format(arr.std())}  "
              f"p5={fmt_s.format(np.percentile(arr, 5))}  "
              f"p95={fmt_s.format(np.percentile(arr, 95))}  "
              f"max={fmt_s.format(arr.max())}")

    _stats("R_au",      R_arr)
    _stats("mu",        mu_arr,  ".3f")
    _stats("phi_off°",  phi_arr, ".1f")
    _stats("snr",       snr_arr, ".2f")
    _stats("drift_px",  drift_arr, ".1f")
    print(f"\n{'clipping_rate':12s}: {clipping_rate:.4f}  "
          f"({clipping_rate*100:.2f}%)  [patch={patch}px]")

    if checks:
        print(f"\n=== Pass/fail (mode={mode}) ===")
        for name, ok in checks.items():
            tag = "PASS" if ok else "FAIL"
            print(f"  [{tag}] {name}")

    # --- Save files ---
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

        txt_path = os.path.join(out_dir, "clipping_rate.txt")
        with open(txt_path, "w") as fh:
            fh.write(f"clipping_rate={clipping_rate:.6f}\n")
            fh.write(f"patch={patch}\nN={N}\nT={T}\n"
                     f"baseline_hours={baseline_hours}\nplate_scale={plate_scale}\n")
            fh.write(f"drift_px_mean={drift_arr.mean():.3f}\n")
            fh.write(f"drift_px_p95={np.percentile(drift_arr,95):.3f}\n")

        if plot:
            _save_plots(
                out_dir, R_arr, mu_arr, phi_arr, snr_arr, drift_arr,
                patch=patch, class_counts=class_counts, N=N,
            )

    return {
        "R_au":          R_arr,
        "mu":            mu_arr,
        "phi_offset":    phi_arr,
        "snr":           snr_arr,
        "drift_px":      drift_arr,
        "clipping_rate": clipping_rate,
        "class_counts":  class_counts,
        "checks":        checks,
    }


def _save_plots(
    out_dir: str,
    R_arr, mu_arr, phi_arr, snr_arr, drift_arr,
    patch: int,
    class_counts: dict,
    N: int,
) -> None:
    """Save five diagnostic plots to out_dir."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not installed; skipping plots.")
        return

    import os

    fig_kw = dict(figsize=(7, 4), tight_layout=True)

    # hist_R.png
    fig, ax = plt.subplots(**fig_kw)
    ax.hist(R_arr, bins=60, color="steelblue", edgecolor="none", alpha=0.8)
    ax.axvline(R_arr.mean(), color="red", lw=1.5, label=f"mean={R_arr.mean():.1f}")
    ax.set_xlabel("R (AU)")
    ax.set_ylabel("Count")
    ax.set_title("Heliocentric Distance Distribution")
    ax.legend()
    fig.savefig(os.path.join(out_dir, "hist_R.png"), dpi=120)
    plt.close(fig)

    # hist_mu.png
    fig, ax = plt.subplots(**fig_kw)
    ax.hist(mu_arr, bins=60, color="darkorange", edgecolor="none", alpha=0.8)
    ax.axvline(4.5, color="crimson", lw=1.5, ls="--", label="4.5 cap")
    ax.axvline(mu_arr.mean(), color="navy", lw=1.5, label=f"mean={mu_arr.mean():.2f}")
    ax.set_xlabel("mu (arcsec/hr)")
    ax.set_ylabel("Count")
    ax.set_title("Sky-Plane Motion Rate")
    ax.legend()
    fig.savefig(os.path.join(out_dir, "hist_mu.png"), dpi=120)
    plt.close(fig)

    # hist_phi_offset.png
    fig, ax = plt.subplots(**fig_kw)
    ax.hist(phi_arr, bins=60, color="mediumseagreen", edgecolor="none", alpha=0.8)
    ax.axvline(-35, color="crimson", lw=1.5, ls="--", label="+/-35 deg")
    ax.axvline(+35, color="crimson", lw=1.5, ls="--")
    ax.set_xlabel("phi offset (deg)")
    ax.set_ylabel("Count")
    ax.set_title("Direction Offset from Ecliptic")
    ax.legend()
    fig.savefig(os.path.join(out_dir, "hist_phi_offset.png"), dpi=120)
    plt.close(fig)

    # hist_snr.png
    fig, ax = plt.subplots(**fig_kw)
    ax.hist(snr_arr, bins=40, color="mediumpurple", edgecolor="none", alpha=0.8)
    for v, ls in [(3, "-"), (6, "--"), (10, "-")]:
        ax.axvline(v, color="crimson", lw=1.2, ls=ls)
    ax.set_xlabel("SNR")
    ax.set_ylabel("Count")
    ax.set_title("SNR Distribution (faint-heavy)")
    fig.savefig(os.path.join(out_dir, "hist_snr.png"), dpi=120)
    plt.close(fig)

    # hist_drift_px.png
    fig, ax = plt.subplots(**fig_kw)
    ax.hist(drift_arr, bins=60, color="salmon", edgecolor="none", alpha=0.8)
    ax.axvline(patch, color="crimson", lw=1.5, ls="--",
               label=f"patch={patch}px")
    ax.axvline(drift_arr.mean(), color="navy", lw=1.5,
               label=f"mean={drift_arr.mean():.1f}px")
    ax.set_xlabel("Total drift (px)")
    ax.set_ylabel("Count")
    ax.set_title("Drift in Pixels")
    ax.legend()
    fig.savefig(os.path.join(out_dir, "hist_drift_px.png"), dpi=120)
    plt.close(fig)

    print(f"  Plots saved to {out_dir}/")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(
        description="InjectionEngine sanity reporter — priors and injection checks."
    )
    parser.add_argument("--priors",         default=None,
                        help="Run prior report for this sample type (e.g. tno)")
    parser.add_argument("--mode",           default="kbo",
                        choices=["kbo", "broad"])
    parser.add_argument("--N",              type=int,   default=100_000)
    parser.add_argument("--T",              type=int,   default=5)
    parser.add_argument("--baseline_hours", type=float, default=4.0)
    parser.add_argument("--plate_scale",    type=float, default=0.187)
    parser.add_argument("--patch",          type=int,   default=128)
    parser.add_argument("--seed",           type=int,   default=None)
    parser.add_argument("--no-plot",        action="store_true",
                        help="Skip saving matplotlib figures")
    parser.add_argument("--out-dir",        default=None,
                        help="Output directory (default: demo/priors_report/)")

    args = parser.parse_args()

    if args.priors is not None:
        out_dir = args.out_dir
        if out_dir is None:
            here = os.path.dirname(os.path.abspath(__file__))
            root = os.path.join(here, "..", "..", "demo", "priors_report")
            out_dir = os.path.normpath(root)

        results = report_priors(
            sample_type=args.priors,
            mode=args.mode,
            N=args.N,
            T=args.T,
            baseline_hours=args.baseline_hours,
            plate_scale=args.plate_scale,
            patch=args.patch,
            out_dir=out_dir,
            seed=args.seed,
            plot=not args.no_plot,
        )

        failed = [k for k, v in results.get("checks", {}).items() if not v]
        if failed:
            print(f"\n[ERROR] {len(failed)} check(s) failed: {failed}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
