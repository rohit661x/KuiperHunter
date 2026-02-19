"""
injector.py – Public API for the InjectionEngine.

Single entry point::

    X, Y, meta = inject(patch_stack, timestamps, plate_scale,
                        psf_params, sample_type, seed)

X : float64 array (n_frames, n_rows, n_cols) – patch_stack + injected signal
Y : float64 array (n_frames, n_rows, n_cols) – the injected signal alone
meta : dict – all parameters used, suitable for logging / provenance
"""

from __future__ import annotations

import numpy as np
from dataclasses import asdict

from .priors import PriorSample, sample as draw_prior
from .trajectory import build_trajectory
from .render_psf import PSFParams, render_stack
from .targets import TargetConfig, draw_target


def inject(
    patch_stack: np.ndarray,
    timestamps: np.ndarray,
    plate_scale: float,
    psf_params: PSFParams,
    sample_type: str = "tno",
    seed: int | None = None,
    *,
    target_config: TargetConfig | None = None,
    sigma_map: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Inject a synthetic moving source into a patch stack.

    Parameters
    ----------
    patch_stack  : float array, shape (n_frames, n_rows, n_cols).
                   The raw image data.  The injected signal is *added* to a
                   copy; the original is not modified.
    timestamps   : float array, shape (n_frames,).
                   Observation times in hours (arbitrary zero-point; the
                   engine uses relative times internally).
    plate_scale  : float – arcsec per pixel.
    psf_params   : PSFParams – model and shape of the point-spread function.
    sample_type  : str – name of the prior to draw from (e.g. 'tno', 'mba',
                   'nea', 'static').
    seed         : int or None – RNG seed for reproducibility.
    target_config: TargetConfig or None – controls where in the patch the
                   source is placed.  Defaults to uniform sampling.

    Returns
    -------
    X    : float64 array (n_frames, n_rows, n_cols) – input + injected signal.
    Y    : float64 array (n_frames, n_rows, n_cols) – injected signal only.
    meta : dict – provenance record containing all drawn / derived parameters.
    """
    # ------------------------------------------------------------------ setup
    patch_stack = np.asarray(patch_stack, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)

    if patch_stack.ndim != 3:
        raise ValueError(
            f"patch_stack must be 3-D (n_frames, n_rows, n_cols), "
            f"got shape {patch_stack.shape}."
        )
    n_frames, n_rows, n_cols = patch_stack.shape
    if len(timestamps) != n_frames:
        raise ValueError(
            f"timestamps length {len(timestamps)} != n_frames {n_frames}."
        )
    if plate_scale <= 0:
        raise ValueError(f"plate_scale must be positive, got {plate_scale}.")

    rng = np.random.default_rng(seed)

    if target_config is None:
        target_config = TargetConfig()

    # ----------------------------------------------------------- draw priors
    prior: PriorSample = draw_prior(sample_type, rng)

    # Override start position with target strategy
    prior.start_x, prior.start_y = draw_target(
        (n_rows, n_cols), target_config, rng
    )

    # ------------------------------------------------------- build trajectory
    from .trajectory import build_trajectory, is_in_patch

    traj = build_trajectory(
        timestamps=timestamps,
        start_x=prior.start_x,
        start_y=prior.start_y,
        motion_ra=prior.motion_ra,
        motion_dec=prior.motion_dec,
        plate_scale=plate_scale,
    )

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

    # ------------------------------------------------------- render & inject
    Y = render_stack(
        patch_shape=(n_rows, n_cols),
        xs=traj.xs,
        ys=traj.ys,
        fluxes=fluxes,
        psf_params=psf_params,
    )

    X = patch_stack + Y

    # ------------------------------------------------------ build meta record
    in_patch = is_in_patch(traj, (n_rows, n_cols))

    psf_meta: dict
    if psf_params.model == "gaussian":
        psf_meta = {"model": "gaussian", "fwhm_pixels": psf_params.fwhm_pixels}
    else:
        psf_meta = {
            "model": "empirical",
            "kernel_shape": list(psf_params.kernel.shape),
        }

    meta = {
        "sample_type": sample_type,
        "seed": seed,
        "plate_scale_arcsec_per_px": plate_scale,
        "n_frames": n_frames,
        "patch_shape": [n_rows, n_cols],
        # Prior draws
        "flux_peak": prior.flux_peak,
        "motion_ra_arcsec_per_hour": prior.motion_ra,
        "motion_dec_arcsec_per_hour": prior.motion_dec,
        # Trajectory
        "start_x_px": prior.start_x,
        "start_y_px": prior.start_y,
        "trajectory_x": traj.xs.tolist(),
        "trajectory_y": traj.ys.tolist(),
        "frames_in_patch": in_patch.tolist(),
        # PSF
        "psf": psf_meta,
        # Target config
        "target_strategy": target_config.strategy,
        "sigma_calibrated": sigma_calibrated,
    }
    if prior.extra:
        meta["prior_extra"] = prior.extra

    return X, Y, meta
