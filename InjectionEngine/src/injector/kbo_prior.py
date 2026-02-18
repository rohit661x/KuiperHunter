"""
kbo_prior.py – Physics-based conditional prior for KBO/TNO injection.

Unit contract
─────────────
mu_arcsec_hr      : scalar sky-plane speed (arcsec/hr) — physics truth
phi_img_rad       : image-plane direction (radians)
motion_ra/dec     : arcsec/hr components — trajectory.py converts to px
vx/vy_px_per_frame: px/frame for sanity checks ONLY (mirrors trajectory.py)
flux_peak         : dimensionless SNR placeholder until Step 3 noise model
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class KBOConfig:
    mode: str = "kbo"
    phi_ecl_sigma_deg: float = 10.0    # spread of KBO directions around ecliptic
    motion_scatter: float = 0.08       # multiplicative scatter on mu(R)
    nominal_sigma_sky: float = 10.0    # ADU — Step 2 placeholder only
    # Canonical sanity defaults (MegaCam)
    plate_scale: float = 0.187         # arcsec/px
    baseline_hours: float = 4.0
    T: int = 5

    def __post_init__(self):
        if self.T < 2:
            raise ValueError(f"KBOConfig.T must be >= 2 (got {self.T}); need at least 2 frames for a time baseline.")

    @property
    def dt_hours(self) -> float:
        """Time step between frames (hours)."""
        return self.baseline_hours / (self.T - 1)


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class KBOSample:
    # --- Physics truth ---
    population_class: str        # classical_cold | classical_hot | plutino | scattering
    R_au: float
    mu_arcsec_hr: float          # sky-plane speed (arcsec/hr)
    phi_offset_deg: float        # offset from ecliptic direction (deg)
    snr: float
    dropout_mask: list | None    # None in Step 2; per-epoch mask in Step 3+

    # --- Derived: direction ---
    phi_img_rad: float           # π + radians(phi_offset_deg)  [Step 2 canonical]

    # --- Derived: injector-compatible (arcsec/hr) ---
    # NOTE: trajectory.py expects arcsec/hr. Do NOT put px/frame here.
    motion_ra: float             # mu * cos(phi_img_rad)   [arcsec/hr]
    motion_dec: float            # mu * sin(phi_img_rad)   [arcsec/hr]

    # --- Derived: sanity-report only (px/frame, mirrors trajectory.py) ---
    vx_px_per_frame: float       # (motion_ra / plate_scale) * dt_hours
    vy_px_per_frame: float       # (motion_dec / plate_scale) * dt_hours

    # --- Adapter ---
    flux_peak: float             # = snr  (dimensionless placeholder, Step 2)
    mode: str


# ---------------------------------------------------------------------------
# KBO class mixture weights
# ---------------------------------------------------------------------------

KBO_MIXTURE: dict[str, float] = {
    "classical_cold": 0.45,
    "classical_hot":  0.25,
    "plutino":        0.20,
    "scattering":     0.10,
}

_CLASSES = list(KBO_MIXTURE.keys())
_WEIGHTS = np.array(list(KBO_MIXTURE.values()), dtype=np.float64)


def _sample_population_class(rng: np.random.Generator) -> str:
    idx = rng.choice(len(_CLASSES), p=_WEIGHTS)
    return _CLASSES[idx]


# ---------------------------------------------------------------------------
# Distance sampling
# ---------------------------------------------------------------------------

def _sample_R(population_class: str, rng: np.random.Generator) -> float:
    """Sample heliocentric distance R (AU) conditional on population class."""
    if population_class in ("classical_cold", "classical_hot"):
        R = rng.normal(44.0, 2.0)
        return float(np.clip(R, 40.0, 50.0))
    elif population_class == "plutino":
        R = rng.normal(39.4, 1.0)
        return float(np.clip(R, 37.0, 42.0))
    elif population_class == "scattering":
        # log-uniform [30, 100]
        log_R = rng.uniform(math.log(30.0), math.log(100.0))
        return float(math.exp(log_R))
    else:
        raise ValueError(f"Unknown population_class: '{population_class}'")


# ---------------------------------------------------------------------------
# Motion model
# ---------------------------------------------------------------------------

# Calibration constant: mu = MU_K / R_au  (arcsec·AU/hr)
# Derived from opposition parallax approximation.
# Gives mu(44 AU) ≈ 2.92 arcsec/hr, mu(39.4) ≈ 3.26, mu(30) ≈ 4.28.
# This is a first-order approximation; will be refined with orbital elements.
MU_K: float = 128.5  # arcsec·AU/hr

_MU_CAP_KBO: float = 4.5  # arcsec/hr hard cap for KBO mode


def mu_of_R(R_au: float) -> float:
    """Nominal opposition motion rate (arcsec/hr) at heliocentric distance R (AU)."""
    return MU_K / R_au


def _sample_mu(
    R_au: float,
    config: KBOConfig,
    rng: np.random.Generator,
) -> float:
    """Sample motion rate with scatter; apply KBO mode cap."""
    nominal = mu_of_R(R_au)
    mu = nominal * rng.normal(1.0, config.motion_scatter)
    mu = max(mu, 0.01)
    if config.mode == "kbo":
        mu = min(mu, _MU_CAP_KBO)
    return float(mu)


# ---------------------------------------------------------------------------
# Direction
# ---------------------------------------------------------------------------

def _sample_phi_offset(config: KBOConfig, rng: np.random.Generator) -> float:
    """Sample angular offset from ecliptic direction (degrees)."""
    return float(rng.normal(0.0, config.phi_ecl_sigma_deg))


def _phi_img_rad(phi_offset_deg: float) -> float:
    """
    Image-plane direction (radians).
    π = retrograde/westward at opposition (Step 2 canonical, no WCS).
    Step 3+: replace π with wcs-derived phi_ecl_img.
    """
    return math.pi + math.radians(phi_offset_deg)


# ---------------------------------------------------------------------------
# SNR
# ---------------------------------------------------------------------------

def _sample_snr(rng: np.random.Generator) -> float:
    """
    Sample SNR from a faint-heavy bimodal mixture.
    75% in [3, 6]  — faint KBO regime
    25% in [6, 10] — moderate brightness
    """
    if rng.random() < 0.75:
        return float(rng.uniform(3.0, 6.0))
    return float(rng.uniform(6.0, 10.0))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sample_kbo(
    rng: np.random.Generator,
    config: KBOConfig | None = None,
) -> KBOSample:
    """
    Draw one KBO-mode injection sample.

    Returns a KBOSample with:
    - physics truth: population_class, R_au, mu_arcsec_hr, phi_offset_deg, snr
    - injector-compatible arcsec/hr components: motion_ra, motion_dec
    - sanity-only px/frame velocities: vx_px_per_frame, vy_px_per_frame
    - dimensionless flux placeholder: flux_peak = snr
    """
    if config is None:
        config = KBOConfig()

    pop = _sample_population_class(rng)
    R = _sample_R(pop, rng)
    mu = _sample_mu(R, config, rng)
    phi_off = _sample_phi_offset(config, rng)
    phi = _phi_img_rad(phi_off)
    snr = _sample_snr(rng)

    # Injector-compatible velocity components (arcsec/hr)
    # trajectory.py converts arcsec/hr → px via plate_scale and dt
    motion_ra = mu * math.cos(phi)
    motion_dec = mu * math.sin(phi)

    # Sanity-only pixel velocities — mirror trajectory.py formula exactly
    # so clipping-rate checks are consistent with production behaviour
    dt = config.dt_hours
    vx = (motion_ra / config.plate_scale) * dt
    vy = (motion_dec / config.plate_scale) * dt

    return KBOSample(
        population_class=pop,
        R_au=R,
        mu_arcsec_hr=mu,
        phi_offset_deg=phi_off,
        snr=snr,
        dropout_mask=None,
        phi_img_rad=phi,
        motion_ra=motion_ra,
        motion_dec=motion_dec,
        vx_px_per_frame=vx,
        vy_px_per_frame=vy,
        flux_peak=snr,          # dimensionless — Step 3 will map to real ADU
        mode=config.mode,
    )
