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
