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
from dataclasses import dataclass, field
from typing import Literal

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
