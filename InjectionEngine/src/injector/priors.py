"""
priors.py – Sampling distributions for synthetic injection parameters.

Each function returns a dict of drawn parameters given a seed and sample type.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PriorSample:
    flux_peak: float          # peak surface brightness (counts / pixel)
    motion_ra: float          # RA motion  (arcsec / hour)
    motion_dec: float         # Dec motion (arcsec / hour)
    start_x: float            # sub-pixel column offset within patch
    start_y: float            # sub-pixel row offset within patch
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_SAMPLE_TYPES: dict[str, Any] = {}

def register(name: str):
    def _dec(fn):
        _SAMPLE_TYPES[name] = fn
        return fn
    return _dec


def sample(sample_type: str, rng: np.random.Generator) -> PriorSample:
    """Draw one set of injection parameters from the named prior."""
    if sample_type not in _SAMPLE_TYPES:
        raise ValueError(
            f"Unknown sample_type '{sample_type}'. "
            f"Available: {sorted(_SAMPLE_TYPES)}"
        )
    return _SAMPLE_TYPES[sample_type](rng)


# ---------------------------------------------------------------------------
# Built-in priors
# ---------------------------------------------------------------------------
@register("tno")
def _tno(rng: np.random.Generator) -> PriorSample:
    """Trans-Neptunian Object: slow, faint."""
    return PriorSample(
        flux_peak=rng.uniform(50, 500),
        motion_ra=rng.uniform(-0.5, 0.5),
        motion_dec=rng.uniform(-0.3, 0.3),
        start_x=rng.uniform(0, 1),
        start_y=rng.uniform(0, 1),
    )


@register("mba")
def _mba(rng: np.random.Generator) -> PriorSample:
    """Main-Belt Asteroid: moderate speed, moderate brightness."""
    return PriorSample(
        flux_peak=rng.uniform(200, 2000),
        motion_ra=rng.uniform(-3.0, 3.0),
        motion_dec=rng.uniform(-1.5, 1.5),
        start_x=rng.uniform(0, 1),
        start_y=rng.uniform(0, 1),
    )


@register("nea")
def _nea(rng: np.random.Generator) -> PriorSample:
    """Near-Earth Asteroid: fast, can be bright or faint."""
    return PriorSample(
        flux_peak=rng.uniform(100, 5000),
        motion_ra=rng.uniform(-30.0, 30.0),
        motion_dec=rng.uniform(-20.0, 20.0),
        start_x=rng.uniform(0, 1),
        start_y=rng.uniform(0, 1),
    )


@register("static")
def _static(rng: np.random.Generator) -> PriorSample:
    """Stationary point source (star-like): useful as a null / sanity check."""
    return PriorSample(
        flux_peak=rng.uniform(100, 3000),
        motion_ra=0.0,
        motion_dec=0.0,
        start_x=rng.uniform(0, 1),
        start_y=rng.uniform(0, 1),
    )
