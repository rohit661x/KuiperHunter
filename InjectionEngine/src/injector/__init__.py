"""InjectionEngine – synthetic moving-source injection into image stacks."""

from .injector import inject
from .render_psf import PSFParams
from .targets import TargetConfig
from .priors import PriorSample, sample as draw_prior
from .trajectory import Trajectory, build_trajectory
from .kbo_prior import KBOSample, KBOConfig, sample_kbo

__all__ = [
    "inject",
    "PSFParams",
    "TargetConfig",
    "PriorSample",
    "draw_prior",
    "Trajectory",
    "build_trajectory",
    "KBOSample",
    "KBOConfig",
    "sample_kbo",
]
