# Step 2: KBO Physics Priors — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace flat uniform priors with a physics-based conditional sampler for KBO-mode injection, and extend `sanity.py` into a CLI report generator.

**Architecture:** New `kbo_prior.py` owns all KBO physics (dataclasses, conditional sampling chain). `priors.py` adapter replaces `@register("tno")` to call `sample_kbo()` and fills `PriorSample` with arcsec/hr motion components and SNR as flux placeholder. `sanity.py` grows a `main()` with argparse, statistical report, and hard pass/fail checks.

**Tech Stack:** Python 3.10+, numpy (already a dep), matplotlib (optional dep — already in pyproject.toml). pytest for tests.

**Design doc:** `docs/plans/2026-02-18-step2-priors-design.md`

---

## Critical unit contract (read before touching any code)

| Field | Units | Used by |
|---|---|---|
| `KBOSample.mu_arcsec_hr` | arcsec/hr | physics truth |
| `KBOSample.phi_img_rad` | radians | decomposition |
| `KBOSample.motion_ra/dec` | **arcsec/hr** | `trajectory.py` (it divides by plate_scale) |
| `KBOSample.vx/vy_px_per_frame` | px/frame | sanity drift_px only |
| `PriorSample.motion_ra/dec` | **arcsec/hr** | `trajectory.py` — do NOT pre-convert to px |
| `PriorSample.flux_peak` | dimensionless SNR | `inject()` (placeholder) |

`trajectory.py` line 57–58 is the **single conversion point**. Never pre-convert.

---

## Task 1: Test infrastructure + KBOConfig / KBOSample dataclasses

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_kbo_prior.py`
- Create: `src/injector/kbo_prior.py`

**Step 1: Create empty test package**

```bash
touch tests/__init__.py
```

**Step 2: Write failing tests for dataclasses**

Create `tests/test_kbo_prior.py`:

```python
"""Tests for kbo_prior dataclasses and configuration."""
import pytest
from src.injector.kbo_prior import KBOConfig, KBOSample


class TestKBOConfig:
    def test_defaults(self):
        c = KBOConfig()
        assert c.mode == "kbo"
        assert c.phi_ecl_sigma_deg == pytest.approx(10.0)
        assert c.motion_scatter == pytest.approx(0.08)
        assert c.nominal_sigma_sky == pytest.approx(10.0)
        assert c.plate_scale == pytest.approx(0.187)
        assert c.baseline_hours == pytest.approx(4.0)
        assert c.T == 5

    def test_dt_hours_property(self):
        c = KBOConfig(baseline_hours=4.0, T=5)
        assert c.dt_hours == pytest.approx(1.0)  # 4/(5-1) = 1

    def test_override(self):
        c = KBOConfig(mode="broad", T=3)
        assert c.mode == "broad"
        assert c.T == 3


class TestKBOSampleFields:
    def test_has_all_fields(self):
        """KBOSample must carry every field specified in the design doc."""
        import dataclasses
        fields = {f.name for f in dataclasses.fields(KBOSample)}
        required = {
            "population_class", "R_au", "mu_arcsec_hr",
            "phi_offset_deg", "snr", "dropout_mask",
            "phi_img_rad", "motion_ra", "motion_dec",
            "vx_px_per_frame", "vy_px_per_frame",
            "flux_peak", "mode",
        }
        assert required <= fields
```

**Step 3: Run tests to confirm they fail**

```bash
pytest tests/test_kbo_prior.py -v
```

Expected: `ImportError` — `kbo_prior` does not exist yet.

**Step 4: Implement `KBOConfig` and `KBOSample` in `kbo_prior.py`**

Create `src/injector/kbo_prior.py`:

```python
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
```

**Step 5: Run tests — expect pass**

```bash
pytest tests/test_kbo_prior.py -v
```

Expected: all 3 tests `PASSED`.

**Step 6: Commit**

```bash
git add tests/__init__.py tests/test_kbo_prior.py src/injector/kbo_prior.py
git commit -m "feat: KBOConfig and KBOSample dataclasses with dt_hours property"
```

---

## Task 2: Population class sampling

**Files:**
- Modify: `tests/test_kbo_prior.py` — add class
- Modify: `src/injector/kbo_prior.py` — add constants + function

**Step 1: Write failing test**

Append to `tests/test_kbo_prior.py`:

```python
class TestPopulationClass:
    VALID = {"classical_cold", "classical_hot", "plutino", "scattering"}

    def test_returns_valid_class(self):
        from src.injector.kbo_prior import _sample_population_class
        rng = np.random.default_rng(0)
        for _ in range(100):
            cls = _sample_population_class(rng)
            assert cls in self.VALID

    def test_mixture_proportions(self):
        """Over 50 000 draws the mixture should be within 2 pp of target."""
        from src.injector.kbo_prior import _sample_population_class, KBO_MIXTURE
        rng = np.random.default_rng(42)
        N = 50_000
        counts = {k: 0 for k in KBO_MIXTURE}
        for _ in range(N):
            counts[_sample_population_class(rng)] += 1
        for cls, target_frac in KBO_MIXTURE.items():
            actual = counts[cls] / N
            assert abs(actual - target_frac) < 0.02, (
                f"{cls}: expected ~{target_frac:.2f}, got {actual:.3f}"
            )
```

**Step 2: Run — confirm FAIL**

```bash
pytest tests/test_kbo_prior.py::TestPopulationClass -v
```

Expected: `ImportError` on `_sample_population_class`.

**Step 3: Implement**

Add to `kbo_prior.py` after the dataclasses:

```python
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
```

**Step 4: Run — confirm PASS**

```bash
pytest tests/test_kbo_prior.py::TestPopulationClass -v
```

**Step 5: Commit**

```bash
git add tests/test_kbo_prior.py src/injector/kbo_prior.py
git commit -m "feat: KBO population class mixture sampler"
```

---

## Task 3: Distance sampling conditional on class

**Files:**
- Modify: `tests/test_kbo_prior.py`
- Modify: `src/injector/kbo_prior.py`

**Step 1: Write failing test**

```python
class TestDistanceSampling:
    def test_classical_cold_range(self):
        from src.injector.kbo_prior import _sample_R
        rng = np.random.default_rng(0)
        for _ in range(1000):
            R = _sample_R("classical_cold", rng)
            assert 40.0 <= R <= 50.0, f"classical_cold R={R} out of [40,50]"

    def test_classical_hot_range(self):
        from src.injector.kbo_prior import _sample_R
        rng = np.random.default_rng(1)
        for _ in range(1000):
            R = _sample_R("classical_hot", rng)
            assert 40.0 <= R <= 50.0

    def test_plutino_range(self):
        from src.injector.kbo_prior import _sample_R
        rng = np.random.default_rng(2)
        for _ in range(1000):
            R = _sample_R("plutino", rng)
            assert 37.0 <= R <= 42.0

    def test_scattering_range(self):
        from src.injector.kbo_prior import _sample_R
        rng = np.random.default_rng(3)
        for _ in range(1000):
            R = _sample_R("scattering", rng)
            assert 30.0 <= R <= 100.0

    def test_classical_cold_mean_near_44(self):
        from src.injector.kbo_prior import _sample_R
        rng = np.random.default_rng(99)
        Rs = [_sample_R("classical_cold", rng) for _ in range(10_000)]
        mean = np.mean(Rs)
        assert 43.0 < mean < 45.0, f"classical_cold mean R={mean:.2f}, expected near 44"

    def test_plutino_mean_near_394(self):
        from src.injector.kbo_prior import _sample_R
        rng = np.random.default_rng(99)
        Rs = [_sample_R("plutino", rng) for _ in range(10_000)]
        mean = np.mean(Rs)
        assert 39.0 < mean < 40.0, f"plutino mean R={mean:.2f}, expected near 39.4"

    def test_unknown_class_raises(self):
        from src.injector.kbo_prior import _sample_R
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="Unknown population_class"):
            _sample_R("centaur", rng)
```

**Step 2: Run — confirm FAIL**

```bash
pytest tests/test_kbo_prior.py::TestDistanceSampling -v
```

**Step 3: Implement**

```python
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
```

**Step 4: Run — confirm PASS**

```bash
pytest tests/test_kbo_prior.py::TestDistanceSampling -v
```

**Step 5: Commit**

```bash
git add tests/test_kbo_prior.py src/injector/kbo_prior.py
git commit -m "feat: distance sampling conditional on KBO population class"
```

---

## Task 4: Motion model mu_of_R

**Files:**
- Modify: `tests/test_kbo_prior.py`
- Modify: `src/injector/kbo_prior.py`

**Step 1: Write failing test**

```python
class TestMotionModel:
    def test_mu_at_44au(self):
        """mu(44 AU) should be near 2.92 arcsec/hr (opposition approximation)."""
        from src.injector.kbo_prior import mu_of_R
        mu = mu_of_R(44.0)
        assert 2.8 < mu < 3.1, f"mu(44)={mu:.3f}, expected ~2.92"

    def test_mu_decreases_with_distance(self):
        from src.injector.kbo_prior import mu_of_R
        assert mu_of_R(30.0) > mu_of_R(44.0) > mu_of_R(80.0)

    def test_mu_sample_cap_kbo(self):
        from src.injector.kbo_prior import _sample_mu, KBOConfig
        cfg = KBOConfig(mode="kbo")
        rng = np.random.default_rng(0)
        for _ in range(5000):
            mu = _sample_mu(30.0, cfg, rng)  # R=30 → high mu; cap must hold
            assert mu <= 4.5 + 1e-9, f"mu={mu:.4f} exceeds 4.5 cap"

    def test_mu_positive(self):
        from src.injector.kbo_prior import _sample_mu, KBOConfig
        cfg = KBOConfig()
        rng = np.random.default_rng(0)
        for _ in range(1000):
            mu = _sample_mu(44.0, cfg, rng)
            assert mu > 0

    def test_mu_scatter_mean_near_nominal(self):
        """With scatter=0.08, sample mean should converge to mu_of_R(44)."""
        from src.injector.kbo_prior import _sample_mu, mu_of_R, KBOConfig
        cfg = KBOConfig(motion_scatter=0.08)
        rng = np.random.default_rng(7)
        samples = [_sample_mu(44.0, cfg, rng) for _ in range(10_000)]
        nominal = mu_of_R(44.0)
        assert abs(np.mean(samples) - nominal) < 0.05
```

**Step 2: Run — confirm FAIL**

```bash
pytest tests/test_kbo_prior.py::TestMotionModel -v
```

**Step 3: Implement**

```python
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
```

**Step 4: Run — confirm PASS**

```bash
pytest tests/test_kbo_prior.py::TestMotionModel -v
```

**Step 5: Commit**

```bash
git add tests/test_kbo_prior.py src/injector/kbo_prior.py
git commit -m "feat: mu_of_R opposition motion model with scatter and KBO cap"
```

---

## Task 5: Direction and SNR sampling

**Files:**
- Modify: `tests/test_kbo_prior.py`
- Modify: `src/injector/kbo_prior.py`

**Step 1: Write failing tests**

```python
class TestDirectionAndSNR:
    def test_phi_offset_is_normal(self):
        """phi_offset ~ Normal(0, sigma); check mean ≈ 0 and std ≈ sigma."""
        from src.injector.kbo_prior import _sample_phi_offset, KBOConfig
        cfg = KBOConfig(phi_ecl_sigma_deg=10.0)
        rng = np.random.default_rng(0)
        samples = [_sample_phi_offset(cfg, rng) for _ in range(20_000)]
        assert abs(np.mean(samples)) < 0.5
        assert abs(np.std(samples) - 10.0) < 0.3

    def test_phi_img_rad_canonical(self):
        """phi_img_rad = pi + radians(phi_offset)."""
        from src.injector.kbo_prior import _phi_img_rad
        assert _phi_img_rad(0.0) == pytest.approx(math.pi)
        assert _phi_img_rad(90.0) == pytest.approx(math.pi + math.pi / 2)

    def test_snr_always_in_range(self):
        from src.injector.kbo_prior import _sample_snr
        rng = np.random.default_rng(0)
        for _ in range(5000):
            snr = _sample_snr(rng)
            assert 3.0 <= snr <= 10.0, f"snr={snr} out of [3, 10]"

    def test_snr_faint_heavy(self):
        """~75% of SNR draws should fall in [3, 6]."""
        from src.injector.kbo_prior import _sample_snr
        rng = np.random.default_rng(42)
        N = 20_000
        faint = sum(1 for _ in range(N) if _sample_snr(rng) <= 6.0)
        frac = faint / N
        assert 0.70 < frac < 0.80, f"faint fraction={frac:.3f}, expected ~0.75"
```

**Step 2: Run — confirm FAIL**

```bash
pytest tests/test_kbo_prior.py::TestDirectionAndSNR -v
```

**Step 3: Implement**

```python
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
```

**Step 4: Run — confirm PASS**

```bash
pytest tests/test_kbo_prior.py::TestDirectionAndSNR -v
```

**Step 5: Commit**

```bash
git add tests/test_kbo_prior.py src/injector/kbo_prior.py
git commit -m "feat: direction (phi_offset) and SNR sampling for KBO prior"
```

---

## Task 6: Derived fields + complete `sample_kbo()`

**Files:**
- Modify: `tests/test_kbo_prior.py`
- Modify: `src/injector/kbo_prior.py`

**Step 1: Write failing tests**

```python
class TestSampleKBO:
    def _draw(self, seed=0):
        from src.injector.kbo_prior import sample_kbo, KBOConfig
        rng = np.random.default_rng(seed)
        return sample_kbo(rng, KBOConfig())

    def test_returns_kbo_sample(self):
        from src.injector.kbo_prior import KBOSample
        s = self._draw()
        assert isinstance(s, KBOSample)

    def test_motion_ra_dec_components(self):
        """motion_ra/dec must equal mu * cos/sin(phi_img_rad)."""
        s = self._draw()
        assert s.motion_ra == pytest.approx(
            s.mu_arcsec_hr * math.cos(s.phi_img_rad), abs=1e-9
        )
        assert s.motion_dec == pytest.approx(
            s.mu_arcsec_hr * math.sin(s.phi_img_rad), abs=1e-9
        )

    def test_vx_vy_mirror_trajectory_formula(self):
        """vx/vy_px_per_frame must use the same formula as trajectory.py."""
        from src.injector.kbo_prior import KBOConfig
        cfg = KBOConfig()
        s = self._draw()
        expected_vx = (s.motion_ra / cfg.plate_scale) * cfg.dt_hours
        expected_vy = (s.motion_dec / cfg.plate_scale) * cfg.dt_hours
        assert s.vx_px_per_frame == pytest.approx(expected_vx, abs=1e-9)
        assert s.vy_px_per_frame == pytest.approx(expected_vy, abs=1e-9)

    def test_flux_peak_equals_snr(self):
        """flux_peak is dimensionless SNR placeholder in Step 2."""
        s = self._draw()
        assert s.flux_peak == pytest.approx(s.snr)

    def test_dropout_mask_is_none(self):
        s = self._draw()
        assert s.dropout_mask is None

    def test_mode_is_kbo(self):
        s = self._draw()
        assert s.mode == "kbo"

    def test_seed_reproducibility(self):
        from src.injector.kbo_prior import sample_kbo, KBOConfig
        cfg = KBOConfig()
        s1 = sample_kbo(np.random.default_rng(99), cfg)
        s2 = sample_kbo(np.random.default_rng(99), cfg)
        assert s1.R_au == s2.R_au
        assert s1.mu_arcsec_hr == s2.mu_arcsec_hr
        assert s1.snr == s2.snr

    def test_mu_capped_at_4_5(self):
        from src.injector.kbo_prior import sample_kbo, KBOConfig
        cfg = KBOConfig()
        rng = np.random.default_rng(0)
        for _ in range(2000):
            s = sample_kbo(rng, cfg)
            assert s.mu_arcsec_hr <= 4.5 + 1e-9
```

**Step 2: Run — confirm FAIL**

```bash
pytest tests/test_kbo_prior.py::TestSampleKBO -v
```

**Step 3: Implement `sample_kbo()`**

Add at the bottom of `kbo_prior.py`:

```python
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
```

**Step 4: Run — confirm all `TestSampleKBO` tests pass**

```bash
pytest tests/test_kbo_prior.py -v
```

Expected: all tests in the file `PASSED`.

**Step 5: Commit**

```bash
git add tests/test_kbo_prior.py src/injector/kbo_prior.py
git commit -m "feat: complete sample_kbo() with all derived fields"
```

---

## Task 7: Adapter — replace @register("tno") in priors.py

**Files:**
- Modify: `tests/test_kbo_prior.py` — add integration test
- Modify: `src/injector/priors.py`

**Step 1: Write failing integration test**

```python
class TestPriorAdapter:
    def test_tno_returns_prior_sample(self):
        from src.injector.priors import sample, PriorSample
        rng = np.random.default_rng(0)
        s = sample("tno", rng)
        assert isinstance(s, PriorSample)

    def test_tno_motion_ra_is_arcsec_hr(self):
        """motion_ra from adapter must be in arcsec/hr (trajectory.py contract)."""
        from src.injector.priors import sample
        from src.injector.kbo_prior import KBOConfig
        # At canonical plate_scale=0.187, T=5, baseline=4hr, even a fast KBO
        # (4.5 arcsec/hr) stays well below 25 arcsec/hr.
        rng = np.random.default_rng(0)
        for _ in range(200):
            s = sample("tno", rng)
            assert abs(s.motion_ra) <= 4.5 + 1e-6, (
                f"motion_ra={s.motion_ra:.3f} exceeds KBO cap — likely px/frame confusion"
            )

    def test_tno_flux_peak_equals_snr_range(self):
        """flux_peak = snr (dimensionless, 3–10)."""
        from src.injector.priors import sample
        rng = np.random.default_rng(0)
        for _ in range(200):
            s = sample("tno", rng)
            assert 3.0 <= s.flux_peak <= 10.0

    def test_other_priors_unchanged(self):
        """mba/nea/static must still work after modifying tno."""
        from src.injector.priors import sample
        rng = np.random.default_rng(0)
        for name in ("mba", "nea", "static"):
            s = sample(name, rng)
            assert s.flux_peak > 0
            assert s.motion_ra is not None
```

**Step 2: Run — confirm tno_motion test FAILS (old flat prior returns large values)**

```bash
pytest tests/test_kbo_prior.py::TestPriorAdapter -v
```

**Step 3: Replace `@register("tno")` in `priors.py`**

Open `src/injector/priors.py`. Replace the `_tno` function only — leave `_mba`, `_nea`, `_static` untouched.

Old code (lines approx 49–58):
```python
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
```

New code:
```python
@register("tno")
def _tno(rng: np.random.Generator) -> PriorSample:
    """
    Trans-Neptunian Object / KBO — delegates to the physics-based KBO prior.

    Unit note: motion_ra/motion_dec are in arcsec/hr.
    trajectory.py divides by plate_scale to convert to pixels.
    flux_peak is dimensionless SNR (placeholder until Step 3 noise model).
    """
    from .kbo_prior import sample_kbo, KBOConfig
    s = sample_kbo(rng, KBOConfig())
    return PriorSample(
        flux_peak=s.flux_peak,    # SNR as dimensionless strength (Step 2)
        motion_ra=s.motion_ra,    # arcsec/hr — trajectory.py converts to px
        motion_dec=s.motion_dec,  # arcsec/hr — trajectory.py converts to px
        start_x=rng.uniform(0, 1),  # overwritten by draw_target() in inject()
        start_y=rng.uniform(0, 1),
    )
```

**Step 4: Run — confirm PASS**

```bash
pytest tests/test_kbo_prior.py -v
```

**Step 5: Verify the existing demo still runs end-to-end**

```bash
python demo/demo.py
```

Expected: same output as before (6/6 sanity checks pass). The demo uses
`sample_type="mba"` for its sanity section, but also exercises `"tno"` in
section 2. Confirm no errors.

**Step 6: Commit**

```bash
git add tests/test_kbo_prior.py src/injector/priors.py
git commit -m "feat: replace flat TNO prior with physics-based KBO adapter"
```

---

## Task 8: Export new symbols from `__init__.py`

**Files:**
- Modify: `src/injector/__init__.py`

**Step 1: Add exports**

Current `__init__.py` exports: `inject`, `PSFParams`, `TargetConfig`, `PriorSample`, `draw_prior`, `Trajectory`, `build_trajectory`.

Add after the existing imports:

```python
from .kbo_prior import KBOSample, KBOConfig, sample_kbo
```

Add to `__all__`:

```python
"KBOSample",
"KBOConfig",
"sample_kbo",
```

**Step 2: Smoke test**

```bash
python -c "from src.injector import KBOSample, KBOConfig, sample_kbo; print('OK')"
```

Expected: `OK`

**Step 3: Commit**

```bash
git add src/injector/__init__.py
git commit -m "feat: export KBOSample, KBOConfig, sample_kbo from package"
```

---

## Task 9: `sanity.py` — add `report_priors()` function

**Files:**
- Create: `tests/test_sanity_priors.py`
- Modify: `src/injector/sanity.py`

**Step 1: Write failing test**

Create `tests/test_sanity_priors.py`:

```python
"""Tests for the prior report generator in sanity.py."""
import numpy as np
import pytest


class TestReportPriors:
    def test_returns_results_dict(self, tmp_path):
        from src.injector.sanity import report_priors
        results = report_priors(
            sample_type="tno",
            mode="kbo",
            N=500,
            T=5,
            baseline_hours=4.0,
            plate_scale=0.187,
            patch=128,
            out_dir=str(tmp_path),
            seed=0,
            plot=False,
        )
        assert isinstance(results, dict)
        for key in ("R_au", "mu", "phi_offset", "snr", "drift_px", "clipping_rate"):
            assert key in results, f"Missing key: {key}"

    def test_clipping_rate_is_fraction(self, tmp_path):
        from src.injector.sanity import report_priors
        results = report_priors(
            sample_type="tno", mode="kbo", N=500,
            T=5, baseline_hours=4.0, plate_scale=0.187, patch=128,
            out_dir=str(tmp_path), seed=0, plot=False,
        )
        assert 0.0 <= results["clipping_rate"] <= 1.0

    def test_all_kbo_checks_pass(self, tmp_path):
        from src.injector.sanity import report_priors
        results = report_priors(
            sample_type="tno", mode="kbo", N=5000,
            T=5, baseline_hours=4.0, plate_scale=0.187, patch=128,
            out_dir=str(tmp_path), seed=0, plot=False,
        )
        checks = results["checks"]
        failed = [name for name, ok in checks.items() if not ok]
        assert not failed, f"KBO checks failed: {failed}"

    def test_clipping_file_written(self, tmp_path):
        from src.injector.sanity import report_priors
        report_priors(
            sample_type="tno", mode="kbo", N=200,
            T=5, baseline_hours=4.0, plate_scale=0.187, patch=128,
            out_dir=str(tmp_path), seed=0, plot=False,
        )
        assert (tmp_path / "clipping_rate.txt").exists()
```

**Step 2: Run — confirm FAIL**

```bash
pytest tests/test_sanity_priors.py -v
```

**Step 3: Implement `report_priors()` in `sanity.py`**

Add to `sanity.py` (append after the existing `run_all` function):

```python
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
    out_dir       : directory for saved files; None → no files written.
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

    half = patch / 2.0
    clipping_rate = float((drift_arr > half).mean())

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

        # clipping_rate.txt
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
    """Save six diagnostic plots to out_dir."""
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
    ax.set_xlabel("μ (arcsec/hr)")
    ax.set_ylabel("Count")
    ax.set_title("Sky-Plane Motion Rate")
    ax.legend()
    fig.savefig(os.path.join(out_dir, "hist_mu.png"), dpi=120)
    plt.close(fig)

    # hist_phi_offset.png
    fig, ax = plt.subplots(**fig_kw)
    ax.hist(phi_arr, bins=60, color="mediumseagreen", edgecolor="none", alpha=0.8)
    ax.axvline(-35, color="crimson", lw=1.5, ls="--", label="±35°")
    ax.axvline(+35, color="crimson", lw=1.5, ls="--")
    ax.set_xlabel("φ offset (deg)")
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
    ax.axvline(patch / 2, color="crimson", lw=1.5, ls="--",
               label=f"patch/2={patch//2}px")
    ax.axvline(drift_arr.mean(), color="navy", lw=1.5,
               label=f"mean={drift_arr.mean():.1f}px")
    ax.set_xlabel("Total drift (px)")
    ax.set_ylabel("Count")
    ax.set_title(f"Drift in Pixels (T={len(R_arr) and 5}, baseline≈)")
    ax.legend()
    fig.savefig(os.path.join(out_dir, "hist_drift_px.png"), dpi=120)
    plt.close(fig)

    print(f"  Plots saved to {out_dir}/")
```

**Step 4: Run — confirm PASS**

```bash
pytest tests/test_sanity_priors.py -v
```

**Step 5: Commit**

```bash
git add tests/test_sanity_priors.py src/injector/sanity.py
git commit -m "feat: report_priors() with stats, pass/fail checks, file output"
```

---

## Task 10: `sanity.py` — add `main()` CLI

**Files:**
- Modify: `src/injector/sanity.py`

**Step 1: Implement `main()`**

Append to `sanity.py`:

```python
# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="InjectionEngine sanity reporter — priors and injection checks."
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- prior report mode ---
    pr = subparsers.add_parser("--priors", add_help=False)

    # Flat flags (no subcommands, for the spec's invocation style)
    parser.add_argument("--priors",   default=None,
                        help="Run prior report for this sample type (e.g. tno)")
    parser.add_argument("--mode",     default="kbo",
                        choices=["kbo", "broad"])
    parser.add_argument("--N",        type=int,   default=100_000)
    parser.add_argument("--T",        type=int,   default=5)
    parser.add_argument("--baseline_hours", type=float, default=4.0)
    parser.add_argument("--plate_scale",    type=float, default=0.187)
    parser.add_argument("--patch",    type=int,   default=128)
    parser.add_argument("--seed",     type=int,   default=None)
    parser.add_argument("--no-plot",  action="store_true",
                        help="Skip saving matplotlib figures")
    parser.add_argument("--out-dir",  default=None,
                        help="Output directory (default: demo/priors_report/)")

    args = parser.parse_args()

    if args.priors is not None:
        out_dir = args.out_dir
        if out_dir is None:
            # Resolve relative to the InjectionEngine root
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

        # Exit non-zero if any check fails
        failed = [k for k, v in results.get("checks", {}).items() if not v]
        if failed:
            import sys
            print(f"\n[ERROR] {len(failed)} check(s) failed: {failed}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

**Step 2: Smoke test the CLI**

```bash
python -m src.injector.sanity --priors tno --mode kbo --N 1000 --no-plot
```

Expected: prints class mixture, stats, 5 `[PASS]` lines, exits 0.

**Step 3: Full run (as per spec)**

```bash
python -m src.injector.sanity --priors tno --mode kbo --N 100000
```

Expected: stats printed, 5 plots saved to `demo/priors_report/`, exits 0.

**Step 4: Commit**

```bash
git add src/injector/sanity.py
git commit -m "feat: sanity CLI main() with --priors flag and non-zero exit on failure"
```

---

## Task 11: Update `.gitignore` + final clean run

**Files:**
- Modify: `.gitignore` (at repo root `KuiperHunter/`)

**Step 1: Add `demo/priors_report/` to `.gitignore`**

In `/Users/rohitsuryadevara/Documents/KuiperHunter/.gitignore`, the generated
report directory is not yet ignored. Add:

```
# Step 2 prior report artefacts
InjectionEngine/demo/priors_report/
```

**Step 2: Full test suite**

```bash
pytest tests/ -v
```

Expected: all tests pass (TaskN tests + TestSanity inject checks if run).

**Step 3: Verify full demo still works**

```bash
python demo/demo.py
```

Expected: 6/6 sanity checks pass, no errors.

**Step 4: Full prior report (the spec's definition-of-done)**

```bash
python -m src.injector.sanity --priors tno --mode kbo --N 100000
```

Expected:
- Counts for all 4 population classes
- R_au mean ≈ 43, std ≈ 2–3
- mu mean ≈ 2.9–3.1, max ≤ 4.50
- phi_offset std ≈ 10°
- drift_px mean ≈ 15–20px (T=5, 4hr baseline, 0.187 arcsec/px)
- clipping_rate < 5%
- 5/5 checks `[PASS]`

**Step 5: Final commit**

```bash
git add .gitignore
git commit -m "chore: gitignore demo/priors_report/; Step 2 complete"
```

---

## Verification checklist

After all tasks, confirm:

- [ ] `pytest tests/ -v` — all green
- [ ] `python demo/demo.py` — 6/6 inject sanity checks pass
- [ ] `python -m src.injector.sanity --priors tno --mode kbo --N 100000` — 5/5 prior checks pass
- [ ] `demo/priors_report/` contains 5 `.png` files and `clipping_rate.txt`
- [ ] `git log --oneline` shows one commit per task (10–11 commits)
- [ ] `git status` is clean
