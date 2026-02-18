# Step 2: KBO Physics Priors — Design Doc

**Date:** 2026-02-18
**Status:** Approved — ready for implementation

---

## Goal

Replace the flat uniform priors in `priors.py` with a physically motivated,
conditionally sampled prior for KBO-mode injection. Prove the priors are
statistically sane before any image data is touched.

Definition of done:

```
python -m src.injector.sanity --priors tno --mode kbo --N 100000
```

prints class mixture counts, distribution stats, pass/fail checks, and saves
six plots/files to `demo/priors_report/`.

---

## Unit Contract (locked)

This is the single source of truth for units across the pipeline.

| Field | Owner | Units | Consumed by |
|---|---|---|---|
| `mu_arcsec_hr` | `KBOSample` | arcsec/hr (scalar speed) | physics truth |
| `phi_img_rad` | `KBOSample` | radians (image-plane direction) | decomposition |
| `vx_px_per_frame`, `vy_px_per_frame` | `KBOSample` | px/frame | sanity report only |
| `PriorSample.motion_ra/dec` | adapter in `priors.py` | **arcsec/hr** (legacy naming) | `trajectory.py` |
| `PriorSample.flux_peak` | adapter in `priors.py` | dimensionless SNR (placeholder) | `inject()` |

**Rules:**
- `motion_ra = mu_arcsec_hr * cos(phi_img_rad)` — arcsec/hr component.
- `motion_dec = mu_arcsec_hr * sin(phi_img_rad)` — arcsec/hr component.
- `trajectory.py` is the **single conversion point** from arcsec/hr → pixels.
  Do not pre-convert before passing to `inject()`.
- `flux_peak = snr` (dimensionless). Proper `flux = snr * local_sigma * f(psf)`
  is a Step 3+ concern once real patches/noise are available.
- `vx/vy_px_per_frame` mirror the production formula exactly:
  `vx = (motion_ra / plate_scale) * dt_hours` so sanity drift_px is consistent.

---

## Architecture

### Option chosen: Option A — `kbo_prior.py` as separate physics module

```
src/injector/
  kbo_prior.py      ← NEW: KBOSample, KBOConfig, sample_kbo()
  priors.py         ← MODIFIED: @register("tno") replaced; others untouched
  sanity.py         ← MODIFIED: add main(), report_priors(), pass/fail checks
  __init__.py       ← MODIFIED: export KBOSample, KBOConfig, sample_kbo
  trajectory.py     ← UNCHANGED
  injector.py       ← UNCHANGED
  render_psf.py     ← UNCHANGED
  targets.py        ← UNCHANGED
demo/priors_report/ ← GENERATED (gitignored)
```

Separation: physics sampling lives in `kbo_prior.py`; framework glue
(`@register`, `PriorSample` adaptation) lives in `priors.py`; reporting
lives in `sanity.py`.

---

## Data Model

### `KBOConfig` (dataclass, all fields have defaults)

```python
@dataclass
class KBOConfig:
    mode: str = "kbo"
    phi_ecl_sigma_deg: float = 10.0   # spread of KBO directions around ecliptic
    motion_scatter: float = 0.08      # ±8% multiplicative scatter on mu(R)
    nominal_sigma_sky: float = 10.0   # ADU — Step 2 placeholder for SNR→flux
    # Canonical sanity defaults
    plate_scale: float = 0.187        # arcsec/px (MegaCam)
    baseline_hours: float = 4.0
    T: int = 5
```

`dt_hours = baseline_hours / (T - 1)` is derived, not stored.

### `KBOSample` (dataclass)

```python
@dataclass
class KBOSample:
    # --- Physics truth ---
    population_class: str        # classical_cold | classical_hot | plutino | scattering
    R_au: float                  # heliocentric distance (AU)
    mu_arcsec_hr: float          # sky-plane motion rate (arcsec/hr)
    phi_offset_deg: float        # offset from ecliptic direction (deg)
    snr: float                   # signal-to-noise ratio (faint-heavy mixture)
    dropout_mask: list[bool] | None  # per-epoch visibility (None in Step 2)

    # --- Derived: direction ---
    phi_img_rad: float           # = π + radians(phi_offset_deg)  [Step 2 canonical]

    # --- Derived: injector-compatible (arcsec/hr components) ---
    motion_ra: float             # = mu_arcsec_hr * cos(phi_img_rad)
    motion_dec: float            # = mu_arcsec_hr * sin(phi_img_rad)

    # --- Derived: sanity-report only (px/frame, mirrors trajectory.py formula) ---
    vx_px_per_frame: float       # = (motion_ra / plate_scale) * dt_hours
    vy_px_per_frame: float       # = (motion_dec / plate_scale) * dt_hours

    # --- Adapter ---
    flux_peak: float             # = snr  (dimensionless placeholder, Step 2)
    mode: str                    # "kbo"
```

---

## Conditional Sampling Chain (`sample_kbo`)

### Step 1 — Population class (KBO mixture)

| Class | Weight |
|---|---|
| `classical_cold` | 0.45 |
| `classical_hot` | 0.25 |
| `plutino` | 0.20 |
| `scattering` | 0.10 |

`numpy.random.Generator.choice` over the class names, weighted.

### Step 2 — Heliocentric distance R conditional on class

| Class | Distribution | Clipping |
|---|---|---|
| `classical_cold` | N(44, 2) | [40, 50] |
| `classical_hot` | N(44, 2) | [40, 50] |
| `plutino` | N(39.4, 1) | [37, 42] |
| `scattering` | log-uniform | [30, 100] |

Log-uniform: `R = exp(Uniform(log(30), log(100)))`.

### Step 3 — Motion rate from R

```
mu_nominal = K / R_au        where K = 128.5 arcsec·AU/hr
```

Calibrated so `mu(44 AU) ≈ 2.92 arcsec/hr`. Cross-check:
`mu(39.4) ≈ 3.26`, `mu(65) ≈ 1.98`, `mu(30) ≈ 4.28`.

This is an approximation to opposition-dominated parallax motion. It will
be refined in a later step when orbital elements are available.

Add multiplicative scatter: `mu = mu_nominal * Normal(1.0, motion_scatter)`,
then `mu = clip(mu, 0.01, 4.5)` for mode=kbo (hard cap at 4.5 arcsec/hr).

### Step 4 — Direction

```
phi_offset_deg ~ Normal(0, phi_ecl_sigma_deg)
phi_img_rad = π + radians(phi_offset_deg)
```

`π` = retrograde/westward at opposition (canonical Step 2 direction).
WCS-aware `phi_ecl_img` replaces the `π` constant in Step 3+.

### Step 5 — SNR (faint-heavy bimodal)

```
75%: snr ~ Uniform(3.0, 6.0)   # faint regime
25%: snr ~ Uniform(6.0, 10.0)  # moderate regime
```

### Step 6 — Derived fields

```
motion_ra       = mu * cos(phi_img_rad)            [arcsec/hr]
motion_dec      = mu * sin(phi_img_rad)             [arcsec/hr]
dt_hours        = config.baseline_hours / (config.T - 1)
vx_px_per_frame = (motion_ra / config.plate_scale) * dt_hours
vy_px_per_frame = (motion_dec / config.plate_scale) * dt_hours
flux_peak       = snr                               [dimensionless placeholder]
```

---

## Adapter (in `priors.py`)

Replace `@register("tno")` only:

```python
@register("tno")
def _tno(rng):
    s = kbo_prior.sample_kbo(rng)
    return PriorSample(
        flux_peak=s.flux_peak,       # SNR as dimensionless strength
        motion_ra=s.motion_ra,       # arcsec/hr — trajectory.py converts
        motion_dec=s.motion_dec,     # arcsec/hr — trajectory.py converts
        start_x=rng.uniform(0, 1),   # overwritten by draw_target() in inject()
        start_y=rng.uniform(0, 1),
    )
```

`mba`, `nea`, `static` priors are **untouched**.

---

## Sanity CLI

### Invocation

```
python -m src.injector.sanity \
  --priors tno --mode kbo --N 100000 \
  --T 5 --baseline_hours 4 \
  --plate_scale 0.187 --patch 128
```

`--priors` selects which prior to benchmark (`tno` routes to KBO sampler).
`--mode` is passed to `KBOConfig`. Defaults match MegaCam DECam-like cadence.

### Console output

```
=== Prior report: priors=tno  mode=kbo  N=100000 ===

Population class counts:
  classical_cold  :  44 982  (45.0%)
  classical_hot   :  24 991  (25.0%)
  plutino         :  20 011  (20.0%)
  scattering      :  10 016  (10.0%)

R_au    : mean=43.01  std=2.94  p5=39.27  p95=47.12
mu      : mean=2.98   std=0.53  p5=2.14   p95=4.21   max=4.50
phi_off : mean=0.02°  std=10.1° p5=-16.5° p95=16.4°
snr     : mean=5.03   std=1.42  p5=3.08   p95=8.97
drift_px: mean=15.2   std=3.1   p95=21.3  (T=5, cadence=4hr, ps=0.187)
clipping_rate (patch=128): 0.12%

=== Pass/fail (mode=kbo) ===
  [PASS] mu_max <= 4.5
  [PASS] mu_mean in [2.5, 3.5]
  [PASS] phi_offset p95 <= 35°
  [PASS] R_mean in [40, 46]
  [PASS] clipping_rate <= 5%
```

### Saved files (`demo/priors_report/`)

| File | Content |
|---|---|
| `hist_R.png` | R_au distribution, coloured by population class |
| `hist_mu.png` | mu_arcsec_hr with 4.5 cap line |
| `hist_phi_offset.png` | phi_offset_deg with ±35° markers |
| `hist_snr.png` | SNR with 3/6/10 markers |
| `hist_drift_px.png` | drift_px with patch_half line |
| `clipping_rate.txt` | plain-text summary of clipping stats |

### Pass/fail thresholds (mode=kbo)

| Check | Threshold |
|---|---|
| `mu.max` | ≤ 4.5 arcsec/hr |
| `mu.mean` | 2.5 – 3.5 arcsec/hr |
| `phi_offset p95` | ≤ 35° |
| `R.mean` | 40 – 46 AU |
| `clipping_rate` | ≤ 5% |

---

## What is NOT in Step 2

- No WCS: `phi_img_rad = π + offset` is a placeholder for the retrograde direction.
- No real noise: `flux_peak = snr` is dimensionless until Step 3 patches are loaded.
- No `dropout_mask`: stored as `None`; per-epoch visibility is a Step 3 feature.
- `mba`, `nea`, `static` priors are not updated to conditional sampling.
- `trajectory.py` is not changed.

---

## `.gitignore` addition needed

`demo/priors_report/` must be added (generated artefacts).
