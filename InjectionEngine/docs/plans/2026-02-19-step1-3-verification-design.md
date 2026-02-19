# Step 1–3 Verification Design

**Date:** 2026-02-19
**Status:** Approved — ready for implementation planning
**Goal:** Produce documented evidence that the InjectionEngine pipeline (Steps 1–3) is correct before model training begins.

---

## Scope & Success Criteria

Done when all of the following are true:

- All existing ~50 tests pass (baseline confirmed)
- Unit tests exist for `trajectory.py`, `render_psf.py`, `targets.py`, and the additive constraint in `injector.py`
- One end-to-end integration test exists: synthetic path always runs; real-data path requires `KBO_REALDATA=1`
- Manual `demo.py --case` replay passes three concrete geometric checks
- Test artifacts committed to `docs/verification/`

---

## Phase 0 — Baseline

Run the existing suite before writing any new tests.

```
cd InjectionEngine
pytest -q 2>&1 | tee docs/verification/pytest_baseline.txt
```

Fix any failures before proceeding. The baseline output is committed so there is a before/after record.

---

## Phase 1 — Unit Tests for Zero-Coverage Critical Modules

Each module below is in the critical path of `inject()` and currently has no tests.

### 1. `trajectory.py`

File: `tests/test_trajectory.py`

| Test | Assertion |
|---|---|
| Displacement from motion_ra/dec in arcsec/hr matches expected pixel delta | within tolerance |
| Pure RA motion shifts x only | `dx != 0`, `dy == 0` |
| Pure DEC motion shifts y only | `dy != 0`, `dx == 0` |
| Zero motion → constant positions across all epochs | all positions equal |
| Output shape and dtype | `(T, 2)` float array, no NaN |

### 2. `render_psf.py`

File: `tests/test_render_psf.py`

| Test | Assertion |
|---|---|
| `render_stamp` output shape matches patch shape | shape == input shape |
| Stamp is non-negative | `np.all(stamp >= 0)` |
| Stamp is normalized (sums to ~1) | `abs(stamp.sum() - 1) < 1e-6` |
| Subpixel shift moves centroid in correct direction | centroid shifts right when dx > 0 |
| `render_stack` output shape matches `(T, H, W)` | shape correct |

### 3. `targets.py`

File: `tests/test_targets.py`

| Test | Assertion |
|---|---|
| Mask peak is at injected center `(x_t, y_t)` each epoch | `argmax == injection center` |
| Gaussian falloff: center > neighbor > far pixel | value ordering holds |
| Negative sample mask is all zeros | `np.all(mask == 0)` |
| `draw_target` returns coordinates within patch bounds | `0 <= x < W`, `0 <= y < H` |

### 4. Additive Constraint in `injector.py`

File: `tests/test_injection.py` (extends existing)

The fundamental invariant:

```
X == patch_stack + injected_signal
```

where `Y` is the target mask (not the injected flux).

| Test | Assertion |
|---|---|
| Positive injection: `X - patch_stack` equals injected contribution where PSF was added, ~0 elsewhere | per-pixel check within float tolerance |
| Positive injection: `Y` matches Gaussian target mask (peak at injection center) | shape, non-negativity, peak location |
| Negative sample: `X == patch_stack` exactly | `np.allclose(X, patch_stack)` |
| Negative sample: `Y == 0` everywhere | `np.all(Y == 0)` |

Test setup: use `patch_stack = np.zeros(...)` and a fixed seed for deterministic injection.

---

## Phase 2 — End-to-End Integration Test

File: `tests/test_integration.py`

### Synthetic path (always runs in CI)

1. Generate a small fake FITS stack using the existing `write_fake_fits()` conftest helper
2. Call `build_one_stack(stack_dir, out_zarr)` directly (Python function, no subprocess)
3. Assert zarr file exists with required datasets: `images`, `timestamps`, `psf_fwhm`, `plate_scale`
4. Assert image shape is `(T, H, W)` with correct values
5. Call `make_demo_cases` directly to produce `.npz` files from the zarr
6. Load one `.npz` case and assert required keys: `X`, `Y`, `patch_stack`, `meta`
7. Assert `X.shape == Y.shape == patch_stack.shape`

### Real-data path (opt-in)

Gated behind environment variable `KBO_REALDATA=1`. If not set, test is skipped with a clear message:

```python
if not os.environ.get("KBO_REALDATA"):
    pytest.skip("Set KBO_REALDATA=1 to run real-data integration tests")
```

When enabled:
1. Locate real FITS stack (configurable path, default `kbmod/kbmod/data/small`)
2. Run `build_one_stack` → `make_demo_cases` → load one case
3. Same shape/key assertions as synthetic path

**Constraint:** All calls are direct Python function imports — no subprocess.

---

## Phase 3 — Manual Visual Sanity

Not automated. Run after all tests pass.

```
python InjectionEngine/demo/demo.py --case InjectionEngine/demo/cases/<pos_hard>.npz
python InjectionEngine/demo/demo.py --case InjectionEngine/demo/cases/<neg>.npz
```

Three concrete pass criteria:

1. **Monotonic motion:** The injected track moves monotonically in the sampled direction across frames (no backwards jumps, no teleports)
2. **Mask alignment:** Target mask peak aligns with injected center at each epoch
3. **No axis swap:** Frame-to-frame movement is visible and time corresponds to the temporal dimension (not height or width)

Document the result (pass/fail + any observations) in `docs/verification/step1-3.md`.

---

## Test Report Artifacts

Committed to `docs/verification/`:

| File | Content |
|---|---|
| `pytest_baseline.txt` | Output of `pytest -q` before new tests |
| `pytest_final.txt` | Output of `pytest -q` after all new tests pass |
| `step1-3.md` | Manual sanity check commands + pass/fail observations |

Generation command (also committed in `step1-3.md`):

```
cd InjectionEngine
pytest -q 2>&1 | tee docs/verification/pytest_final.txt
```

---

## Files to Create

| File | Purpose |
|---|---|
| `tests/test_trajectory.py` | Trajectory unit tests |
| `tests/test_render_psf.py` | PSF rendering unit tests |
| `tests/test_targets.py` | Target mask unit tests |
| `tests/test_integration.py` | End-to-end pipeline integration test |
| `docs/verification/pytest_baseline.txt` | Baseline test run output |
| `docs/verification/pytest_final.txt` | Final test run output after new tests |
| `docs/verification/step1-3.md` | Manual sanity check record |

Existing file modified:
- `tests/test_injection.py` — extended with additive constraint tests
