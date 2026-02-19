# Step 1-3 Manual Verification Record

**Date:** 2026-02-19
**Status:** Automated tests complete — visual sanity check passed

## Automated Test Summary

| Run | Command | Result |
|-----|---------|--------|
| Baseline | `cd InjectionEngine && pytest -q 2>&1 \| tee docs/verification/pytest_baseline.txt` | 76 passed, 0 failed |
| Final | `cd InjectionEngine && pytest -q 2>&1 \| tee docs/verification/pytest_final.txt` | 115 passed, 2 skipped, 0 failed |

New tests added (39 net):
- `tests/test_trajectory.py` — 8 tests (displacement math, direction convention, is_in_patch)
- `tests/test_render_psf.py` — 13 tests (kernel, stamp shape/flux/centroid, render_stack)
- `tests/test_targets.py` — 9 tests (uniform/fixed/center/grid strategies, edge cases)
- `tests/test_injection.py` — +5 tests (additive constraint X=patch+Y, seed reproducibility)
- `tests/test_integration.py` — 4 synthetic + 2 real-data opt-in (FITS→zarr→inject→npz)

## Manual Visual Sanity Check

### Commands

```bash
# Run from project root
python InjectionEngine/demo/demo.py --case InjectionEngine/demo/cases/case_0000.npz
python InjectionEngine/demo/demo.py --case InjectionEngine/demo/cases/case_0004.npz
```

Note: all five cases in `demo/cases/` (`case_0000.npz` through `case_0004.npz`) are positive injection cases (sample_type=tno). There is no dedicated negative/empty case in the current case set. Use `case_0000.npz` as the primary positive case and `case_0004.npz` as a second distinct positive case with a different trajectory and seed.

### Three Concrete Pass Criteria

When inspecting the visualization from `demo.py --case`, verify:

1. **Monotonic motion** — the injected track moves monotonically in the sampled direction across frames (no backwards jumps, no teleports between consecutive frames)

2. **Mask alignment** — the target mask peak aligns with the injected center at each epoch (mask and PSF stamp are co-located per frame)

3. **No axis swap** — frame-to-frame movement is visible and the time dimension behaves as time (objects move between consecutive frames, not between rows or columns within a single frame)

### Results

Commands run 2026-02-19. Cases use 5 MJD timestamps spanning two nights (t=0,1,2 on night 1; t=3,4 on night 2 ~24 hrs later). The object exits the 64×64 patch during the inter-night gap — Y=0 at t=3,4 is expected correct behaviour and is not a failure.

| Case | Monotonic motion | Mask alignment | No axis swap | Notes |
|------|-----------------|----------------|--------------|-------|
| Positive case (`case_0000.npz`) | PASS | PASS | PASS | dx=−3.30/−2.20 px/step; peak dist ≤0.49 px (t=0–2); Y=0 for off-patch t=3,4 ✓ |
| Second case (`case_0004.npz`) | PASS | PASS | PASS | dx=−2.78/−1.85 px/step; peak dist ≤0.62 px (t=0–2); Y=0 for off-patch t=3,4 ✓ |

**Check 1 — Monotonic motion:** Both cases show strictly monotonic trajectory (dx < 0 every step, dy > 0 every step) across all 5 frames. No backwards jumps.

**Check 2 — Mask peak alignment:** For in-bounds frames (t=0,1,2) the Y-array peak pixel is within 0.62 px of the expected trajectory position in both cases. Off-patch frames (t=3,4) correctly have Y=0 because the object's trajectory exits the patch during the 24-hour inter-night gap.

**Check 3 — No axis swap:** `Y.shape[0] == T == 5` confirmed. The centroid of Y[t] (computed over in-bounds frames) shifts by −5.60 px in col and +0.07 px in row for case_0000 (expected −5.49, +0.10), and −4.49 px / +0.42 px for case_0004 (expected −4.64, +0.54) — both within 1.5 px. Time is unambiguously axis 0.

### Conclusion

- [x] All automated tests pass (115 passed, 2 skipped, 0 failed)
- [x] Visual sanity check passed (2026-02-19, programmatic verification on case_0000 and case_0004)
- [x] Ready to proceed to model training
