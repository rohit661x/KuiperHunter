# Step 1-3 Manual Verification Record

**Date:** 2026-02-19
**Status:** Automated tests complete — ready for visual sanity check

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

> **TODO:** Run the commands above and fill in results before marking Step 1-3 as complete.

| Case | Monotonic motion | Mask alignment | No axis swap | Notes |
|------|-----------------|----------------|--------------|-------|
| Positive case (`case_0000.npz`) | [ ] | [ ] | [ ] | |
| Second case (`case_0004.npz`) | [ ] | [ ] | [ ] | |

### Conclusion

> **TODO:** Fill in after visual sanity check.

- [ ] All automated tests pass (115 passed, 2 skipped, 0 failed)
- [ ] Visual sanity check passed
- [ ] Ready to proceed to model training
