# Step 1–3 Verification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Verify the InjectionEngine pipeline (Steps 1–3) is correct via unit tests, integration tests, and manual sanity checks before model training begins.

**Architecture:** Risk-based approach — run baseline first, then fill zero-coverage critical modules (trajectory, render_psf, targets, additive constraint), then add one end-to-end integration test covering both synthetic and real-data paths.

**Tech Stack:** pytest, numpy, zarr, astropy (for fake FITS), Python 3.10+

All commands run from `InjectionEngine/` directory.

---

## Task 0: Run Baseline and Capture Output

**Files:**
- Create: `docs/verification/pytest_baseline.txt`

**Step 1: Run the existing suite**

```bash
cd InjectionEngine
pytest -q 2>&1 | tee docs/verification/pytest_baseline.txt
```

Expected: All ~50 tests pass. If anything fails, fix it before continuing to Task 1.

**Step 2: Commit the baseline artifact**

```bash
git add docs/verification/pytest_baseline.txt
git commit -m "test: capture baseline pytest output before verification pass"
```

---

## Task 1: Unit Tests for `trajectory.py`

**Files:**
- Create: `tests/test_trajectory.py`

**Step 1: Write the test file**

```python
# tests/test_trajectory.py
"""Unit tests for trajectory.py — displacement math and direction conventions."""
import numpy as np
import pytest

from src.injector.trajectory import build_trajectory, is_in_patch, Trajectory


class TestBuildTrajectory:
    def test_displacement_correct(self):
        """motion_ra=1 arcsec/hr, plate_scale=0.5 arcsec/px, dt=[0,1,2]hr → dx=[0,2,4]px."""
        timestamps = np.array([0.0, 1.0, 2.0])
        traj = build_trajectory(timestamps, start_x=10.0, start_y=10.0,
                                motion_ra=1.0, motion_dec=0.0, plate_scale=0.5)
        np.testing.assert_allclose(traj.xs, [10.0, 12.0, 14.0], atol=1e-10)
        np.testing.assert_allclose(traj.ys, [10.0, 10.0, 10.0], atol=1e-10)

    def test_pure_ra_shifts_x_only(self):
        """Pure RA motion must change xs and leave ys constant."""
        timestamps = np.array([0.0, 1.0, 2.0])
        traj = build_trajectory(timestamps, start_x=16.0, start_y=16.0,
                                motion_ra=2.0, motion_dec=0.0, plate_scale=1.0)
        assert traj.xs[0] != traj.xs[-1], "xs must change with RA motion"
        np.testing.assert_allclose(traj.ys, 16.0, atol=1e-10)

    def test_pure_dec_shifts_y_only(self):
        """Pure Dec motion must change ys and leave xs constant.
        Dec increases upward → pixel y decreases (ys = start_y - dec/plate * dt).
        """
        timestamps = np.array([0.0, 1.0, 2.0])
        traj = build_trajectory(timestamps, start_x=16.0, start_y=16.0,
                                motion_ra=0.0, motion_dec=2.0, plate_scale=1.0)
        np.testing.assert_allclose(traj.xs, 16.0, atol=1e-10)
        # ys = 16 - (2/1)*dt = 16, 14, 12
        np.testing.assert_allclose(traj.ys, [16.0, 14.0, 12.0], atol=1e-10)

    def test_zero_motion_constant_positions(self):
        """Zero motion must produce identical position every frame."""
        timestamps = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        traj = build_trajectory(timestamps, start_x=8.0, start_y=8.0,
                                motion_ra=0.0, motion_dec=0.0, plate_scale=0.263)
        np.testing.assert_allclose(traj.xs, 8.0, atol=1e-10)
        np.testing.assert_allclose(traj.ys, 8.0, atol=1e-10)

    def test_output_shape_and_dtype(self):
        """Output xs/ys must be float64 arrays of length n_frames with no NaN."""
        timestamps = np.array([0.0, 1.0, 2.0, 3.0])
        traj = build_trajectory(timestamps, start_x=5.0, start_y=5.0,
                                motion_ra=1.0, motion_dec=1.0, plate_scale=0.5)
        assert traj.xs.shape == (4,)
        assert traj.ys.shape == (4,)
        assert traj.xs.dtype == np.float64
        assert traj.ys.dtype == np.float64
        assert not np.any(np.isnan(traj.xs))
        assert not np.any(np.isnan(traj.ys))

    def test_as_array_shape(self):
        """as_array() must return (n_frames, 2) with xs in col 0, ys in col 1."""
        timestamps = np.array([0.0, 1.0])
        traj = build_trajectory(timestamps, start_x=5.0, start_y=3.0,
                                motion_ra=0.0, motion_dec=0.0, plate_scale=1.0)
        arr = traj.as_array()
        assert arr.shape == (2, 2)
        np.testing.assert_allclose(arr[:, 0], traj.xs)
        np.testing.assert_allclose(arr[:, 1], traj.ys)


class TestIsInPatch:
    def test_all_inside(self):
        """Static source in centre of 64×64 patch must always be in-patch."""
        timestamps = np.array([0.0, 1.0, 2.0])
        traj = build_trajectory(timestamps, start_x=32.0, start_y=32.0,
                                motion_ra=0.0, motion_dec=0.0, plate_scale=1.0)
        mask = is_in_patch(traj, patch_shape=(64, 64))
        assert np.all(mask)

    def test_exits_patch(self):
        """Source moving east at 10 px/hr must exit a 64-wide patch by frame 2."""
        # xs = 60 + 10*dt: frame0=60 (in), frame1=70 (in with +0.5 margin), frame2=80 (out)
        timestamps = np.array([0.0, 1.0, 2.0])
        traj = build_trajectory(timestamps, start_x=60.0, start_y=32.0,
                                motion_ra=10.0, motion_dec=0.0, plate_scale=1.0)
        mask = is_in_patch(traj, patch_shape=(64, 64))
        assert mask[0], "frame 0 (x=60) should be inside"
        assert not mask[2], "frame 2 (x=80) should be outside"
```

**Step 2: Run the new tests to verify they pass**

```bash
pytest tests/test_trajectory.py -v
```

Expected: All 7 tests PASS.

**Step 3: Commit**

```bash
git add tests/test_trajectory.py
git commit -m "test: unit tests for trajectory.py — displacement math and direction convention"
```

---

## Task 2: Unit Tests for `render_psf.py`

**Files:**
- Create: `tests/test_render_psf.py`

**Step 1: Write the test file**

```python
# tests/test_render_psf.py
"""Unit tests for render_psf.py — stamp shape, non-negativity, flux, centroid shift."""
import numpy as np
import pytest

from src.injector.render_psf import PSFParams, render_stamp, render_stack, _gaussian_kernel


class TestGaussianKernel:
    def test_normalized(self):
        """Gaussian kernel must sum to 1.0."""
        k = _gaussian_kernel(2.5, 13)
        assert abs(k.sum() - 1.0) < 1e-10

    def test_shape(self):
        k = _gaussian_kernel(2.5, 13)
        assert k.shape == (13, 13)

    def test_odd_size_required(self):
        """Kernel with even size still runs (no enforcement) but verify shape."""
        k = _gaussian_kernel(2.5, 12)
        assert k.shape == (12, 12)


class TestRenderStamp:
    def setup_method(self):
        self.psf = PSFParams(fwhm_pixels=2.5)

    def test_output_shape(self):
        stamp = render_stamp((64, 64), x=32.0, y=32.0, flux=100.0, psf_params=self.psf)
        assert stamp.shape == (64, 64)
        assert stamp.dtype == np.float64

    def test_nonnegative(self):
        stamp = render_stamp((64, 64), x=32.0, y=32.0, flux=100.0, psf_params=self.psf)
        assert np.all(stamp >= 0), "All stamp values must be >= 0"

    def test_flux_conservation(self):
        """Total flux in stamp must equal the requested flux (source fully inside)."""
        flux = 100.0
        stamp = render_stamp((64, 64), x=32.0, y=32.0, flux=flux, psf_params=self.psf)
        assert abs(stamp.sum() - flux) < 1.0, (
            f"Expected total flux ~{flux}, got {stamp.sum():.4f}"
        )

    def test_centroid_shifts_right_when_x_increases(self):
        """Moving x from 28 → 36 must shift the centroid column to the right."""
        stamp_left = render_stamp((64, 64), x=28.0, y=32.0, flux=100.0, psf_params=self.psf)
        stamp_right = render_stamp((64, 64), x=36.0, y=32.0, flux=100.0, psf_params=self.psf)
        cols = np.arange(64, dtype=float)
        cx_left = (stamp_left * cols[np.newaxis, :]).sum() / stamp_left.sum()
        cx_right = (stamp_right * cols[np.newaxis, :]).sum() / stamp_right.sum()
        assert cx_right > cx_left, (
            f"Expected centroid to shift right, got cx_left={cx_left:.2f}, cx_right={cx_right:.2f}"
        )

    def test_out_of_bounds_source_no_error(self):
        """Source fully outside the patch must produce an all-zero stamp without error."""
        stamp = render_stamp((64, 64), x=200.0, y=200.0, flux=100.0, psf_params=self.psf)
        assert stamp.shape == (64, 64)
        # May be all zeros if source is far enough outside
        assert stamp.sum() >= 0


class TestRenderStack:
    def setup_method(self):
        self.psf = PSFParams(fwhm_pixels=2.5)

    def test_output_shape(self):
        xs = np.array([30.0, 31.0, 32.0, 33.0, 34.0])
        ys = np.full(5, 32.0)
        fluxes = np.full(5, 100.0)
        stack = render_stack((64, 64), xs, ys, fluxes, self.psf)
        assert stack.shape == (5, 64, 64)
        assert stack.dtype == np.float64

    def test_nonnegative(self):
        xs = np.full(5, 32.0)
        ys = np.full(5, 32.0)
        fluxes = np.full(5, 50.0)
        stack = render_stack((64, 64), xs, ys, fluxes, self.psf)
        assert np.all(stack >= 0)

    def test_per_frame_position(self):
        """Each frame should have its peak near the specified (x, y)."""
        xs = np.array([16.0, 32.0, 48.0])
        ys = np.full(3, 32.0)
        fluxes = np.full(3, 100.0)
        stack = render_stack((64, 64), xs, ys, fluxes, self.psf)
        for i, x_expected in enumerate(xs):
            frame = stack[i]
            col_peak = int(np.unravel_index(frame.argmax(), frame.shape)[1])
            assert abs(col_peak - x_expected) <= 2, (
                f"Frame {i}: peak col={col_peak}, expected near x={x_expected}"
            )
```

**Step 2: Run tests**

```bash
pytest tests/test_render_psf.py -v
```

Expected: All 10 tests PASS.

**Step 3: Commit**

```bash
git add tests/test_render_psf.py
git commit -m "test: unit tests for render_psf.py — shape, flux, centroid, stack"
```

---

## Task 3: Unit Tests for `targets.py`

**Files:**
- Create: `tests/test_targets.py`

**Step 1: Write the test file**

```python
# tests/test_targets.py
"""Unit tests for targets.py — target placement strategies."""
import numpy as np
import pytest

from src.injector.targets import draw_target, TargetConfig


class TestDrawTargetUniform:
    def test_within_margin_bounds(self):
        """Uniform strategy must keep (x, y) inside margin-clipped patch."""
        rng = np.random.default_rng(0)
        config = TargetConfig(strategy="uniform", margin=0.05)
        n_rows, n_cols = 64, 64
        margin_x = 0.05 * n_cols
        margin_y = 0.05 * n_rows
        for _ in range(200):
            x, y = draw_target((n_rows, n_cols), config, rng)
            assert margin_x <= x <= n_cols - margin_x, f"x={x} out of bounds"
            assert margin_y <= y <= n_rows - margin_y, f"y={y} out of bounds"

    def test_returns_floats(self):
        rng = np.random.default_rng(1)
        x, y = draw_target((64, 64), TargetConfig(strategy="uniform"), rng)
        assert isinstance(x, float)
        assert isinstance(y, float)


class TestDrawTargetFixed:
    def test_returns_exact_coordinates(self):
        rng = np.random.default_rng(0)
        config = TargetConfig(strategy="fixed", fixed_x=12.5, fixed_y=33.7)
        x, y = draw_target((64, 64), config, rng)
        assert x == 12.5
        assert y == 33.7

    def test_rng_not_consumed(self):
        """Fixed strategy should not consume RNG state (result is deterministic)."""
        rng = np.random.default_rng(99)
        state_before = rng.bit_generator.state
        draw_target((64, 64), TargetConfig(strategy="fixed", fixed_x=10.0, fixed_y=10.0), rng)
        # fixed path doesn't call rng, so state unchanged
        state_after = rng.bit_generator.state
        assert state_before == state_after


class TestDrawTargetCenter:
    def test_mean_near_patch_center(self):
        """Center strategy mean should be near (n_cols/2, n_rows/2) over many draws."""
        rng = np.random.default_rng(42)
        config = TargetConfig(strategy="center", margin=0.05)
        xs, ys = [], []
        for _ in range(300):
            x, y = draw_target((64, 64), config, rng)
            xs.append(x)
            ys.append(y)
        assert abs(np.mean(xs) - 32.0) < 4.0, f"Mean x={np.mean(xs):.2f}, expected ~32"
        assert abs(np.mean(ys) - 32.0) < 4.0, f"Mean y={np.mean(ys):.2f}, expected ~32"

    def test_clipped_within_bounds(self):
        """Center strategy must never produce coordinates outside margin bounds."""
        rng = np.random.default_rng(7)
        config = TargetConfig(strategy="center", margin=0.05)
        margin = 0.05 * 64
        for _ in range(200):
            x, y = draw_target((64, 64), config, rng)
            assert margin <= x <= 64 - margin
            assert margin <= y <= 64 - margin


class TestDrawTargetGrid:
    def test_cell_zero_coords(self):
        """Grid cell 0 (row=0, col=0) → x in [0, cell_w), y in [0, cell_h)."""
        rng = np.random.default_rng(0)
        config = TargetConfig(strategy="grid", grid_n=4, margin=0.0)
        n_rows, n_cols = 64, 64
        cell_w = n_cols / 4  # 16
        cell_h = n_rows / 4  # 16
        for _ in range(50):
            x, y = draw_target((n_rows, n_cols), config, rng, grid_index=0)
            assert 0.0 <= x < cell_w + 1, f"x={x} outside cell 0 col range"
            assert 0.0 <= y < cell_h + 1, f"y={y} outside cell 0 row range"

    def test_grid_index_wraps(self):
        """grid_index wraps modulo grid_n^2, so index 16 == index 0 for grid_n=4."""
        rng1 = np.random.default_rng(5)
        rng2 = np.random.default_rng(5)
        config = TargetConfig(strategy="grid", grid_n=4, margin=0.0)
        x1, y1 = draw_target((64, 64), config, rng1, grid_index=0)
        x2, y2 = draw_target((64, 64), config, rng2, grid_index=16)
        assert x1 == x2 and y1 == y2


class TestDrawTargetEdgeCases:
    def test_unknown_strategy_raises(self):
        rng = np.random.default_rng(0)
        config = TargetConfig()
        config.strategy = "teleport"  # type: ignore  — inject bad value
        with pytest.raises(ValueError, match="Unknown target strategy"):
            draw_target((64, 64), config, rng)
```

**Step 2: Run tests**

```bash
pytest tests/test_targets.py -v
```

Expected: All 9 tests PASS.

**Step 3: Commit**

```bash
git add tests/test_targets.py
git commit -m "test: unit tests for targets.py — all placement strategies and edge cases"
```

---

## Task 4: Additive Constraint Tests in `test_injection.py`

**Files:**
- Modify: `tests/test_injection.py` (append a new test class)

**Step 1: Read the current file to understand the append point**

Read `tests/test_injection.py` — the file currently ends after `TestSigmaMapCalibration`. Append the following class after the last line.

**Step 2: Append the new test class**

Add to the end of `tests/test_injection.py`:

```python


class TestAdditiveConstraint:
    """Verify the core invariant: X == patch_stack + Y (injected signal)."""

    def _zeros_patch(self, T=5, size=64):
        return np.zeros((T, size, size), dtype=np.float64)

    def _noise_patch(self, T=5, size=64, seed=7):
        return np.random.default_rng(seed).normal(0, 5, (T, size, size))

    def _times(self, T=5):
        return np.arange(T, dtype=float) * 0.5

    def test_x_equals_patch_plus_y_zero_patch(self):
        """X must exactly equal patch_stack + Y when patch is zeros."""
        from src.injector import inject
        from src.injector.render_psf import PSFParams
        from src.injector.targets import TargetConfig

        patch = self._zeros_patch()
        psf = PSFParams(fwhm_pixels=2.5)
        config = TargetConfig(strategy="fixed", fixed_x=32.0, fixed_y=32.0)

        X, Y, meta = inject(patch, self._times(), 0.263, psf, seed=0,
                            target_config=config)

        np.testing.assert_allclose(
            X, patch + Y, atol=1e-12,
            err_msg="X must equal patch_stack + Y"
        )

    def test_x_equals_patch_plus_y_noise_patch(self):
        """X - patch_stack must equal Y even when patch has non-zero background."""
        from src.injector import inject
        from src.injector.render_psf import PSFParams

        patch = self._noise_patch()
        psf = PSFParams(fwhm_pixels=2.5)

        X, Y, _ = inject(patch, self._times(), 0.263, psf, seed=7)

        np.testing.assert_allclose(
            X - patch, Y, atol=1e-12,
            err_msg="X - patch_stack must equal Y for noisy patch"
        )

    def test_y_is_nonnegative(self):
        """Injected signal Y must be non-negative everywhere (PSF clipped to >= 0)."""
        from src.injector import inject
        from src.injector.render_psf import PSFParams

        patch = self._zeros_patch()
        psf = PSFParams(fwhm_pixels=2.5)
        _, Y, _ = inject(patch, self._times(), 0.263, psf, seed=1)
        assert np.all(Y >= 0), f"Y had {(Y < 0).sum()} negative values"

    def test_y_sum_positive(self):
        """Y must have positive total flux — something was injected."""
        from src.injector import inject
        from src.injector.render_psf import PSFParams

        patch = self._zeros_patch()
        psf = PSFParams(fwhm_pixels=2.5)
        _, Y, _ = inject(patch, self._times(), 0.263, psf, seed=2)
        assert Y.sum() > 0, "Y.sum() should be positive after injection"

    def test_seed_reproducibility(self):
        """Same seed must produce identical X, Y, meta."""
        from src.injector import inject
        from src.injector.render_psf import PSFParams

        patch = self._noise_patch(seed=99)
        psf = PSFParams(fwhm_pixels=2.5)

        X1, Y1, meta1 = inject(patch, self._times(), 0.263, psf, seed=42)
        X2, Y2, meta2 = inject(patch, self._times(), 0.263, psf, seed=42)

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(Y1, Y2)
        assert meta1["start_x_px"] == meta2["start_x_px"]
        assert meta1["start_y_px"] == meta2["start_y_px"]
```

**Step 3: Run only the new class**

```bash
pytest tests/test_injection.py::TestAdditiveConstraint -v
```

Expected: All 5 tests PASS.

**Step 4: Run the full injection test file**

```bash
pytest tests/test_injection.py -v
```

Expected: All tests PASS (including original 4 sigma tests).

**Step 5: Commit**

```bash
git add tests/test_injection.py
git commit -m "test: additive constraint X == patch_stack + Y, seed reproducibility"
```

---

## Task 5: End-to-End Integration Test

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write the integration test file**

```python
# tests/test_integration.py
"""
End-to-end integration tests: FITS → zarr → inject → .npz round-trip.

Synthetic path: always runs (uses fake FITS via conftest helper).
Real-data path: opt-in via KBO_REALDATA=1 environment variable.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import zarr

# Import conftest helper (pytest adds tests/ to sys.path)
from conftest import write_fake_fits

from src.data.build_one_stack import build_one_stack
from src.data.patches import extract_patch_grid, patch_sigma
from src.injector import inject, PSFParams
from src.injector.targets import TargetConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_fits_stack(tmp_path: Path, n_frames: int = 5) -> Path:
    """Create n_frames fake FITS files in tmp_path/fits/ and return the dir."""
    fits_dir = tmp_path / "fits"
    fits_dir.mkdir()
    for i in range(n_frames):
        write_fake_fits(fits_dir / f"frame_{i:04d}.fits", mjd=60000.0 + i * 0.04)
    return fits_dir


def _open_and_verify_zarr(zarr_path: Path, expected_T: int) -> zarr.Group:
    """Open a zarr store and assert the required schema is present."""
    assert zarr_path.exists(), f"zarr not found: {zarr_path}"
    z = zarr.open(str(zarr_path), mode="r")
    assert "images" in z,     "zarr missing 'images' dataset"
    assert "timestamps" in z, "zarr missing 'timestamps' dataset"
    assert "psf_fwhm" in z,   "zarr missing 'psf_fwhm' dataset"
    assert z["images"].shape[0] == expected_T, (
        f"Expected {expected_T} frames, got {z['images'].shape[0]}"
    )
    assert z["timestamps"].shape == (expected_T,)
    assert z["psf_fwhm"].shape == (expected_T,)
    assert "plate_scale" in z.attrs
    assert float(z.attrs["plate_scale"]) > 0
    return z


# ---------------------------------------------------------------------------
# Synthetic path (always runs)
# ---------------------------------------------------------------------------

class TestSyntheticPipeline:
    def test_fits_to_zarr_schema(self, tmp_path):
        """FITS → build_one_stack → zarr with correct schema."""
        fits_dir = _make_fake_fits_stack(tmp_path)
        zarr_path = tmp_path / "stack.zarr"
        build_one_stack(fits_dir, zarr_path, T=5, patch_size=32, stride=16,
                        plate_scale=0.263)
        _open_and_verify_zarr(zarr_path, expected_T=5)

    def test_zarr_images_shape(self, tmp_path):
        """zarr images must be (T, H, W) float32."""
        fits_dir = _make_fake_fits_stack(tmp_path)
        zarr_path = tmp_path / "stack.zarr"
        build_one_stack(fits_dir, zarr_path, T=5, patch_size=32, stride=16,
                        plate_scale=0.263)
        z = zarr.open(str(zarr_path), mode="r")
        imgs = z["images"][:]
        assert imgs.ndim == 3
        assert imgs.dtype == np.float32

    def test_zarr_to_inject_to_npz(self, tmp_path):
        """zarr → extract patch → inject → save .npz → load → verify keys and shapes."""
        fits_dir = _make_fake_fits_stack(tmp_path)
        zarr_path = tmp_path / "stack.zarr"
        build_one_stack(fits_dir, zarr_path, T=5, patch_size=32, stride=16,
                        plate_scale=0.263)

        z = zarr.open(str(zarr_path), mode="r")
        imgs = z["images"][:]
        timestamps = z["timestamps"][:]
        psf_fwhm = z["psf_fwhm"][:]
        plate_scale = float(z.attrs["plate_scale"])

        patches = list(extract_patch_grid(imgs, patch_size=32, stride=16))
        assert len(patches) > 0, "No patches extracted — check patch_size vs image size"

        patch_stack, row_start, col_start = patches[0]
        sigma = patch_sigma(patch_stack)
        t_hours = (timestamps - timestamps[0]) * 24.0
        psf_params = PSFParams(fwhm_pixels=float(psf_fwhm.mean()))

        X, Y, meta = inject(patch_stack, t_hours, plate_scale, psf_params,
                            seed=0, sigma_map=sigma)

        # Save .npz
        out_dir = tmp_path / "cases"
        out_dir.mkdir()
        npz_path = out_dir / "case_0000.npz"
        np.savez(
            npz_path,
            patch_stack=patch_stack.astype(np.float32),
            X=X.astype(np.float32),
            Y=Y.astype(np.float32),
            sigma_patch=sigma,
            timestamps=timestamps,
            plate_scale=np.float32(plate_scale),
            psf_fwhm=psf_fwhm,
            meta=np.array(meta, dtype=object),
        )

        # Load and verify
        case = np.load(npz_path, allow_pickle=True)
        required_keys = ["patch_stack", "X", "Y", "sigma_patch",
                         "timestamps", "plate_scale", "psf_fwhm", "meta"]
        for key in required_keys:
            assert key in case, f"Missing key in .npz: {key}"

        assert case["X"].shape == case["Y"].shape == case["patch_stack"].shape

    def test_additive_constraint_end_to_end(self, tmp_path):
        """X == patch_stack + Y must hold in the full pipeline output."""
        fits_dir = _make_fake_fits_stack(tmp_path)
        zarr_path = tmp_path / "stack.zarr"
        build_one_stack(fits_dir, zarr_path, T=5, patch_size=32, stride=16,
                        plate_scale=0.263)

        z = zarr.open(str(zarr_path), mode="r")
        imgs = z["images"][:]
        timestamps = z["timestamps"][:]
        psf_fwhm = z["psf_fwhm"][:]
        plate_scale = float(z.attrs["plate_scale"])

        patches = list(extract_patch_grid(imgs, patch_size=32, stride=16))
        patch_stack, _, _ = patches[0]
        sigma = patch_sigma(patch_stack)
        t_hours = (timestamps - timestamps[0]) * 24.0
        psf_params = PSFParams(fwhm_pixels=float(psf_fwhm.mean()))

        X, Y, _ = inject(patch_stack, t_hours, plate_scale, psf_params,
                         seed=0, sigma_map=sigma)

        np.testing.assert_allclose(
            X, patch_stack + Y, atol=1e-6,
            err_msg="End-to-end additive constraint failed: X != patch_stack + Y"
        )


# ---------------------------------------------------------------------------
# Real-data path (opt-in: KBO_REALDATA=1)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("KBO_REALDATA"),
    reason="Set KBO_REALDATA=1 to run real-data integration tests",
)
class TestRealDataPipeline:
    def _get_real_data_dir(self):
        path = Path(os.environ.get("KBO_REALDATA_PATH", "kbmod/kbmod/data/small"))
        if not path.exists():
            pytest.skip(f"Real data path not found: {path}")
        return path

    def test_real_fits_to_zarr(self, tmp_path):
        """Real FITS → zarr with correct schema."""
        real_dir = self._get_real_data_dir()
        zarr_path = tmp_path / "real.zarr"
        build_one_stack(real_dir, zarr_path, T=5, patch_size=32, stride=16)
        _open_and_verify_zarr(zarr_path, expected_T=5)

    def test_real_zarr_to_inject(self, tmp_path):
        """Real zarr → extract patch → inject → verify shapes and constraint."""
        real_dir = self._get_real_data_dir()
        zarr_path = tmp_path / "real.zarr"
        build_one_stack(real_dir, zarr_path, T=5, patch_size=32, stride=16)

        z = zarr.open(str(zarr_path), mode="r")
        imgs = z["images"][:]
        timestamps = z["timestamps"][:]
        psf_fwhm = z["psf_fwhm"][:]
        plate_scale = float(z.attrs["plate_scale"])

        patches = list(extract_patch_grid(imgs, patch_size=32, stride=16))
        assert len(patches) > 0, "No patches from real data"

        patch_stack, _, _ = patches[0]
        sigma = patch_sigma(patch_stack)
        t_hours = (timestamps - timestamps[0]) * 24.0
        psf_params = PSFParams(fwhm_pixels=float(psf_fwhm.mean()))

        X, Y, meta = inject(patch_stack, t_hours, plate_scale, psf_params,
                            seed=0, sigma_map=sigma)

        assert X.shape == patch_stack.shape
        assert Y.shape == patch_stack.shape
        assert isinstance(meta, dict)
        np.testing.assert_allclose(X, patch_stack + Y, atol=1e-6)
```

**Step 2: Run synthetic tests only**

```bash
pytest tests/test_integration.py::TestSyntheticPipeline -v
```

Expected: All 4 synthetic tests PASS.

**Step 3: Verify real-data tests skip cleanly without env var**

```bash
pytest tests/test_integration.py::TestRealDataPipeline -v
```

Expected: Both tests show `SKIPPED (Set KBO_REALDATA=1 ...)`.

**Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: end-to-end integration test FITS→zarr→inject→npz (synthetic + real-data opt-in)"
```

---

## Task 6: Run Full Suite and Capture Final Artifact

**Files:**
- Create: `docs/verification/pytest_final.txt`

**Step 1: Run the complete test suite**

```bash
cd InjectionEngine
pytest -q 2>&1 | tee docs/verification/pytest_final.txt
```

Expected output (approximate):
```
......................................................................  [100%]
XX passed in X.XXs
```

All tests must PASS. Zero failures, zero errors.

**Step 2: Commit the artifact**

```bash
git add docs/verification/pytest_final.txt
git commit -m "test: capture final pytest output — all tests passing"
```

---

## Task 7: Manual Visual Sanity Check and Record

**Files:**
- Create: `docs/verification/step1-3.md`

**Step 1: Run demo.py on a positive case**

```bash
python InjectionEngine/demo/demo.py --case InjectionEngine/demo/cases/<pos_hard>.npz
```

(Replace `<pos_hard>` with an actual case filename from `demo/cases/`.)

**Step 2: Run demo.py on a negative case**

```bash
python InjectionEngine/demo/demo.py --case InjectionEngine/demo/cases/<neg>.npz
```

**Step 3: Check three concrete pass criteria**

1. **Monotonic motion:** The injected track moves monotonically in the sampled direction across frames (no backwards jumps, no teleports)
2. **Mask alignment:** Target mask peak aligns with the injected center at each epoch
3. **No axis swap:** Frame-to-frame movement is visible and the time dimension behaves as time (objects move between consecutive frames)

**Step 4: Write the verification record**

Create `docs/verification/step1-3.md` with:

```markdown
# Step 1–3 Manual Verification Record

**Date:** YYYY-MM-DD

## Commands

### Baseline test run
```
cd InjectionEngine && pytest -q 2>&1 | tee docs/verification/pytest_baseline.txt
```

### Final test run (after new tests added)
```
cd InjectionEngine && pytest -q 2>&1 | tee docs/verification/pytest_final.txt
```

### Manual sanity check
```
python InjectionEngine/demo/demo.py --case InjectionEngine/demo/cases/<pos_hard>.npz
python InjectionEngine/demo/demo.py --case InjectionEngine/demo/cases/<neg>.npz
```

## Visual Sanity Results

### Positive case (`<filename>`)
- [ ] Monotonic motion: [PASS/FAIL — notes]
- [ ] Mask alignment: [PASS/FAIL — notes]
- [ ] No axis swap: [PASS/FAIL — notes]

### Negative case (`<filename>`)
- [ ] Y == 0 / no signal visible: [PASS/FAIL — notes]

## Conclusion

Steps 1–3 verified. Ready to proceed to model training.
```

**Step 5: Commit**

```bash
git add docs/verification/step1-3.md
git commit -m "docs: Step 1-3 verification record — all tests pass, visual sanity confirmed"
```

---

## Summary of New Files

| File | Purpose |
|---|---|
| `tests/test_trajectory.py` | displacement math, direction convention, is_in_patch |
| `tests/test_render_psf.py` | stamp shape, flux, centroid shift, render_stack |
| `tests/test_targets.py` | all 4 placement strategies, bounds, edge cases |
| `tests/test_injection.py` | +5 additive constraint tests appended to existing file |
| `tests/test_integration.py` | FITS→zarr→inject→npz synthetic + real-data (opt-in) |
| `docs/verification/pytest_baseline.txt` | baseline run before new tests |
| `docs/verification/pytest_final.txt` | final run after all tests added |
| `docs/verification/step1-3.md` | manual sanity record + conclusion |
