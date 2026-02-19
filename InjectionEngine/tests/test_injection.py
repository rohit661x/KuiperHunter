"""Tests for inject() public API – Task 8: per-epoch sigma calibration."""
import numpy as np
import pytest


class TestSigmaMapCalibration:
    """Test Step 3.6: per-epoch sigma calibration."""

    def _patch(self, T=5, size=64):
        rng = np.random.default_rng(0)
        return rng.normal(0, 1, (T, size, size))

    def test_sigma_map_scales_flux(self):
        """With sigma_map, injected peak should scale proportionally."""
        from src.injector import inject
        from src.injector.render_psf import PSFParams

        patch = self._patch()
        times = np.arange(5, dtype=float) * 0.5
        psf = PSFParams(fwhm_pixels=2.5)

        sigma_low  = np.ones(5) * 1.0
        sigma_high = np.ones(5) * 10.0

        _, Y_low,  _ = inject(patch, times, 0.263, psf, seed=42, sigma_map=sigma_low)
        _, Y_high, _ = inject(patch, times, 0.263, psf, seed=42, sigma_map=sigma_high)

        ratio = Y_high.sum() / Y_low.sum()
        assert abs(ratio - 10.0) < 0.5  # should be ~10x

    def test_no_sigma_map_backward_compat(self):
        """inject() without sigma_map must still work (backward compat)."""
        from src.injector import inject
        from src.injector.render_psf import PSFParams

        patch = self._patch()
        times = np.arange(5, dtype=float) * 0.5
        psf = PSFParams(fwhm_pixels=2.5)

        X, Y, meta = inject(patch, times, 0.263, psf, seed=42)
        assert Y.sum() > 0

    def test_sigma_map_wrong_length_raises(self):
        from src.injector import inject
        from src.injector.render_psf import PSFParams

        patch = self._patch(T=5)
        times = np.arange(5, dtype=float) * 0.5
        psf = PSFParams(fwhm_pixels=2.5)

        with pytest.raises(ValueError, match="sigma_map"):
            inject(patch, times, 0.263, psf, seed=42, sigma_map=np.ones(3))

    def test_meta_records_sigma_map_used(self):
        """meta should indicate whether sigma calibration was applied."""
        from src.injector import inject
        from src.injector.render_psf import PSFParams

        patch = self._patch()
        times = np.arange(5, dtype=float) * 0.5
        psf = PSFParams(fwhm_pixels=2.5)

        _, _, meta_no  = inject(patch, times, 0.263, psf, seed=42)
        _, _, meta_yes = inject(patch, times, 0.263, psf, seed=42,
                                sigma_map=np.ones(5) * 5.0)

        assert meta_no["sigma_calibrated"] is False
        assert meta_yes["sigma_calibrated"] is True


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
