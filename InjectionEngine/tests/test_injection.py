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
