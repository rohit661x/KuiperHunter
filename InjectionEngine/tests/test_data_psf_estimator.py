# tests/test_data_psf_estimator.py
"""Tests for psf_estimator.py — per-epoch PSF FWHM estimation."""
import numpy as np
import pytest


def _gauss2d(shape, cx, cy, sigma):
    H, W = shape
    rows, cols = np.mgrid[0:H, 0:W]
    return np.exp(-((cols - cx)**2 + (rows - cy)**2) / (2 * sigma**2)).astype(np.float32)


class TestEstimatePsfFwhm:
    def test_output_shape(self):
        from src.data.psf_estimator import estimate_psf_fwhm
        imgs = np.random.default_rng(0).normal(0, 1, (4, 64, 64)).astype(np.float32)
        fwhm = estimate_psf_fwhm(imgs)
        assert fwhm.shape == (4,)
        assert fwhm.dtype == np.float32

    def test_all_positive(self):
        from src.data.psf_estimator import estimate_psf_fwhm
        imgs = np.random.default_rng(0).normal(0, 1, (3, 64, 64)).astype(np.float32)
        fwhm = estimate_psf_fwhm(imgs)
        assert (fwhm > 0).all()

    def test_recovers_known_fwhm(self):
        """For a clean Gaussian star, returned FWHM should be within 20% of truth."""
        from src.data.psf_estimator import estimate_psf_fwhm
        sigma_true = 2.0
        fwhm_true = sigma_true * 2.3548
        # Plant a Gaussian star at (32, 32) in a noise-free image
        img = (_gauss2d((64, 64), 32, 32, sigma_true) * 100.0)
        imgs = img[np.newaxis]  # T=1
        fwhm = estimate_psf_fwhm(imgs)
        assert abs(fwhm[0] - fwhm_true) / fwhm_true < 0.20

    def test_fallback_on_flat_image(self):
        """All-zero image should return the fallback FWHM, not raise."""
        from src.data.psf_estimator import estimate_psf_fwhm
        imgs = np.zeros((2, 64, 64), dtype=np.float32)
        fallback = 2.5
        fwhm = estimate_psf_fwhm(imgs, fallback_fwhm=fallback)
        np.testing.assert_allclose(fwhm, fallback)
