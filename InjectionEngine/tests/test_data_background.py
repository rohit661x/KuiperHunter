# tests/test_data_background.py
"""Tests for background.py — per-frame background subtraction and noise estimation."""
import numpy as np
import pytest


class TestSubtractBackground:
    def _make_imgs(self, T=3, H=64, W=64, bg=100.0, sigma=5.0, seed=0):
        rng = np.random.default_rng(seed)
        return (bg + rng.normal(0, sigma, (T, H, W))).astype(np.float32)

    def test_output_shapes(self):
        from src.data.background import subtract_background
        imgs = self._make_imgs()
        imgs_out, sigma = subtract_background(imgs)
        assert imgs_out.shape == imgs.shape
        assert sigma.shape == (3,)

    def test_background_removed(self):
        """After subtraction, per-frame median should be near zero."""
        from src.data.background import subtract_background
        imgs = self._make_imgs(bg=200.0)
        imgs_out, _ = subtract_background(imgs)
        for t in range(imgs_out.shape[0]):
            assert abs(np.median(imgs_out[t])) < 1.0

    def test_sigma_positive(self):
        from src.data.background import subtract_background
        imgs = self._make_imgs()
        _, sigma = subtract_background(imgs)
        assert (sigma > 0).all()

    def test_sigma_estimates_noise(self):
        """Sigma should be within 20% of the true noise std for Gaussian noise."""
        from src.data.background import subtract_background
        true_sigma = 7.5
        imgs = self._make_imgs(sigma=true_sigma, T=1, H=128, W=128)
        _, sigma = subtract_background(imgs)
        assert abs(sigma[0] - true_sigma) / true_sigma < 0.20

    def test_does_not_modify_input(self):
        from src.data.background import subtract_background
        imgs = self._make_imgs()
        orig = imgs.copy()
        subtract_background(imgs)
        np.testing.assert_array_equal(imgs, orig)
