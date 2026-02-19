# tests/test_data_patches.py
"""Tests for patches.py — grid-based patch extraction and per-patch sigma."""
import numpy as np
import pytest


class TestExtractPatchGrid:
    def _make_imgs(self, T=5, H=64, W=64, seed=0):
        return np.random.default_rng(seed).normal(0, 5, (T, H, W)).astype(np.float32)

    def test_yields_correct_shape(self):
        from src.data.patches import extract_patch_grid
        imgs = self._make_imgs(T=5, H=64, W=64)
        patches = list(extract_patch_grid(imgs, patch_size=32, stride=32))
        assert len(patches) > 0
        patch_stack, row_start, col_start = patches[0]
        assert patch_stack.shape == (5, 32, 32)

    def test_grid_covers_image(self):
        """All non-overlapping 32×32 patches in a 64×64 image: expect 4 patches."""
        from src.data.patches import extract_patch_grid
        imgs = self._make_imgs(H=64, W=64)
        patches = list(extract_patch_grid(imgs, patch_size=32, stride=32))
        assert len(patches) == 4  # (64/32) * (64/32)

    def test_row_col_starts_are_correct(self):
        from src.data.patches import extract_patch_grid
        imgs = self._make_imgs(H=64, W=64)
        patches = list(extract_patch_grid(imgs, patch_size=32, stride=32))
        starts = [(r, c) for _, r, c in patches]
        assert (0, 0) in starts
        assert (0, 32) in starts
        assert (32, 0) in starts
        assert (32, 32) in starts

    def test_patch_data_matches_source(self):
        from src.data.patches import extract_patch_grid
        imgs = self._make_imgs(H=64, W=64)
        patches = list(extract_patch_grid(imgs, patch_size=32, stride=32))
        for patch_stack, r, c in patches:
            np.testing.assert_array_equal(
                patch_stack, imgs[:, r:r + 32, c:c + 32]
            )


class TestPatchSigma:
    def test_output_shape(self):
        from src.data.patches import patch_sigma
        patch_stack = np.random.default_rng(0).normal(0, 5, (5, 32, 32)).astype(np.float32)
        sigma = patch_sigma(patch_stack)
        assert sigma.shape == (5,)
        assert sigma.dtype == np.float32

    def test_sigma_positive(self):
        from src.data.patches import patch_sigma
        patch_stack = np.random.default_rng(0).normal(0, 5, (3, 32, 32)).astype(np.float32)
        sigma = patch_sigma(patch_stack)
        assert (sigma > 0).all()

    def test_sigma_estimates_noise(self):
        """Sigma from patch should be within 30% of true noise std."""
        from src.data.patches import patch_sigma
        true_sigma = 8.0
        imgs = np.random.default_rng(7).normal(0, true_sigma, (2, 64, 64)).astype(np.float32)
        sigma = patch_sigma(imgs)
        for s in sigma:
            assert abs(s - true_sigma) / true_sigma < 0.30
