# tests/test_data_register.py
"""Tests for register.py — sidereal alignment."""
import numpy as np
import pytest


def _make_identical_hdrs(n=3) -> list[dict]:
    """All epochs share the same WCS (typical for same-pointing stacks)."""
    hdr = {
        "CRVAL1": 200.615, "CRVAL2": -7.789,
        "CRPIX1": 1033.9,  "CRPIX2": 2043.5,
        "CD1_1": -1.14e-07, "CD1_2": 7.318e-05,
        "CD2_1": -7.301e-05, "CD2_2": -1.28e-07,
    }
    return [dict(hdr) for _ in range(n)]


def _make_imgs(T=3, H=64, W=64, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 5, (T, H, W)).astype(np.float32)


class TestRegisterToEpoch0:
    def test_identical_wcs_returns_copy(self):
        from src.data.register import register_to_epoch0
        imgs = _make_imgs()
        hdrs = _make_identical_hdrs(3)
        out = register_to_epoch0(imgs, hdrs)
        np.testing.assert_array_equal(out, imgs)
        assert out is not imgs  # must be a copy

    def test_output_shape_unchanged(self):
        from src.data.register import register_to_epoch0
        imgs = _make_imgs(T=5, H=64, W=64)
        hdrs = _make_identical_hdrs(5)
        out = register_to_epoch0(imgs, hdrs)
        assert out.shape == imgs.shape

    def test_shifted_wcs_shifts_image(self):
        """One epoch shifted by 1 pixel should produce a shifted output."""
        from src.data.register import register_to_epoch0
        import numpy as np
        # Build a simple image with a bright point source at (32, 32)
        T, H, W = 2, 64, 64
        imgs = np.zeros((T, H, W), dtype=np.float32)
        imgs[0, 32, 32] = 100.0
        imgs[1, 32, 32] = 100.0  # same pixel coords before registration

        # Build hdrs: epoch-0 is reference; epoch-1 is shifted by 1 px in x
        base_hdr = {
            "CRVAL1": 200.615, "CRVAL2": -7.789,
            "CRPIX1": 1033.9,  "CRPIX2": 2043.5,
            "CD1_1": -1.14e-07, "CD1_2": 7.318e-05,
            "CD2_1": -7.301e-05, "CD2_2": -1.28e-07,
        }
        shifted_hdr = dict(base_hdr)
        shifted_hdr["CRPIX1"] = base_hdr["CRPIX1"] + 1.0  # shift ref pixel by 1

        hdrs = [base_hdr, shifted_hdr]
        out = register_to_epoch0(imgs, hdrs)
        # Epoch-0 stays unchanged
        assert out[0, 32, 32] == pytest.approx(100.0, abs=1.0)
        # Epoch-1 peak should move by ~1 pixel
        assert out[1].max() > 50.0  # still bright somewhere
        assert out[1, 32, 32] < 100.0  # no longer exactly at (32, 32)
