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

# Import conftest helper
from tests.conftest import write_fake_fits

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
