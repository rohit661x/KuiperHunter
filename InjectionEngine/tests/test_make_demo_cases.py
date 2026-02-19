# tests/test_make_demo_cases.py
"""Tests for make_demo_cases.py — injection cases from a zarr stack."""
import numpy as np
import pytest
import zarr
from tests.conftest import write_fake_fits


def _write_fake_zarr(tmp_path, T=5, H=64, W=64, seed=0):
    """Create a minimal zarr store matching build_one_stack output."""
    rng = np.random.default_rng(seed)
    imgs = rng.normal(0, 5, (T, H, W)).astype(np.float32)
    mjds = np.array([57130.0 + i * 0.01 for i in range(T)])
    psf_fwhm = np.full(T, 2.5, dtype=np.float32)

    zarr_path = tmp_path / "stack.zarr"
    z = zarr.open(str(zarr_path), mode="w")
    z.create_dataset("images",     data=imgs, overwrite=True)
    z.create_dataset("timestamps", data=mjds, overwrite=True)
    z.create_dataset("psf_fwhm",   data=psf_fwhm, overwrite=True)
    z.attrs["plate_scale"] = 0.263
    z.attrs["T"] = T
    z.attrs["patch_size"] = 32
    z.attrs["stride"] = 32
    return zarr_path


class TestMakeDemoCases:
    def test_creates_npz_files(self, tmp_path):
        from demo.make_demo_cases import make_demo_cases
        zarr_path = _write_fake_zarr(tmp_path)
        out_dir = tmp_path / "cases"
        make_demo_cases(str(zarr_path), str(out_dir), n_cases=2, seed=0)
        npz_files = list(out_dir.glob("*.npz"))
        assert len(npz_files) == 2

    def test_npz_has_required_keys(self, tmp_path):
        from demo.make_demo_cases import make_demo_cases
        zarr_path = _write_fake_zarr(tmp_path)
        out_dir = tmp_path / "cases"
        make_demo_cases(str(zarr_path), str(out_dir), n_cases=1, seed=0)
        npz = np.load(list((out_dir).glob("*.npz"))[0], allow_pickle=True)
        for key in ("patch_stack", "X", "Y", "sigma_patch",
                    "timestamps", "plate_scale"):
            assert key in npz, f"Missing key: {key}"

    def test_x_equals_patch_plus_y(self, tmp_path):
        """X = patch_stack + Y must hold exactly."""
        from demo.make_demo_cases import make_demo_cases
        zarr_path = _write_fake_zarr(tmp_path)
        out_dir = tmp_path / "cases"
        make_demo_cases(str(zarr_path), str(out_dir), n_cases=1, seed=0)
        npz = np.load(list((out_dir).glob("*.npz"))[0], allow_pickle=True)
        np.testing.assert_allclose(
            npz["X"].astype(np.float64),
            npz["patch_stack"].astype(np.float64) + npz["Y"].astype(np.float64),
            atol=1e-5,
        )
