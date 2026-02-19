# tests/test_data_build_one_stack.py
"""Tests for build_one_stack.py — end-to-end zarr pipeline."""
import numpy as np
import pytest
import zarr
from tests.conftest import write_fake_fits


class TestBuildOneStack:
    def _make_stack_dir(self, tmp_path, T=5):
        stack_dir = tmp_path / "raw"
        stack_dir.mkdir()
        for i in range(T):
            write_fake_fits(stack_dir / f"{i:06d}.fits", mjd=57130.0 + i * 0.01, seed=i)
        return stack_dir

    def test_creates_zarr(self, tmp_path):
        from src.data.build_one_stack import build_one_stack
        stack_dir = self._make_stack_dir(tmp_path)
        out = tmp_path / "stack.zarr"
        build_one_stack(stack_dir, out, T=5)
        assert out.exists()

    def test_zarr_has_required_datasets(self, tmp_path):
        from src.data.build_one_stack import build_one_stack
        stack_dir = self._make_stack_dir(tmp_path)
        out = tmp_path / "stack.zarr"
        build_one_stack(stack_dir, out, T=5)
        z = zarr.open(str(out), mode="r")
        assert "images" in z
        assert "timestamps" in z
        assert "psf_fwhm" in z

    def test_zarr_images_shape(self, tmp_path):
        from src.data.build_one_stack import build_one_stack
        stack_dir = self._make_stack_dir(tmp_path)
        out = tmp_path / "stack.zarr"
        build_one_stack(stack_dir, out, T=5)
        z = zarr.open(str(out), mode="r")
        assert z["images"].shape == (5, 64, 64)

    def test_zarr_plate_scale_attr(self, tmp_path):
        from src.data.build_one_stack import build_one_stack
        stack_dir = self._make_stack_dir(tmp_path)
        out = tmp_path / "stack.zarr"
        build_one_stack(stack_dir, out, T=5)
        z = zarr.open(str(out), mode="r")
        assert "plate_scale" in z.attrs
        ps = z.attrs["plate_scale"]
        assert 0.1 < ps < 1.0  # reasonable arcsec/px

    def test_plate_scale_override(self, tmp_path):
        from src.data.build_one_stack import build_one_stack
        stack_dir = self._make_stack_dir(tmp_path)
        out = tmp_path / "stack.zarr"
        build_one_stack(stack_dir, out, T=5, plate_scale=0.187)
        z = zarr.open(str(out), mode="r")
        assert z.attrs["plate_scale"] == pytest.approx(0.187)

    def test_too_few_fits_raises(self, tmp_path):
        from src.data.build_one_stack import build_one_stack
        stack_dir = tmp_path / "raw"
        stack_dir.mkdir()
        write_fake_fits(stack_dir / "000000.fits", mjd=57130.0)
        with pytest.raises(ValueError, match="at least"):
            build_one_stack(stack_dir, tmp_path / "stack.zarr", T=5)
