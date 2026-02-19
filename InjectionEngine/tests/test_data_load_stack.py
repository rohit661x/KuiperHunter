# tests/test_data_load_stack.py
"""Tests for load_stack.py — uses synthetic in-memory FITS, no real files."""
from pathlib import Path
import numpy as np
import pytest
from astropy.io import fits


def _write_fake_fits(path: Path, mjd: float, shape=(64, 64), seed=0):
    """Write a minimal 4-HDU FITS matching kbmod format."""
    rng = np.random.default_rng(seed)
    sci = rng.normal(0, 5, shape).astype(np.float32)
    variance = np.zeros(shape, dtype=np.float32)
    mask = np.full(shape, 4, dtype=np.float32)

    primary = fits.PrimaryHDU()
    primary.header["MJD"] = mjd

    sci_hdu = fits.ImageHDU(data=sci)
    sci_hdu.header["WCSAXES"] = 2
    sci_hdu.header["CTYPE1"] = "RA---TAN-SIP"
    sci_hdu.header["CTYPE2"] = "DEC--TAN-SIP"
    sci_hdu.header["CRVAL1"] = 200.615
    sci_hdu.header["CRVAL2"] = -7.789
    sci_hdu.header["CRPIX1"] = 1033.9
    sci_hdu.header["CRPIX2"] = 2043.5
    sci_hdu.header["CD1_1"] = -1.14e-07
    sci_hdu.header["CD1_2"] = 7.318e-05
    sci_hdu.header["CD2_1"] = -7.301e-05
    sci_hdu.header["CD2_2"] = -1.28e-07

    hdul = fits.HDUList([primary, sci_hdu, fits.ImageHDU(variance), fits.ImageHDU(mask)])
    hdul.writeto(path, overwrite=True)


class TestLoadFitsStack:
    def test_returns_correct_shapes(self, tmp_path):
        paths = []
        for i in range(3):
            p = tmp_path / f"{i:06d}.fits"
            _write_fake_fits(p, mjd=57130.0 + i * 0.01)
            paths.append(p)

        from src.data.load_stack import load_fits_stack
        imgs, hdrs, mjds = load_fits_stack(paths)

        assert imgs.shape == (3, 64, 64)
        assert imgs.dtype == np.float32
        assert len(hdrs) == 3
        assert mjds.shape == (3,)
        assert mjds.dtype == np.float64

    def test_mjds_correct(self, tmp_path):
        paths = []
        expected_mjds = [57130.0, 57130.5, 57131.0]
        for i, mjd in enumerate(expected_mjds):
            p = tmp_path / f"{i:06d}.fits"
            _write_fake_fits(p, mjd=mjd)
            paths.append(p)

        from src.data.load_stack import load_fits_stack
        _, _, mjds = load_fits_stack(paths)

        np.testing.assert_allclose(mjds, expected_mjds)

    def test_hdr_contains_wcs_keys(self, tmp_path):
        p = tmp_path / "000000.fits"
        _write_fake_fits(p, mjd=57130.0)

        from src.data.load_stack import load_fits_stack
        _, hdrs, _ = load_fits_stack([p])

        assert "CRVAL1" in hdrs[0]
        assert "CD1_2" in hdrs[0]

    def test_science_data_is_hdu1(self, tmp_path):
        """HDU[1] data should be loaded, not HDU[2] or HDU[3]."""
        p = tmp_path / "000000.fits"
        rng = np.random.default_rng(0)
        sci = rng.normal(0, 5, (64, 64)).astype(np.float32)
        primary = fits.PrimaryHDU()
        primary.header["MJD"] = 57130.0
        sci_hdu = fits.ImageHDU(data=sci)
        junk = fits.ImageHDU(data=np.zeros((64, 64), dtype=np.float32))
        fits.HDUList([primary, sci_hdu, junk, junk]).writeto(p, overwrite=True)

        from src.data.load_stack import load_fits_stack
        imgs, _, _ = load_fits_stack([p])

        np.testing.assert_array_equal(imgs[0], sci)
