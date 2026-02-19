# tests/conftest.py
"""Shared pytest fixtures and helpers for src/data tests."""
from pathlib import Path
import numpy as np
from astropy.io import fits


def write_fake_fits(path: Path, mjd: float, shape=(64, 64), seed=0):
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

    fits.HDUList(
        [primary, sci_hdu, fits.ImageHDU(variance), fits.ImageHDU(mask)]
    ).writeto(path, overwrite=True)
