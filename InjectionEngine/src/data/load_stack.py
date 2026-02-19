"""
load_stack.py – Read a list of FITS files (kbmod 4-HDU format).

HDU layout expected:
  HDU[0]: PrimaryHDU  — MJD in header['MJD']
  HDU[1]: ImageHDU    — science image (float32) with TAN-SIP WCS
  HDU[2]: ImageHDU    — variance plane (may be zeros)
  HDU[3]: ImageHDU    — mask plane

Returns imgs[t], hdrs[t], mjds[t].
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from astropy.io import fits


def load_fits_stack(
    paths: list[str | Path],
) -> tuple[np.ndarray, list[dict], np.ndarray]:
    """
    Read FITS files and return stacked science images, headers, and MJDs.

    Parameters
    ----------
    paths : list of str or Path
        Ordered list of FITS file paths.  Each must follow the kbmod
        4-HDU convention (PrimaryHDU + science + variance + mask).

    Returns
    -------
    imgs   : float32 array, shape (T, H, W)
    hdrs   : list of T dicts — HDU[1] header key/value pairs (contains WCS)
    mjds   : float64 array, shape (T,)  — MJD timestamps from HDU[0].header
    """
    imgs_list: list[np.ndarray] = []
    hdrs_list: list[dict] = []
    mjds_list: list[float] = []

    for p in paths:
        with fits.open(p) as hdul:
            imgs_list.append(hdul[1].data.astype(np.float32))
            hdrs_list.append(dict(hdul[1].header))
            mjds_list.append(float(hdul[0].header["MJD"]))

    return (
        np.stack(imgs_list, axis=0),
        hdrs_list,
        np.array(mjds_list, dtype=np.float64),
    )
