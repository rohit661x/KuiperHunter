# tests/test_render_psf.py
"""Unit tests for render_psf.py — stamp shape, non-negativity, flux, centroid shift."""
import numpy as np
import pytest

from src.injector.render_psf import PSFParams, render_stamp, render_stack, _gaussian_kernel


class TestGaussianKernel:
    def test_normalized(self):
        """Gaussian kernel must sum to 1.0."""
        k = _gaussian_kernel(2.5, 13)
        assert abs(k.sum() - 1.0) < 1e-10

    def test_shape(self):
        k = _gaussian_kernel(2.5, 13)
        assert k.shape == (13, 13)

    def test_odd_size_required(self):
        """Kernel with even size still runs (no enforcement) but verify shape."""
        k = _gaussian_kernel(2.5, 12)
        assert k.shape == (12, 12)


class TestRenderStamp:
    def setup_method(self):
        self.psf = PSFParams(fwhm_pixels=2.5)

    def test_output_shape(self):
        stamp = render_stamp((64, 64), x=32.0, y=32.0, flux=100.0, psf_params=self.psf)
        assert stamp.shape == (64, 64)
        assert stamp.dtype == np.float64

    def test_nonnegative(self):
        stamp = render_stamp((64, 64), x=32.0, y=32.0, flux=100.0, psf_params=self.psf)
        assert np.all(stamp >= 0), "All stamp values must be >= 0"

    def test_flux_conservation(self):
        """Total flux in stamp must equal the requested flux (source fully inside)."""
        flux = 100.0
        stamp = render_stamp((64, 64), x=32.0, y=32.0, flux=flux, psf_params=self.psf)
        assert abs(stamp.sum() - flux) < 0.01, (
            f"Expected total flux ~{flux}, got {stamp.sum():.4f}"
        )

    def test_flux_conservation_subpixel(self):
        """Flux must be conserved to within 1.0 count even with sub-pixel shift applied."""
        flux = 100.0
        stamp = render_stamp((64, 64), x=32.5, y=32.3, flux=flux, psf_params=self.psf)
        assert abs(stamp.sum() - flux) < 1.0, (
            f"Expected total flux ~{flux}, got {stamp.sum():.4f}"
        )

    def test_centroid_shifts_right_when_x_increases(self):
        """Moving x from 28 → 36 must shift the centroid column to the right."""
        stamp_left = render_stamp((64, 64), x=28.0, y=32.0, flux=100.0, psf_params=self.psf)
        stamp_right = render_stamp((64, 64), x=36.0, y=32.0, flux=100.0, psf_params=self.psf)
        cols = np.arange(64, dtype=float)
        cx_left = (stamp_left * cols[np.newaxis, :]).sum() / stamp_left.sum()
        cx_right = (stamp_right * cols[np.newaxis, :]).sum() / stamp_right.sum()
        assert cx_right > cx_left, (
            f"Expected centroid to shift right, got cx_left={cx_left:.2f}, cx_right={cx_right:.2f}"
        )

    def test_centroid_shifts_down_when_y_increases(self):
        """Moving y from 24 → 40 must shift the centroid row downward."""
        stamp_up = render_stamp((64, 64), x=32.0, y=24.0, flux=100.0, psf_params=self.psf)
        stamp_down = render_stamp((64, 64), x=32.0, y=40.0, flux=100.0, psf_params=self.psf)
        rows = np.arange(64, dtype=float)
        cy_up = (stamp_up * rows[:, np.newaxis]).sum() / stamp_up.sum()
        cy_down = (stamp_down * rows[:, np.newaxis]).sum() / stamp_down.sum()
        assert cy_down > cy_up, (
            f"Expected centroid to shift down, got cy_up={cy_up:.2f}, cy_down={cy_down:.2f}"
        )

    def test_out_of_bounds_source_no_error(self):
        """Source fully outside the patch must produce a stamp without error."""
        stamp = render_stamp((64, 64), x=200.0, y=200.0, flux=100.0, psf_params=self.psf)
        assert stamp.shape == (64, 64)
        assert np.all(stamp == 0.0), "Fully OOB source must produce an all-zero stamp"


class TestRenderStack:
    def setup_method(self):
        self.psf = PSFParams(fwhm_pixels=2.5)

    def test_output_shape(self):
        xs = np.array([30.0, 31.0, 32.0, 33.0, 34.0])
        ys = np.full(5, 32.0)
        fluxes = np.full(5, 100.0)
        stack = render_stack((64, 64), xs, ys, fluxes, self.psf)
        assert stack.shape == (5, 64, 64)
        assert stack.dtype == np.float64

    def test_nonnegative(self):
        xs = np.full(5, 32.0)
        ys = np.full(5, 32.0)
        fluxes = np.full(5, 50.0)
        stack = render_stack((64, 64), xs, ys, fluxes, self.psf)
        assert np.all(stack >= 0)

    def test_per_frame_position(self):
        """Each frame should have its peak near the specified (x, y)."""
        xs = np.array([16.0, 32.0, 48.0])
        ys = np.full(3, 32.0)
        fluxes = np.full(3, 100.0)
        stack = render_stack((64, 64), xs, ys, fluxes, self.psf)
        for i, x_expected in enumerate(xs):
            frame = stack[i]
            col_peak = int(np.unravel_index(frame.argmax(), frame.shape)[1])
            assert abs(col_peak - x_expected) <= 2, (
                f"Frame {i}: peak col={col_peak}, expected near x={x_expected}"
            )
