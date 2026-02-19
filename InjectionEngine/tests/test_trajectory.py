# tests/test_trajectory.py
"""Unit tests for trajectory.py — displacement math and direction conventions."""
import numpy as np
from src.injector.trajectory import build_trajectory, is_in_patch, Trajectory


class TestBuildTrajectory:
    def test_displacement_correct(self):
        """motion_ra=1 arcsec/hr, plate_scale=0.5 arcsec/px, dt=[0,1,2]hr → dx=[0,2,4]px."""
        timestamps = np.array([0.0, 1.0, 2.0])
        traj = build_trajectory(timestamps, start_x=10.0, start_y=10.0,
                                motion_ra=1.0, motion_dec=0.0, plate_scale=0.5)
        np.testing.assert_allclose(traj.xs, [10.0, 12.0, 14.0], atol=1e-10)
        np.testing.assert_allclose(traj.ys, [10.0, 10.0, 10.0], atol=1e-10)

    def test_pure_ra_shifts_x_only(self):
        """Pure RA motion must change xs and leave ys constant."""
        timestamps = np.array([0.0, 1.0, 2.0])
        traj = build_trajectory(timestamps, start_x=16.0, start_y=16.0,
                                motion_ra=2.0, motion_dec=0.0, plate_scale=1.0)
        np.testing.assert_allclose(traj.xs, [16.0, 18.0, 20.0], atol=1e-10)
        np.testing.assert_allclose(traj.ys, 16.0, atol=1e-10)

    def test_pure_dec_shifts_y_only(self):
        """Pure Dec motion must change ys and leave xs constant.
        Dec increases upward → pixel y decreases (ys = start_y - dec/plate * dt).
        """
        timestamps = np.array([0.0, 1.0, 2.0])
        traj = build_trajectory(timestamps, start_x=16.0, start_y=16.0,
                                motion_ra=0.0, motion_dec=2.0, plate_scale=1.0)
        np.testing.assert_allclose(traj.xs, 16.0, atol=1e-10)
        # ys = 16 - (2/1)*dt = 16, 14, 12
        np.testing.assert_allclose(traj.ys, [16.0, 14.0, 12.0], atol=1e-10)

    def test_zero_motion_constant_positions(self):
        """Zero motion must produce identical position every frame."""
        timestamps = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        traj = build_trajectory(timestamps, start_x=8.0, start_y=8.0,
                                motion_ra=0.0, motion_dec=0.0, plate_scale=0.263)
        np.testing.assert_allclose(traj.xs, 8.0, atol=1e-10)
        np.testing.assert_allclose(traj.ys, 8.0, atol=1e-10)

    def test_output_shape_and_dtype(self):
        """Output xs/ys must be float64 arrays of length n_frames with no NaN."""
        timestamps = np.array([0.0, 1.0, 2.0, 3.0])
        traj = build_trajectory(timestamps, start_x=5.0, start_y=5.0,
                                motion_ra=1.0, motion_dec=1.0, plate_scale=0.5)
        assert traj.xs.shape == (4,)
        assert traj.ys.shape == (4,)
        assert traj.xs.dtype == np.float64
        assert traj.ys.dtype == np.float64
        assert not np.any(np.isnan(traj.xs))
        assert not np.any(np.isnan(traj.ys))

    def test_as_array_shape(self):
        """as_array() must return (n_frames, 2) with xs in col 0, ys in col 1."""
        timestamps = np.array([0.0, 1.0])
        traj = build_trajectory(timestamps, start_x=5.0, start_y=3.0,
                                motion_ra=0.0, motion_dec=0.0, plate_scale=1.0)
        arr = traj.as_array()
        assert arr.shape == (2, 2)
        np.testing.assert_allclose(arr[:, 0], traj.xs)
        np.testing.assert_allclose(arr[:, 1], traj.ys)


class TestIsInPatch:
    def test_all_inside(self):
        """Static source in centre of 64×64 patch must always be in-patch."""
        timestamps = np.array([0.0, 1.0, 2.0])
        traj = build_trajectory(timestamps, start_x=32.0, start_y=32.0,
                                motion_ra=0.0, motion_dec=0.0, plate_scale=1.0)
        mask = is_in_patch(traj, patch_shape=(64, 64))
        assert np.all(mask)

    def test_exits_patch(self):
        """Source moving east at 10 px/hr must exit a 64-wide patch by frame 2."""
        # xs = 60 + 10*dt: frame0=60 (in), frame1=70 (outside, 70 >= 64.5), frame2=80 (out)
        timestamps = np.array([0.0, 1.0, 2.0])
        traj = build_trajectory(timestamps, start_x=60.0, start_y=32.0,
                                motion_ra=10.0, motion_dec=0.0, plate_scale=1.0)
        mask = is_in_patch(traj, patch_shape=(64, 64))
        assert mask[0], "frame 0 (x=60) should be inside"
        assert not mask[1], "frame 1 (x=70) should be outside (70 >= 64.5)"
        assert not mask[2], "frame 2 (x=80) should be outside"
