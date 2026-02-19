# tests/test_targets.py
"""Unit tests for targets.py — target placement strategies."""
import numpy as np
import pytest

from src.injector.targets import draw_target, TargetConfig


class TestDrawTargetUniform:
    def test_within_margin_bounds(self):
        """Uniform strategy must keep (x, y) inside margin-clipped patch."""
        rng = np.random.default_rng(0)
        config = TargetConfig(strategy="uniform", margin=0.05)
        n_rows, n_cols = 64, 64
        margin_x = 0.05 * n_cols
        margin_y = 0.05 * n_rows
        for _ in range(200):
            x, y = draw_target((n_rows, n_cols), config, rng)
            assert margin_x <= x <= n_cols - margin_x, f"x={x} out of bounds"
            assert margin_y <= y <= n_rows - margin_y, f"y={y} out of bounds"

    def test_returns_floats(self):
        rng = np.random.default_rng(1)
        x, y = draw_target((64, 64), TargetConfig(strategy="uniform"), rng)
        assert isinstance(x, float)
        assert isinstance(y, float)


class TestDrawTargetFixed:
    def test_returns_exact_coordinates(self):
        rng = np.random.default_rng(0)
        config = TargetConfig(strategy="fixed", fixed_x=12.5, fixed_y=33.7)
        x, y = draw_target((64, 64), config, rng)
        assert x == 12.5
        assert y == 33.7

    def test_rng_not_consumed(self):
        """Fixed strategy should not consume RNG state (result is deterministic)."""
        rng = np.random.default_rng(99)
        state_before = rng.bit_generator.state
        draw_target((64, 64), TargetConfig(strategy="fixed", fixed_x=10.0, fixed_y=10.0), rng)
        state_after = rng.bit_generator.state
        assert state_before == state_after


class TestDrawTargetCenter:
    def test_mean_near_patch_center(self):
        """Center strategy mean should be near (n_cols/2, n_rows/2) over many draws."""
        rng = np.random.default_rng(42)
        config = TargetConfig(strategy="center", margin=0.05)
        xs, ys = [], []
        for _ in range(300):
            x, y = draw_target((64, 64), config, rng)
            xs.append(x)
            ys.append(y)
        assert abs(np.mean(xs) - 32.0) < 4.0, f"Mean x={np.mean(xs):.2f}, expected ~32"
        assert abs(np.mean(ys) - 32.0) < 4.0, f"Mean y={np.mean(ys):.2f}, expected ~32"

    def test_clipped_within_bounds(self):
        """Center strategy must never produce coordinates outside margin bounds."""
        rng = np.random.default_rng(7)
        config = TargetConfig(strategy="center", margin=0.05)
        margin = 0.05 * 64
        for _ in range(200):
            x, y = draw_target((64, 64), config, rng)
            assert margin <= x <= 64 - margin
            assert margin <= y <= 64 - margin


class TestDrawTargetGrid:
    def test_cell_zero_coords(self):
        """Grid cell 0 (row=0, col=0) → x in [0, cell_w), y in [0, cell_h)."""
        rng = np.random.default_rng(0)
        config = TargetConfig(strategy="grid", grid_n=4, margin=0.0)
        n_rows, n_cols = 64, 64
        cell_w = n_cols / 4  # 16
        cell_h = n_rows / 4  # 16
        for _ in range(50):
            x, y = draw_target((n_rows, n_cols), config, rng, grid_index=0)
            assert 0.0 <= x < cell_w + 1, f"x={x} outside cell 0 col range"
            assert 0.0 <= y < cell_h + 1, f"y={y} outside cell 0 row range"

    def test_grid_index_wraps(self):
        """grid_index wraps modulo grid_n^2, so index 16 == index 0 for grid_n=4."""
        rng1 = np.random.default_rng(5)
        rng2 = np.random.default_rng(5)
        config = TargetConfig(strategy="grid", grid_n=4, margin=0.0)
        x1, y1 = draw_target((64, 64), config, rng1, grid_index=0)
        x2, y2 = draw_target((64, 64), config, rng2, grid_index=16)
        assert x1 == x2 and y1 == y2


class TestDrawTargetEdgeCases:
    def test_unknown_strategy_raises(self):
        rng = np.random.default_rng(0)
        config = TargetConfig()
        config.strategy = "teleport"  # type: ignore
        with pytest.raises(ValueError, match="Unknown target strategy"):
            draw_target((64, 64), config, rng)
