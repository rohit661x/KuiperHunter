"""
test_step4_smoke.py — Unit smoke tests for Step 4 model modules.

Tests run on CPU only. No training required.
"""
import numpy as np
import pytest
import torch


class TestStep4Smoke:

    def test_dataset_returns_correct_shapes(self, tmp_path):
        """Dataset loads a case and returns (T,H,W) float32 tensors."""
        # Write one minimal fake .npz
        T, H, W = 5, 64, 64
        X = np.random.randn(T, H, W).astype(np.float32)
        Y = np.abs(np.random.randn(T, H, W)).astype(np.float32)
        patch_stack = np.random.randn(T, H, W).astype(np.float32)
        np.savez(tmp_path / "case_0000.npz", X=X, Y=Y, patch_stack=patch_stack)

        from model.dataset import CaseDataset
        ds = CaseDataset(tmp_path)
        X_t, Y_t = ds[0]
        assert X_t.shape == (T, H, W), f"X shape {X_t.shape}"
        assert Y_t.shape == (T, H, W), f"Y shape {Y_t.shape}"
        assert X_t.dtype == torch.float32
        assert Y_t.dtype == torch.float32

    def test_model_forward_shape(self):
        """Baseline2DNet output shape matches input shape."""
        from model.model import Baseline2DNet
        model = Baseline2DNet(n_frames=5)
        x = torch.zeros(2, 5, 64, 64)
        out = model(x)
        assert out.shape == (2, 5, 64, 64), f"Wrong shape: {out.shape}"

    def test_model_output_nonnegative(self):
        """Model output is non-negative (final ReLU)."""
        from model.model import Baseline2DNet
        model = Baseline2DNet(n_frames=5)
        x = torch.randn(1, 5, 32, 32)  # intentionally negative values
        out = model(x)
        assert (out >= 0).all(), f"Output has {(out < 0).sum()} negative values"
