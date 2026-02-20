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

    def test_dataset_reconstructs_x_from_patch_stack(self, tmp_path):
        """When X is absent, dataset returns patch_stack + Y."""
        T, H, W = 3, 32, 32
        patch_stack = np.ones((T, H, W), dtype=np.float32)
        Y = np.ones((T, H, W), dtype=np.float32) * 2.0
        np.savez(tmp_path / "case_0000.npz", Y=Y, patch_stack=patch_stack)

        from model.dataset import CaseDataset
        ds = CaseDataset(tmp_path, use_X_if_present=False)
        X_t, Y_t = ds[0]
        np.testing.assert_allclose(X_t.numpy(), patch_stack + Y, rtol=0)

    # ---- New tests for UNet3DMinimal and 3D dataset mode ----

    def test_dataset_3d_mode_shapes(self, tmp_path):
        """Dataset in 3d mode returns (1, T, H, W) tensors."""
        T, H, W = 5, 64, 64
        X = np.random.randn(T, H, W).astype(np.float32)
        Y = np.abs(np.random.randn(T, H, W)).astype(np.float32)
        np.savez(tmp_path / "case_0000.npz", X=X, Y=Y, patch_stack=X)

        from model.dataset import CaseDataset
        ds = CaseDataset(tmp_path, mode="3d")
        X_t, Y_t = ds[0]
        assert X_t.shape == (1, T, H, W), f"X shape {X_t.shape}"
        assert Y_t.shape == (1, T, H, W), f"Y shape {Y_t.shape}"
        assert X_t.dtype == torch.float32

    def test_dataset_normalization(self, tmp_path):
        """Per-case normalization produces near zero-mean, unit-std X."""
        T, H, W = 5, 32, 32
        X = np.random.randn(T, H, W).astype(np.float32) * 100 + 500
        Y = np.abs(np.random.randn(T, H, W)).astype(np.float32)
        np.savez(tmp_path / "case_0000.npz", X=X, Y=Y, patch_stack=X)

        from model.dataset import CaseDataset
        ds = CaseDataset(tmp_path, normalize="per_case")
        X_t, Y_t = ds[0]
        assert abs(X_t.mean().item()) < 0.01, f"mean={X_t.mean():.4f}"
        assert abs(X_t.std().item() - 1.0) < 0.01, f"std={X_t.std():.4f}"
        # Y should be untouched
        np.testing.assert_allclose(Y_t.numpy(), Y, rtol=1e-5)

    def test_unet3d_forward_shape(self):
        """UNet3DMinimal output shape matches input (B,1,T,H,W)."""
        from model.model import UNet3DMinimal
        model = UNet3DMinimal(base_channels=8)
        x = torch.zeros(1, 1, 5, 64, 64)
        out = model(x)
        assert out.shape == (1, 1, 5, 64, 64), f"Wrong shape: {out.shape}"

    def test_unet3d_no_nans(self):
        """UNet3DMinimal output has no NaNs."""
        from model.model import UNet3DMinimal
        model = UNet3DMinimal(base_channels=8)
        x = torch.randn(1, 1, 5, 32, 32)
        out = model(x)
        assert not torch.isnan(out).any(), "Output contains NaN values"

