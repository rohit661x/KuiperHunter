# Step 4: 2D Baseline ML Pipeline Implementation Plan

**Date:** 2026-02-19
**Goal:** Implement the end-to-end ML training and inference pipeline for the KBO injection engine using a 2D baseline (frames-as-channels) model. Must be verifiably correct on CPU before GPU use.

## Context

- Repo root: `KuiperHunter/`
- Package installed editable: `pip install -e InjectionEngine/`
- Installed package names: `data`, `injector`, `model` (new, under `src/model/`)
- Training data: `InjectionEngine/data/smoke_cases/` — 50 × `.npz` files
- Each `.npz`: `X (5,64,64) float32`, `Y (5,64,64) float32`, `patch_stack (5,64,64) float32`
- Model convention: input `(B, T, H, W)` where T=5 is treated as 2D channels

## Package Layout

```
InjectionEngine/src/model/
    __init__.py
    dataset.py
    model.py
    train.py
    infer.py
InjectionEngine/checkpoints/       (created at runtime)
InjectionEngine/tests/test_step4_smoke.py
InjectionEngine/docs/verification/step4_local.txt
```

## pyproject.toml additions (after implementing)

```toml
[project.scripts]
# existing entries...
kuiper-train   = "model.train:main"
kuiper-infer   = "model.infer:main"
```

---

## Task 1: `src/model/__init__.py` + `src/model/dataset.py`

**Files:**
- Create: `InjectionEngine/src/model/__init__.py`
- Create: `InjectionEngine/src/model/dataset.py`

### Spec

`CaseDataset(case_dir, use_X_if_present=True)` is a `torch.utils.data.Dataset` that:
- Globs `case_*.npz` in `case_dir`, sorted
- `__len__` returns number of files
- `__getitem__(i)` returns `(X_tensor, Y_tensor)`:
  - `X_tensor`: float32 torch tensor, shape `(T, H, W)`
  - `Y_tensor`: float32 torch tensor, shape `(T, H, W)`
  - If `use_X_if_present=True` and `'X'` key present in npz: load X directly
  - Else: reconstruct `X = patch_stack + Y`
  - Both tensors are `torch.float32`

```python
# test_step4_smoke.py will verify:
ds = CaseDataset("InjectionEngine/data/smoke_cases")
X, Y = ds[0]
assert X.shape == (5, 64, 64)
assert Y.shape == (5, 64, 64)
assert X.dtype == torch.float32
assert Y.dtype == torch.float32
```

**`__init__.py`:** export `CaseDataset`, `Baseline2DNet`

---

## Task 2: `src/model/model.py`

**Files:**
- Create: `InjectionEngine/src/model/model.py`

### Spec

`Baseline2DNet(n_frames: int = 5)`:
- `nn.Module`
- `__init__`: build 3-layer conv stack
  - `Conv2d(n_frames, 32, kernel_size=3, padding=1)`
  - `ReLU`
  - `Conv2d(32, 32, kernel_size=3, padding=1)`
  - `ReLU`
  - `Conv2d(32, n_frames, kernel_size=3, padding=1)`
  - `ReLU` (clamp non-negative, consistent with Y >= 0)
- `forward(x)`:
  - Input: `(B, T, H, W)`
  - Pass through sequential stack
  - Output: `(B, T, H, W)`

```python
# test_step4_smoke.py will verify:
model = Baseline2DNet(n_frames=5)
x = torch.zeros(2, 5, 64, 64)
out = model(x)
assert out.shape == (2, 5, 64, 64)
```

---

## Task 3: `src/model/train.py`

**Files:**
- Create: `InjectionEngine/src/model/train.py`

### Spec

CLI entry point `main()` with argparse:

| Arg | Default | Description |
|-----|---------|-------------|
| `--train_dir` | required | Path to dir of case_*.npz |
| `--val_dir` | None | Optional val dir |
| `--epochs` | 2 | Training epochs |
| `--batch` | 4 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--out_ckpt` | required | Path to save best.pt |

Algorithm:
1. Build `CaseDataset(train_dir)`, `DataLoader(shuffle=True, batch_size=batch)`
2. If `val_dir` provided: build val `CaseDataset` + `DataLoader(shuffle=False)`
3. Build `Baseline2DNet(n_frames=T)` where T is inferred from first batch
4. `Adam(lr=lr)`, `MSELoss()`
5. Per epoch:
   - Train loop: zero_grad → forward → loss → backward → step
   - Print `Epoch {e}/{epochs}  train_loss={avg:.6f}`
   - If val: eval loop (no grad), print val_loss
6. Track best loss (val if available, else train); save `best.pt` on improvement
7. After training: save `{stem}_log.json` alongside `out_ckpt` with `{"epochs": N, "final_train_loss": f, "final_val_loss": f_or_null}`
8. Checkpoint format: `torch.save({"model_state": model.state_dict(), "n_frames": T, "epoch": e, "loss": best_loss}, out_ckpt)`

Console_scripts entry: `kuiper-train = "model.train:main"`

---

## Task 4: `src/model/infer.py`

**Files:**
- Create: `InjectionEngine/src/model/infer.py`

### Spec

CLI entry point `main()` with argparse:

| Arg | Description |
|-----|-------------|
| `--ckpt` | Path to .pt checkpoint |
| `--case` | Path to .npz case file |
| `--out_png` | Path to save output PNG |

Algorithm:
1. Load checkpoint: `ckpt = torch.load(path, map_location="cpu")`
2. Reconstruct model: `Baseline2DNet(n_frames=ckpt["n_frames"])`, load state dict
3. Load case: `np.load(case_path, allow_pickle=True)`; build X, Y tensors `(1, T, H, W)` float32
4. `model.eval(); with torch.no_grad(): Y_hat = model(X_tensor)`
5. Print:
   ```
   MSE(Y, Y_hat) = {value:.6f}
   max(Y)        = {value:.4f}
   max(Y_hat)    = {value:.4f}
   ```
6. Plot 3 panels for `t = T // 2`:
   - Panel 1: `X[0, t]` — title "X (input, frame t)"
   - Panel 2: `Y[0, t]` — title "Y (target, frame t)"
   - Panel 3: `Y_hat[0, t]` — title "Ŷ (predicted, frame t)"
7. `plt.savefig(out_png, dpi=120, bbox_inches="tight")`
8. Print `Saved → {out_png}`

Console_scripts entry: `kuiper-infer = "model.infer:main"`

---

## Task 5: Smoke test + verification gate

**Files:**
- Create: `InjectionEngine/tests/test_step4_smoke.py`
- Create: `InjectionEngine/docs/verification/step4_local.txt`
- Create: `InjectionEngine/demo/step4_pred_case0000.png`
- Create: `InjectionEngine/checkpoints/baseline2d_best.pt`

### test_step4_smoke.py spec

Three tests (no GPU required, no actual training):

```python
class TestStep4Smoke:
    def test_dataset_returns_correct_shapes(self, tmp_path):
        # Write one minimal fake .npz
        # Load via CaseDataset
        # Assert X.shape == (5,64,64), Y.shape == (5,64,64), dtype float32

    def test_model_forward_shape(self):
        # Baseline2DNet(n_frames=5)
        # x = torch.zeros(2, 5, 64, 64)
        # assert model(x).shape == (2, 5, 64, 64)

    def test_model_output_nonnegative(self):
        # Baseline2DNet(n_frames=5)
        # x = torch.randn(1, 5, 32, 32)
        # assert (model(x) >= 0).all()
```

### Verification gate commands

Run in this order from repo root (`KuiperHunter/`), capture output:

```bash
# 1. Smoke train (2 epochs, batch 4)
python -m model.train \
  --train_dir InjectionEngine/data/smoke_cases \
  --epochs 2 --batch 4 \
  --out_ckpt InjectionEngine/checkpoints/baseline2d_best.pt

# 2. Inference demo
python -m model.infer \
  --ckpt InjectionEngine/checkpoints/baseline2d_best.pt \
  --case InjectionEngine/data/smoke_cases/case_0000.npz \
  --out_png InjectionEngine/demo/step4_pred_case0000.png
```

Capture all stdout/stderr to `InjectionEngine/docs/verification/step4_local.txt`.

Assert artifacts exist:
- `InjectionEngine/checkpoints/baseline2d_best.pt`
- `InjectionEngine/demo/step4_pred_case0000.png`

---

## pyproject.toml additions

After all modules work, add to `[project.scripts]`:

```toml
kuiper-train = "model.train:main"
kuiper-infer = "model.infer:main"
```

Then `pip install -e InjectionEngine/` to register.
