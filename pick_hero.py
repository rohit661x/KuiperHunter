import numpy as np
from pathlib import Path
import shutil

val_dir = Path("InjectionEngine/data/local_val_cases")
demo_dir = Path("InjectionEngine/demo/cases_demo")
demo_dir.mkdir(parents=True, exist_ok=True)

easy, hard = None, None

for p in val_dir.glob("case_*.npz"):
    data = np.load(p, allow_pickle=True)
    meta = data["meta"].item()
    fp = meta.get("flux_peak", 0)
    if easy is None and fp > 400:
        easy = p
    if hard is None and 100 < fp < 150:
        hard = p
    if easy and hard:
        break

if easy and hard:
    shutil.copy(easy, demo_dir / "easy_positive.npz")
    shutil.copy(hard, demo_dir / "hard_positive.npz")

    # For negative, just take 'easy', but zero out Y and replace X with patch_stack
    data = np.load(easy, allow_pickle=True)
    patch = data["patch_stack"]
    np.savez(
        demo_dir / "negative.npz",
        patch_stack=patch,
        X=patch,
        Y=np.zeros_like(patch),
        sigma_patch=data["sigma_patch"],
        timestamps=data["timestamps"],
        plate_scale=data["plate_scale"],
        psf_fwhm=data["psf_fwhm"],
        meta=data["meta"]
    )
    print(f"Easy: {easy.name} (flux {np.load(easy, allow_pickle=True)['meta'].item()['flux_peak']:.1f})")
    print(f"Hard: {hard.name} (flux {np.load(hard, allow_pickle=True)['meta'].item()['flux_peak']:.1f})")
    print("Negative generated (Y=0, X=patch_stack)")
else:
    print("Could not find easy/hard cases satisfying criteria!")
