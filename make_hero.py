import numpy as np
import shutil
from pathlib import Path

val_dir = Path("InjectionEngine/data/local_val_cases")
demo_dir = Path("InjectionEngine/demo/cases_demo")
demo_dir.mkdir(parents=True, exist_ok=True)

easy = val_dir / "case_0000.npz"
hard = val_dir / "case_0001.npz"
neg_src = val_dir / "case_0002.npz"

shutil.copy(easy, demo_dir / "easy_positive.npz")
shutil.copy(hard, demo_dir / "hard_positive.npz")

data = np.load(neg_src, allow_pickle=True)
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
print("Hero cases created successfully.")
