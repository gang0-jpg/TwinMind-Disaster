import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

base = Path("/home/team-010/data/twinmind_disaster/datasets/dem_slope_patches_s16")
dem = np.load(base / "dem_patches.npy")
slope = np.load(base / "slope_patches.npy")

sample_ids = [0, 10, 50, 100, 200, 400]

out_dir = base / "samples"
out_dir.mkdir(parents=True, exist_ok=True)

for i in sample_ids:
    if i >= len(dem):
        continue

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(dem[i], origin="upper")
    plt.colorbar()
    plt.title(f"DEM patch {i}")

    plt.subplot(1, 2, 2)
    plt.imshow(slope[i], origin="upper")
    plt.colorbar()
    plt.title(f"Slope patch {i}")

    plt.tight_layout()
    out_path = out_dir / f"patch_{i:04d}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    print("[OK]", out_path)
