import numpy as np
from pathlib import Path

base = Path("/home/team-010/data/twinmind_disaster/datasets/dem_slope_patches_s16")
out = Path("/home/team-010/data/twinmind_disaster/datasets/dem_slope_patches_s16_split")
out.mkdir(parents=True, exist_ok=True)

dem = np.load(base / "dem_patches.npy")
slope = np.load(base / "slope_patches.npy")

n = len(dem)
idx = np.arange(n)

rng = np.random.default_rng(42)
rng.shuffle(idx)

n_train = int(n * 0.8)

train_idx = idx[:n_train]
val_idx = idx[n_train:]

np.save(out / "dem_train.npy", dem[train_idx])
np.save(out / "slope_train.npy", slope[train_idx])
np.save(out / "dem_val.npy", dem[val_idx])
np.save(out / "slope_val.npy", slope[val_idx])
np.save(out / "train_idx.npy", train_idx)
np.save(out / "val_idx.npy", val_idx)

print("train:", len(train_idx))
print("val  :", len(val_idx))
