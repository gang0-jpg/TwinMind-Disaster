import numpy as np
from pathlib import Path

src = Path("/home/team-010/data/twinmind_disaster/datasets/dem_slope_patches_s16_split")
out = Path("/home/team-010/data/twinmind_disaster/datasets/dummy_multichannel_dataset")
out.mkdir(parents=True, exist_ok=True)

dem_train = np.load(src / "dem_train.npy").astype(np.float32)
slope_train = np.load(src / "slope_train.npy").astype(np.float32)
dem_val = np.load(src / "dem_val.npy").astype(np.float32)
slope_val = np.load(src / "slope_val.npy").astype(np.float32)

rng = np.random.default_rng(42)

def build_x_y(dem, slope):
    n, h, w = dem.shape

    rainfall = []
    for t in range(6):
        noise = rng.random((n, h, w), dtype=np.float32) * 0.2
        rain_t = (1.0 - dem) * (0.3 + 0.1 * t) + noise
        rain_t = np.clip(rain_t, 0.0, 1.0).astype(np.float32)
        rainfall.append(rain_t)

    x = np.stack([dem, slope] + rainfall, axis=1).astype(np.float32)

    y = (
        0.5 * (1.0 - dem)
        + 0.3 * rainfall[-1]
        + 0.2 * np.clip(1.0 - slope, 0.0, 1.0)
    )
    y = np.clip(y, 0.0, 1.0).astype(np.float32)
    y = y[:, None, :, :]

    return x, y

x_train, y_train = build_x_y(dem_train, slope_train)
x_val, y_val = build_x_y(dem_val, slope_val)

np.save(out / "X_train.npy", x_train)
np.save(out / "Y_train.npy", y_train)
np.save(out / "X_val.npy", x_val)
np.save(out / "Y_val.npy", y_val)

print("X_train:", x_train.shape, x_train.dtype, x_train.min(), x_train.max())
print("Y_train:", y_train.shape, y_train.dtype, y_train.min(), y_train.max())
print("X_val  :", x_val.shape, x_val.dtype, x_val.min(), x_val.max())
print("Y_val  :", y_val.shape, y_val.dtype, y_val.min(), y_val.max())
