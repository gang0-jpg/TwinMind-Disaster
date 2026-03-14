import numpy as np

IN_PATH = r"data/dem_npy_grid/dem_for_training.npy"
OUT_PATH = r"data/dem_npy_grid/slope_for_training.npy"

dem = np.load(IN_PATH).astype(np.float32)

# x方向, y方向の勾配
gy, gx = np.gradient(dem)

# 傾斜の強さ
slope = np.sqrt(gx**2 + gy**2).astype(np.float32)

# 正規化
slope = (slope - slope.mean()) / (slope.std() + 1e-6)

np.save(OUT_PATH, slope)

print("[saved]", OUT_PATH)
print("shape =", slope.shape)
print("min/max =", slope.min(), slope.max())
