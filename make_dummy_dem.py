import numpy as np
from pathlib import Path

Path("data/raw_dem").mkdir(parents=True, exist_ok=True)
Path("data/raw_mask").mkdir(parents=True, exist_ok=True)

h, w = 256, 256
y, x = np.mgrid[0:h, 0:w]

# 簡易DEM（斜面 + くぼみ）
dem = 50 + 0.03*x + 0.02*y - 8*np.exp(-((x-180)**2 + (y-140)**2)/(2*35**2))

mask = np.ones((h, w), dtype=np.uint8)

np.save("data/raw_dem/dem_tile_0001.npy", dem.astype(np.float32))
np.save("data/raw_mask/mask_tile_0001.npy", mask)

print("dummy DEM tile created")
