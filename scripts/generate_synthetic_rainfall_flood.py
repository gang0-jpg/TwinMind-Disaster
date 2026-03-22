import numpy as np
from pathlib import Path

dem_path = "/home/team-010/data/twinmind_disaster/processed_dem/mosaic/dem_mosaic.npy"

rain_dir = Path("/home/team-010/data/twinmind_disaster/rainfall_real")
flood_dir = Path("/home/team-010/data/twinmind_disaster/flood_real")

rain_dir.mkdir(parents=True, exist_ok=True)
flood_dir.mkdir(parents=True, exist_ok=True)

dem = np.load(dem_path)

h, w = dem.shape

print("DEM shape:", dem.shape)

# -------------------------
# rainfall生成
# -------------------------

rainfalls = []

for t in range(6):

    base = np.random.gamma(2.0, 5.0, size=(h, w))

    # 台風中心を作る
    cy = np.random.randint(h)
    cx = np.random.randint(w)

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    dist = np.sqrt((yy-cy)**2 + (xx-cx)**2)

    storm = np.exp(-(dist/200)**2) * 80

    rain = base + storm

    rain = rain.astype(np.float32)

    rainfalls.append(rain)

    np.save(rain_dir / f"rainfall_t{t}.npy", rain)

    print("saved rainfall_t", t)

# -------------------------
# flood depth生成
# -------------------------

rain_total = sum(rainfalls)

# 標高低いほど浸水
dem_norm = (dem - np.nanmin(dem)) / (np.nanmax(dem)-np.nanmin(dem))

flood = rain_total * (1 - dem_norm)

# 平滑化
from scipy.ndimage import gaussian_filter

flood = gaussian_filter(flood, sigma=5)

flood = flood.astype(np.float32)

np.save(flood_dir / "flood_depth_mosaic.npy", flood)

print("saved flood_depth_mosaic")
print("flood range:", flood.min(), flood.max())
