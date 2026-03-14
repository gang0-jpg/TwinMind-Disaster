import os
import glob
import re
import math
import numpy as np

IN_DIR = r"data/dem_npy_grid"
OUT_PATH = r"data/dem_npy_grid/dem_mosaic.npy"


def tile_key(path):
    name = os.path.basename(path)
    m = re.search(r"FG-GML-(\d+)-(\d+)-(\d+)-DEM5A", name)
    if not m:
        raise ValueError(f"cannot parse tile name: {name}")
    return int(m.group(2)), int(m.group(3))  # row, col


files = sorted(glob.glob(os.path.join(IN_DIR, "*.npy")))
files = [f for f in files if not f.endswith("dem_mosaic.npy")]

tiles = {}
rows_all = []
cols_all = []

for f in files:
    r, c = tile_key(f)
    arr = np.load(f)
    tiles[(r, c)] = arr
    rows_all.append(r)
    cols_all.append(c)

rmin, rmax = min(rows_all), max(rows_all)
cmin, cmax = min(cols_all), max(cols_all)

tile_h, tile_w = next(iter(tiles.values())).shape

canvas = np.full(
    ((rmax - rmin + 1) * tile_h, (cmax - cmin + 1) * tile_w),
    np.nan,
    dtype=np.float32
)

for (r, c), arr in tiles.items():
    y0 = (r - rmin) * tile_h
    x0 = (c - cmin) * tile_w
    canvas[y0:y0 + tile_h, x0:x0 + tile_w] = arr

np.save(OUT_PATH, canvas)

valid = canvas[np.isfinite(canvas)]
print("[saved]", OUT_PATH)
print("shape =", canvas.shape)
print("valid =", valid.size)
print("min/max =", np.nanmin(valid), np.nanmax(valid))
