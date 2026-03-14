import numpy as np
from PIL import Image

IN_PATH = r"data/dem_npy_grid/dem_mosaic.npy"
OUT_PATH = r"data/dem_npy_grid/dem_for_training.npy"

TARGET = 64

dem = np.load(IN_PATH)

# NaN処理
mean_val = np.nanmean(dem)
dem = np.nan_to_num(dem, nan=mean_val)

# PIL resize
img = Image.fromarray(dem.astype(np.float32))
img = img.resize((TARGET, TARGET), resample=Image.BILINEAR)

dem2 = np.array(img, dtype=np.float32)

# 正規化
dem2 = (dem2 - dem2.mean()) / (dem2.std() + 1e-6)

np.save(OUT_PATH, dem2)

print("saved:", OUT_PATH)
print("shape:", dem2.shape)
print("min/max:", dem2.min(), dem2.max())
