import glob
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

RAW_DIR = Path("data/raw_dem")
OUT_DIR = Path("data/raw_slope")
PREVIEW_DIR = Path("data/raw_slope_preview")

def compute_slope(dem: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(dem.astype(np.float32))
    slope = np.sqrt(gx**2 + gy**2)
    if np.isfinite(slope).any():
        smin = np.nanmin(slope)
        smax = np.nanmax(slope)
        if smax > smin:
            slope = (slope - smin) / (smax - smin)
    return slope.astype(np.float32)

def save_preview(arr, out_png, title):
    plt.figure(figsize=(5,5))
    plt.imshow(arr, cmap="magma")
    plt.colorbar(label="Normalized slope")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob("data/raw_dem/*.npy"))
    print(f"[info] found {len(files)} DEM tiles")

    for path in files:
        dem = np.load(path).astype(np.float32)
        slope = compute_slope(dem)

        name = os.path.basename(path).replace("dem_tile", "slope_tile")
        np.save(OUT_DIR / name, slope)

        png_name = name.replace(".npy", ".png")
        save_preview(slope, PREVIEW_DIR / png_name, png_name)

        print(f"[save] {name} shape={slope.shape} min={float(np.nanmin(slope)):.4f} max={float(np.nanmax(slope)):.4f}")

    print("[done]")

if __name__ == "__main__":
    main()
