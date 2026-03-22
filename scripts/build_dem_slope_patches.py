import argparse
import csv
from pathlib import Path

import numpy as np


def normalize_valid(arr: np.ndarray):
    out = arr.copy().astype(np.float32)
    mask = ~np.isnan(out)
    if mask.sum() == 0:
        return out

    vmin = np.nanmin(out)
    vmax = np.nanmax(out)
    if vmax > vmin:
        out[mask] = (out[mask] - vmin) / (vmax - vmin)
    else:
        out[mask] = 0.0
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dem", required=True)
    parser.add_argument("--slope", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--max_nan_ratio", type=float, default=0.30)
    args = parser.parse_args()

    dem = np.load(args.dem).astype(np.float32)
    slope = np.load(args.slope).astype(np.float32)

    if dem.shape != slope.shape:
        raise ValueError(f"Shape mismatch: dem={dem.shape}, slope={slope.shape}")

    patch_size = args.patch_size
    stride = args.stride
    max_nan_ratio = args.max_nan_ratio

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    dem_norm = normalize_valid(dem)
    slope_norm = normalize_valid(slope)

    dem_patches = []
    slope_patches = []
    rows = []

    h, w = dem.shape
    total = 0
    kept = 0

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            total += 1

            dem_patch = dem_norm[y:y+patch_size, x:x+patch_size]
            slope_patch = slope_norm[y:y+patch_size, x:x+patch_size]

            valid_mask = ~np.isnan(dem_patch)
            nan_ratio = 1.0 - (valid_mask.sum() / valid_mask.size)

            if nan_ratio > max_nan_ratio:
                continue

            # 学習用に NaN を 0 埋め
            dem_patch = np.nan_to_num(dem_patch, nan=0.0)
            slope_patch = np.nan_to_num(slope_patch, nan=0.0)

            dem_patches.append(dem_patch)
            slope_patches.append(slope_patch)

            rows.append([kept, y, x, patch_size, patch_size, float(nan_ratio)])
            kept += 1

    if kept == 0:
        raise RuntimeError("No valid patches kept. Relax max_nan_ratio or inspect DEM.")

    dem_patches = np.stack(dem_patches).astype(np.float32)
    slope_patches = np.stack(slope_patches).astype(np.float32)

    np.save(outdir / "dem_patches.npy", dem_patches)
    np.save(outdir / "slope_patches.npy", slope_patches)

    with open(outdir / "patch_index.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["patch_id", "y", "x", "height", "width", "nan_ratio"])
        writer.writerows(rows)

    print("[OK] patches saved:", outdir)
    print("total windows:", total)
    print("kept patches :", kept)
    print("dem_patches  :", dem_patches.shape, dem_patches.dtype)
    print("slope_patches:", slope_patches.shape, slope_patches.dtype)
    print("max_nan_ratio:", max_nan_ratio)


if __name__ == "__main__":
    main()
