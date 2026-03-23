import argparse
from pathlib import Path
import numpy as np


def fill_nan_nearest_simple(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    mask = np.isnan(out)
    if not mask.any():
        return out

    # 簡易補間: 前後左右に1回ずつ伝播
    for _ in range(8):
        if not np.isnan(out).any():
            break

        up = np.roll(out, 1, axis=0)
        down = np.roll(out, -1, axis=0)
        left = np.roll(out, 1, axis=1)
        right = np.roll(out, -1, axis=1)

        for neigh in (up, down, left, right):
            fill_mask = np.isnan(out) & ~np.isnan(neigh)
            out[fill_mask] = neigh[fill_mask]

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dem", required=True)
    parser.add_argument("--output_slope", required=True)
    args = parser.parse_args()

    dem = np.load(args.input_dem).astype(np.float32)

    # gradient前に簡易補間
    dem_filled = fill_nan_nearest_simple(dem)

    gy, gx = np.gradient(dem_filled)
    slope = np.sqrt(gx**2 + gy**2).astype(np.float32)

    # 元の NaN 領域は slope も NaN に戻す
    slope[np.isnan(dem)] = np.nan

    out = Path(args.output_slope)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, slope)

    print("[OK] slope saved:", out)
    print("shape:", slope.shape)
    print("nan count:", int(np.isnan(slope).sum()))
    print("min/max:", float(np.nanmin(slope)), float(np.nanmax(slope)))


if __name__ == "__main__":
    main()
