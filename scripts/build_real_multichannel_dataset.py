import argparse
import csv
import json
from pathlib import Path

import numpy as np


def normalize_valid(arr: np.ndarray):
    out = arr.astype(np.float32).copy()
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


def load_patch_index(csv_path: Path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "patch_id": int(row["patch_id"]),
                    "y": int(row["y"]),
                    "x": int(row["x"]),
                    "height": int(row["height"]),
                    "width": int(row["width"]),
                    "nan_ratio": float(row["nan_ratio"]),
                }
            )
    return rows


def slice_patch(arr: np.ndarray, y: int, x: int, h: int, w: int):
    return arr[y:y+h, x:x+w]


def build_split(
    split_name: str,
    indices: np.ndarray,
    patch_rows,
    dem_mosaic: np.ndarray,
    slope_mosaic: np.ndarray,
    rainfall_list,
    flood_mosaic: np.ndarray,
):
    x_list = []
    y_list = []
    meta_rows = []

    for idx in indices:
        row = patch_rows[int(idx)]

        y0 = row["y"]
        x0 = row["x"]
        h = row["height"]
        w = row["width"]

        dem_patch = slice_patch(dem_mosaic, y0, x0, h, w)
        slope_patch = slice_patch(slope_mosaic, y0, x0, h, w)
        rain_patches = [slice_patch(r, y0, x0, h, w) for r in rainfall_list]
        flood_patch = slice_patch(flood_mosaic, y0, x0, h, w)

        # NaN を 0 埋め
        dem_patch = np.nan_to_num(dem_patch, nan=0.0).astype(np.float32)
        slope_patch = np.nan_to_num(slope_patch, nan=0.0).astype(np.float32)
        rain_patches = [np.nan_to_num(r, nan=0.0).astype(np.float32) for r in rain_patches]
        flood_patch = np.nan_to_num(flood_patch, nan=0.0).astype(np.float32)

        # X = [DEM, slope, rainfall x 6] => 8 channels
        x_patch = np.stack([dem_patch, slope_patch] + rain_patches, axis=0).astype(np.float32)
        y_patch = flood_patch[None, :, :].astype(np.float32)

        x_list.append(x_patch)
        y_list.append(y_patch)

        meta_rows.append(
            {
                "patch_id": int(row["patch_id"]),
                "y": y0,
                "x": x0,
                "height": h,
                "width": w,
                "nan_ratio": float(row["nan_ratio"]),
            }
        )

    X = np.stack(x_list).astype(np.float32)
    Y = np.stack(y_list).astype(np.float32)

    print(f"[{split_name}] X shape:", X.shape, X.dtype, float(X.min()), float(X.max()))
    print(f"[{split_name}] Y shape:", Y.shape, Y.dtype, float(Y.min()), float(Y.max()))

    return X, Y, meta_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dem_mosaic", required=True)
    parser.add_argument("--slope_mosaic", required=True)
    parser.add_argument("--rainfall_dir", required=True)
    parser.add_argument("--flood_mosaic", required=True)
    parser.add_argument("--patch_index_csv", required=True)
    parser.add_argument("--train_idx", required=True)
    parser.add_argument("--val_idx", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--rainfall_pattern", default="*.npy")
    parser.add_argument("--expected_rainfall_steps", type=int, default=6)
    parser.add_argument("--normalize_rainfall_each", action="store_true")
    parser.add_argument("--normalize_flood", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    dem_mosaic = np.load(args.dem_mosaic).astype(np.float32)
    slope_mosaic = np.load(args.slope_mosaic).astype(np.float32)
    flood_mosaic = np.load(args.flood_mosaic).astype(np.float32)

    rainfall_files = sorted(Path(args.rainfall_dir).glob(args.rainfall_pattern))
    if len(rainfall_files) != args.expected_rainfall_steps:
        raise ValueError(
            f"Expected {args.expected_rainfall_steps} rainfall files, got {len(rainfall_files)}: "
            f"{[p.name for p in rainfall_files]}"
        )

    rainfall_list = [np.load(p).astype(np.float32) for p in rainfall_files]

    ref_shape = dem_mosaic.shape
    if slope_mosaic.shape != ref_shape:
        raise ValueError(f"slope shape mismatch: {slope_mosaic.shape} vs {ref_shape}")
    if flood_mosaic.shape != ref_shape:
        raise ValueError(f"flood shape mismatch: {flood_mosaic.shape} vs {ref_shape}")

    for i, r in enumerate(rainfall_list):
        if r.shape != ref_shape:
            raise ValueError(f"rainfall[{i}] shape mismatch: {r.shape} vs {ref_shape}")

    # 正規化
    dem_mosaic = normalize_valid(dem_mosaic)
    slope_mosaic = normalize_valid(slope_mosaic)

    if args.normalize_rainfall_each:
        rainfall_list = [normalize_valid(r) for r in rainfall_list]

    if args.normalize_flood:
        flood_mosaic = normalize_valid(flood_mosaic)

    patch_rows = load_patch_index(Path(args.patch_index_csv))
    train_idx = np.load(args.train_idx)
    val_idx = np.load(args.val_idx)

    X_train, Y_train, meta_train = build_split(
        "train",
        train_idx,
        patch_rows,
        dem_mosaic,
        slope_mosaic,
        rainfall_list,
        flood_mosaic,
    )

    X_val, Y_val, meta_val = build_split(
        "val",
        val_idx,
        patch_rows,
        dem_mosaic,
        slope_mosaic,
        rainfall_list,
        flood_mosaic,
    )

    np.save(outdir / "X_train.npy", X_train)
    np.save(outdir / "Y_train.npy", Y_train)
    np.save(outdir / "X_val.npy", X_val)
    np.save(outdir / "Y_val.npy", Y_val)

    with open(outdir / "meta_train.json", "w", encoding="utf-8") as f:
        json.dump(meta_train, f, ensure_ascii=False, indent=2)

    with open(outdir / "meta_val.json", "w", encoding="utf-8") as f:
        json.dump(meta_val, f, ensure_ascii=False, indent=2)

    with open(outdir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dem_mosaic": args.dem_mosaic,
                "slope_mosaic": args.slope_mosaic,
                "rainfall_files": [str(p) for p in rainfall_files],
                "flood_mosaic": args.flood_mosaic,
                "patch_index_csv": args.patch_index_csv,
                "train_idx": args.train_idx,
                "val_idx": args.val_idx,
                "output_dir": str(outdir),
                "ref_shape": list(ref_shape),
                "rainfall_steps": len(rainfall_files),
                "channels": 2 + len(rainfall_files),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("[OK] saved:")
    print(outdir / "X_train.npy")
    print(outdir / "Y_train.npy")
    print(outdir / "X_val.npy")
    print(outdir / "Y_val.npy")


if __name__ == "__main__":
    main()
