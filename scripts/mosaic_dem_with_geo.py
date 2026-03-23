import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_npy", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    npy_files = sorted(input_dir.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No npy files found in {input_dir}")

    items = []
    for npy_path in npy_files:
        json_path = npy_path.with_suffix(".json")
        if not json_path.exists():
            raise FileNotFoundError(f"Missing metadata: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        arr = np.load(npy_path)
        lower = meta["lower_corner"]   # [lat, lon]
        upper = meta["upper_corner"]   # [lat, lon]

        items.append({
            "name": npy_path.name,
            "npy_path": str(npy_path),
            "json_path": str(json_path),
            "arr": arr,
            "lower_lat": float(lower[0]),
            "lower_lon": float(lower[1]),
            "upper_lat": float(upper[0]),
            "upper_lon": float(upper[1]),
            "shape": list(arr.shape),
            "nan_count": int(np.isnan(arr).sum()),
        })

    # 列は西→東で lower_lon 昇順
    unique_lons = sorted({round(x["lower_lon"], 12) for x in items})
    lon_to_col = {lon: i for i, lon in enumerate(unique_lons)}

    # 行は北→南で upper_lat 降順
    unique_upper_lats = sorted({round(x["upper_lat"], 12) for x in items}, reverse=True)
    lat_to_row = {lat: i for i, lat in enumerate(unique_upper_lats)}

    for item in items:
        item["grid_row"] = lat_to_row[round(item["upper_lat"], 12)]
        item["grid_col"] = lon_to_col[round(item["lower_lon"], 12)]

    tile_h, tile_w = items[0]["arr"].shape
    nrows = len(unique_upper_lats)
    ncols = len(unique_lons)

    blank = np.full((tile_h, tile_w), np.nan, dtype=np.float32)

    grid = [[blank.copy() for _ in range(ncols)] for _ in range(nrows)]
    placement = []

    for item in items:
        r = item["grid_row"]
        c = item["grid_col"]
        grid[r][c] = item["arr"]
        placement.append({
            "name": item["name"],
            "grid_row": r,
            "grid_col": c,
            "lower_corner": [item["lower_lat"], item["lower_lon"]],
            "upper_corner": [item["upper_lat"], item["upper_lon"]],
            "shape": item["shape"],
            "nan_count": item["nan_count"],
        })

    row_blocks = [np.concatenate(row_tiles, axis=1) for row_tiles in grid]
    mosaic = np.concatenate(row_blocks, axis=0)

    out_npy = Path(args.output_npy)
    out_json = Path(args.output_json)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    np.save(out_npy, mosaic)

    meta = {
        "tile_shape": [tile_h, tile_w],
        "grid_rows": nrows,
        "grid_cols": ncols,
        "mosaic_shape": list(mosaic.shape),
        "nan_count": int(np.isnan(mosaic).sum()),
        "min": None if np.all(np.isnan(mosaic)) else float(np.nanmin(mosaic)),
        "max": None if np.all(np.isnan(mosaic)) else float(np.nanmax(mosaic)),
        "lon_values_west_to_east": unique_lons,
        "upper_lat_values_north_to_south": unique_upper_lats,
        "placement": sorted(placement, key=lambda x: (x["grid_row"], x["grid_col"])),
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[OK] mosaic saved:", out_npy)
    print("grid:", nrows, "x", ncols)
    print("shape:", mosaic.shape)
    print("nan count:", meta["nan_count"])
    print("min/max:", meta["min"], meta["max"])


if __name__ == "__main__":
    main()
