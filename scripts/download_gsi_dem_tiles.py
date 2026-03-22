import io
import math
import os
from pathlib import Path

import numpy as np
import requests
from PIL import Image
import matplotlib.pyplot as plt

AREAS = {
    "osaka": {
        "lat_min": 34.60,
        "lat_max": 34.73,
        "lon_min": 135.43,
        "lon_max": 135.58,
    },
    "kyoto": {
        "lat_min": 34.96,
        "lat_max": 35.08,
        "lon_min": 135.70,
        "lon_max": 135.84,
    },
    "shizuoka": {
        "lat_min": 34.93,
        "lat_max": 35.03,
        "lon_min": 138.34,
        "lon_max": 138.48,
    },
}

ZOOM = 14
LAYERS = [
    ("dem1a_png", "DEM1A"),
    ("dem5a_png", "DEM5A"),
    ("dem5b_png", "DEM5B"),
    ("dem5c_png", "DEM5C"),
    ("dem_png",   "DEM10B"),
]

RAW_DIR = Path("data/raw_dem")
PREVIEW_DIR = Path("data/raw_dem_preview")


def lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[int, int]:
    lat_rad = math.radians(lat)
    n = 2 ** z
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int(
        (1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    )
    return xtile, ytile


def bbox_to_tiles(lat_min, lat_max, lon_min, lon_max, z):
    x0, y0 = lonlat_to_tile(lon_min, lat_max, z)  # 左上
    x1, y1 = lonlat_to_tile(lon_max, lat_min, z)  # 右下

    xs = range(min(x0, x1), max(x0, x1) + 1)
    ys = range(min(y0, y1), max(y0, y1) + 1)
    return [(x, y) for x in xs for y in ys]


def decode_elevation_png(content: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(content)).convert("RGB")
    rgb = np.asarray(img).astype(np.int32)

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    x = r * 256 * 256 + g * 256 + b
    elev = np.where(x < 2**23, x, x - 2**24).astype(np.float32) * 0.01

    nodata = (r == 128) & (g == 0) & (b == 0)
    elev[nodata] = np.nan
    return elev


def fetch_tile(z: int, x: int, y: int):
    for layer_path, layer_name in LAYERS:
        url = f"https://cyberjapandata.gsi.go.jp/xyz/{layer_path}/{z}/{x}/{y}.png"
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            continue
        try:
            arr = decode_elevation_png(r.content)
            if np.isfinite(arr).any():
                return arr, layer_name, url
        except Exception:
            continue
    return None, None, None


def save_preview(arr: np.ndarray, out_path: Path, title: str):
    plt.figure(figsize=(5, 5))
    plt.imshow(arr, cmap="terrain")
    plt.colorbar(label="Elevation (m)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main():

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    tile_id = 1

    for area_name, bbox in AREAS.items():

        tiles = bbox_to_tiles(
            bbox["lat_min"], bbox["lat_max"],
            bbox["lon_min"], bbox["lon_max"],
            ZOOM
        )

        print(f"[area] {area_name}: {len(tiles)} candidate tiles")

        for x, y in tiles:

            arr, layer, url = fetch_tile(ZOOM, x, y)

            if arr is None:
                print(f"[skip] {area_name} z={ZOOM} x={x} y={y}: no DEM tile")
                continue

            npy_name = f"dem_tile_{tile_id:04d}_{area_name}_z{ZOOM}_x{x}_y{y}_{layer}.npy"
            png_name = f"dem_tile_{tile_id:04d}_{area_name}_z{ZOOM}_x{x}_y{y}_{layer}.png"

            np.save(RAW_DIR / npy_name, arr.astype(np.float32))
            save_preview(arr, PREVIEW_DIR / png_name, f"{area_name} {layer} z{ZOOM} x{x} y{y}")

            print(
                f"[save] {npy_name} shape={arr.shape} "
                f"min={np.nanmin(arr):.2f} max={np.nanmax(arr):.2f} url={url}"
            )

            tile_id += 1


if __name__ == "__main__":
    main()

