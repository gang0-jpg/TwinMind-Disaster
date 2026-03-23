from pathlib import Path
import numpy as np
import cv2
import re

RAW_DEM_GSI_DIR = Path("data/raw_dem_gsi")
RAW_SLOPE_GSI_DIR = Path("data/raw_slope_gsi")
INFERENCE_CASE_DIR = Path("data/inference_cases")

def fill_nan_nearest_simple(arr):
    out = arr.copy()
    mask = np.isnan(out)
    if not mask.any():
        return out

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

    return np.nan_to_num(out, nan=0.0)

def resize_2d(arr, out_hw=(256,256)):
    return cv2.resize(arr.astype(np.float32), out_hw[::-1])

def make_rain(out_hw=(256,256), steps=6):
    h,w = out_hw
    return np.ones((steps,h,w), dtype=np.float16)

def safe_name(f):
    stem = Path(f).stem
    return f"case_{re.sub(r'[^A-Za-z0-9_-]+','_',stem)}.npz"

def build_case_from_gsi(gsi_filename):
    INFERENCE_CASE_DIR.mkdir(parents=True, exist_ok=True)

    dem = np.load(RAW_DEM_GSI_DIR / gsi_filename)
    dem = fill_nan_nearest_simple(dem)

    mask = np.isfinite(dem).astype(np.uint8)

    dem = resize_2d(dem)
    mask = resize_2d(mask)
    mask = (mask>0.5).astype(np.uint8)

    rain = make_rain()
    flood_gt = np.zeros_like(dem, dtype=np.float16)

    case_name = safe_name(gsi_filename)

    np.savez_compressed(
        INFERENCE_CASE_DIR / case_name,
        dem=dem.astype(np.float32),
        rain=rain,
        flood_gt=flood_gt,
        mask=mask,
        meta=np.array("GSI case"),
    )

    return case_name
