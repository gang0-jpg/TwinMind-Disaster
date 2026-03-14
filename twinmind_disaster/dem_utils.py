from pathlib import Path
import numpy as np

from twinmind_disaster.config import Config


def tile_id_to_filename(prefix: str, tile_id: int) -> str:
    return f"{prefix}_tile_{tile_id:04d}.npy"


def load_dem_tile(tile_id: int, config: Config) -> np.ndarray:
    path = config.raw_dem_dir / tile_id_to_filename("dem", tile_id)
    if not path.exists():
        raise FileNotFoundError(f"DEM tile not found: {path}")
    dem = np.load(path)
    return dem.astype(np.float32)


def load_mask_tile(tile_id: int, config: Config) -> np.ndarray:
    path = config.raw_mask_dir / tile_id_to_filename("mask", tile_id)
    if not path.exists():
        raise FileNotFoundError(f"Mask tile not found: {path}")
    mask = np.load(path)
    return mask.astype(np.uint8)


def validate_dem_and_mask(dem: np.ndarray, mask: np.ndarray, config: Config) -> None:
    expected_shape = (config.grid_height, config.grid_width)

    if dem.shape != expected_shape:
        raise ValueError(f"DEM shape mismatch: {dem.shape} != {expected_shape}")

    if mask.shape != expected_shape:
        raise ValueError(f"Mask shape mismatch: {mask.shape} != {expected_shape}")

    if dem.dtype not in (np.float32, np.float64):
        raise TypeError(f"DEM dtype must be float32/float64, got {dem.dtype}")

    if mask.dtype != np.uint8:
        raise TypeError(f"Mask dtype must be uint8, got {mask.dtype}")

    unique_vals = np.unique(mask)
    if not np.all(np.isin(unique_vals, [0, 1])):
        raise ValueError(f"Mask must contain only 0/1, got {unique_vals}")
