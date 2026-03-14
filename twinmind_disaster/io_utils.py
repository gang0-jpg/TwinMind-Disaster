import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_case_npz(
    output_path: Path,
    dem: np.ndarray,
    rain: np.ndarray,
    flood_gt: np.ndarray,
    mask: np.ndarray,
    meta: dict,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        dem=dem.astype(np.float32),
        rain=rain.astype(np.float16),
        flood_gt=flood_gt.astype(np.float16),
        mask=mask.astype(np.uint8),
        meta=json.dumps(meta, ensure_ascii=False),
    )


def save_case_preview(
    preview_dir: Path,
    case_id: int,
    dem: np.ndarray,
    rain: np.ndarray,
    flood_gt: np.ndarray,
) -> None:
    preview_dir.mkdir(parents=True, exist_ok=True)

    cid = f"{case_id:06d}"

    plt.figure(figsize=(5, 4))
    plt.imshow(dem, cmap="terrain")
    plt.colorbar()
    plt.title(f"DEM case_{cid}")
    plt.tight_layout()
    plt.savefig(preview_dir / f"case_{cid}_dem.png", dpi=120)
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.imshow(rain[0], cmap="Blues")
    plt.colorbar()
    plt.title(f"Rain t0 case_{cid}")
    plt.tight_layout()
    plt.savefig(preview_dir / f"case_{cid}_rain_t0.png", dpi=120)
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.imshow(flood_gt, cmap="viridis")
    plt.colorbar()
    plt.title(f"Flood case_{cid}")
    plt.tight_layout()
    plt.savefig(preview_dir / f"case_{cid}_flood.png", dpi=120)
    plt.close()
