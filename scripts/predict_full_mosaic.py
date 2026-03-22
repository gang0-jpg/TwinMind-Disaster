import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class SmallUNet(nn.Module):
    def __init__(self, in_channels=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dem", required=True)
    parser.add_argument("--slope", required=True)
    parser.add_argument("--rainfall_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--rainfall_pattern", default="*.npy")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dem = np.load(args.dem).astype(np.float32)
    slope = np.load(args.slope).astype(np.float32)

    rain_files = sorted(Path(args.rainfall_dir).glob(args.rainfall_pattern))
    if len(rain_files) != 6:
        raise ValueError(f"Expected 6 rainfall files, got {len(rain_files)}")

    rains = [np.load(p).astype(np.float32) for p in rain_files]

    ref_shape = dem.shape
    if slope.shape != ref_shape:
        raise ValueError("DEM and slope shape mismatch")
    for i, r in enumerate(rains):
        if r.shape != ref_shape:
            raise ValueError(f"rain[{i}] shape mismatch: {r.shape} vs {ref_shape}")

    dem_n = normalize_valid(dem)
    slope_n = normalize_valid(slope)
    rains_n = [normalize_valid(r) for r in rains]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SmallUNet(in_channels=8).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    h, w = dem.shape
    ps = args.patch_size
    st = args.stride

    pred_sum = np.zeros((h, w), dtype=np.float32)
    pred_count = np.zeros((h, w), dtype=np.float32)

    with torch.no_grad():
        for y in range(0, h - ps + 1, st):
            for x in range(0, w - ps + 1, st):
                dem_patch = dem_n[y:y+ps, x:x+ps]
                slope_patch = slope_n[y:y+ps, x:x+ps]
                rain_patches = [r[y:y+ps, x:x+ps] for r in rains_n]

                nan_ratio = np.isnan(dem_patch).mean()
                if nan_ratio > 0.40:
                    continue

                dem_patch = np.nan_to_num(dem_patch, nan=0.0)
                slope_patch = np.nan_to_num(slope_patch, nan=0.0)
                rain_patches = [np.nan_to_num(r, nan=0.0) for r in rain_patches]

                x_patch = np.stack([dem_patch, slope_patch] + rain_patches, axis=0)
                x_tensor = torch.from_numpy(x_patch[None, ...]).to(device)

                pred = model(x_tensor).cpu().numpy()[0, 0]

                pred_sum[y:y+ps, x:x+ps] += pred
                pred_count[y:y+ps, x:x+ps] += 1.0

    pred_full = np.divide(
        pred_sum,
        np.maximum(pred_count, 1e-6),
        out=np.zeros_like(pred_sum),
        where=pred_count > 0
    )

    # DEM が NaN の場所は予測も NaN に戻す
    pred_full[np.isnan(dem)] = np.nan

    np.save(output_dir / "pred_flood_full.npy", pred_full)

    plt.figure(figsize=(8, 8))
    plt.imshow(pred_full, origin="upper")
    plt.colorbar(label="Predicted Flood Depth")
    plt.title("Predicted Flood Depth - Full Mosaic")
    plt.tight_layout()
    plt.savefig(output_dir / "pred_flood_full.png", dpi=150)
    plt.close()

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(dem_n, origin="upper")
    plt.colorbar()
    plt.title("DEM Mosaic")

    plt.subplot(1, 2, 2)
    plt.imshow(pred_full, origin="upper")
    plt.colorbar()
    plt.title("Predicted Flood Depth Mosaic")

    plt.tight_layout()
    plt.savefig(output_dir / "dem_vs_pred_full.png", dpi=150)
    plt.close()

    print("[OK] saved:", output_dir / "pred_flood_full.npy")
    print("[OK] saved:", output_dir / "pred_flood_full.png")
    print("[OK] saved:", output_dir / "dem_vs_pred_full.png")
    print("pred range:", float(np.nanmin(pred_full)), float(np.nanmax(pred_full)))


if __name__ == "__main__":
    main()
