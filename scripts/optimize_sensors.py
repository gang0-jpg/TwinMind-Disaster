import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt

from twinmind_disaster.dataset_loader import FloodNPZDataset
from twinmind_disaster.model_unet import UNetSmall


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases_dir", type=str, default="data/cases")
    parser.add_argument("--model_glob", type=str, default="runs/unet_seed*_best.pt")
    parser.add_argument("--output_dir", type=str, default="runs/sensor_optimization")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--min_dist", type=int, default=20)
    return parser.parse_args()


def load_models(model_paths, device):
    models = []
    for path in model_paths:
        ckpt = torch.load(path, map_location=device)
        model = UNetSmall(in_channels=7, out_channels=1, base_ch=32).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models.append((Path(path).name, model))
    return models


def select_topk_points(
    score_map: np.ndarray,
    mask: np.ndarray,
    top_k: int,
    min_dist: int,
) -> list[tuple[int, int, float]]:
    """
    score_map: (H, W) 高いほど選びたい
    mask:      (H, W) 1=有効
    return: [(y, x, score), ...]
    """
    h, w = score_map.shape
    valid = mask.astype(bool)

    # 無効領域は -inf
    score = score_map.copy()
    score[~valid] = -np.inf

    flat_indices = np.argsort(score.ravel())[::-1]  # descending

    selected: list[tuple[int, int, float]] = []

    for idx in flat_indices:
        y = idx // w
        x = idx % w
        val = score[y, x]

        if not np.isfinite(val):
            continue

        too_close = False
        for sy, sx, _ in selected:
            dist = np.sqrt((y - sy) ** 2 + (x - sx) ** 2)
            if dist < min_dist:
                too_close = True
                break

        if too_close:
            continue

        selected.append((y, x, float(val)))

        if len(selected) >= top_k:
            break

    return selected


def save_sensor_figure(
    output_path: Path,
    dem: np.ndarray,
    gt: np.ndarray,
    pred_mean: np.ndarray,
    unc_map: np.ndarray,
    sensor_points: list[tuple[int, int, float]],
    case_name: str,
):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    im0 = axes[0].imshow(dem, cmap="terrain")
    axes[0].set_title("DEM")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(gt, cmap="viridis")
    axes[1].set_title("GT Flood")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(pred_mean, cmap="viridis")
    axes[2].set_title("Pred Mean")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    im3 = axes[3].imshow(unc_map, cmap="magma")
    axes[3].set_title("Uncertainty + Sensors")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    for (y, x, _) in sensor_points:
        axes[3].plot(x, y, "co", markersize=7, markeredgecolor="black")
        axes[3].text(x + 2, y + 2, "S", fontsize=9, color="white")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"{case_name} | sensors={len(sensor_points)}")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=140)
    plt.close(fig)


def save_sensor_txt(
    output_path: Path,
    sensor_points: list[tuple[int, int, float]],
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("sensor_id,y,x,uncertainty_score\n")
        for i, (y, x, score) in enumerate(sensor_points, start=1):
            f.write(f"S{i},{y},{x},{score:.6f}\n")


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device = {device}")

    model_paths = sorted(glob.glob(args.model_glob))
    if not model_paths:
        raise RuntimeError(f"No model files matched: {args.model_glob}")

    print("[info] models:")
    for p in model_paths:
        print("  ", p)

    models = load_models(model_paths, device)
    dataset = FloodNPZDataset(args.cases_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    end_idx = min(args.start_idx + args.num_samples, len(dataset))

    with torch.no_grad():
        for idx in range(args.start_idx, end_idx):
            sample = dataset[idx]

            x = sample["x"].unsqueeze(0).to(device)      # (1, C, H, W)
            y = sample["y"].squeeze(0).cpu().numpy()     # (H, W)
            mask = sample["mask"].squeeze(0).cpu().numpy()
            case_path = Path(sample["path"])

            preds = []
            for _, model in models:
                pred = model(x).squeeze(0).squeeze(0).cpu().numpy()
                pred = pred * mask
                preds.append(pred)

            preds = np.stack(preds, axis=0)  # (M, H, W)
            pred_mean = preds.mean(axis=0)
            pred_std = preds.std(axis=0) * mask

            sensor_points = select_topk_points(
                score_map=pred_std,
                mask=mask,
                top_k=args.top_k,
                min_dist=args.min_dist,
            )

            x_np = sample["x"].cpu().numpy()
            dem = x_np[0]

            fig_path = output_dir / f"{case_path.stem}_sensors.png"
            txt_path = output_dir / f"{case_path.stem}_sensors.csv"

            save_sensor_figure(
                output_path=fig_path,
                dem=dem,
                gt=y,
                pred_mean=pred_mean,
                unc_map=pred_std,
                sensor_points=sensor_points,
                case_name=case_path.stem,
            )

            save_sensor_txt(txt_path, sensor_points)

            unc_mean = float((pred_std * mask).sum() / np.clip(mask.sum(), 1, None))
            print(
                f"[OK] {case_path.stem} -> {fig_path} | "
                f"uncertainty_mean={unc_mean:.6f} sensors={len(sensor_points)}"
            )

    print("[done]")


if __name__ == "__main__":
    main()
