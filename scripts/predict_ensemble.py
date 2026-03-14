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
    parser.add_argument("--output_dir", type=str, default="runs/ensemble_predictions")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--start_idx", type=int, default=0)
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


def save_uncertainty_figure(
    output_path: Path,
    dem: np.ndarray,
    rain0: np.ndarray,
    gt: np.ndarray,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    err: np.ndarray,
    case_name: str,
):
    fig, axes = plt.subplots(1, 6, figsize=(24, 4))

    im0 = axes[0].imshow(dem, cmap="terrain")
    axes[0].set_title("DEM")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(rain0, cmap="Blues")
    axes[1].set_title("Rain t0")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(gt, cmap="viridis")
    axes[2].set_title("GT Flood")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    im3 = axes[3].imshow(pred_mean, cmap="viridis")
    axes[3].set_title("Pred Mean")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    im4 = axes[4].imshow(pred_std, cmap="magma")
    axes[4].set_title("Uncertainty (Std)")
    plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)

    im5 = axes[5].imshow(err, cmap="magma")
    axes[5].set_title("Abs Error")
    plt.colorbar(im5, ax=axes[5], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(case_name)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=140)
    plt.close(fig)


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
            for model_name, model in models:
                pred = model(x).squeeze(0).squeeze(0).cpu().numpy()
                pred = pred * mask
                preds.append(pred)

            preds = np.stack(preds, axis=0)  # (M, H, W)
            pred_mean = preds.mean(axis=0)
            pred_var = preds.var(axis=0)
            pred_std = np.sqrt(pred_var)

            err = np.abs(pred_mean - y) * mask

            x_np = sample["x"].cpu().numpy()
            dem = x_np[0]
            rain0 = x_np[1]

            out_path = output_dir / f"{case_path.stem}_ensemble.png"
            save_uncertainty_figure(
                output_path=out_path,
                dem=dem,
                rain0=rain0,
                gt=y,
                pred_mean=pred_mean,
                pred_std=pred_std,
                err=err,
                case_name=case_path.stem,
            )

            mae = float(err.sum() / np.clip(mask.sum(), 1, None))
            unc_mean = float((pred_std * mask).sum() / np.clip(mask.sum(), 1, None))
            print(
                f"[OK] {case_path.stem} -> {out_path} | "
                f"masked_MAE={mae:.6f} uncertainty_mean={unc_mean:.6f}"
            )

    print("[done]")


if __name__ == "__main__":
    main()
