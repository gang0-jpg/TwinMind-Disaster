import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class SimpleUNet(nn.Module):
    def __init__(self, in_ch=7, hidden=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, 1, 1)
        )

    def forward(self, x):
        return self.model(x)


def load_case(npz_path):
    d = np.load(npz_path)

    dem = d["dem"].astype(np.float32)          # (H, W)
    rain = d["rain"].astype(np.float32)        # (6, H, W)
    flood = d["flood_gt"].astype(np.float32)   # (H, W)
    mask = d["mask"].astype(np.float32)        # (H, W)

    x = np.concatenate([dem[None, ...], rain], axis=0)  # (7, H, W)
    return x, flood, mask


def save_preview(x, pred, target, mask, out_png):
    err = np.abs(pred - target) * (mask > 0)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    axes[0].imshow(x[0], cmap="terrain")
    axes[0].set_title("DEM")
    axes[0].axis("off")

    axes[1].imshow(x[1], cmap="Blues")
    axes[1].set_title("Rain t0")
    axes[1].axis("off")

    axes[2].imshow(target, cmap="viridis")
    axes[2].set_title("Ground Truth Flood")
    axes[2].axis("off")

    axes[3].imshow(pred, cmap="viridis")
    axes[3].set_title("Predicted Flood")
    axes[3].axis("off")

    axes[4].imshow(err, cmap="magma")
    axes[4].set_title("Absolute Error")
    axes[4].axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases_dir", default="data/cases")
    parser.add_argument("--model_path", default="runs/unet_cases_best.pt")
    parser.add_argument("--output_dir", default="outputs_cases")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--start_idx", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device = {device}")

    model = SimpleUNet(in_ch=7, hidden=32).to(device)
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    case_files = sorted(glob.glob(os.path.join(args.cases_dir, "*.npz")))
    selected = case_files[args.start_idx: args.start_idx + args.num_samples]

    for path in selected:
        x, target, mask = load_case(path)
        xt = torch.tensor(x[None, ...], dtype=torch.float32).to(device)

        with torch.no_grad():
            pred = model(xt).squeeze().detach().cpu().numpy()

        pred = pred * (mask > 0)

        base = os.path.splitext(os.path.basename(path))[0]
        out_png = os.path.join(args.output_dir, f"{base}_pred.png")
        save_preview(x, pred, target, mask, out_png)
        print(f"[save] {out_png}")

    print("[done]")


if __name__ == "__main__":
    main()
