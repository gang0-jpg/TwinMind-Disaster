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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--x", required=True)
    parser.add_argument("--y", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    X = np.load(args.x).astype(np.float32)
    Y = np.load(args.y).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SmallUNet(in_channels=8).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    n = min(args.num_samples, len(X))

    with torch.no_grad():
        for i in range(n):
            x = torch.from_numpy(X[i:i+1]).to(device)
            pred = model(x).cpu().numpy()[0, 0]
            target = Y[i, 0]
            dem = X[i, 0]
            slope = X[i, 1]
            rain_last = X[i, 7]

            fig = plt.figure(figsize=(12, 8))

            ax1 = fig.add_subplot(2, 3, 1)
            im1 = ax1.imshow(dem, origin="upper")
            ax1.set_title("DEM")
            plt.colorbar(im1, ax=ax1, fraction=0.046)

            ax2 = fig.add_subplot(2, 3, 2)
            im2 = ax2.imshow(slope, origin="upper")
            ax2.set_title("Slope")
            plt.colorbar(im2, ax=ax2, fraction=0.046)

            ax3 = fig.add_subplot(2, 3, 3)
            im3 = ax3.imshow(rain_last, origin="upper")
            ax3.set_title("Rainfall t5")
            plt.colorbar(im3, ax=ax3, fraction=0.046)

            ax4 = fig.add_subplot(2, 3, 4)
            im4 = ax4.imshow(target, origin="upper")
            ax4.set_title("Target Flood Depth")
            plt.colorbar(im4, ax=ax4, fraction=0.046)

            ax5 = fig.add_subplot(2, 3, 5)
            im5 = ax5.imshow(pred, origin="upper")
            ax5.set_title("Predicted Flood Depth")
            plt.colorbar(im5, ax=ax5, fraction=0.046)

            ax6 = fig.add_subplot(2, 3, 6)
            err = np.abs(pred - target)
            im6 = ax6.imshow(err, origin="upper")
            ax6.set_title("Absolute Error")
            plt.colorbar(im6, ax=ax6, fraction=0.046)

            plt.tight_layout()
            out_path = outdir / f"infer_sample_{i:03d}.png"
            plt.savefig(out_path, dpi=150)
            plt.close(fig)
            print("[OK]", out_path)


if __name__ == "__main__":
    main()
