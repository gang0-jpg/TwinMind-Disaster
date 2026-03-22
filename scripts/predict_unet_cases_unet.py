import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import yaml

from twinmind_disaster.model_unet_reservoir import ReservoirUNet


def enable_mc_dropout(model: nn.Module) -> None:
    """
    model 全体は eval のまま、Dropout 層だけ train にする
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()


@torch.no_grad()
def mc_dropout_predict(model, x: torch.Tensor, mc_runs: int = 16):
    """
    x: [B, C, H, W]
    returns:
        mean_map: [B, 1, H, W]
        std_map : [B, 1, H, W]
        preds    : [T, B, 1, H, W]
    """
    enable_mc_dropout(model)

    preds = []
    for _ in range(mc_runs):
        y = model(x)
        preds.append(y)

    preds = torch.stack(preds, dim=0)
    mean_map = preds.mean(dim=0)
    std_map = preds.std(dim=0, unbiased=False)
    return mean_map, std_map, preds

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, p_drop: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.drop = nn.Dropout2d(p=p_drop) if p_drop > 0 else nn.Identity()

    def forward(self, x):
        x = self.net(x)
        x = self.drop(x)
        return x


class UNetSmall(nn.Module):
    """
    旧 checkpoint 互換の 2-down / 2-up U-Net
    key 名を合わせるため、層名は旧版を維持
    """
    def __init__(self, in_ch=7, out_ch=1, base=32, p_drop: float = 0.1):
        super().__init__()
        self.inc = DoubleConv(in_ch, base, p_drop=0.0)

        # checkpoint 互換のため nn.Sequential の形を維持
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base, base * 2, p_drop=0.0)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base * 2, base * 4, p_drop=p_drop)
        )

        self.up1 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(base * 4, base * 2, p_drop=p_drop)

        self.up2 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(base * 2, base, p_drop=0.0)

        self.outc = nn.Conv2d(base, out_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x = self.up1(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv2(x)

        return self.outc(x)

def load_case(npz_path):
    d = np.load(npz_path)
    dem = d["dem"].astype(np.float32)
    rain = d["rain"].astype(np.float32)
    flood = d["flood_gt"].astype(np.float32)
    mask = d["mask"].astype(np.float32)

    x = np.concatenate([dem[None, ...], rain], axis=0)
    return x, flood, mask


def save_preview(x, pred, target, mask, out_png):
    valid = (mask > 0)
    pred = pred * valid
    target = target * valid
    err = np.abs(pred - target)

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
    parser.add_argument("--config", default=None)
    parser.add_argument("--cases_dir", default="data/cases")
    parser.add_argument("--model_path", default="runs/unet_cases_unet_best.pt")
    parser.add_argument("--output_dir", default="outputs_cases_unet")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--use_reservoir", action="store_true")
    parser.add_argument("--reservoir_dim", type=int, default=16)
    parser.add_argument("--mc_runs", type=int, default=12)

    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            setattr(args, k, v)

    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device = {device}")

    # ★ここが重要（分岐）
    if args.use_reservoir:
        model = ReservoirUNet(
            reservoir_dim=args.reservoir_dim,
            base_channels=args.base_channels,
            rain_timesteps=5,
        ).to(device)
    else:
        model = UNetSmall(
            in_ch=7,
            out_ch=1,
            base=args.base_channels,
            p_drop=0.1,
        ).to(device)

    ckpt = torch.load(args.model_path, map_location=device, weights_only=True)

    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print("[info] missing_keys   =", missing)
    print("[info] unexpected_keys=", unexpected)

    print("[info] use_reservoir =", args.use_reservoir)
    print("[info] model class   =", model.__class__.__name__)
    print("[info] model_path    =", args.model_path)

    model.eval()

    case_files = sorted(glob.glob(os.path.join(args.cases_dir, "*.npz")))
    selected = case_files[args.start_idx: args.start_idx + args.num_samples]

    for path in selected:
        x, target, mask = load_case(path)
        xt = torch.tensor(x[None, ...], dtype=torch.float32).to(device)

        mean_pred, std_pred, _ = mc_dropout_predict(
            model,
            xt,
            mc_runs=args.mc_runs,
        )

        pred = mean_pred.squeeze().detach().cpu().numpy()
        pred_std = std_pred.squeeze().detach().cpu().numpy()

        print("[debug] file =", os.path.basename(path))
        print("[debug] pred mean =", float(pred.mean()), "pred std =", float(pred.std()))
        print("[debug] unc  mean =", float(pred_std.mean()), "unc  std =", float(pred_std.std()))

        base = os.path.splitext(os.path.basename(path))[0]

        out_png = os.path.join(args.output_dir, f"{base}_pred.png")
        save_preview(x, pred, target, mask, out_png)
        print(f"[save] {out_png}")

        out_mean_npy = os.path.join(args.output_dir, f"{base}_pred_mean.npy")
        np.save(out_mean_npy, pred)
        print(f"[save] {out_mean_npy}")

        out_std_npy = os.path.join(args.output_dir, f"{base}_pred_std.npy")
        np.save(out_std_npy, pred_std)
        print(f"[save] {out_std_npy}")

    print("[done]")


if __name__ == "__main__":
    main()
