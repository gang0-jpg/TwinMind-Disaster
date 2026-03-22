import os
import glob
import random
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mlflow
import yaml
from twinmind_disaster.model_unet import UNetSmall
from twinmind_disaster.model_unet_reservoir import ReservoirUNet


class CaseDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        d = np.load(path)

        dem = d["dem"].astype(np.float32)              # (H, W)
        rain = d["rain"].astype(np.float32)            # (6, H, W)
        flood = d["flood_gt"].astype(np.float32)       # (H, W)
        mask = d["mask"].astype(np.float32)            # (H, W)

        x = np.concatenate([dem[None, ...], rain], axis=0)   # (7, H, W)
        y = flood[None, ...]                                 # (1, H, W)
        mask = (mask > 0).astype(np.float32)[None, ...]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
        )


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, in_ch=7, out_ch=1, base=32):
        super().__init__()

        self.inc = DoubleConv(in_ch, base)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base, base * 2)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base * 2, base * 4)
        )

        self.up1 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(base * 4, base * 2)

        self.up2 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(base * 2, base)

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

        x = self.outc(x)
        return x


def masked_mse(pred, target, mask):
    diff = (pred - target) ** 2
    diff = diff * mask
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom


def train_one_epoch(model, loader, optimizer, device, scaler, use_amp):
    model.train()
    total = 0.0
    steps = 0

    for x, y, mask in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = model(x)
            loss = masked_mse(pred, y, mask)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total += loss.item()
        steps += 1

    return total / max(steps, 1)


@torch.no_grad()
def eval_one_epoch(model, loader, device, use_amp):
    model.eval()
    total = 0.0
    steps = 0

    for x, y, mask in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = model(x)
            loss = masked_mse(pred, y, mask)

        total += loss.item()
        steps += 1

    return total / max(steps, 1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--cases_dir", default="data/cases")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", default="runs/unet_cases_unet_best.pt")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--use_reservoir", action="store_true")
    parser.add_argument("--reservoir_dim", type=int, default=16)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--split_info_path", default="runs/train_val_split.json")
    return parser.parse_args()


def merge_config(args):
    if args.config is None:
        return args

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for key, value in cfg.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


def main():
    args = parse_args()
    args = merge_config(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs("runs", exist_ok=True)
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.split_info_path) or ".", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device = {device}")
    print(f"[info] amp = {args.amp}")

    files = sorted(glob.glob(os.path.join(args.cases_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No npz files found in {args.cases_dir}")

    random.shuffle(files)
    n_train = int(len(files) * args.train_ratio)
    train_files = files[:n_train]
    val_files = files[n_train:]

    print(f"[info] total={len(files)} train={len(train_files)} val={len(val_files)}")

    with open(args.split_info_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": args.seed,
                "train_ratio": args.train_ratio,
                "train_files": train_files,
                "val_files": val_files,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    train_ds = CaseDataset(train_files)
    val_ds = CaseDataset(val_files)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

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
        ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    mlflow.set_experiment("TwinMind_Disaster_Cases_UNet")

    best_val = float("inf")

    with mlflow.start_run():
        mlflow.log_params({
            "cases_dir": args.cases_dir,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "lr": args.lr,
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "input_channels": 7,
            "target": "flood_gt",
            "resolution": "256x256",
            "model": "UNetSmall",
            "base_channels": args.base_channels,
            "amp": args.amp,
            "save_path": args.save_path,
            "split_info_path": args.split_info_path,
        })

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, args.amp)
            val_loss = eval_one_epoch(model, val_loader, device, args.amp)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
            }, step=epoch)

            print(f"[epoch {epoch:03d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), args.save_path)
                print(f"[save] best model -> {args.save_path}")

    print("[done]")


if __name__ == "__main__":
    main()
