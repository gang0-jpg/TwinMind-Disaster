import os
import glob
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mlflow


class CaseDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        d = np.load(path)

        dem = d["dem"].astype(np.float32)
        rain = d["rain"].astype(np.float32)
        flood = d["flood_gt"].astype(np.float32)
        mask = d["mask"].astype(np.float32)

        x = np.concatenate([dem[None, ...], rain], axis=0)   # (7, H, W)
        y = flood[None, ...]                                 # (1, H, W)
        mask = (mask > 0).astype(np.float32)[None, ...]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
        )


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


def masked_mse(pred, target, mask):
    diff = (pred - target) ** 2
    diff = diff * mask
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    steps = 0

    for x, y, mask in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad()
        pred = model(x)
        loss = masked_mse(pred, y, mask)
        loss.backward()
        optimizer.step()

        total += loss.item()
        steps += 1

    return total / max(steps, 1)


@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    total = 0.0
    steps = 0

    for x, y, mask in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        pred = model(x)
        loss = masked_mse(pred, y, mask)

        total += loss.item()
        steps += 1

    return total / max(steps, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases_dir", default="data/cases")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", default="runs/unet_cases_best.pt")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs("runs", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device = {device}")

    files = sorted(glob.glob(os.path.join(args.cases_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No npz files found in {args.cases_dir}")

    random.shuffle(files)
    n_train = int(len(files) * args.train_ratio)
    train_files = files[:n_train]
    val_files = files[n_train:]

    print(f"[info] total={len(files)} train={len(train_files)} val={len(val_files)}")

    train_ds = CaseDataset(train_files)
    val_ds = CaseDataset(val_files)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = SimpleUNet(in_ch=7, hidden=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    mlflow.set_experiment("TwinMind_Disaster_Cases")

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
        })

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            val_loss = eval_one_epoch(model, val_loader, device)

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
