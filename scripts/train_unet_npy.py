import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class NpyFloodDataset(Dataset):
    def __init__(self, x_path: str, y_path: str):
        self.x = np.load(x_path).astype(np.float32)
        self.y = np.load(y_path).astype(np.float32)

        if len(self.x) != len(self.y):
            raise ValueError(f"X/Y size mismatch: {self.x.shape} vs {self.y.shape}")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.x[idx])
        y = torch.from_numpy(self.y[idx])
        return x, y


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


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item() * x.size(0)
            count += x.size(0)

    return total_loss / max(count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_train", required=True)
    parser.add_argument("--y_train", required=True)
    parser.add_argument("--x_val", required=True)
    parser.add_argument("--y_val", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[info] device =", device)

    train_ds = NpyFloodDataset(args.x_train, args.y_train)
    val_ds = NpyFloodDataset(args.x_val, args.y_val)

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

    model = SmallUNet(in_channels=8).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        count = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            count += x.size(0)

        train_loss = running_loss / max(count, 1)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"[epoch {epoch:03d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val,
                },
                save_path,
            )
            print(f"[saved] {save_path} (best_val={best_val:.6f})")

    print("[done] best_val_loss =", best_val)


if __name__ == "__main__":
    main()
