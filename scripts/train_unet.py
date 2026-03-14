import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow

# Git警告を抑制
os.environ["GIT_PYTHON_REFRESH"] = "quiet"


# ==========================
# Dummy U-Net（例）
# ※既存の UNet がある場合は置き換えてください
# ==========================
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1)
        )

    def forward(self, x):
        return self.model(x)

# ==========================
# ダミーデータ + DEM + slope
# ==========================
def get_dummy_loader(batch_size=2, steps=10):
    import numpy as np
    import torch

    dem = np.load("data/dem_npy_grid/dem_for_training.npy").astype(np.float32)
    slope = np.load("data/dem_npy_grid/slope_for_training.npy").astype(np.float32)

    for _ in range(steps):
        x_list = []
        y_list = []

        for _b in range(batch_size):
            sensor_map = np.random.randn(64, 64).astype(np.float32)

            # 3ch: [sensor, dem, slope]
            x = np.stack([sensor_map, dem, slope], axis=0)

            y = np.random.randn(1, 64, 64).astype(np.float32)

            x_list.append(x)
            y_list.append(y)

        x_batch = torch.tensor(np.array(x_list), dtype=torch.float32)
        y_batch = torch.tensor(np.array(y_list), dtype=torch.float32)

        yield x_batch, y_batch


# ==========================
# train
# ==========================
def train_one_epoch(model, optimizer, criterion, device):

    model.train()

    total_loss = 0
    steps = 0

    for x, y in get_dummy_loader():

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        pred = model(x)

        loss = criterion(pred, y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        steps += 1

    return total_loss / steps


# ==========================
# validation
# ==========================
def validate(model, criterion, device):

    model.eval()

    total_loss = 0
    steps = 0

    with torch.no_grad():

        for x, y in get_dummy_loader():

            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            loss = criterion(pred, y)

            total_loss += loss.item()
            steps += 1

    return total_loss / steps


# ==========================
# main
# ==========================
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="runs/unet_best.pt")

    parser.add_argument(
        "--mlflow_tracking_uri",
        type=str,
        default="sqlite:///mlflow.db"
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[info] device =", device)

    # ==========================
    # MLflow設定
    # ==========================
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment("TwinMind_Disaster")

    run_name = f"unet_seed{args.seed}"

    start_time = time.time()

    model = SimpleUNet().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = 0

    with mlflow.start_run(run_name=run_name):

        # ----------------------
        # params
        # ----------------------
        mlflow.log_param("model", "unet_small")
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("device", str(device))
        mlflow.log_param("use_dem", 1)
        mlflow.log_param("use_slope", 1)

        for epoch in range(1, args.epochs + 1):

            train_loss = train_one_epoch(
                model,
                optimizer,
                criterion,
                device
            )

            val_loss = validate(
                model,
                criterion,
                device
            )

            print(
                f"[epoch {epoch:03d}] "
                f"train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f}"
            )

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

            if val_loss < best_val_loss:

                best_val_loss = val_loss
                best_epoch = epoch

                os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

                torch.save(model.state_dict(), args.save_path)

                print("[save] best model ->", args.save_path)

                mlflow.log_artifact(args.save_path)

        train_time_sec = time.time() - start_time

        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_metric("best_epoch", best_epoch)
        mlflow.log_metric("train_time_sec", train_time_sec)

    print("[done]")


if __name__ == "__main__":
    main()
