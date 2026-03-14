import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import mlflow

from twinmind_disaster.dataset_loader import FloodNPZDataset
from twinmind_disaster.model_unet import UNetSmall


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases_dir", type=str, default="data/cases")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="runs/unet_small_best.pt")

    # MLflow
    parser.add_argument("--mlflow_experiment", type=str, default="TwinMind_Disaster")
    parser.add_argument("--mlflow_run_name", type=str, default=None)

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, optimizer, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    criterion = torch.nn.MSELoss()
    total_loss = 0.0
    total_count = 0

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            pred = model(x)
            loss = criterion(pred, y)

            if train:
                loss.backward()
                optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_count += bs

    return total_loss / max(total_count, 1)


def main():
    args = parse_args()
    set_seed(args.seed)

    # -------------------------------
    # MLflow tracking URI: local folder
    # -------------------------------
    mlflow.set_tracking_uri("file:./mlruns")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device = {device}")

    dataset = FloodNPZDataset(args.cases_dir)
    n_total = len(dataset)
    n_train = int(n_total * 0.8)
    n_val = n_total - n_train

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = UNetSmall(in_channels=7, out_channels=1, base_ch=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment(args.mlflow_experiment)
    run_name = args.mlflow_run_name or f"seed{args.seed}_bs{args.batch_size}_lr{args.lr}"

    best_val_loss = float("inf")
    best_epoch = -1

    with mlflow.start_run(run_name=run_name):
        # params
        mlflow.log_param("cases_dir", args.cases_dir)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("lr", args.lr)
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("num_workers", args.num_workers)
        mlflow.log_param("save_path", str(save_path))
        mlflow.log_param("device", str(device))
        mlflow.log_param("dataset_size", n_total)
        mlflow.log_param("train_size", n_train)
        mlflow.log_param("val_size", n_val)

        for epoch in range(1, args.epochs + 1):
            train_loss = run_epoch(model, train_loader, optimizer, device, train=True)
            val_loss = run_epoch(model, val_loader, optimizer, device, train=False)

            print(f"[epoch {epoch:03d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_loss": best_val_loss,
                        "args": vars(args),
                    },
                    save_path,
                )

                print(f"[save] best model -> {save_path}")

                mlflow.log_metric("best_val_loss", best_val_loss, step=epoch)
                mlflow.log_metric("best_epoch", best_epoch, step=epoch)

        # artifact
        if save_path.exists():
            mlflow.log_artifact(str(save_path))

        mlflow.log_metric("final_best_val_loss", best_val_loss)
        mlflow.log_metric("final_best_epoch", best_epoch)

    print("[done]")


if __name__ == "__main__":
    main()
