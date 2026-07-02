"""
Train conditional Flow Matching on MNIST.

Objective:
    x_0 ~ N(0, I)
    x_1 ~ MNIST
    x_t = (1 - t) x_0 + t x_1
    v_target = x_1 - x_0
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model import MnistFlowUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MNIST conditional Flow Matching")
    parser.add_argument("--data-dir", default="dataset/torchvision", help="MNIST download/cache directory")
    parser.add_argument("--output-dir", default="vis-flow-matching/runs/mnist_flow", help="training output directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=0)
    return parser.parse_args()


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_metrics(path: Path, row: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def plot_metrics(csv_path: Path, output_path: Path) -> None:
    """Overwrite a JPG loss curve after each epoch. Labels stay English for font safety."""
    if not csv_path.exists():
        return

    rows: list[dict[str, float]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({key: float(value) for key, value in row.items()})

    if not rows:
        return

    epochs = [row["epoch"] for row in rows]
    train_losses = [row["train_loss"] for row in rows]
    val_losses = [row["val_loss"] for row in rows]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="train_loss")
    plt.plot(epochs, val_losses, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("mse_loss")
    plt.title("MNIST Flow Matching Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_epoch(
    model: MnistFlowUNet,
    loader: DataLoader,
    device: torch.device,
    optimizer: AdamW | None,
    epoch: int,
    phase: str,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_samples = 0
    progress = tqdm(loader, desc=f"{phase} epoch {epoch}", leave=False)

    for images, labels in progress:
        x1 = images.to(device)
        labels = labels.to(device)
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=device)
        t_view = t[:, None, None, None]
        xt = (1 - t_view) * x0 + t_view * x1
        target_velocity = x1 - x0

        with torch.set_grad_enabled(is_train):
            predicted_velocity = model(xt, t, labels)
            loss = F.mse_loss(predicted_velocity, target_velocity)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        batch_size = x1.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        progress.set_postfix(loss=total_loss / total_samples)

    return total_loss / total_samples


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = select_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )
    train_dataset = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = MnistFlowUNet(base_channels=args.base_channels).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config = vars(args) | {"device": str(device)}
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"使用设备: {device}")
    print(f"训练 batch 数: {len(train_loader)}，验证 batch 数: {len(val_loader)}")

    best_loss = float("inf")
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer, epoch, "train")
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, device, None, epoch, "val")

        metrics = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        metrics_csv = output_dir / "metrics.csv"
        append_metrics(metrics_csv, metrics)
        plot_metrics(metrics_csv, output_dir / "metrics.jpg")
        (output_dir / "last_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        checkpoint = {
            "model": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        torch.save(checkpoint, output_dir / "last.pt")

        improved = val_loss < best_loss - args.min_delta
        if improved:
            best_loss = val_loss
            bad_epochs = 0
            torch.save(checkpoint, output_dir / "best.pt")
        else:
            bad_epochs += 1

        if args.checkpoint_every_epochs > 0 and epoch % args.checkpoint_every_epochs == 0:
            torch.save(checkpoint, output_dir / f"epoch_{epoch:04d}.pt")

        print(
            f"epoch={epoch} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} best_val_loss={best_loss:.6f} bad_epochs={bad_epochs}"
        )
        if bad_epochs >= args.patience:
            print(f"Early stopping: val loss 连续 {args.patience} 个 epoch 没有有效改善")
            break


if __name__ == "__main__":
    main()
