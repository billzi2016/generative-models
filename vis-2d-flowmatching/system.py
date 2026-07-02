"""Shared train, sampling, GIF, and vector-field utilities for 2D Flow Matching."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from tqdm import tqdm

from distributions import sample_distribution
from model import VelocityMLP


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


def build_parser(distribution: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"2D Flow Matching visualization: {distribution}")
    parser.add_argument("--distribution", default=distribution, choices=("ring", "moons"))
    parser.add_argument("--output-dir", default=f"vis-2d-flowmatching/runs/{distribution}")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-count", type=int, default=1200)
    parser.add_argument("--gif-steps", type=int, default=160)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--hold-final-frames", type=int, default=30)
    parser.add_argument("--field-grid", type=int, default=25)
    parser.add_argument("--checkpoint", default=None, help="load checkpoint instead of training when --skip-train is set")
    parser.add_argument("--skip-train", action="store_true", help="skip training and render from --checkpoint")
    return parser


def append_metrics(path: Path, row: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def plot_metrics(csv_path: Path, output_path: Path) -> None:
    rows: list[dict[str, float]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append({key: float(value) for key, value in row.items()})
    if not rows:
        return

    plt.figure(figsize=(8, 5))
    plt.plot([row["epoch"] for row in rows], [row["loss"] for row in rows], label="loss")
    plt.xlabel("epoch")
    plt.ylabel("mse_loss")
    plt.title("2D Flow Matching Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def train(args: argparse.Namespace, device: torch.device) -> VelocityMLP:
    model = VelocityMLP(hidden_dim=args.hidden_dim, depth=args.depth).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args) | {"device": str(device)}, indent=2), encoding="utf-8")

    best_loss = float("inf")
    progress = tqdm(range(1, args.epochs + 1), desc=f"train {args.distribution}")
    for epoch in progress:
        model.train()
        x1 = sample_distribution(args.distribution, args.batch_size, device)
        x0 = torch.randn_like(x1)
        t = torch.rand(args.batch_size, device=device)
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        target_velocity = x1 - x0

        predicted_velocity = model(xt, t)
        loss = F.mse_loss(predicted_velocity, target_velocity)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_value = float(loss.item())
        metrics = {"epoch": epoch, "loss": loss_value}
        append_metrics(output_dir / "metrics.csv", metrics)
        plot_metrics(output_dir / "metrics.csv", output_dir / "metrics.jpg")
        progress.set_postfix(loss=loss_value)

        checkpoint = {
            "model": model.state_dict(),
            "config": vars(args),
            "epoch": epoch,
            "loss": loss_value,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if loss_value < best_loss:
            best_loss = loss_value
            torch.save(checkpoint, output_dir / "best.pt")

    return model


@torch.no_grad()
def sample_trajectory(model: VelocityMLP, count: int, steps: int, device: torch.device, seed: int) -> list[torch.Tensor]:
    torch.manual_seed(seed)
    x = torch.randn(count, 2, device=device)
    trajectory = [x.detach().cpu()]
    dt = 1.0 / steps
    model.eval()
    for step in range(steps):
        t = torch.full((count,), step / steps, device=device)
        velocity = model(x, t)
        x = x + dt * velocity
        trajectory.append(x.detach().cpu())
    return trajectory


def render_frame(points: torch.Tensor, target: torch.Tensor, title: str) -> Image.Image:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(target[:, 0], target[:, 1], s=4, alpha=0.16, color="#a8a8a8", label="target")
    ax.scatter(points[:, 0], points[:, 1], s=6, alpha=0.75, color="#0066cc", label="flow")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = Image.frombuffer("RGBA", (width, height), fig.canvas.buffer_rgba(), "raw", "RGBA", 0, 1).convert("RGB")
    plt.close(fig)
    return image


def save_gif(args: argparse.Namespace, model: VelocityMLP, device: torch.device) -> None:
    output_dir = Path(args.output_dir)
    target = sample_distribution(args.distribution, args.sample_count, torch.device("cpu"))
    trajectory = sample_trajectory(model, args.sample_count, args.gif_steps, device, args.seed)
    frames = [
        render_frame(points, target, f"{args.distribution} flow t={index / args.gif_steps:.2f}")
        for index, points in enumerate(trajectory)
    ]
    if args.hold_final_frames > 0:
        frames.extend([frames[-1].copy() for _ in range(args.hold_final_frames)])
    output = output_dir / f"{args.distribution}_flow.gif"
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=max(1, int(1000 / args.fps)),
        loop=0,
        optimize=False,
    )
    print(f"saved gif: {output}")


@torch.no_grad()
def save_vector_field(args: argparse.Namespace, model: VelocityMLP, device: torch.device) -> None:
    output_dir = Path(args.output_dir)
    grid_values = torch.linspace(-3, 3, args.field_grid, device=device)
    yy, xx = torch.meshgrid(grid_values, grid_values, indexing="ij")
    points = torch.stack([xx.flatten(), yy.flatten()], dim=1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, t_value in zip(axes, (0.0, 0.5, 1.0)):
        t = torch.full((points.shape[0],), t_value, device=device)
        velocity = model(points, t).detach().cpu()
        pts = points.detach().cpu()
        axis.quiver(pts[:, 0], pts[:, 1], velocity[:, 0], velocity[:, 1], angles="xy", scale_units="xy", scale=18)
        axis.set_xlim(-3, 3)
        axis.set_ylim(-3, 3)
        axis.set_aspect("equal")
        axis.set_title(f"velocity field t={t_value:.1f}")
        axis.grid(True, alpha=0.2)
    fig.tight_layout()
    output = output_dir / f"{args.distribution}_vector_field.jpg"
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"saved vector field: {output}")


def run(distribution: str) -> None:
    parser = build_parser(distribution)
    args = parser.parse_args()
    seed_everything(args.seed)
    device = select_device()
    print(f"device: {device}")
    if args.skip_train:
        if not args.checkpoint:
            raise ValueError("--skip-train requires --checkpoint")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model = VelocityMLP(hidden_dim=args.hidden_dim, depth=args.depth).to(device)
        model.load_state_dict(checkpoint["model"])
    else:
        model = train(args, device)
    save_gif(args, model, device)
    save_vector_field(args, model, device)
