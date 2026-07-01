"""
latent GAN 训练入口。

本脚本作为生成基线使用，不替代 DDPM / DiT / Flow Matching 主线。
训练配置采用常见稳定组合：hinge loss、spectral norm discriminator、EMA generator。
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.latent import DEFAULT_LATENTS_H5, select_device
from common.training import create_latent_dataloaders, prune_numbered_checkpoints, save_json, seed_everything
from gan.models import LatentDiscriminator, LatentGenerator


@dataclass
class TrainConfig:
    latents_h5: str
    output_dir: str
    noise_dim: int
    batch_size: int
    epochs: int
    lr_g: float
    lr_d: float
    betas: tuple[float, float]
    val_ratio: float
    num_workers: int
    seed: int
    ema_decay: float
    checkpoint_every_epochs: int
    max_epoch_checkpoints: int


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="训练 latent GAN")
    parser.add_argument("--latents-h5", default=str(DEFAULT_LATENTS_H5))
    parser.add_argument("--output-dir", default="gan/runs/latent_gan")
    parser.add_argument("--noise-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr-g", type=float, default=2e-4)
    parser.add_argument("--lr-d", type=float, default=2e-4)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.0, 0.99))
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=10)
    parser.add_argument("--max-epoch-checkpoints", type=int, default=10)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


def update_ema(source: torch.nn.Module, target: torch.nn.Module, decay: float) -> None:
    """更新 EMA generator 权重。"""
    with torch.no_grad():
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.mul_(decay).add_(source_param.data, alpha=1 - decay)


def save_gan(path: Path, generator, discriminator, ema_generator, config: TrainConfig, metrics: dict) -> None:
    """保存 GAN 权重。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "ema_generator": ema_generator.state_dict(),
            "config": asdict(config),
            "metrics": metrics,
        },
        path,
    )


def train_epoch(generator, discriminator, ema_generator, dataloader, device, opt_g, opt_d, config: TrainConfig) -> dict[str, float]:
    """训练一个 epoch。"""
    generator.train()
    discriminator.train()

    total_d = 0.0
    total_g = 0.0
    total_samples = 0
    progress = tqdm(dataloader, leave=False)

    for real_latents in progress:
        real_latents = real_latents.to(device)
        batch_size = real_latents.shape[0]

        noise = torch.randn(batch_size, config.noise_dim, device=device)
        with torch.no_grad():
            fake_latents = generator(noise)

        real_scores = discriminator(real_latents)
        fake_scores = discriminator(fake_latents)
        d_loss = F.relu(1 - real_scores).mean() + F.relu(1 + fake_scores).mean()

        opt_d.zero_grad(set_to_none=True)
        d_loss.backward()
        opt_d.step()

        noise = torch.randn(batch_size, config.noise_dim, device=device)
        fake_latents = generator(noise)
        g_loss = -discriminator(fake_latents).mean()

        opt_g.zero_grad(set_to_none=True)
        g_loss.backward()
        opt_g.step()
        update_ema(generator, ema_generator, config.ema_decay)

        total_d += d_loss.item() * batch_size
        total_g += g_loss.item() * batch_size
        total_samples += batch_size
        progress.set_postfix(d=total_d / total_samples, g=total_g / total_samples)

    return {"d_loss": total_d / total_samples, "g_loss": total_g / total_samples}


@torch.no_grad()
def validate_generator(generator, discriminator, dataloader, device, config: TrainConfig) -> float:
    """
    简单验证指标：生成 latent 的 discriminator score。

    GAN 没有和 DDPM 一样直接可靠的 val loss，这里只用于保存趋势，不把它解释成最终质量。
    """
    generator.eval()
    discriminator.eval()
    total_score = 0.0
    total_samples = 0
    for real_latents in dataloader:
        batch_size = real_latents.shape[0]
        noise = torch.randn(batch_size, config.noise_dim, device=device)
        fake_latents = generator(noise)
        score = discriminator(fake_latents).mean()
        total_score += score.item() * batch_size
        total_samples += batch_size
    return total_score / total_samples


def main() -> None:
    config = parse_args()
    seed_everything(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config.json", config)

    device = select_device()
    print(f"使用设备: {device}")
    train_loader, val_loader = create_latent_dataloaders(
        config.latents_h5,
        config.batch_size,
        config.val_ratio,
        config.seed,
        config.num_workers,
    )

    generator = LatentGenerator(noise_dim=config.noise_dim).to(device)
    discriminator = LatentDiscriminator().to(device)
    ema_generator = copy.deepcopy(generator).eval()

    opt_g = AdamW(generator.parameters(), lr=config.lr_g, betas=tuple(config.betas))
    opt_d = AdamW(discriminator.parameters(), lr=config.lr_d, betas=tuple(config.betas))

    best_score = -float("inf")
    for epoch in range(1, config.epochs + 1):
        train_metrics = train_epoch(generator, discriminator, ema_generator, train_loader, device, opt_g, opt_d, config)
        val_score = validate_generator(ema_generator, discriminator, val_loader, device, config)
        metrics = {"epoch": epoch, **train_metrics, "val_fake_score": val_score}
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        save_json(output_dir / "last_metrics.json", metrics)

        save_gan(output_dir / "last.pt", generator, discriminator, ema_generator, config, metrics)
        if val_score > best_score:
            best_score = val_score
            save_gan(output_dir / "best.pt", generator, discriminator, ema_generator, config, metrics)
            save_json(output_dir / "best_metrics.json", metrics)

        if config.checkpoint_every_epochs > 0 and epoch % config.checkpoint_every_epochs == 0:
            save_gan(output_dir / f"epoch_{epoch:04d}.pt", generator, discriminator, ema_generator, config, metrics)
            prune_numbered_checkpoints(output_dir, "epoch_*.pt", config.max_epoch_checkpoints)


if __name__ == "__main__":
    main()
