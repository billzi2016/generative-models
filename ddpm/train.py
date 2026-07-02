"""
latent DDPM 训练入口。

本脚本使用成熟 Diffusers 组件：
- UNet2DModel 作为 latent denoiser。
- DDPMScheduler 提供标准前向加噪公式。

训练数据来自 VAE best 生成的 HDF5 latent 缓存，形状固定为 [B, 4, 16, 16]。
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

warnings.filterwarnings("ignore", message=r"urllib3 .* doesn't match a supported version!")

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.latent import DEFAULT_LATENTS_H5, select_device
from common.training import create_latent_dataloaders, prune_numbered_checkpoints, record_metrics, save_json, seed_everything


@dataclass
class TrainConfig:
    latents_h5: str
    output_dir: str
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    val_ratio: float
    num_workers: int
    seed: int
    num_train_timesteps: int
    checkpoint_every_epochs: int
    max_epoch_checkpoints: int
    patience: int
    min_delta: float


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="训练 latent DDPM")
    parser.add_argument("--latents-h5", default=str(DEFAULT_LATENTS_H5), help="VAE latent HDF5 缓存")
    parser.add_argument("--output-dir", default="ddpm/runs/latent_ddpm", help="输出目录")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-train-timesteps", type=int, default=1000)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=10)
    parser.add_argument("--max-epoch-checkpoints", type=int, default=10)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    return TrainConfig(**vars(parser.parse_args()))


def build_model() -> UNet2DModel:
    """
    构建 latent DDPM U-Net。

    配置保持常规 U-Net diffusion 结构，但输入输出通道固定为 VAE latent 的 4 通道。
    """
    return UNet2DModel(
        sample_size=16,
        in_channels=4,
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(128, 256, 512),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        norm_num_groups=32,
        attention_head_dim=8,
    )


def run_epoch(model, scheduler, dataloader, device, optimizer=None) -> float:
    """执行一个 train 或 val epoch，返回平均噪声预测 MSE。"""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_samples = 0
    progress = tqdm(dataloader, leave=False)

    for clean_latents in progress:
        clean_latents = clean_latents.to(device)
        noise = torch.randn_like(clean_latents)
        timesteps = torch.randint(
            0,
            scheduler.config.num_train_timesteps,
            (clean_latents.shape[0],),
            device=device,
        ).long()
        noisy_latents = scheduler.add_noise(clean_latents, noise, timesteps)

        with torch.set_grad_enabled(is_train):
            noise_pred = model(noisy_latents, timesteps).sample
            loss = F.mse_loss(noise_pred, noise)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        batch_size = clean_latents.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        progress.set_postfix(loss=total_loss / total_samples)

    return total_loss / total_samples


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

    model = build_model().to(device)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config.num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_val_loss = float("inf")
    bad_epochs = 0
    for epoch in range(1, config.epochs + 1):
        train_loss = run_epoch(model, noise_scheduler, train_loader, device, optimizer)
        val_loss = run_epoch(model, noise_scheduler, val_loader, device)
        lr_scheduler.step()

        improved = val_loss < best_val_loss - config.min_delta
        if improved:
            best_val_loss = val_loss
            bad_epochs = 0
        else:
            bad_epochs += 1

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "bad_epochs": bad_epochs,
            "lr": optimizer.param_groups[0]["lr"],
        }
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        record_metrics(output_dir, metrics)

        model.save_pretrained(output_dir / "last_model", safe_serialization=True)
        noise_scheduler.save_pretrained(output_dir / "scheduler")

        if improved:
            model.save_pretrained(output_dir / "best_model", safe_serialization=True)
            save_json(output_dir / "best_metrics.json", metrics)

        if config.checkpoint_every_epochs > 0 and epoch % config.checkpoint_every_epochs == 0:
            model.save_pretrained(output_dir / f"epoch_{epoch:04d}_model", safe_serialization=True)
            prune_numbered_checkpoints(output_dir, "epoch_*_model", config.max_epoch_checkpoints)

        if bad_epochs >= config.patience:
            print(f"Early stopping: val_loss 连续 {config.patience} 个 epoch 没有有效改善")
            break


if __name__ == "__main__":
    main()
