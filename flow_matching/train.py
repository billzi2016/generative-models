"""
latent Flow Matching / Rectified Flow 训练入口。

训练目标采用标准线性路径：
z_t = (1 - t) * z_0 + t * z_1
v_target = z_1 - z_0

其中 z_0 是高斯噪声 latent，z_1 是 VAE HDF5 缓存中的真实图片 latent。
模型使用 Diffusers UNet2DModel 学习 velocity field。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import UNet2DModel
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
    checkpoint_every_epochs: int
    max_epoch_checkpoints: int


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="训练 latent Flow Matching")
    parser.add_argument("--latents-h5", default=str(DEFAULT_LATENTS_H5))
    parser.add_argument("--output-dir", default="flow_matching/runs/latent_fm")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=10)
    parser.add_argument("--max-epoch-checkpoints", type=int, default=10)
    return TrainConfig(**vars(parser.parse_args()))


def build_model() -> UNet2DModel:
    """构建 velocity field U-Net。"""
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


def run_epoch(model, dataloader, device, optimizer=None) -> float:
    """执行一个 train 或 val epoch，返回 velocity MSE。"""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_samples = 0
    progress = tqdm(dataloader, leave=False)

    for data_latents in progress:
        z1 = data_latents.to(device)
        z0 = torch.randn_like(z1)
        t = torch.rand(z1.shape[0], 1, 1, 1, device=device)
        zt = (1 - t) * z0 + t * z1
        target_velocity = z1 - z0

        # UNet2DModel 使用整数 timestep embedding，这里把连续 t 映射到 0..999。
        timesteps = (t.flatten() * 999).long()

        with torch.set_grad_enabled(is_train):
            velocity = model(zt, timesteps).sample
            loss = F.mse_loss(velocity, target_velocity)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        batch_size = z1.shape[0]
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
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_val_loss = float("inf")
    for epoch in range(1, config.epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer)
        val_loss = run_epoch(model, val_loader, device)
        lr_scheduler.step()

        metrics = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"]}
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        record_metrics(output_dir, metrics)

        model.save_pretrained(output_dir / "last_model", safe_serialization=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(output_dir / "best_model", safe_serialization=True)
            save_json(output_dir / "best_metrics.json", metrics)

        if config.checkpoint_every_epochs > 0 and epoch % config.checkpoint_every_epochs == 0:
            model.save_pretrained(output_dir / f"epoch_{epoch:04d}_model", safe_serialization=True)
            prune_numbered_checkpoints(output_dir, "epoch_*_model", config.max_epoch_checkpoints)


if __name__ == "__main__":
    main()
