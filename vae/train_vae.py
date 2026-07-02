"""
VAE 训练入口。

本文件是当前项目第一阶段的主程序：
1. 从已解压图片目录递归读取动漫头像。
2. 使用 Diffusers AutoencoderKL 把图片压缩到 SD 风格空间 latent map。
3. 保存 Diffusers 格式 checkpoint，供后续 latent diffusion / stable_diffusion 模块复用。

默认数据目录直接匹配当前仓库中的 DAF 解压结构：dataset/raw/fullMin256。
实际训练时仍可以通过 --data-dir 显式指定其他图片目录。
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
from tqdm import tqdm

from dataset import ImageFolderRecursiveDataset
from model import DEFAULT_PRETRAINED_VAE, load_autoencoder_kl


@dataclass
class TrainConfig:
    """训练配置，保存到 checkpoint 中方便后续复现实验。"""

    data_dir: str
    output_dir: str
    pretrained_vae: str
    init_from_scratch: bool
    image_size: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    kl_weight: float
    val_ratio: float
    num_workers: int
    seed: int
    patience: int
    min_delta: float
    save_last: bool
    save_training_state: bool
    checkpoint_every_epochs: int
    max_epoch_checkpoints: int


class EarlyStopping:
    """
    简单早停控制。

    PRD 中要求默认支持 Early Stopping。这里保持实现很薄，只负责判断验证集 loss
    是否持续没有改善；学习率调度仍使用 PyTorch 原生 ReduceLROnPlateau。
    """

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.bad_epochs = 0

    def step(self, loss: float) -> bool:
        improved = loss < self.best_loss - self.min_delta
        if improved:
            self.best_loss = loss
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def parse_args() -> TrainConfig:
    """解析命令行参数，并转换成训练配置对象。"""
    parser = argparse.ArgumentParser(description="训练动漫头像 VAE")
    parser.add_argument("--data-dir", default="dataset/raw/fullMin256", help="DAF 已解压图片目录")
    parser.add_argument("--output-dir", default="vae/runs/vae_daf", help="checkpoint 和样例图输出目录")
    parser.add_argument("--pretrained-vae", default=DEFAULT_PRETRAINED_VAE, help="Diffusers VAE 权重名或本地目录")
    parser.add_argument("--init-from-scratch", action="store_true", help="不用预训练权重，仅用 Diffusers AutoencoderKL 架构从零训练")
    parser.add_argument("--image-size", type=int, default=128, help="训练分辨率，当前主线固定使用 128")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--epochs", type=int, default=50, help="最大训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW 权重衰减")
    parser.add_argument("--kl-weight", type=float, default=1e-4, help="KL loss 权重")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="验证集比例")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker 数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--patience", type=int, default=8, help="早停等待轮数")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="判定验证 loss 改善的最小幅度")
    parser.add_argument("--save-last", action="store_true", help="额外保存 last_diffusers；默认只强制保存 best_diffusers")
    parser.add_argument("--save-training-state", action="store_true", help="保存 optimizer/scheduler 状态 .pt；体积较大，默认关闭")
    parser.add_argument("--checkpoint-every-epochs", type=int, default=0, help="每隔多少个 epoch 保存一个 Diffusers checkpoint；0 表示不保存周期点")
    parser.add_argument("--max-epoch-checkpoints", type=int, default=10, help="最多保留多少个周期 checkpoint，避免占满 SSD")
    args = parser.parse_args()
    return TrainConfig(**vars(args))


def select_device() -> torch.device:
    """优先选择 CUDA，其次 Apple Silicon MPS，最后 CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """固定主要随机源，降低每次训练划分和初始化的波动。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def vae_loss(reconstruction: Tensor, target: Tensor, mu: Tensor, logvar: Tensor, kl_weight: float) -> tuple[Tensor, Tensor, Tensor]:
    """
    计算 VAE 损失。

    reconstruction loss 使用 MSE，符合 PRD 第一版要求；KL loss 约束空间 latent map
    接近标准正态，为后续 latent diffusion 提供更规整的输入空间。
    """
    reconstruction_loss = F.mse_loss(reconstruction, target, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = reconstruction_loss + kl_weight * kl_loss
    return total_loss, reconstruction_loss, kl_loss


def encode_decode(model: nn.Module, images: Tensor, sample_posterior: bool) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    使用 Diffusers AutoencoderKL 的标准接口完成 encode/decode。

    返回：
        reconstruction: 重建图像，范围约为 [-1, 1]。
        latents: 乘过 scaling_factor 的 latent，后续 diffusion 训练应使用这个版本。
        mu/logvar: KL loss 使用的后验分布参数。
    """
    posterior = model.encode(images).latent_dist
    latents = posterior.sample() if sample_posterior else posterior.mode()
    scaled_latents = latents * model.config.scaling_factor
    reconstruction = model.decode(latents).sample
    return reconstruction, scaled_latents, posterior.mean, posterior.logvar


def create_dataloaders(config: TrainConfig) -> tuple[DataLoader, DataLoader]:
    """创建训练和验证 DataLoader。"""
    dataset = ImageFolderRecursiveDataset(
        config.data_dir,
        image_size=config.image_size,
    )
    val_size = max(1, int(len(dataset) * config.val_ratio))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise RuntimeError("数据量太少，无法划分训练集和验证集")

    generator = torch.Generator().manual_seed(config.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: AdamW | None,
    kl_weight: float,
    epoch: int,
    phase: str,
) -> dict[str, float]:
    """执行一个训练或验证 epoch，并返回平均 loss 指标。"""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_reconstruction_loss = 0.0
    total_kl_loss = 0.0
    total_samples = 0

    progress = tqdm(dataloader, desc=f"{phase} epoch {epoch}", leave=False)
    for images in progress:
        images = images.to(device)

        with torch.set_grad_enabled(is_train):
            reconstruction, _, mu, logvar = encode_decode(model, images, sample_posterior=is_train)
            loss, reconstruction_loss, kl_loss = vae_loss(reconstruction, images, mu, logvar, kl_weight)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_reconstruction_loss += reconstruction_loss.item() * batch_size
        total_kl_loss += kl_loss.item() * batch_size

        progress.set_postfix(
            loss=total_loss / total_samples,
            recon=total_reconstruction_loss / total_samples,
            kl=total_kl_loss / total_samples,
        )

    return {
        "loss": total_loss / total_samples,
        "reconstruction_loss": total_reconstruction_loss / total_samples,
        "kl_loss": total_kl_loss / total_samples,
    }


@torch.no_grad()
def save_reconstruction_samples(model: nn.Module, dataloader: DataLoader, device: torch.device, output_path: Path) -> None:
    """保存原图和重建图对比，便于肉眼检查 VAE 是否真的学到重建能力。"""
    model.eval()
    images = next(iter(dataloader)).to(device)
    reconstruction, _, _, _ = encode_decode(model, images, sample_posterior=False)

    # save_image 期望 [0, 1]，而训练张量是 [-1, 1]，这里转换回可视化范围。
    comparison = torch.cat([images[:8], reconstruction[:8]], dim=0)
    comparison = (comparison.clamp(-1, 1) + 1) / 2
    save_image(comparison, output_path, nrow=8)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: ReduceLROnPlateau,
    config: TrainConfig,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    """保存训练状态，后续可用于恢复训练；模型权重另存为 Diffusers 格式。"""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": asdict(config),
            "metrics": metrics,
        },
        path,
    )


def save_diffusers_model(model: nn.Module, path: Path) -> None:
    """以 Diffusers 原生格式保存 VAE，并优先使用 safetensors，减少损坏风险。"""
    model.save_pretrained(path, safe_serialization=True)


def prune_epoch_checkpoints(output_dir: Path, max_checkpoints: int) -> None:
    """
    限制周期 checkpoint 数量。

    只删除命名为 epoch_XXXX_diffusers 的目录，避免误删 best_diffusers 或用户手动保存的目录。
    """
    if max_checkpoints <= 0:
        return

    checkpoints = sorted(output_dir.glob("epoch_*_diffusers"))
    extra_count = len(checkpoints) - max_checkpoints
    for checkpoint_dir in checkpoints[:max(0, extra_count)]:
        shutil.rmtree(checkpoint_dir)


def main() -> None:
    """训练主流程。"""
    config = parse_args()
    seed_everything(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")

    device = select_device()
    print(f"使用设备: {device}")

    train_loader, val_loader = create_dataloaders(config)
    print(f"训练 batch 数: {len(train_loader)}，验证 batch 数: {len(val_loader)}")

    model = load_autoencoder_kl(
        config.pretrained_vae,
        init_from_scratch=config.init_from_scratch,
    ).to(device)
    print(
        "VAE latent 配置:",
        f"latent_channels={model.config.latent_channels},",
        f"scaling_factor={model.config.scaling_factor}",
    )
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    early_stopping = EarlyStopping(patience=config.patience, min_delta=config.min_delta)

    best_val_loss = float("inf")
    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer, config.kl_weight, epoch, "train")
        val_metrics = run_epoch(model, val_loader, device, None, config.kl_weight, epoch, "val")
        scheduler.step(val_metrics["loss"])

        metrics = {
            "train_loss": train_metrics["loss"],
            "train_reconstruction_loss": train_metrics["reconstruction_loss"],
            "train_kl_loss": train_metrics["kl_loss"],
            "val_loss": val_metrics["loss"],
            "val_reconstruction_loss": val_metrics["reconstruction_loss"],
            "val_kl_loss": val_metrics["kl_loss"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        print(json.dumps({"epoch": epoch, **metrics}, ensure_ascii=False, indent=2))

        if config.save_training_state:
            save_checkpoint(output_dir / "last.pt", model, optimizer, scheduler, config, epoch, metrics)

        if config.save_last:
            save_diffusers_model(model, output_dir / "last_diffusers")

        if config.checkpoint_every_epochs > 0 and epoch % config.checkpoint_every_epochs == 0:
            save_diffusers_model(model, output_dir / f"epoch_{epoch:04d}_diffusers")
            prune_epoch_checkpoints(output_dir, config.max_epoch_checkpoints)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            if config.save_training_state:
                save_checkpoint(output_dir / "best.pt", model, optimizer, scheduler, config, epoch, metrics)
            save_diffusers_model(model, output_dir / "best_diffusers")
            save_reconstruction_samples(model, val_loader, device, output_dir / f"reconstruction_epoch_{epoch:03d}.png")

        if early_stopping.step(val_metrics["loss"]):
            print(f"验证 loss 连续 {config.patience} 轮没有明显改善，提前停止训练。")
            break


if __name__ == "__main__":
    main()
