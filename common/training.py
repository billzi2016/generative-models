"""
训练共享工具。

本模块只放通用工程逻辑，不放具体生成模型算法：
- 固定随机种子。
- HDF5 latent 数据集 train/val 划分。
- 有上限的 checkpoint 清理。
- JSON 配置保存。
"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, random_split

from common.latent import LatentH5Dataset


def seed_everything(seed: int) -> None:
    """固定主要随机源，降低划分和初始化波动。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: str | Path, data: Any) -> None:
    """保存 JSON 配置或指标。"""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(data) if is_dataclass(data) else data
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def create_latent_dataloaders(
    h5_path: str | Path,
    batch_size: int,
    val_ratio: float,
    seed: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    """从 HDF5 latent 创建 train / val DataLoader。"""
    dataset = LatentH5Dataset(h5_path)
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise RuntimeError("latent 数据量太少，无法划分训练集和验证集")

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def prune_numbered_checkpoints(output_dir: str | Path, pattern: str, max_checkpoints: int) -> None:
    """
    删除超过上限的周期 checkpoint。

    pattern 例如 epoch_*_model 或 epoch_*.pt。只清理匹配该 pattern 的路径。
    """
    if max_checkpoints <= 0:
        return

    root = Path(output_dir)
    checkpoints = sorted(root.glob(pattern))
    extra_count = len(checkpoints) - max_checkpoints
    for checkpoint in checkpoints[: max(0, extra_count)]:
        if checkpoint.is_dir():
            shutil.rmtree(checkpoint)
        else:
            checkpoint.unlink()
