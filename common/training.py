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
import csv
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def append_metrics_csv(path: str | Path, metrics: dict[str, Any]) -> None:
    """追加一行训练指标到 CSV。"""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output.exists()

    with output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)


def plot_metrics_jpg(csv_path: str | Path, output_path: str | Path, dpi: int = 200) -> None:
    """
    从 metrics.csv 读取数值列并覆盖保存曲线图。

    每个 epoch 后覆盖写同一个 jpg，避免生成大量历史图片。
    """
    source = Path(csv_path)
    if not source.exists():
        return

    rows: list[dict[str, float]] = []
    with source.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: dict[str, float] = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    continue
            rows.append(parsed)

    if not rows or "epoch" not in rows[0]:
        return

    x_values = [row["epoch"] for row in rows]
    metric_names = [key for key in rows[0].keys() if key != "epoch"]
    metric_names = [key for key in metric_names if any(key in row for row in rows)]
    if not metric_names:
        return

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for metric_name in metric_names:
        y_values = [row.get(metric_name, float("nan")) for row in rows]
        plt.plot(x_values, y_values, label=metric_name)
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.title("Training metrics")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=dpi)
    plt.close()


def record_metrics(output_dir: str | Path, metrics: dict[str, Any]) -> None:
    """同时保存 last_metrics.json、追加 metrics.csv，并覆盖 metrics.jpg。"""
    root = Path(output_dir)
    save_json(root / "last_metrics.json", metrics)
    csv_path = root / "metrics.csv"
    append_metrics_csv(csv_path, metrics)
    plot_metrics_jpg(csv_path, root / "metrics.jpg", dpi=200)


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
