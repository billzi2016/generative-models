"""
latent 训练共享工具。

本模块服务于 ddpm / dit / flow_matching / gan 等方法目录，统一处理：
- 设备选择。
- HDF5 latent 数据集读取。
- VAE best 权重路径检查。
- scaled latent 到图片的解码保存。

所有生成方法默认都读取同一份 HDF5 latent 缓存，不在训练时重复跑 VAE encoder。
"""

from __future__ import annotations

from pathlib import Path

import h5py
import torch
from diffusers import AutoencoderKL
from torch.utils.data import Dataset
from torchvision.utils import save_image


DEFAULT_VAE_DIR = Path("vae/runs/vae_daf/best_diffusers")
DEFAULT_LATENTS_H5 = Path("dataset/latents/vae_daf_128_best.h5")


def select_device() -> torch.device:
    """优先选择 CUDA，其次 Apple Silicon MPS，最后 CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def require_path(path: str | Path, description: str) -> Path:
    """检查必须存在的路径，并给出明确错误信息。"""
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"{description} 不存在: {resolved}")
    return resolved


def load_best_vae(vae_dir: str | Path = DEFAULT_VAE_DIR, device: torch.device | None = None) -> AutoencoderKL:
    """
    读取 VAE best Diffusers 权重。

    默认路径固定为 vae/runs/vae_daf/best_diffusers，避免后续方法误用未确认的 VAE。
    """
    vae_path = require_path(vae_dir, "VAE best_diffusers 权重目录")
    target_device = device or select_device()
    vae = AutoencoderKL.from_pretrained(vae_path).to(target_device)
    vae.eval()
    return vae


class LatentH5Dataset(Dataset):
    """
    HDF5 latent 数据集。

    DataLoader 多 worker 时不能在主进程共享同一个 h5py 文件句柄，所以这里使用懒加载：
    每个 worker 第一次读取样本时打开自己的只读文件句柄。
    """

    def __init__(self, h5_path: str | Path = DEFAULT_LATENTS_H5) -> None:
        self.h5_path = str(require_path(h5_path, "latent HDF5 缓存文件"))
        self._h5 = None

        with h5py.File(self.h5_path, "r") as h5:
            self.length = int(h5["latents"].shape[0])
            self.latent_shape = tuple(h5["latents"].shape[1:])
            self.scaling_factor = float(h5.attrs.get("scaling_factor", 0.18215))

    def _file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        h5 = self._file()
        latent = h5["latents"][index]
        return torch.from_numpy(latent).float()


@torch.no_grad()
def decode_scaled_latents(vae: AutoencoderKL, scaled_latents: torch.Tensor) -> torch.Tensor:
    """
    将 scaled latent 解码成 [0, 1] 图片张量。

    HDF5 中保存的是乘过 scaling_factor 的 latent；AutoencoderKL decoder 接收未缩放 latent。
    """
    latents = scaled_latents / vae.config.scaling_factor
    images = vae.decode(latents).sample
    images = (images.clamp(-1, 1) + 1) / 2
    return images


@torch.no_grad()
def save_latent_grid(
    vae: AutoencoderKL,
    scaled_latents: torch.Tensor,
    output_path: str | Path,
    nrow: int = 4,
) -> None:
    """解码一批 latent，并保存成图片网格。"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    images = decode_scaled_latents(vae, scaled_latents)
    save_image(images, output, nrow=nrow)
