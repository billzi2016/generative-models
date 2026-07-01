"""
latent GAN 模型。

GAN 没有像 diffusers 那样统一的官方 latent GAN 组件，因此这里使用常见稳定配置：
- Generator 从噪声向量生成 [4, 16, 16] latent。
- Discriminator 使用 spectral norm。
- 训练脚本使用 hinge loss 和 EMA generator。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import spectral_norm


class LatentGenerator(nn.Module):
    """从标准高斯噪声生成 VAE scaled latent。"""

    def __init__(self, noise_dim: int = 256, base_channels: int = 512) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.net = nn.Sequential(
            nn.Linear(noise_dim, base_channels * 4 * 4),
            nn.SiLU(inplace=True),
            nn.Unflatten(1, (base_channels, 4, 4)),
            nn.ConvTranspose2d(base_channels, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 4, kernel_size=3, padding=1),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.net(noise)


class LatentDiscriminator(nn.Module):
    """判别真实/生成 latent，卷积层使用 spectral norm。"""

    def __init__(self, base_channels: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(4, base_channels, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            spectral_norm(nn.Linear(base_channels * 4 * 4 * 4, 1)),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.net(latents).flatten()
