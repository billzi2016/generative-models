"""
Small conditional U-Net for MNIST Flow Matching.

The model predicts a velocity field v_theta(x_t, t, y) for grayscale 28x28
digits.  It is intentionally small so the demo can run on CPU or Apple MPS.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Encode continuous t in [0, 1] with sinusoidal features."""
    half_dim = dim // 2
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(half_dim, device=t.device, dtype=t.dtype) / max(half_dim - 1, 1)
    )
    angles = t[:, None] * frequencies[None, :]
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ResidualBlock(nn.Module):
    """Conv residual block with additive time/class conditioning."""

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.cond = nn.Linear(cond_dim, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.cond(cond)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class MnistFlowUNet(nn.Module):
    """A compact class-conditional U-Net for 28x28 MNIST velocity prediction."""

    def __init__(self, base_channels: int = 32, cond_dim: int = 128) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.label_embedding = nn.Embedding(10, cond_dim)

        self.input = nn.Conv2d(1, base_channels, kernel_size=3, padding=1)
        self.down1 = ResidualBlock(base_channels, base_channels, cond_dim)
        self.downsample1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.down2 = ResidualBlock(base_channels * 2, base_channels * 2, cond_dim)
        self.downsample2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)

        self.middle1 = ResidualBlock(base_channels * 4, base_channels * 4, cond_dim)
        self.middle2 = ResidualBlock(base_channels * 4, base_channels * 4, cond_dim)

        self.upsample2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.up2 = ResidualBlock(base_channels * 4, base_channels * 2, cond_dim)
        self.upsample1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.up1 = ResidualBlock(base_channels * 2, base_channels, cond_dim)

        self.output = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cond = self.time_mlp(sinusoidal_embedding(t, self.cond_dim)) + self.label_embedding(labels)

        h0 = self.input(x)
        h1 = self.down1(h0, cond)
        h2 = self.down2(self.downsample1(h1), cond)
        h3 = self.middle2(self.middle1(self.downsample2(h2), cond), cond)

        h = self.upsample2(h3)
        h = self.up2(torch.cat([h, h2], dim=1), cond)
        h = self.upsample1(h)
        h = self.up1(torch.cat([h, h1], dim=1), cond)
        return self.output(h)
