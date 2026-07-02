"""Shared MLP velocity field for 2D Flow Matching."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half_dim = dim // 2
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(half_dim, device=t.device, dtype=t.dtype) / max(half_dim - 1, 1)
    )
    angles = t[:, None] * frequencies[None, :]
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class VelocityMLP(nn.Module):
    """Predict v_theta(x_t, t) for 2D points."""

    def __init__(self, hidden_dim: int = 128, time_dim: int = 64, depth: int = 4) -> None:
        super().__init__()
        self.time_dim = time_dim
        layers: list[nn.Module] = []
        input_dim = 2 + time_dim
        for index in range(depth):
            layers.append(nn.Linear(input_dim if index == 0 else hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_features = sinusoidal_embedding(t, self.time_dim)
        return self.net(torch.cat([x, t_features], dim=1))
