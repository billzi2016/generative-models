"""2D target distributions for Flow Matching visualization."""

from __future__ import annotations

import math

import torch


def sample_ring(count: int, device: torch.device, noise: float = 0.06) -> torch.Tensor:
    """Sample a noisy ring centered at the origin."""
    theta = torch.rand(count, device=device) * (2 * math.pi)
    radius = 1.6 + noise * torch.randn(count, device=device)
    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)
    return torch.stack([x, y], dim=1)


def sample_moons(count: int, device: torch.device, noise: float = 0.06) -> torch.Tensor:
    """Sample two interleaving half circles."""
    half = count // 2
    rest = count - half

    theta_a = torch.rand(half, device=device) * math.pi
    moon_a = torch.stack([torch.cos(theta_a), torch.sin(theta_a)], dim=1)

    theta_b = torch.rand(rest, device=device) * math.pi
    moon_b = torch.stack([1.0 - torch.cos(theta_b), -torch.sin(theta_b) - 0.5], dim=1)

    points = torch.cat([moon_a, moon_b], dim=0)
    points = points + noise * torch.randn_like(points)
    points = points * 1.35
    points[:, 0] -= 0.7
    points[:, 1] += 0.35
    return points


def sample_distribution(name: str, count: int, device: torch.device) -> torch.Tensor:
    """Dispatch distribution sampler by name."""
    if name == "ring":
        return sample_ring(count, device)
    if name == "moons":
        return sample_moons(count, device)
    raise ValueError(f"unknown distribution: {name}")
