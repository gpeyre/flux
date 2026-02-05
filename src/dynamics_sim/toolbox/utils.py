"""Shared utility helpers for the simulation."""

import torch


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def random_positions(count: int, box_size: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.rand((count, 2), device=device, dtype=dtype) * box_size


def random_positions_in_bounds(
    count: int,
    min_corner: torch.Tensor,
    max_corner: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    span = (max_corner - min_corner).clamp(min=1e-6)
    return torch.rand((count, 2), device=device, dtype=dtype) * span + min_corner


def random_velocities(count: int, speed_scale: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return (torch.rand((count, 2), device=device, dtype=dtype) - 0.5) * speed_scale
