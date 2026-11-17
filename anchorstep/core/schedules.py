"""Deployment and teacher-integration time grids."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor


def build_time_grid(steps: Literal[1, 2, 4]) -> Tensor:
    """Return the inclusive descending deployment grid."""
    if type(steps) is not int or steps not in {1, 2, 4}:
        raise ValueError("steps must be one of {1, 2, 4}")
    return torch.linspace(1.0, 0.0, steps + 1)


def sample_two_times(
    batch_size: int, device: torch.device, generator: torch.Generator
) -> tuple[Tensor, Tensor]:
    """Sample per-example finite intervals with ``1 >= t > r >= 0``."""
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    t = torch.rand(batch_size, device=device, generator=generator)
    epsilon = torch.finfo(t.dtype).eps
    t = t * (1.0 - epsilon) + epsilon
    r = torch.rand(batch_size, device=device, generator=generator) * t
    return t, r


def interval_grid(t: Tensor, r: Tensor, steps: int) -> Tensor:
    """Interpolate a per-example interval into ``steps`` teacher transitions."""
    if t.shape != r.shape or t.ndim != 1:
        raise ValueError("t and r must be matching rank-one tensors")
    if type(steps) is not int or steps <= 0:
        raise ValueError("steps must be a positive integer")
    fractions = torch.linspace(0.0, 1.0, steps + 1, device=t.device, dtype=t.dtype)
    return t[:, None] + (r - t)[:, None] * fractions[None, :]
