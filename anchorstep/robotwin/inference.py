"""Teacher-free one-, two-, and four-step student inference."""

from __future__ import annotations

from typing import Literal

import torch

from anchorstep.backends.base import WAMBackend
from anchorstep.core.schedules import build_time_grid
from anchorstep.interfaces import JointBatch, JointLatent


def rollout_student(
    backend: WAMBackend,
    state: JointLatent,
    batch: JointBatch,
    steps: Literal[1, 2, 4],
) -> JointLatent:
    """Run exactly the selected deployment budget without a teacher query."""
    grid = build_time_grid(steps).to(
        device=state.video.device, dtype=state.video.dtype
    )
    batch_size = state.video.shape[0]
    current = state
    with torch.no_grad():
        for index in range(steps):
            t = grid[index].expand(batch_size)
            r = grid[index + 1].expand(batch_size)
            current = backend.student_transition(
                backend.student_module, current, t, r, batch
            )
    return current
