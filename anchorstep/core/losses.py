"""Explicit modality losses used by Stage I and Stage II."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from anchorstep.interfaces import JointLatent


@dataclass(frozen=True)
class LossOutput:
    """Differentiable objective with named terms and detached metrics."""

    total: Tensor
    terms: dict[str, Tensor]
    metrics: dict[str, Tensor]


def modality_mse(
    prediction: JointLatent,
    target: JointLatent,
    *,
    video_weight: float,
    action_weight: float,
) -> LossOutput:
    """Reduce video and action errors separately before weighting."""
    video_mse = torch.mean((prediction.video.float() - target.video.float()) ** 2)
    action_mse = torch.mean((prediction.action.float() - target.action.float()) ** 2)
    video_weighted = float(video_weight) * video_mse
    action_weighted = float(action_weight) * action_mse
    return LossOutput(
        total=video_weighted + action_weighted,
        terms={
            "video_mse": video_mse,
            "action_mse": action_mse,
            "video_weighted": video_weighted,
            "action_weighted": action_weighted,
        },
        metrics={},
    )
