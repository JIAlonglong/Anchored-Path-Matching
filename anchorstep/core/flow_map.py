"""Stage I arbitrary finite-interval flow-map distillation.

Adapted from the Apache-2.0 Flash-WAM finite-map objective at revision
``4cbecf208836739a86377c0efa4e7835d66b1dbb`` and re-expressed behind the
public :class:`~anchorstep.backends.base.WAMBackend` contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from anchorstep.backends.base import WAMBackend
from anchorstep.core.losses import LossOutput, modality_mse
from anchorstep.core.schedules import interval_grid, sample_two_times
from anchorstep.interfaces import JointBatch, JointLatent


def _per_sample(value: Tensor, reference: Tensor) -> Tensor:
    return value.reshape(value.shape[0], *((1,) * (reference.ndim - 1)))


def interpolate(prior: JointLatent, clean: JointLatent, time: Tensor) -> JointLatent:
    """Interpolate clean data at zero to the sampled prior at one."""
    video_time = _per_sample(time, clean.video)
    action_time = _per_sample(time, clean.action)
    return JointLatent(
        (1.0 - video_time) * clean.video + video_time * prior.video,
        (1.0 - action_time) * clean.action + action_time * prior.action,
    )


def stage1_objective(
    backend: WAMBackend,
    batch: JointBatch,
    config: Mapping[str, Any],
    generator: torch.Generator,
) -> LossOutput:
    """Distill one arbitrary finite interval from an eight-step teacher target."""
    batch_size = batch.clean.video.shape[0]
    if batch.clean.action.shape[0] != batch_size:
        raise ValueError("video and action batch sizes must match")
    train = config.get("train", {})
    loss = config.get("loss", {})
    teacher_steps = int(train.get("teacher_reference_steps", 8))
    if teacher_steps != 8:
        raise ValueError("the public Stage I contract uses eight teacher steps")
    try:
        video_weight = float(loss["video_weight"])
        action_weight = float(loss["action_weight"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("loss video_weight/action_weight must be explicit") from error

    prior = backend.sample_prior(batch, generator)
    t, r = sample_two_times(batch_size, batch.clean.video.device, generator)
    state_t = interpolate(prior, batch.clean, t)
    teacher_state = state_t.detach()
    with torch.no_grad():
        continuation = backend.teacher_continuation(
            teacher_state, interval_grid(t, r, teacher_steps), batch
        )
    if len(continuation) != teacher_steps:
        raise ValueError("teacher continuation must return one state per transition")
    target_r = continuation[-1].detach()
    prediction_r = backend.student_transition(
        backend.student_module, state_t, t, r, batch
    )
    output = modality_mse(
        prediction_r,
        target_r,
        video_weight=video_weight,
        action_weight=action_weight,
    )
    return LossOutput(
        total=output.total,
        terms=output.terms,
        metrics={
            "time/t_mean": t.detach().mean(),
            "time/r_mean": r.detach().mean(),
            "time/gap_mean": (t - r).detach().mean(),
        },
    )
