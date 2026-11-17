"""Stage II restart-anchor and instantaneous-field path objective."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from anchorstep.backends.base import WAMBackend
from anchorstep.core.flow_map import stage1_objective
from anchorstep.core.schedules import build_time_grid, interval_grid
from anchorstep.interfaces import JointBatch, JointLatent


@dataclass(frozen=True)
class APMObjectiveOutput:
    """Separate loss groups retained for anchor-priority gradient capture."""

    base: Tensor
    anchor_video: Tensor
    anchor_action: Tensor
    field_video: Tensor
    field_action: Tensor
    metrics: Mapping[str, Tensor]

    @property
    def total(self) -> Tensor:
        return (
            self.base
            + self.anchor_video
            + self.anchor_action
            + self.field_video
            + self.field_action
        )


def _batch_time(value: Tensor, batch_size: int, reference: Tensor) -> Tensor:
    return value.to(device=reference.device, dtype=reference.dtype).expand(batch_size)


def _beta_five_two(generator: torch.Generator, device: torch.device) -> float:
    """Sample Beta(5,2) through its monotone polynomial CDF."""
    uniform = float(torch.rand((), generator=generator, device=device))
    low, high = 0.0, 1.0
    for _ in range(48):
        midpoint = (low + high) / 2.0
        cdf = 6.0 * midpoint**5 - 5.0 * midpoint**6
        if cdf < uniform:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def _correction_index(
    steps: int, generator: torch.Generator, device: torch.device
) -> int:
    if steps == 1:
        return 0
    if steps == 2:
        return 1
    sample = _beta_five_two(generator, device)
    return 1 + min(steps - 2, int(sample * (steps - 1)))


def _mse(prediction: Tensor, target: Tensor, weight: float) -> Tensor:
    return float(weight) * torch.mean((prediction.float() - target.float()) ** 2)


def _require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def anchored_path_objective(
    backend: WAMBackend,
    batch: JointBatch,
    config: Mapping[str, Any],
    generator: torch.Generator,
) -> APMObjectiveOutput:
    """Return retained Stage I, endpoint-anchor, and instantaneous-field losses.

    The correction prefix is always detached. A frozen teacher continuation
    provides the zero-noise restart endpoint, while raw teacher and student
    directions are matched at the exact same correction state and time.
    """
    apm = _require_mapping(config, "apm")
    loss = _require_mapping(config, "loss")
    train = _require_mapping(config, "train")
    steps = apm.get("rollout_steps")
    if type(steps) is not int or steps not in {1, 2, 4}:
        raise ValueError("apm.rollout_steps must be one of {1, 2, 4}")
    if apm.get("field_target", "instantaneous") != "instantaneous":
        raise ValueError("public APM requires an instantaneous field target")
    teacher_steps = int(train.get("teacher_reference_steps", 8))
    if teacher_steps != 8:
        raise ValueError("public APM uses eight teacher continuation steps")

    state_index = _correction_index(steps, generator, batch.clean.video.device)
    base = stage1_objective(backend, batch, config, generator).total
    prior = backend.sample_prior(batch, generator).detach()
    grid = build_time_grid(steps).to(
        device=prior.video.device, dtype=prior.video.dtype
    )
    batch_size = prior.video.shape[0]
    correction_state = prior
    for index in range(state_index):
        t = _batch_time(grid[index], batch_size, prior.video)
        r = _batch_time(grid[index + 1], batch_size, prior.video)
        with torch.no_grad():
            correction_state = backend.student_transition(
                backend.student_module, correction_state.detach(), t, r, batch
            )
        correction_state = correction_state.detach()

    state_t = _batch_time(grid[state_index], batch_size, prior.video)
    zero = torch.zeros_like(state_t)
    with torch.no_grad():
        continuation = backend.teacher_continuation(
            correction_state,
            interval_grid(state_t, zero, teacher_steps),
            batch,
        )
    if len(continuation) != teacher_steps:
        raise ValueError("teacher continuation must return eight target states")
    restart_endpoint = continuation[-1].detach()

    anchor_source = apm.get("anchor_source", "restart")
    if anchor_source == "restart":
        anchor_target = restart_endpoint
    elif anchor_source == "original":
        one = torch.ones_like(state_t)
        with torch.no_grad():
            original = backend.teacher_continuation(
                prior,
                interval_grid(one, zero, teacher_steps),
                batch,
            )
        anchor_target = original[-1].detach()
    else:
        raise ValueError("apm.anchor_source must be 'restart' or 'original'")

    predicted_endpoint = backend.student_transition(
        backend.student_module, correction_state, state_t, zero, batch
    )

    anchor_video = _mse(
        predicted_endpoint.video,
        anchor_target.video,
        float(loss["anchor_video_weight"]),
    )
    anchor_action = _mse(
        predicted_endpoint.action,
        anchor_target.action,
        float(loss["anchor_action_weight"]),
    )
    if bool(apm.get("field_enabled", True)):
        predicted_field = backend.student_direction(
            backend.student_module, correction_state, state_t, batch
        )
        with torch.no_grad():
            teacher_field = backend.teacher_direction(
                correction_state, state_t, batch
            ).detach()
        field_video = _mse(
            predicted_field.video,
            teacher_field.video,
            float(loss["field_video_weight"]),
        )
        field_action = _mse(
            predicted_field.action,
            teacher_field.action,
            float(loss["field_action_weight"]),
        )
    else:
        field_video = predicted_endpoint.video.sum() * 0.0
        field_action = predicted_endpoint.action.sum() * 0.0

    return APMObjectiveOutput(
        base=base,
        anchor_video=anchor_video,
        anchor_action=anchor_action,
        field_video=field_video,
        field_action=field_action,
        metrics={
            "apm/rollout_steps": torch.tensor(float(steps)),
            "apm/state_index": torch.tensor(float(state_index)),
            "apm/state_time": state_t.detach().mean(),
            "apm/field_time": state_t.detach().mean(),
        },
    )
