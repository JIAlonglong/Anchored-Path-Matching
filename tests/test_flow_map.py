from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from anchorstep.core.flow_map import stage1_objective
from anchorstep.interfaces import JointBatch, JointLatent


class RecordingBackend:
    def __init__(self) -> None:
        self.student_module = nn.Linear(1, 1, bias=False)
        nn.init.constant_(self.student_module.weight, 0.75)
        self.teacher_state: JointLatent | None = None
        self.student_state: JointLatent | None = None
        self.teacher_times: Tensor | None = None
        self.teacher_target_requires_grad: tuple[bool, bool] | None = None

    def sample_prior(
        self, batch: JointBatch, generator: torch.Generator
    ) -> JointLatent:
        del generator
        return JointLatent(
            torch.full_like(batch.clean.video, 2.0),
            torch.full_like(batch.clean.action, -1.0),
        )

    def teacher_continuation(
        self, state: JointLatent, times: Tensor, batch: JointBatch
    ) -> list[JointLatent]:
        del batch
        self.teacher_state = state
        self.teacher_times = times
        outputs = [
            JointLatent(state.video - time[:, None], state.action + time[:, None])
            for time in times[:, 1:].unbind(dim=1)
        ]
        endpoint = outputs[-1]
        self.teacher_target_requires_grad = (
            endpoint.video.requires_grad,
            endpoint.action.requires_grad,
        )
        return outputs

    def student_transition(
        self,
        model: nn.Module,
        state: JointLatent,
        t: Tensor,
        r: Tensor,
        batch: JointBatch,
    ) -> JointLatent:
        del t, r, batch
        self.student_state = state
        weight = next(model.parameters()).reshape(())
        return JointLatent(state.video * weight, state.action * weight)


def _batch() -> JointBatch:
    clean = JointLatent(
        torch.tensor([[0.0], [1.0]], requires_grad=True),
        torch.tensor([[0.5], [-0.5]], requires_grad=True),
    )
    return JointBatch(clean=clean, conditioning={}, metadata={})


def test_stage1_uses_same_state_and_detached_eight_step_teacher_target() -> None:
    backend = RecordingBackend()
    config: Mapping[str, Any] = {
        "train": {"teacher_reference_steps": 8},
        "loss": {"video_weight": 2.0, "action_weight": 0.25},
    }

    output = stage1_objective(
        backend, _batch(), config, torch.Generator().manual_seed(7)
    )

    assert backend.teacher_state is not None
    assert backend.student_state is not None
    assert torch.equal(backend.teacher_state.video, backend.student_state.video)
    assert torch.equal(backend.teacher_state.action, backend.student_state.action)
    assert backend.teacher_times is not None
    assert backend.teacher_times.shape == (2, 9)
    assert backend.teacher_target_requires_grad == (False, False)
    assert torch.equal(output.terms["video_weighted"], 2.0 * output.terms["video_mse"])
    assert torch.equal(
        output.terms["action_weighted"], 0.25 * output.terms["action_mse"]
    )
    output.total.backward()
    assert backend.student_module.weight.grad is not None
