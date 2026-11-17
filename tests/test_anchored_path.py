from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import Tensor, nn

from anchorstep.core.anchored_path import anchored_path_objective
from anchorstep.interfaces import JointBatch, JointLatent


@dataclass
class TeacherCall:
    state: JointLatent
    times: Tensor


@dataclass
class StudentCall:
    state: JointLatent
    t: Tensor
    r: Tensor


class RecordingAPMBackend:
    def __init__(self) -> None:
        self.student_module = nn.Linear(1, 1, bias=False)
        nn.init.constant_(self.student_module.weight, 0.2)
        self.teacher_calls: list[TeacherCall] = []
        self.student_calls: list[StudentCall] = []

    def sample_prior(
        self, batch: JointBatch, generator: torch.Generator
    ) -> JointLatent:
        del generator
        return JointLatent(
            torch.zeros_like(batch.clean.video), torch.zeros_like(batch.clean.action)
        )

    def teacher_continuation(
        self, state: JointLatent, times: Tensor, batch: JointBatch
    ) -> list[JointLatent]:
        del batch
        self.teacher_calls.append(TeacherCall(state, times.detach().clone()))
        start = times[:, :1]
        outputs: list[JointLatent] = []
        for time in times[:, 1:].unbind(dim=1):
            gap = start - time[:, None]
            outputs.append(JointLatent(state.video - gap, state.action + 2.0 * gap))
        return outputs

    def student_transition(
        self,
        model: nn.Module,
        state: JointLatent,
        t: Tensor,
        r: Tensor,
        batch: JointBatch,
    ) -> JointLatent:
        del batch
        self.student_calls.append(StudentCall(state, t.detach().clone(), r.detach().clone()))
        weight = next(model.parameters()).reshape(())
        gap = (t - r)[:, None]
        return JointLatent(state.video + gap * weight, state.action - gap * weight)


def _batch() -> JointBatch:
    return JointBatch(
        clean=JointLatent(torch.ones(1, 1), -torch.ones(1, 1)),
        conditioning={},
        metadata={},
    )


def _config(steps: int) -> dict[str, object]:
    return {
        "train": {"teacher_reference_steps": 8},
        "loss": {
            "video_weight": 1.0,
            "action_weight": 0.1,
            "anchor_video_weight": 1.0,
            "anchor_action_weight": 0.1,
            "field_video_weight": 1.0,
            "field_action_weight": 0.1,
        },
        "apm": {
            "rollout_steps": steps,
            "anchor_source": "restart",
            "field_target": "finite_local",
            "field_enabled": True,
        },
    }


@pytest.mark.parametrize(
    ("steps", "seed", "expected_state_index"),
    [(1, 0, 0), (2, 0, 1), (4, 11, 2)],
)
def test_apm_selects_documented_student_reached_state(
    steps: int, seed: int, expected_state_index: int
) -> None:
    backend = RecordingAPMBackend()

    output = anchored_path_objective(
        backend,
        _batch(),
        _config(steps),
        torch.Generator().manual_seed(seed),
    )

    assert output.metrics["apm/state_index"].item() == expected_state_index
    correction = [call for call in backend.teacher_calls if call.times[0, -1] == 0.0]
    assert len(correction) == 1
    assert correction[0].state.video.requires_grad is False
    assert correction[0].state.action.requires_grad is False


def test_apm_uses_one_continuation_for_same_state_local_and_endpoint_targets() -> None:
    backend = RecordingAPMBackend()

    output = anchored_path_objective(
        backend,
        _batch(),
        _config(2),
        torch.Generator().manual_seed(0),
    )

    correction = [call for call in backend.teacher_calls if call.times[0, -1] == 0.0]
    assert len(correction) == 1
    endpoint_call, local_call = backend.student_calls[-2:]
    assert endpoint_call.state.video.data_ptr() == local_call.state.video.data_ptr()
    assert endpoint_call.state.action.data_ptr() == local_call.state.action.data_ptr()
    assert endpoint_call.r.item() == 0.0
    assert 0.0 < local_call.r.item() < endpoint_call.t.item()
    assert output.anchor_video.requires_grad
    assert output.anchor_action.requires_grad
    assert output.field_video.requires_grad
    assert output.field_action.requires_grad
