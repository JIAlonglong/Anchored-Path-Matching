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


@dataclass
class DirectionCall:
    state: JointLatent
    time: Tensor


class RecordingAPMBackend:
    def __init__(self) -> None:
        self.student_module = nn.Linear(1, 1, bias=False)
        nn.init.constant_(self.student_module.weight, 0.2)
        self.teacher_calls: list[TeacherCall] = []
        self.student_calls: list[StudentCall] = []
        self.teacher_direction_calls: list[DirectionCall] = []
        self.student_direction_calls: list[DirectionCall] = []

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

    def teacher_direction(
        self, state: JointLatent, time: Tensor, batch: JointBatch
    ) -> JointLatent:
        del batch
        self.teacher_direction_calls.append(
            DirectionCall(state, time.detach().clone())
        )
        return JointLatent(
            torch.full_like(state.video, -1.0),
            torch.full_like(state.action, 2.0),
        )

    def student_direction(
        self,
        model: nn.Module,
        state: JointLatent,
        time: Tensor,
        batch: JointBatch,
    ) -> JointLatent:
        del batch
        self.student_direction_calls.append(
            DirectionCall(state, time.detach().clone())
        )
        weight = next(model.parameters()).reshape(())
        return JointLatent(
            torch.zeros_like(state.video) + weight,
            torch.zeros_like(state.action) - weight,
        )


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
            "field_target": "instantaneous",
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


def test_apm_matches_instantaneous_fields_at_restart_anchor_query_state() -> None:
    backend = RecordingAPMBackend()

    output = anchored_path_objective(
        backend,
        _batch(),
        _config(2),
        torch.Generator().manual_seed(0),
    )

    correction = [call for call in backend.teacher_calls if call.times[0, -1] == 0.0]
    assert len(correction) == 1
    endpoint_call = backend.student_calls[-1]
    assert len(backend.teacher_direction_calls) == 1
    assert len(backend.student_direction_calls) == 1
    teacher_field_call = backend.teacher_direction_calls[0]
    student_field_call = backend.student_direction_calls[0]
    assert endpoint_call.state.video.data_ptr() == teacher_field_call.state.video.data_ptr()
    assert endpoint_call.state.video.data_ptr() == student_field_call.state.video.data_ptr()
    assert endpoint_call.state.action.data_ptr() == teacher_field_call.state.action.data_ptr()
    assert endpoint_call.state.action.data_ptr() == student_field_call.state.action.data_ptr()
    assert endpoint_call.r.item() == 0.0
    assert torch.equal(teacher_field_call.time, endpoint_call.t)
    assert torch.equal(student_field_call.time, endpoint_call.t)
    assert teacher_field_call.state.video.requires_grad is False
    assert teacher_field_call.state.action.requires_grad is False
    assert output.anchor_video.requires_grad
    assert output.anchor_action.requires_grad
    assert output.field_video.requires_grad
    assert output.field_action.requires_grad
