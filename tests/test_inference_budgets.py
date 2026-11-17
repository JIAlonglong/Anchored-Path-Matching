from __future__ import annotations

import pytest
import torch

from anchorstep.backends.mock import MockBackend
from anchorstep.interfaces import JointBatch, JointLatent
from anchorstep.robotwin.inference import rollout_student


@pytest.mark.parametrize("steps", [1, 2, 4])
def test_inference_uses_exact_student_budget_and_no_teacher(steps: int) -> None:
    backend = MockBackend()
    batch = JointBatch(
        clean=JointLatent(torch.ones(2, 1), torch.ones(2, 1)),
        conditioning={},
        metadata={},
    )
    prior = JointLatent(torch.zeros(2, 1), torch.zeros(2, 1))
    backend.reset_call_counts()

    result = rollout_student(backend, prior, batch, steps)

    assert result.video.shape == prior.video.shape
    assert result.action.shape == prior.action.shape
    assert backend.student_call_count == steps
    assert backend.teacher_call_count == 0
