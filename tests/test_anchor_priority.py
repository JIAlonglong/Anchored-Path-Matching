from __future__ import annotations

import math

import torch

from anchorstep.core.anchor_priority import project_conflicting_gradient


def test_conflicting_field_is_projected_orthogonal_to_anchor() -> None:
    anchor = [torch.tensor([1.0, 0.0])]
    field = [torch.tensor([-1.0, 2.0])]

    corrected, stats = project_conflicting_gradient(anchor, field)

    assert torch.allclose(corrected[0], torch.tensor([0.0, 2.0]))
    assert stats.conflict == 1.0
    assert abs(stats.post_dot) < 1e-7


def test_compatible_field_is_unchanged() -> None:
    field = [torch.tensor([1.0, 2.0])]
    corrected, stats = project_conflicting_gradient([torch.tensor([2.0, 0.0])], field)

    assert torch.equal(corrected[0], field[0])
    assert corrected[0] is not field[0]
    assert stats.conflict == 0.0


def test_zero_anchor_and_none_gradients_stay_finite_and_routed() -> None:
    field_only = torch.tensor([3.0])
    corrected, stats = project_conflicting_gradient(
        [torch.zeros(2), None, torch.tensor([1.0])],
        [torch.tensor([-2.0, 4.0]), field_only, None],
    )

    assert torch.equal(corrected[0], torch.tensor([-2.0, 4.0]))
    assert torch.equal(corrected[1], field_only)
    assert torch.equal(corrected[2], torch.zeros(1))
    assert all(
        math.isfinite(value)
        for value in (
            stats.cosine,
            stats.anchor_norm,
            stats.field_norm,
            stats.removed_norm,
            stats.post_dot,
        )
    )


def test_projection_calls_do_not_mix_modality_groups() -> None:
    video, _ = project_conflicting_gradient(
        [torch.tensor([1.0])], [torch.tensor([-1.0])]
    )
    action, _ = project_conflicting_gradient(
        [torch.tensor([1.0])], [torch.tensor([2.0])]
    )

    assert torch.equal(video[0], torch.tensor([0.0]))
    assert torch.equal(action[0], torch.tensor([2.0]))
