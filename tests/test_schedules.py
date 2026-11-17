from __future__ import annotations

import pytest
import torch

from anchorstep.core.schedules import build_time_grid


@pytest.mark.parametrize(
    ("steps", "expected"),
    [
        (1, [1.0, 0.0]),
        (2, [1.0, 0.5, 0.0]),
        (4, [1.0, 0.75, 0.5, 0.25, 0.0]),
    ],
)
def test_time_grid_matches_deployment_budget(steps: int, expected: list[float]) -> None:
    assert torch.equal(build_time_grid(steps), torch.tensor(expected))


def test_time_grid_rejects_unsupported_budget() -> None:
    with pytest.raises(ValueError, match="1, 2, 4"):
        build_time_grid(3)
