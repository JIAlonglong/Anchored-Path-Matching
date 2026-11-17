"""Small replicated-gradient synchronization boundary."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, distributed


def average_gradients(gradients: Sequence[Tensor | None]) -> list[Tensor | None]:
    """All-reduce replicated gradients, or clone them in a single process."""
    world_size = distributed.get_world_size() if distributed.is_initialized() else 1
    result: list[Tensor | None] = []
    for gradient in gradients:
        if gradient is None:
            result.append(None)
            continue
        value = gradient.detach().clone()
        if world_size > 1:
            distributed.all_reduce(value, op=distributed.ReduceOp.SUM)
            value.div_(world_size)
        result.append(value)
    return result
