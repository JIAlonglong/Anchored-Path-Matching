"""One-sided anchor-priority projection from APM Equation 10."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ProjectionStats:
    """Finite diagnostics for one independently routed modality group."""

    cosine: float
    conflict: float
    anchor_norm: float
    field_norm: float
    removed_norm: float
    post_dot: float


@torch.no_grad()
def project_conflicting_gradient(
    anchor: Sequence[Tensor | None], field: Sequence[Tensor | None]
) -> tuple[list[Tensor | None], ProjectionStats]:
    """Remove only the field component that conflicts with the anchor."""
    if len(anchor) != len(field):
        raise ValueError("anchor and field gradient sequences must have equal length")
    reference = next(
        (value for value in (*anchor, *field) if value is not None), None
    )
    if reference is None:
        return [None] * len(anchor), ProjectionStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    zero = reference.new_zeros((), dtype=torch.float32)
    dot = zero.clone()
    anchor_sq = zero.clone()
    field_sq = zero.clone()
    for anchor_gradient, field_gradient in zip(anchor, field, strict=True):
        if anchor_gradient is not None:
            anchor_sq = anchor_sq + anchor_gradient.float().square().sum()
        if field_gradient is not None:
            field_sq = field_sq + field_gradient.float().square().sum()
        if anchor_gradient is not None and field_gradient is not None:
            if anchor_gradient.shape != field_gradient.shape:
                raise ValueError("paired anchor and field gradients must match shape")
            dot = dot + (anchor_gradient.float() * field_gradient.float()).sum()
    if not torch.isfinite(torch.stack((dot, anchor_sq, field_sq))).all().item():
        raise FloatingPointError("non-finite anchor/field gradient statistics")

    coefficient = (
        dot.clamp(max=0.0) / anchor_sq.clamp_min(torch.finfo(torch.float32).tiny)
        if anchor_sq.item() > 0.0
        else zero
    )
    corrected: list[Tensor | None] = []
    for anchor_gradient, field_gradient in zip(anchor, field, strict=True):
        if field_gradient is None:
            corrected.append(
                None if anchor_gradient is None else torch.zeros_like(anchor_gradient)
            )
            continue
        value = field_gradient.clone()
        if anchor_gradient is not None:
            value = value - coefficient.to(value.dtype) * anchor_gradient
        corrected.append(value)

    post_dot = zero.clone()
    for anchor_gradient, corrected_gradient in zip(anchor, corrected, strict=True):
        if anchor_gradient is not None and corrected_gradient is not None:
            post_dot = post_dot + (
                anchor_gradient.float() * corrected_gradient.float()
            ).sum()
    denominator = (anchor_sq * field_sq).sqrt()
    cosine = dot / denominator if denominator.item() > 0.0 else zero
    return corrected, ProjectionStats(
        cosine=float(cosine),
        conflict=float(dot.item() < 0.0),
        anchor_norm=float(anchor_sq.sqrt()),
        field_norm=float(field_sq.sqrt()),
        removed_norm=float(coefficient.abs() * anchor_sq.sqrt()),
        post_dot=float(post_dot),
    )
