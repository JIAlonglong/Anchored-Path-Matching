"""Typed backend-neutral contracts shared by APM objectives."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from torch import Tensor


@dataclass(frozen=True)
class JointLatent:
    """Paired video and action latent state."""

    video: Tensor
    action: Tensor

    def detach(self) -> "JointLatent":
        return JointLatent(self.video.detach(), self.action.detach())


@dataclass(frozen=True)
class JointBatch:
    """Encoded clean target with model-specific conditioning and metadata."""

    clean: JointLatent
    conditioning: Mapping[str, Any]
    metadata: Mapping[str, Any]
