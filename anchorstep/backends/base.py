"""Abstract world-action-model boundary used by public objectives."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from torch import Generator, Tensor, nn

from anchorstep.interfaces import JointBatch, JointLatent


class WAMBackend(ABC):
    """Operations required from a concrete world-action-model backend."""

    @property
    @abstractmethod
    def student_module(self) -> nn.Module:
        """Return the online student module."""

    @abstractmethod
    def encode_batch(self, raw_batch: Mapping[str, Any]) -> JointBatch:
        """Encode a model-specific raw batch."""

    @abstractmethod
    def sample_prior(self, batch: JointBatch, generator: Generator) -> JointLatent:
        """Sample a joint prior using the supplied generator."""

    @abstractmethod
    def teacher_continuation(
        self, state: JointLatent, times: Tensor, batch: JointBatch
    ) -> list[JointLatent]:
        """Return frozen teacher states after every adjacent time transition."""

    @abstractmethod
    def student_transition(
        self,
        model: nn.Module,
        state: JointLatent,
        t: Tensor,
        r: Tensor,
        batch: JointBatch,
    ) -> JointLatent:
        """Apply one student finite transition from ``t`` to ``r``."""

    @abstractmethod
    def teacher_direction(
        self, state: JointLatent, time: Tensor, batch: JointBatch
    ) -> JointLatent:
        """Query the frozen teacher's instantaneous direction field."""

    @abstractmethod
    def student_direction(
        self,
        model: nn.Module,
        state: JointLatent,
        time: Tensor,
        batch: JointBatch,
    ) -> JointLatent:
        """Query the student's instantaneous direction at equal clocks."""

    @abstractmethod
    def decode_prediction(
        self, state: JointLatent, batch: JointBatch
    ) -> Mapping[str, Any]:
        """Decode a latent state to backend-specific predictions."""
