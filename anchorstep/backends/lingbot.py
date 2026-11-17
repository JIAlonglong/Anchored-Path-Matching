"""Lazy compatibility boundary for an external LingBot-VA runtime.

LingBot-VA is intentionally not imported at module import time. A configured
runtime factory constructs the upstream teacher/student objects only after the
checkout and pinned revision have been validated.
"""

from __future__ import annotations

import importlib
import subprocess
from collections.abc import Mapping, Set
from pathlib import Path
from typing import Any

from torch import Generator, Tensor, nn

from anchorstep.interfaces import JointBatch, JointLatent

LINGBOT_CODE_REVISION = "58c2ae5bac46bd8114065bea9d7d256eb67c16c3"
LINGBOT_MODEL_REVISION = "8c9dea8abbc5c91cc9e18bc3264b8915083bbe70"


def verify_checkout_revision(path: Path, expected: str) -> str:
    """Return the checkout SHA or fail with both expected and actual values."""
    checkout = Path(path)
    try:
        actual = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"paths.lingbot_repo is not a readable Git checkout: {checkout}"
        ) from error
    if actual != expected:
        raise RuntimeError(
            f"LingBot revision mismatch: expected {expected}, actual {actual}"
        )
    return actual


def load_routed_adapter_state(
    model: nn.Module, state: Mapping[str, Tensor], expected_keys: Set[str]
) -> None:
    """Copy an adapter state only when its names and shapes match exactly."""
    missing = sorted(set(expected_keys) - set(state))
    unexpected = sorted(set(state) - set(expected_keys))
    if missing:
        raise ValueError(f"routed adapter state has missing keys: {missing}")
    if unexpected:
        raise ValueError(f"routed adapter state has unexpected keys: {unexpected}")
    live = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    absent_live = sorted(set(expected_keys) - set(live))
    if absent_live:
        raise ValueError(f"model topology is missing adapter keys: {absent_live}")
    for name in sorted(expected_keys):
        if live[name].shape != state[name].shape:
            raise ValueError(
                f"routed adapter shape mismatch for {name}: "
                f"expected {tuple(live[name].shape)}, got {tuple(state[name].shape)}"
            )
        live[name].data.copy_(
            state[name].to(device=live[name].device, dtype=live[name].dtype)
        )


class LingBotBackend:
    """Delegating backend around a lazily constructed upstream runtime."""

    def __init__(self, runtime: Any) -> None:
        required = (
            "student_module",
            "target_student_module",
            "encode_batch",
            "sample_prior",
            "teacher_continuation",
            "student_transition",
            "teacher_direction",
            "student_direction",
            "decode_prediction",
        )
        missing = [name for name in required if not hasattr(runtime, name)]
        if missing:
            raise TypeError(f"LingBot runtime is missing required members: {missing}")
        self._runtime = runtime

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> LingBotBackend:
        paths = config.get("paths")
        if not isinstance(paths, Mapping) or not paths.get("lingbot_repo"):
            raise RuntimeError("LingBot backend requires paths.lingbot_repo")
        checkout = Path(str(paths["lingbot_repo"])).expanduser()
        if not checkout.is_dir():
            raise RuntimeError(
                f"paths.lingbot_repo does not exist or is not a directory: {checkout}"
            )
        expected = str(
            config.get("compatibility", {}).get(
                "lingbot_code_revision", LINGBOT_CODE_REVISION
            )
        )
        verify_checkout_revision(checkout, expected)
        backend = config.get("backend", {})
        if not isinstance(backend, Mapping):
            raise RuntimeError("backend must be a mapping")
        factory_path = backend.get("runtime_factory")
        if not isinstance(factory_path, str) or ":" not in factory_path:
            raise RuntimeError(
                "LingBot backend requires backend.runtime_factory='module:function'; "
                "install the compatible external checkout first"
            )
        module_name, function_name = factory_path.split(":", 1)
        try:
            factory = getattr(importlib.import_module(module_name), function_name)
        except (ImportError, AttributeError) as error:
            raise RuntimeError(
                f"cannot import configured LingBot runtime factory {factory_path}"
            ) from error
        return cls(factory(config))

    @property
    def student_module(self) -> nn.Module:
        return self._runtime.student_module

    @property
    def target_student_module(self) -> nn.Module:
        return self._runtime.target_student_module

    def encode_batch(self, raw_batch: Mapping[str, Any]) -> JointBatch:
        return self._runtime.encode_batch(raw_batch)

    def sample_prior(self, batch: JointBatch, generator: Generator) -> JointLatent:
        return self._runtime.sample_prior(batch, generator)

    def teacher_continuation(
        self, state: JointLatent, times: Tensor, batch: JointBatch
    ) -> list[JointLatent]:
        return self._runtime.teacher_continuation(state, times, batch)

    def student_transition(
        self,
        model: nn.Module,
        state: JointLatent,
        t: Tensor,
        r: Tensor,
        batch: JointBatch,
    ) -> JointLatent:
        return self._runtime.student_transition(model, state, t, r, batch)

    def teacher_direction(
        self, state: JointLatent, time: Tensor, batch: JointBatch
    ) -> JointLatent:
        return self._runtime.teacher_direction(state, time, batch)

    def student_direction(
        self,
        model: nn.Module,
        state: JointLatent,
        time: Tensor,
        batch: JointBatch,
    ) -> JointLatent:
        return self._runtime.student_direction(model, state, time, batch)

    def decode_prediction(
        self, state: JointLatent, batch: JointBatch
    ) -> Mapping[str, Any]:
        return self._runtime.decode_prediction(state, batch)
