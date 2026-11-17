"""Deterministic differentiable CPU backend for tests and smoke runs."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from anchorstep.adapters.modality_routed import AdapterTopology, RoutedAdapterConfig
from anchorstep.interfaces import JointBatch, JointLatent


class _MockBranch(nn.Module):
    def __init__(self, initial: float) -> None:
        super().__init__()
        self.adapter_up = nn.Linear(1, 1, bias=False)
        nn.init.constant_(self.adapter_up.weight, initial)


class MockStudent(nn.Module):
    """Two scalar modality adapters with an intentionally frozen base."""

    def __init__(self) -> None:
        super().__init__()
        self.base_scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.video_ffn = _MockBranch(0.25)
        self.action_ffn = _MockBranch(0.25)


class MockBackend:
    """A backend with deterministic teacher paths and real adapter gradients."""

    def __init__(self) -> None:
        self.student_module = MockStudent()
        self.target_student_module = copy.deepcopy(self.student_module)
        names = dict(self.student_module.named_parameters())
        self.topology = AdapterTopology(
            config=RoutedAdapterConfig(),
            video_parameter_names=frozenset({"video_ffn.adapter_up.weight"}),
            action_parameter_names=frozenset({"action_ffn.adapter_up.weight"}),
            adapted_module_names=("video_ffn", "action_ffn"),
        )
        if not self.topology.trainable_parameter_names <= names.keys():
            raise RuntimeError("mock adapter topology is inconsistent")
        self.student_call_count = 0
        self.teacher_call_count = 0

    def encode_batch(self, raw_batch: Mapping[str, Any] | JointBatch) -> JointBatch:
        if isinstance(raw_batch, JointBatch):
            return raw_batch
        try:
            video = raw_batch["video"]
            action = raw_batch["action"]
        except KeyError as error:
            raise ValueError("mock batch requires video and action tensors") from error
        if not isinstance(video, Tensor) or not isinstance(action, Tensor):
            raise TypeError("mock video and action must be tensors")
        return JointBatch(JointLatent(video, action), {}, {})

    def sample_prior(
        self, batch: JointBatch, generator: torch.Generator
    ) -> JointLatent:
        random_device = generator.device
        return JointLatent(
            torch.randn(
                batch.clean.video.shape,
                generator=generator,
                device=random_device,
                dtype=batch.clean.video.dtype,
            ).to(batch.clean.video.device),
            torch.randn(
                batch.clean.action.shape,
                generator=generator,
                device=random_device,
                dtype=batch.clean.action.dtype,
            ).to(batch.clean.action.device),
        )

    def teacher_continuation(
        self, state: JointLatent, times: Tensor, batch: JointBatch
    ) -> list[JointLatent]:
        self.teacher_call_count += 1
        if times.ndim != 2 or times.shape[1] < 2:
            raise ValueError("teacher times must be [batch, transitions + 1]")
        start = times[:, 0].clamp_min(torch.finfo(times.dtype).eps)
        outputs: list[JointLatent] = []
        for time in times[:, 1:].unbind(dim=1):
            video_fraction = (time / start).reshape(
                time.shape[0], *((1,) * (state.video.ndim - 1))
            )
            action_fraction = (time / start).reshape(
                time.shape[0], *((1,) * (state.action.ndim - 1))
            )
            outputs.append(
                JointLatent(
                    batch.clean.video
                    + video_fraction * (state.video - batch.clean.video),
                    batch.clean.action
                    + action_fraction * (state.action - batch.clean.action),
                )
            )
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
        self.student_call_count += 1
        if not isinstance(model, MockStudent):
            raise TypeError("mock backend requires MockStudent")
        video_gap = (t - r).reshape(t.shape[0], *((1,) * (state.video.ndim - 1)))
        action_gap = (t - r).reshape(t.shape[0], *((1,) * (state.action.ndim - 1)))
        video_weight = model.video_ffn.adapter_up.weight.reshape(())
        action_weight = model.action_ffn.adapter_up.weight.reshape(())
        return JointLatent(
            state.video - video_gap * video_weight,
            state.action - action_gap * action_weight,
        )

    def teacher_direction(
        self, state: JointLatent, time: Tensor, batch: JointBatch
    ) -> JointLatent:
        video_time = time.reshape(time.shape[0], *((1,) * (state.video.ndim - 1)))
        action_time = time.reshape(time.shape[0], *((1,) * (state.action.ndim - 1)))
        return JointLatent(
            (batch.clean.video - state.video) / video_time,
            (batch.clean.action - state.action) / action_time,
        )

    def student_direction(
        self,
        model: nn.Module,
        state: JointLatent,
        time: Tensor,
        batch: JointBatch,
    ) -> JointLatent:
        del time, batch
        if not isinstance(model, MockStudent):
            raise TypeError("mock backend requires MockStudent")
        video_weight = model.video_ffn.adapter_up.weight.reshape(())
        action_weight = model.action_ffn.adapter_up.weight.reshape(())
        return JointLatent(
            torch.zeros_like(state.video) - video_weight,
            torch.zeros_like(state.action) - action_weight,
        )

    def decode_prediction(
        self, state: JointLatent, batch: JointBatch
    ) -> Mapping[str, Any]:
        del batch
        return {"video": state.video, "action": state.action}

    def reset_call_counts(self) -> None:
        self.student_call_count = 0
        self.teacher_call_count = 0

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for parameter in self.student_module.parameters()
            if parameter.requires_grad
        ]

    def adapter_state(self, *, target: bool = False) -> dict[str, Tensor]:
        model = self.target_student_module if target else self.student_module
        parameters = dict(model.named_parameters())
        return {
            name: parameters[name].detach().cpu().clone()
            for name in self.topology.trainable_parameter_names
        }

    def load_adapter_state(
        self, state: Mapping[str, Tensor], *, target: bool = False
    ) -> None:
        if set(state) != self.topology.trainable_parameter_names:
            raise ValueError("mock adapter state has missing or unexpected keys")
        model = self.target_student_module if target else self.student_module
        parameters = dict(model.named_parameters())
        for name, value in state.items():
            if parameters[name].shape != value.shape:
                raise ValueError(f"mock adapter shape mismatch: {name}")
            parameters[name].data.copy_(value)

    def copy_online_to_target(self) -> None:
        self.target_student_module.load_state_dict(self.student_module.state_dict())
