from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from anchorstep.core.anchored_path import APMObjectiveOutput
from anchorstep.core.losses import LossOutput
from anchorstep.training.trainer import Trainer


class AdapterBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapter_down = nn.Linear(1, 1, bias=False)
        self.adapter_up = nn.Linear(1, 1, bias=False)
        self.adapter_down.weight.requires_grad_(False)
        nn.init.ones_(self.adapter_down.weight)
        nn.init.ones_(self.adapter_up.weight)


class TinyRoutedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared_weight = nn.Parameter(torch.tensor(7.0), requires_grad=False)
        self.video_ffn = AdapterBranch()
        self.action_ffn = AdapterBranch()


class TinyBackend:
    def __init__(self) -> None:
        self.student_module = TinyRoutedModel()
        self.target_student_module = copy.deepcopy(self.student_module)

    def encode_batch(self, raw_batch: Mapping[str, Any]) -> Mapping[str, Any]:
        return raw_batch


def separated_objective(
    backend: TinyBackend,
    batch: Mapping[str, Any],
    config: Mapping[str, Any],
    generator: torch.Generator,
) -> APMObjectiveOutput:
    del batch, config, generator
    video = backend.student_module.video_ffn.adapter_up.weight.reshape(())
    action = backend.student_module.action_ffn.adapter_up.weight.reshape(())
    return APMObjectiveOutput(
        base=video + action,
        anchor_video=video,
        anchor_action=action,
        # Locally compatible with the anchor. The reducer below turns only the
        # globally averaged video field into a conflict.
        field_video=video,
        field_action=2.0 * action,
        metrics={},
    )


class RecordingReducer:
    def __init__(self) -> None:
        self.components: list[str] = []

    def __call__(
        self,
        component: str,
        names: Sequence[str],
        gradients: Sequence[Tensor | None],
    ) -> list[Tensor | None]:
        self.components.append(component)
        result: list[Tensor | None] = []
        for name, gradient in zip(names, gradients, strict=True):
            if gradient is None:
                result.append(None)
            elif component == "field" and "video" in name:
                result.append(-gradient)
            else:
                result.append(gradient.clone())
        return result


def test_stage2_projects_after_global_average_and_updates_only_adapters() -> None:
    backend = TinyBackend()
    reducer = RecordingReducer()
    optimizer = torch.optim.SGD(
        [
            parameter
            for parameter in backend.student_module.parameters()
            if parameter.requires_grad
        ],
        lr=0.1,
    )
    trainer = Trainer(
        backend=backend,
        optimizer=optimizer,
        config={"train": {"ema_decay": 0.0, "gradient_clip_norm": 100.0}},
        objective=separated_objective,
        gradient_reducer=reducer,
        generator=torch.Generator().manual_seed(0),
    )

    metrics = trainer.train_accumulated_step([{"sample": 1}, {"sample": 2}])

    assert reducer.components == ["base", "anchor", "field"]
    assert backend.student_module.video_ffn.adapter_up.weight.item() == pytest.approx(
        0.8
    )
    assert backend.student_module.action_ffn.adapter_up.weight.item() == pytest.approx(
        0.6
    )
    assert backend.student_module.shared_weight.item() == 7.0
    assert metrics["gradient/video/base_norm"] == pytest.approx(1.0)
    assert metrics["gradient/video/anchor_norm"] == pytest.approx(1.0)
    assert metrics["gradient/video/conflict"] == 1.0
    assert abs(metrics["gradient/video/post_dot"]) < 1e-7
    assert metrics["gradient/action/conflict"] == 0.0
    assert torch.equal(
        backend.target_student_module.video_ffn.adapter_up.weight,
        backend.student_module.video_ffn.adapter_up.weight,
    )
    assert torch.equal(
        backend.target_student_module.action_ffn.adapter_up.weight,
        backend.student_module.action_ffn.adapter_up.weight,
    )


def test_stage2_can_disable_anchor_priority_projection() -> None:
    backend = TinyBackend()
    reducer = RecordingReducer()
    optimizer = torch.optim.SGD(
        [
            parameter
            for parameter in backend.student_module.parameters()
            if parameter.requires_grad
        ],
        lr=0.1,
    )
    trainer = Trainer(
        backend=backend,
        optimizer=optimizer,
        config={
            "apm": {"anchor_priority_projection": False},
            "train": {"ema_decay": 0.0, "gradient_clip_norm": 100.0},
        },
        objective=separated_objective,
        gradient_reducer=reducer,
        generator=torch.Generator().manual_seed(0),
    )

    metrics = trainer.train_accumulated_step([{"sample": 1}])

    # video: base + anchor + conflicting field = 1 + 1 - 1
    assert backend.student_module.video_ffn.adapter_up.weight.item() == pytest.approx(
        0.9
    )
    assert metrics["gradient/video/conflict"] == 1.0
    assert metrics["gradient/video/removed_norm"] == 0.0
    assert metrics["gradient/video/projection_enabled"] == 0.0


def test_continued_ablation_routes_to_stage1_without_anchor_or_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = TinyBackend()
    optimizer = torch.optim.SGD(
        [
            parameter
            for parameter in backend.student_module.parameters()
            if parameter.requires_grad
        ],
        lr=0.1,
    )
    calls: list[str] = []

    def fake_stage1_objective(
        backend: TinyBackend,
        batch: Mapping[str, Any],
        config: Mapping[str, Any],
        generator: torch.Generator,
    ) -> LossOutput:
        del batch, config, generator
        calls.append("stage1")
        video = backend.student_module.video_ffn.adapter_up.weight.reshape(())
        action = backend.student_module.action_ffn.adapter_up.weight.reshape(())
        total = video + action
        return LossOutput(total=total, terms={}, metrics={"continued": total.detach()})

    monkeypatch.setattr(
        "anchorstep.training.trainer.stage1_objective", fake_stage1_objective
    )
    trainer = Trainer(
        backend=backend,
        optimizer=optimizer,
        config={
            "ablation": {"objective": "continued-stage1"},
            "train": {"ema_decay": 0.0, "gradient_clip_norm": 100.0},
        },
        generator=torch.Generator().manual_seed(0),
    )

    metrics = trainer.train_accumulated_step([{"sample": 1}])

    assert calls == ["stage1"]
    assert backend.student_module.video_ffn.adapter_up.weight.item() == pytest.approx(
        0.9
    )
    assert backend.student_module.action_ffn.adapter_up.weight.item() == pytest.approx(
        0.9
    )
    assert metrics["continued"] == pytest.approx(2.0)
    assert metrics["gradient/video/anchor_norm"] == 0.0
    assert metrics["gradient/video/field_norm"] == 0.0
