"""Direct Stage II gradient composition with anchor-priority routing."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from anchorstep.adapters.modality_routed import parameter_modality
from anchorstep.core.anchor_priority import project_conflicting_gradient
from anchorstep.core.anchored_path import APMObjectiveOutput, anchored_path_objective
from anchorstep.core.flow_map import stage1_objective
from anchorstep.training.distributed import average_gradients
from anchorstep.training.ema import update_ema

GradientReducer = Callable[
    [str, Sequence[str], Sequence[Tensor | None]], list[Tensor | None]
]
Objective = Callable[[Any, Any, Mapping[str, Any], torch.Generator], APMObjectiveOutput]


def _continued_stage1_objective(
    backend: Any,
    batch: Any,
    config: Mapping[str, Any],
    generator: torch.Generator,
) -> APMObjectiveOutput:
    output = stage1_objective(backend, batch, config, generator)
    zero = output.total * 0.0
    return APMObjectiveOutput(
        base=output.total,
        anchor_video=zero,
        anchor_action=zero,
        field_video=zero,
        field_action=zero,
        metrics=output.metrics,
    )


def _configured_objective(config: Mapping[str, Any]) -> Objective:
    ablation = config.get("ablation", {})
    if not isinstance(ablation, Mapping):
        raise ValueError("ablation must be a mapping")
    name = ablation.get("objective", "correction")
    if name == "correction":
        return anchored_path_objective
    if name == "continued-stage1":
        return _continued_stage1_objective
    raise ValueError("ablation.objective must be 'correction' or 'continued-stage1'")


def _add_gradients(*gradients: Tensor | None) -> Tensor | None:
    values = [value for value in gradients if value is not None]
    if not values:
        return None
    result = values[0].clone()
    for value in values[1:]:
        result.add_(value)
    return result


def _norm(gradients: Sequence[Tensor | None]) -> float:
    values = [value.float().square().sum() for value in gradients if value is not None]
    if not values:
        return 0.0
    return float(torch.stack(values).sum().sqrt())


class Trainer:
    """Accumulate, globally average, project, clip, and update routed adapters."""

    def __init__(
        self,
        *,
        backend: Any,
        optimizer: torch.optim.Optimizer,
        config: Mapping[str, Any],
        objective: Objective | None = None,
        gradient_reducer: GradientReducer | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        self.backend = backend
        self.optimizer = optimizer
        self.config = config
        self.objective = objective or _configured_objective(config)
        self.gradient_reducer = gradient_reducer
        self.generator = generator
        self.step = 0

    def _batch_generator(self, batch: Any) -> torch.Generator:
        if self.generator is None:
            clean = getattr(batch, "clean", None)
            video = getattr(clean, "video", None)
            device = video.device if isinstance(video, Tensor) else torch.device("cpu")
            self.generator = torch.Generator(device=device).manual_seed(0)
        return self.generator

    def _reduce(
        self,
        component: str,
        names: Sequence[str],
        gradients: Sequence[Tensor | None],
    ) -> list[Tensor | None]:
        if self.gradient_reducer is not None:
            return self.gradient_reducer(component, names, gradients)
        return average_gradients(gradients)

    def train_accumulated_step(
        self, raw_batches: Sequence[Mapping[str, Any]]
    ) -> dict[str, float]:
        """Perform one optimizer step from one or more raw microbatches."""
        if not raw_batches:
            raise ValueError("raw_batches must not be empty")
        model: nn.Module = self.backend.student_module
        named = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        if not named:
            raise ValueError("student has no trainable adapter parameters")
        names = [name for name, _ in named]
        parameters = [parameter for _, parameter in named]
        modalities = [parameter_modality(name) for name in names]
        totals: dict[str, list[Tensor | None]] = {
            component: [None] * len(parameters)
            for component in ("base", "anchor", "field")
        }
        objective_metrics: dict[str, float] = {}

        for raw_batch in raw_batches:
            batch = self.backend.encode_batch(raw_batch)
            output = self.objective(
                self.backend, batch, self.config, self._batch_generator(batch)
            )
            losses = {
                "base": output.base,
                "anchor": output.anchor_video + output.anchor_action,
                "field": output.field_video + output.field_action,
            }
            for component, loss in losses.items():
                gradients = torch.autograd.grad(
                    loss,
                    parameters,
                    retain_graph=component != "field",
                    allow_unused=True,
                )
                totals[component] = [
                    _add_gradients(existing, gradient)
                    for existing, gradient in zip(
                        totals[component], gradients, strict=True
                    )
                ]
            objective_metrics.update(
                {name: float(value.detach()) for name, value in output.metrics.items()}
            )

        batch_count = float(len(raw_batches))
        for component, gradients in totals.items():
            averaged = [
                None if gradient is None else gradient / batch_count
                for gradient in gradients
            ]
            reduced = self._reduce(component, names, averaged)
            if len(reduced) != len(parameters):
                raise ValueError("gradient reducer changed the parameter count")
            totals[component] = reduced

        metrics = dict(objective_metrics)
        apm = self.config.get("apm", {})
        if not isinstance(apm, Mapping):
            raise ValueError("apm must be a mapping")
        projection_enabled = bool(apm.get("anchor_priority_projection", True))
        corrected_field: list[Tensor | None] = [None] * len(parameters)
        for modality in ("video", "action"):
            indices = [
                index for index, value in enumerate(modalities) if value == modality
            ]
            base = [totals["base"][index] for index in indices]
            anchor = [totals["anchor"][index] for index in indices]
            field = [totals["field"][index] for index in indices]
            projected, stats = project_conflicting_gradient(anchor, field)
            corrected = projected if projection_enabled else field
            for index, gradient in zip(indices, corrected, strict=True):
                corrected_field[index] = gradient
            raw_dot = sum(
                float((anchor_gradient.float() * field_gradient.float()).sum())
                for anchor_gradient, field_gradient in zip(anchor, field, strict=True)
                if anchor_gradient is not None and field_gradient is not None
            )
            metrics[f"gradient/{modality}/base_norm"] = _norm(base)
            metrics[f"gradient/{modality}/anchor_norm"] = stats.anchor_norm
            metrics[f"gradient/{modality}/field_norm"] = stats.field_norm
            metrics[f"gradient/{modality}/cosine"] = stats.cosine
            metrics[f"gradient/{modality}/conflict"] = stats.conflict
            metrics[f"gradient/{modality}/removed_norm"] = (
                stats.removed_norm if projection_enabled else 0.0
            )
            metrics[f"gradient/{modality}/post_dot"] = (
                stats.post_dot if projection_enabled else raw_dot
            )
            metrics[f"gradient/{modality}/projection_enabled"] = float(
                projection_enabled
            )

        self.optimizer.zero_grad(set_to_none=True)
        for index, parameter in enumerate(parameters):
            parameter.grad = _add_gradients(
                totals["base"][index],
                totals["anchor"][index],
                corrected_field[index],
            )
        train = self.config.get("train", {})
        if not isinstance(train, Mapping):
            raise ValueError("train must be a mapping")
        clip_norm = float(train.get("gradient_clip_norm", math.inf))
        if not math.isfinite(clip_norm) and clip_norm != math.inf:
            raise ValueError("gradient_clip_norm must be positive or infinity")
        if clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        if math.isfinite(clip_norm):
            total_norm = nn.utils.clip_grad_norm_(parameters, clip_norm)
            metrics["gradient/total_norm_before_clip"] = float(total_norm)
        self.optimizer.step()

        decay = float(train.get("ema_decay", 0.999))
        update_ema(
            self.backend.target_student_module,
            self.backend.student_module,
            decay,
        )
        self.step += 1
        metrics["train/step"] = float(self.step)
        return metrics
