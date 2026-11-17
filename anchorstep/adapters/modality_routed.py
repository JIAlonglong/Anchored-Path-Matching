"""Static video/action low-rank routing for LingBot-shaped transformers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class RoutedAdapterConfig:
    """Paper adapter ranks for modality FFNs and temporal projections."""

    ffn_video_rank: int = 64
    ffn_action_rank: int = 64
    temporal_rank: int = 128

    def __post_init__(self) -> None:
        for name, value in (
            ("ffn_video_rank", self.ffn_video_rank),
            ("ffn_action_rank", self.ffn_action_rank),
            ("temporal_rank", self.temporal_rank),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class AdapterTopology:
    """Serializable identity of the installed trainable parameter graph."""

    config: RoutedAdapterConfig
    video_parameter_names: frozenset[str]
    action_parameter_names: frozenset[str]
    adapted_module_names: tuple[str, ...]

    @property
    def trainable_parameter_names(self) -> frozenset[str]:
        return self.video_parameter_names | self.action_parameter_names


class RoutedLowRankAdapter(nn.Module):
    """Frozen linear projection plus a zero-initialized low-rank residual."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        modality: Literal["video", "action"],
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("base must be an nn.Linear")
        self.base = base
        self.rank = rank
        self.modality = modality
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.adapter_down = nn.Linear(
            base.in_features,
            rank,
            bias=False,
            device=base.weight.device,
            dtype=torch.float32,
        )
        self.adapter_up = nn.Linear(
            rank,
            base.out_features,
            bias=False,
            device=base.weight.device,
            dtype=torch.float32,
        )
        nn.init.kaiming_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_up.weight)

    def forward(self, inputs: Tensor) -> Tensor:
        base = self.base(inputs)
        residual = self.adapter_up(self.adapter_down(inputs.float()))
        return base + residual.to(dtype=base.dtype)


def _adapter_target(name: str) -> tuple[int, Literal["video", "action"]] | None:
    normalized = name.lower()
    if "attention" in normalized or "attn" in normalized:
        return None
    if "temporal" in normalized or "time_projection" in normalized:
        return 128, "video"
    if "ffn" in normalized and "action" in normalized:
        return 64, "action"
    if "ffn" in normalized and "video" in normalized:
        return 64, "video"
    return None


def _replace_module(model: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, _, name = path.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    if name.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
        parent[int(name)] = replacement
    else:
        setattr(parent, name, replacement)


def parameter_modality(name: str) -> Literal["video", "action"]:
    """Classify one trainable adapter name without ambiguous fallbacks."""
    normalized = name.lower()
    is_adapter = ".adapter_down." in normalized or ".adapter_up." in normalized
    if is_adapter and "action" in normalized:
        return "action"
    if is_adapter and ("video" in normalized or "temporal" in normalized):
        return "video"
    raise ValueError(f"Unclassified trainable parameter: {name}")


def install_modality_routed_adapters(
    model: nn.Module, config: RoutedAdapterConfig
) -> AdapterTopology:
    """Freeze a model and install statically routed paper adapters."""
    if not isinstance(config, RoutedAdapterConfig):
        raise TypeError("config must be a RoutedAdapterConfig")
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    replacements: list[tuple[str, RoutedLowRankAdapter]] = []
    for name, module in tuple(model.named_modules()):
        if not name or not isinstance(module, nn.Linear):
            continue
        target = _adapter_target(name)
        if target is None:
            continue
        default_rank, modality = target
        if "temporal" in name.lower() or "time_projection" in name.lower():
            rank = config.temporal_rank
        elif modality == "action":
            rank = config.ffn_action_rank
        else:
            rank = config.ffn_video_rank
        if default_rank <= 0:
            raise AssertionError("invalid built-in rank")
        replacements.append(
            (name, RoutedLowRankAdapter(module, rank=rank, modality=modality))
        )
    if not replacements:
        raise ValueError("no modality-routed FFN or temporal projections were found")
    for name, replacement in replacements:
        _replace_module(model, name, replacement)

    video: set[str] = set()
    action: set[str] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        modality = parameter_modality(name)
        (video if modality == "video" else action).add(name)
    if not video or not action:
        raise ValueError("routed topology must contain video and action parameters")
    return AdapterTopology(
        config=config,
        video_parameter_names=frozenset(video),
        action_parameter_names=frozenset(action),
        adapted_module_names=tuple(name for name, _ in replacements),
    )
