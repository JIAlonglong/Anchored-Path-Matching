"""Configuration composition, redaction, and formal CFG validation."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass(frozen=True)
class GuidanceContract:
    """All guidance values needed to interpret a training or evaluation run."""

    text_dropout_probability: float
    teacher_video_cfg: float
    teacher_action_cfg: float
    deployment_video_cfg: float
    deployment_action_cfg: float
    target_guidance_mode: str

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "GuidanceContract":
        data = _mapping(config, "data")
        teacher = _mapping(config, "teacher_target_cfg")
        deployment = _mapping(config, "deployment_cfg")
        guidance = _mapping(config, "guidance")
        try:
            return cls(
                text_dropout_probability=float(data["text_dropout_probability"]),
                teacher_video_cfg=float(teacher["video"]),
                teacher_action_cfg=float(teacher["action"]),
                deployment_video_cfg=float(deployment["video"]),
                deployment_action_cfg=float(deployment["action"]),
                target_guidance_mode=str(guidance["target_guidance_mode"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "formal config requires text dropout and explicit teacher/deployment CFG"
            ) from error


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an explicit mapping")
    return value


def _deep_merge(left: dict[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(left)
    for key, value in right.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(
    paths: Sequence[Path], overrides: Sequence[str] = ()
) -> dict[str, Any]:
    """Compose YAML mappings in order and apply typed dotted-key overrides."""
    result: dict[str, Any] = {}
    for path in paths:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping):
            raise ValueError(f"config must contain a mapping: {path}")
        result = _deep_merge(result, loaded)
    for override in overrides:
        key, separator, raw_value = override.partition("=")
        if not separator or not key or any(not part for part in key.split(".")):
            raise ValueError(f"invalid override: {override}")
        value = yaml.safe_load(raw_value)
        cursor = result
        parts = key.split(".")
        for part in parts[:-1]:
            child = cursor.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(f"override crosses a non-mapping key: {key}")
            cursor = child
        cursor[parts[-1]] = value
    return result


def validate_config(
    config: Mapping[str, Any], mode: Literal["stage1", "stage2", "eval"]
) -> None:
    """Fail closed when a formal run has an ambiguous guidance contract."""
    if mode not in {"stage1", "stage2", "eval"}:
        raise ValueError("mode must be stage1, stage2, or eval")
    data = _mapping(config, "data")
    if "cfg_probability" in data and "text_dropout_probability" not in data:
        raise ValueError(
            "data.cfg_probability is not guidance; set text_dropout_probability explicitly"
        )
    contract = GuidanceContract.from_config(config)
    if not 0.0 <= contract.text_dropout_probability <= 1.0:
        raise ValueError("data.text_dropout_probability must be in [0, 1]")
    cfg_values = (
        contract.teacher_video_cfg,
        contract.teacher_action_cfg,
        contract.deployment_video_cfg,
        contract.deployment_action_cfg,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in cfg_values):
        raise ValueError("all teacher and deployment CFG values must be positive")
    if contract.target_guidance_mode not in {
        "inference_composed",
        "target_distilled",
    }:
        raise ValueError(
            "guidance.target_guidance_mode must be inference_composed or target_distilled"
        )
    if contract.target_guidance_mode == "target_distilled" and (
        contract.deployment_video_cfg != 1.0
        or contract.deployment_action_cfg != 1.0
    ):
        raise ValueError("target_distilled guidance requires deployment CFG 1/1")


def redact_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a serializable copy without credentials or absolute host paths."""
    def redact(value: Any, key: str = "") -> Any:
        normalized = key.lower()
        if any(marker in normalized for marker in ("token", "password", "secret")):
            return "<redacted>"
        if isinstance(value, Mapping):
            return {str(child_key): redact(child, str(child_key)) for child_key, child in value.items()}
        if isinstance(value, list):
            return [redact(child, key) for child in value]
        if isinstance(value, tuple):
            return [redact(child, key) for child in value]
        if isinstance(value, Path):
            value = str(value)
        if isinstance(value, str) and value.startswith("/"):
            return "<redacted-path>"
        return copy.deepcopy(value)

    return redact(config)
