"""Discover immutable EMA weights and adjacent semantic evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

try:
    from tools.release.checkpoint_schema import (
        CandidateSpec,
        CheckpointRecord,
        CheckpointSemantics,
    )
except ModuleNotFoundError:  # Direct execution through audit_checkpoints.py.
    from checkpoint_schema import CandidateSpec, CheckpointRecord, CheckpointSemantics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"sidecar must be a JSON object: {path.name}")
    return value


def _config_path(spec: CandidateSpec) -> Path | None:
    if spec.config_path is not None:
        return spec.config_path
    for candidate in (
        spec.checkpoint_dir / "config.yaml",
        spec.checkpoint_dir.parent / "config.yaml",
    ):
        if candidate.is_file():
            return candidate
    return None


def _topology(config: dict[str, Any], contract: dict[str, Any]) -> str | None:
    routed = config.get("routed_stage2")
    if isinstance(routed, dict):
        return (
            f"{routed.get('version')}:r{routed.get('ffn_video_rank')}"
            f"/r{routed.get('ffn_action_rank')}/t{routed.get('time_rank')}"
        )
    lora = contract.get("distillation_lora")
    if isinstance(lora, dict):
        return f"{lora.get('topology')}:rank{lora.get('rank')}"
    return None


def _extract_semantics(
    metadata: dict[str, Any],
    protocol: dict[str, Any],
    contract: dict[str, Any],
    config: dict[str, Any],
) -> CheckpointSemantics:
    train = config.get("train", {}) if isinstance(config.get("train", {}), dict) else {}
    loss_weights = train.get("loss_weights", {}) if isinstance(train.get("loss_weights", {}), dict) else {}
    guidance = protocol.get("guidance")
    if not isinstance(guidance, dict):
        guidance = contract.get("guidance_contract", {})
    teacher_cfg = None
    deployment_cfg = None
    dropout = None
    if isinstance(guidance, dict):
        try:
            teacher_cfg = (
                float(guidance["teacher_target_video_cfg"]),
                float(guidance["teacher_target_action_cfg"]),
            )
            deployment_cfg = (
                float(guidance["deployment_video_cfg"]),
                float(guidance["deployment_action_cfg"]),
            )
            dropout = float(guidance["text_dropout_probability"])
        except (KeyError, TypeError, ValueError):
            pass
    initialization = protocol.get("initialization")
    base_identity = (
        initialization.get("selected_sha256")
        if isinstance(initialization, dict)
        else None
    )
    raw_anchor = train.get("apm_anchor_source")
    anchor_source = {
        "reached_state": "restart",
        "restart": "restart",
        "original_prior": "original",
        "original": "original",
    }.get(raw_anchor)
    field_weights = (
        loss_weights.get("apm_field_video"),
        loss_weights.get("apm_field_action"),
    )
    field_enabled = None
    if any(value is not None for value in field_weights):
        field_enabled = any(float(value or 0.0) > 0.0 for value in field_weights)
    anchor_priority = config.get("anchor_priority")
    projection = None
    if field_enabled is not None:
        projection = bool(
            field_enabled
            and isinstance(anchor_priority, dict)
            and anchor_priority.get("enabled", True)
        )
    return CheckpointSemantics(
        stage=int(metadata["stage"]) if "stage" in metadata else None,
        step=int(metadata["step"]) if "step" in metadata else None,
        data_domains=None,
        text_dropout_probability=dropout,
        teacher_cfg=teacher_cfg,
        deployment_cfg=deployment_cfg,
        base_identity=base_identity,
        topology=_topology(config, contract),
        anchor_source=anchor_source,
        field_enabled=field_enabled,
        field_target=(str(train["apm_field_target"]) if "apm_field_target" in train else None),
        projection=projection,
        objective=("continued" if metadata.get("stage") == 1 else "apm"),
    )


def _apply_evidence(
    semantics: CheckpointSemantics, evidence: dict[str, Any]
) -> CheckpointSemantics:
    updates: dict[str, Any] = {}
    for key, raw_value in evidence.items():
        if not hasattr(semantics, key):
            raise ValueError(f"unknown semantic evidence field: {key}")
        value = tuple(raw_value) if key in {"data_domains", "teacher_cfg", "deployment_cfg"} else raw_value
        current = getattr(semantics, key)
        if current is not None and current != value:
            raise ValueError(
                f"semantic evidence conflicts with resolved config for {key}: "
                f"{current!r} != {value!r}"
            )
        updates[key] = value
    return replace(semantics, **updates)


def discover_candidate(spec: CandidateSpec) -> CheckpointRecord:
    directory = spec.checkpoint_dir
    weight = directory / "target_student.safetensors"
    if not weight.is_file():
        raise ValueError(f"missing target_student.safetensors in {directory}")
    source_sha256 = _sha256(weight)
    if spec.expected_sha256 is not None and source_sha256 != spec.expected_sha256:
        raise ValueError(
            f"target weight hash mismatch for {spec.candidate_id}: "
            f"expected {spec.expected_sha256}, found {source_sha256}"
        )
    required = {
        "metadata.json": directory / "metadata.json",
        "budget_protocol.json": directory / "budget_protocol.json",
        "robotwin_contract.json": directory / "robotwin_contract.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise ValueError(f"incomplete checkpoint metadata: missing {missing}")
    metadata = _json(required["metadata.json"])
    if not {"stage", "step", "config_sha256"} <= metadata.keys():
        raise ValueError("metadata.json is incomplete")
    protocol = _json(required["budget_protocol.json"])
    contract = _json(required["robotwin_contract.json"])
    config_path = _config_path(spec)
    config: dict[str, Any] = {}
    sidecars = dict(required)
    if config_path is not None:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("resolved config must be a YAML mapping")
        config = loaded
        sidecars["config.yaml"] = config_path
    inference = directory / "inference_export.json"
    contracts: dict[str, Any] = {"budget_protocol": protocol, "robotwin_contract": contract}
    if inference.is_file():
        contracts["inference_export"] = _json(inference)
        sidecars["inference_export.json"] = inference
        declared = contracts["inference_export"].get("weights_sha256")
        if declared is not None and declared != source_sha256:
            raise ValueError("inference export weight hash mismatch")
    semantics = _apply_evidence(
        _extract_semantics(metadata, protocol, contract, config),
        spec.semantic_evidence,
    )
    return CheckpointRecord(
        candidate_id=spec.candidate_id,
        requested_role=spec.requested_role,
        checkpoint_dir=directory,
        weight_path=weight,
        source_sha256=source_sha256,
        sidecar_sha256={name: _sha256(path) for name, path in sidecars.items()},
        resolved_config=config,
        metadata=metadata,
        contracts=contracts,
        semantics=semantics,
    )


def discover_candidates(specs: list[CandidateSpec]) -> list[CheckpointRecord]:
    records = [discover_candidate(spec) for spec in specs]
    seen: dict[str, str] = {}
    for record in records:
        previous = seen.get(record.source_sha256)
        if previous is not None:
            raise ValueError(
                f"duplicate source identity {record.source_sha256}: "
                f"{previous}, {record.candidate_id}"
            )
        seen[record.source_sha256] = record.candidate_id
    return records
