"""Typed records for checkpoint discovery and paper-role audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


def _redact_config(value: Any, key: str = "") -> Any:
    normalized = key.lower()
    if any(marker in normalized for marker in ("token", "password", "secret")):
        return "<redacted>"
    if isinstance(value, dict):
        return {
            str(child_key): _redact_config(child, str(child_key))
            for child_key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_config(child, key) for child in value]
    if isinstance(value, str) and value.startswith("/"):
        return "<redacted-path>"
    return value


PaperStatus = Literal["exact", "unresolved", "auxiliary"]


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    requested_role: str
    checkpoint_dir: Path
    expected_sha256: str | None
    semantic_evidence: dict[str, Any]
    config_path: Path | None = None


@dataclass(frozen=True)
class CheckpointSemantics:
    stage: int | None
    step: int | None
    data_domains: tuple[str, ...] | None
    text_dropout_probability: float | None
    teacher_cfg: tuple[float, float] | None
    deployment_cfg: tuple[float, float] | None
    base_identity: str | None
    topology: str | None
    anchor_source: str | None
    field_enabled: bool | None
    field_target: str | None
    projection: bool | None
    objective: str | None


@dataclass(frozen=True)
class CheckpointRecord:
    candidate_id: str
    requested_role: str
    checkpoint_dir: Path
    weight_path: Path
    source_sha256: str
    sidecar_sha256: dict[str, str]
    resolved_config: dict[str, Any]
    metadata: dict[str, Any]
    contracts: dict[str, Any]
    semantics: CheckpointSemantics

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "requested_role": self.requested_role,
            "source_ref": f"candidate://{self.candidate_id}/{self.weight_path.name}",
            "source_sha256": self.source_sha256,
            "sidecar_sha256": dict(sorted(self.sidecar_sha256.items())),
            "resolved_config": _redact_config(self.resolved_config),
            "metadata": _redact_config(self.metadata),
            "semantics": asdict(self.semantics),
        }


@dataclass(frozen=True)
class PaperRoleAudit:
    candidate_id: str
    requested_role: str
    paper_status: PaperStatus
    reasons: tuple[str, ...]
    source_sha256: str
    semantics: CheckpointSemantics

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["semantics"] = asdict(self.semantics)
        return payload


@dataclass(frozen=True)
class ReleaseInventory:
    audits: tuple[PaperRoleAudit, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "apm.release-inventory.v1",
            "audits": [item.to_dict() for item in self.audits],
        }


def load_candidate_specs(path: Path) -> list[CandidateSpec]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("candidates") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("candidate file must contain a candidates list")
    result: list[CandidateSpec] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("every candidate must be a mapping")
        result.append(
            CandidateSpec(
                candidate_id=str(row["id"]),
                requested_role=str(row["role"]),
                checkpoint_dir=Path(str(row["path"])),
                expected_sha256=(
                    None if row.get("sha256") is None else str(row["sha256"])
                ),
                semantic_evidence=dict(row.get("semantic_evidence", {})),
                config_path=(
                    None if row.get("config") is None else Path(str(row["config"]))
                ),
            )
        )
    return result
