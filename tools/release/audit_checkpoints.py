"""Audit discovered checkpoints against semantic paper-role predicates."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

try:
    from tools.release.checkpoint_schema import (
        CheckpointRecord,
        PaperRoleAudit,
        ReleaseInventory,
        load_candidate_specs,
    )
    from tools.release.discover_checkpoints import discover_candidates
except ModuleNotFoundError:  # Support the documented direct-script invocation.
    from checkpoint_schema import (
        CheckpointRecord,
        PaperRoleAudit,
        ReleaseInventory,
        load_candidate_specs,
    )
    from discover_checkpoints import discover_candidates


PAPER_STAGE1_BASE_SHA256 = "0408df97114d4be32f38d59e1be0b57455cd0f1c50c086140347df1fd88ed2bd"


ROLE_EXPECTATIONS: dict[str, dict[str, Any]] = {
    "shared": {"stage": 1, "step": 2000},
    "continued": {"stage": 1, "step": 4000},
    "original-no-field": {"stage": 2, "anchor_source": "original", "field_enabled": False},
    "original-field-projected": {
        "stage": 2,
        "anchor_source": "original",
        "field_enabled": True,
        "field_target": "finite_local",
        "projection": True,
    },
    "restart-no-field": {"stage": 2, "anchor_source": "restart", "field_enabled": False},
    "restart-field-no-projection": {
        "stage": 2,
        "anchor_source": "restart",
        "field_enabled": True,
        "field_target": "finite_local",
        "projection": False,
    },
    "apm": {
        "stage": 2,
        "anchor_source": "restart",
        "field_enabled": True,
        "field_target": "finite_local",
        "projection": True,
    },
}


def _common_reasons(record: CheckpointRecord) -> list[str]:
    semantic = record.semantics
    if semantic.stage != 2:
        return []
    reasons: list[str] = []
    if semantic.data_domains != ("Clean",):
        reasons.append(f"data_domains expected Clean-only, found {semantic.data_domains}")
    if semantic.text_dropout_probability != 0.1:
        reasons.append(
            "text_dropout_probability expected 0.1, "
            f"found {semantic.text_dropout_probability}"
        )
    if semantic.teacher_cfg != (5.0, 1.0):
        reasons.append(f"teacher_cfg expected 5/1, found {semantic.teacher_cfg}")
    if semantic.deployment_cfg != (5.0, 1.0):
        reasons.append(f"deployment_cfg expected 5/1, found {semantic.deployment_cfg}")
    if semantic.base_identity != PAPER_STAGE1_BASE_SHA256:
        reasons.append(
            "base_identity expected paper Stage I EMA "
            f"{PAPER_STAGE1_BASE_SHA256}, found {semantic.base_identity}"
        )
    expected_topology = "frozen-stage1-routed-ffn-time-v1:r64/r64/t128"
    if semantic.topology != expected_topology:
        reasons.append(f"topology expected {expected_topology}, found {semantic.topology}")
    return reasons


def _role_reasons(record: CheckpointRecord) -> list[str]:
    expected = ROLE_EXPECTATIONS[record.requested_role]
    reasons = _common_reasons(record)
    for key, value in expected.items():
        actual = getattr(record.semantics, key)
        if actual != value:
            rendered = str(value).lower() if isinstance(value, bool) else value
            reasons.append(f"{key} expected {rendered}, found {actual}")
    return reasons


def audit_candidates(candidates: Sequence[CheckpointRecord]) -> ReleaseInventory:
    audits: list[PaperRoleAudit] = []
    for record in candidates:
        if record.requested_role not in ROLE_EXPECTATIONS:
            status = "auxiliary"
            reasons = ("candidate is not assigned to a formal paper role",)
        else:
            role_reasons = _role_reasons(record)
            status = "exact" if not role_reasons else "unresolved"
            reasons = tuple(role_reasons)
        audits.append(
            PaperRoleAudit(
                candidate_id=record.candidate_id,
                requested_role=record.requested_role,
                paper_status=status,
                reasons=reasons,
                source_sha256=record.source_sha256,
                semantics=record.semantics,
            )
        )
    return ReleaseInventory(tuple(audits))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    specs = load_candidate_specs(args.candidates)
    records = discover_candidates(specs)
    inventory = audit_candidates(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(inventory.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
