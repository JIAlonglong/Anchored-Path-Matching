from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tools.release.audit_checkpoints import PAPER_STAGE1_BASE_SHA256, audit_candidates
from tools.release.checkpoint_schema import CheckpointRecord, CheckpointSemantics


def _record(**changes) -> CheckpointRecord:
    semantics = CheckpointSemantics(
        stage=2,
        step=5000,
        data_domains=("Clean",),
        text_dropout_probability=0.1,
        teacher_cfg=(5.0, 1.0),
        deployment_cfg=(5.0, 1.0),
        base_identity=PAPER_STAGE1_BASE_SHA256,
        topology="frozen-stage1-routed-ffn-time-v1:r64/r64/t128",
        anchor_source="restart",
        field_enabled=True,
        field_target="instantaneous",
        projection=True,
        objective="apm",
    )
    semantics = replace(semantics, **changes)
    return CheckpointRecord(
        candidate_id="display-name-says-apm",
        requested_role="apm",
        checkpoint_dir=Path("/private/source"),
        weight_path=Path("/private/source/target_student.safetensors"),
        source_sha256="1" * 64,
        sidecar_sha256={},
        resolved_config={},
        metadata={"stage": 2, "step": 5000},
        contracts={},
        semantics=semantics,
    )


def test_only_semantically_matching_apm_is_exact() -> None:
    inventory = audit_candidates([_record()])
    audit = inventory.audits[0]
    assert audit.paper_status == "exact"
    assert audit.reasons == ()


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"field_target": "finite_local"}, "field_target expected instantaneous"),
        ({"projection": False}, "projection expected true"),
        ({"data_domains": ("Clean", "Random")}, "data_domains expected Clean-only"),
        ({"teacher_cfg": (1.0, 1.0)}, "teacher_cfg expected 5/1"),
        ({"base_identity": "other"}, "base_identity expected paper Stage I EMA"),
    ],
)
def test_apm_display_name_cannot_override_semantic_mismatch(change, reason) -> None:
    audit = audit_candidates([_record(**change)]).audits[0]
    assert audit.paper_status == "unresolved"
    assert any(reason in item for item in audit.reasons)
