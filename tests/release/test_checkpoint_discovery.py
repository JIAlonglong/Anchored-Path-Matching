from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from tools.release.checkpoint_schema import CandidateSpec
from tools.release.discover_checkpoints import discover_candidate, discover_candidates


def _tree(root: Path) -> tuple[Path, str]:
    checkpoint = root / "run/stage2/step_10"
    checkpoint.mkdir(parents=True)
    weight = checkpoint / "target_student.safetensors"
    weight.write_bytes(b"target-ema")
    (checkpoint / "online_student.safetensors").write_bytes(b"online")
    (checkpoint / "trainer_state.pt").write_bytes(b"trainer")
    (checkpoint / "recovery-rank0.pt").write_bytes(b"recovery")
    config = {
        "train": {
            "stage": 2,
            "apm_anchor_source": "reached_state",
            "apm_field_target": "finite_local",
            "loss_weights": {"apm_field_video": 1.0, "apm_field_action": 0.1},
        },
        "anchor_priority": {"version": "global-update-anchor-priority-v1"},
    }
    (checkpoint.parent / "config.yaml").write_text(yaml.safe_dump(config))
    (checkpoint / "metadata.json").write_text(
        json.dumps({"stage": 2, "step": 10, "config_sha256": "config-id"})
    )
    (checkpoint / "budget_protocol.json").write_text(
        json.dumps(
            {
                "guidance": {
                    "text_dropout_probability": 0.1,
                    "teacher_target_video_cfg": 5.0,
                    "teacher_target_action_cfg": 1.0,
                    "deployment_video_cfg": 5.0,
                    "deployment_action_cfg": 1.0,
                },
                "initialization": {"selected_sha256": "base-id"},
            }
        )
    )
    (checkpoint / "robotwin_contract.json").write_text(
        json.dumps(
            {"contract": {"version": "fixture"}, "distillation_lora": {"rank": 128}}
        )
    )
    return checkpoint, hashlib.sha256(b"target-ema").hexdigest()


def test_discovery_selects_target_and_redacts_source_root(tmp_path: Path) -> None:
    checkpoint, digest = _tree(tmp_path)
    record = discover_candidate(
        CandidateSpec(
            candidate_id="fixture-apm",
            requested_role="apm",
            checkpoint_dir=checkpoint,
            expected_sha256=digest,
            semantic_evidence={"data_domains": ["Clean"]},
        )
    )

    assert record.weight_path.name == "target_student.safetensors"
    assert record.source_sha256 == digest
    assert set(record.sidecar_sha256) == {
        "config.yaml",
        "metadata.json",
        "budget_protocol.json",
        "robotwin_contract.json",
    }
    public = record.to_public_dict()
    assert str(tmp_path) not in json.dumps(public)
    assert public["source_ref"] == "candidate://fixture-apm/target_student.safetensors"
    assert "online_student.safetensors" not in json.dumps(public)
    assert "trainer_state.pt" not in json.dumps(public)


def test_discovery_fails_on_hash_mismatch_missing_target_and_duplicate_identity(
    tmp_path: Path,
) -> None:
    checkpoint, digest = _tree(tmp_path)
    with pytest.raises(ValueError, match="hash mismatch"):
        discover_candidate(CandidateSpec("bad", "apm", checkpoint, "0" * 64, {}))
    (checkpoint / "target_student.safetensors").unlink()
    with pytest.raises(ValueError, match="target_student"):
        discover_candidate(CandidateSpec("missing", "apm", checkpoint, None, {}))

    other, _ = _tree(tmp_path / "other")
    first = CandidateSpec("one", "apm", other, digest, {"data_domains": ["Clean"]})
    second = CandidateSpec("two", "apm", other, digest, {"data_domains": ["Clean"]})
    with pytest.raises(ValueError, match="duplicate source identity"):
        discover_candidates([first, second])
