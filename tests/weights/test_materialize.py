from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from anchorstep.weights.export import classify_artifact, export_artifact
from anchorstep.weights.materialize import materialize
from anchorstep.weights.safetensors_io import tensor_state_digest
from anchorstep.weights.verify import verify_logical_state


def _setup(tmp_path: Path):
    base = tmp_path / "base.safetensors"
    source = tmp_path / "source.safetensors"
    save_file({"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}, base)
    save_file(
        {"a": torch.tensor([3.0]), "b": torch.tensor([2.0]), "c": torch.tensor([4.0])},
        source,
    )
    artifact = tmp_path / "artifact"
    export_artifact(
        classify_artifact(base, source, {"a", "c"}),
        artifact,
        variant_id="fixture",
        paper_role="apm",
        topology="fixture",
        cfg_contract={},
    )
    return base, source, artifact


def test_materialization_is_sharded_and_exact(tmp_path: Path) -> None:
    base, source, artifact = _setup(tmp_path)
    index = materialize(base, artifact, tmp_path / "materialized", max_shard_bytes=4)

    assert index.name == "model.safetensors.index.json"
    assert tensor_state_digest(index) == tensor_state_digest(source)
    report = verify_logical_state(base, artifact, source)
    assert report.ok
    assert report.expected_digest == report.materialized_digest
    payload = json.loads(index.read_text())
    assert len(set(payload["weight_map"].values())) == 3


def test_materialization_rejects_wrong_base_and_corrupted_artifact(
    tmp_path: Path,
) -> None:
    base, source, artifact = _setup(tmp_path)
    wrong = tmp_path / "wrong.safetensors"
    save_file({"a": torch.tensor([0.0]), "b": torch.tensor([2.0])}, wrong)
    with pytest.raises(ValueError, match="base digest"):
        materialize(wrong, artifact, tmp_path / "wrong-output")

    manifest = json.loads((artifact / "manifest.json").read_text())
    artifact_file = artifact / manifest["artifact_file"]
    data = bytearray(artifact_file.read_bytes())
    data[-1] ^= 1
    artifact_file.write_bytes(data)
    with pytest.raises(ValueError, match="artifact checksum"):
        materialize(base, artifact, tmp_path / "corrupt-output")
