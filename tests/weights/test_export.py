from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from anchorstep.weights.export import classify_artifact, export_artifact


def _file(path: Path, **values: torch.Tensor) -> Path:
    save_file(values, path)
    return path


def test_classifies_adapter_only_when_all_changes_are_allowlisted(
    tmp_path: Path,
) -> None:
    base = _file(
        tmp_path / "base.safetensors",
        frozen=torch.tensor([1.0]),
        adapter=torch.tensor([0.0]),
    )
    adapter_variant = _file(
        tmp_path / "adapter.safetensors",
        frozen=torch.tensor([1.0]),
        adapter=torch.tensor([2.0]),
        new_adapter=torch.tensor([3.0]),
    )
    frozen_variant = _file(
        tmp_path / "frozen.safetensors",
        frozen=torch.tensor([9.0]),
        adapter=torch.tensor([2.0]),
    )

    adapter = classify_artifact(base, adapter_variant, {"adapter", "new_adapter"})
    replacement = classify_artifact(base, frozen_variant, {"adapter"})

    assert adapter.kind == "adapter"
    assert adapter.changed_keys == ("adapter", "new_adapter")
    assert replacement.kind == "replacement"
    assert replacement.changed_keys == ("adapter", "frozen")


def test_classification_fails_closed_on_empty_removed_or_incompatible_state(
    tmp_path: Path,
) -> None:
    base = _file(
        tmp_path / "base.safetensors", a=torch.tensor([1.0]), b=torch.tensor([2.0])
    )
    equal = _file(
        tmp_path / "equal.safetensors", a=torch.tensor([1.0]), b=torch.tensor([2.0])
    )
    removed = _file(tmp_path / "removed.safetensors", a=torch.tensor([1.0]))
    shape = _file(
        tmp_path / "shape.safetensors", a=torch.tensor([[1.0]]), b=torch.tensor([2.0])
    )

    with pytest.raises(ValueError, match="empty"):
        classify_artifact(base, equal, {"a"})
    with pytest.raises(ValueError, match="removed"):
        classify_artifact(base, removed, {"a"})
    with pytest.raises(ValueError, match="shape|dtype"):
        classify_artifact(base, shape, {"a"})


def test_export_writes_only_changed_tensors_and_auditable_manifest(
    tmp_path: Path,
) -> None:
    base = _file(
        tmp_path / "base.safetensors",
        frozen=torch.tensor([1.0]),
        adapter=torch.tensor([0.0]),
    )
    variant = _file(
        tmp_path / "variant.safetensors",
        frozen=torch.tensor([1.0]),
        adapter=torch.tensor([4.0]),
    )
    plan = classify_artifact(base, variant, {"adapter"})

    manifest = export_artifact(
        plan,
        tmp_path / "export",
        variant_id="apm-fixture",
        paper_role="apm",
        topology="fixture",
        cfg_contract={"teacher": [5, 1], "deployment": [5, 1]},
    )

    assert manifest.kind == "adapter"
    with safe_open(
        tmp_path / "export" / manifest.artifact_file, framework="pt"
    ) as handle:
        assert handle.keys() == ["adapter"]
        assert torch.equal(handle.get_tensor("adapter"), torch.tensor([4.0]))
    assert manifest.keys["adapter"]["shape"] == [1]
    assert manifest.license == "Apache-2.0"
