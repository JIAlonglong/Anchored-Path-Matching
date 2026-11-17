from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from anchorstep.weights.safetensors_io import tensor_entries, tensor_state_digest


def test_logical_digest_covers_values_dtype_and_shape(tmp_path: Path, monkeypatch) -> None:
    paths = {name: tmp_path / f"{name}.safetensors" for name in ("same", "value", "dtype", "shape")}
    save_file({"weight": torch.tensor([1.0, 2.0])}, paths["same"])
    save_file({"weight": torch.tensor([1.0, 3.0])}, paths["value"])
    save_file({"weight": torch.tensor([1.0, 2.0], dtype=torch.float64)}, paths["dtype"])
    save_file({"weight": torch.tensor([[1.0, 2.0]])}, paths["shape"])
    monkeypatch.setattr("safetensors.torch.load_file", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("full load forbidden")))

    digest = tensor_state_digest(paths["same"])

    assert tensor_state_digest(paths["same"]) == digest
    assert tensor_state_digest(paths["value"]) != digest
    assert tensor_state_digest(paths["dtype"]) != digest
    assert tensor_state_digest(paths["shape"]) != digest


def test_sharded_index_matches_single_file_digest(tmp_path: Path) -> None:
    single = tmp_path / "single.safetensors"
    save_file({"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}, single)
    shard_a = tmp_path / "model-00001-of-00002.safetensors"
    shard_b = tmp_path / "model-00002-of-00002.safetensors"
    save_file({"a": torch.tensor([1.0])}, shard_a)
    save_file({"b": torch.tensor([2.0])}, shard_b)
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(
        json.dumps(
            {
                "metadata": {"total_size": 8},
                "weight_map": {"a": shard_a.name, "b": shard_b.name},
            }
        )
    )

    assert tensor_state_digest(index) == tensor_state_digest(single)
    assert set(tensor_entries(index)) == {"a", "b"}


def test_sharded_index_rejects_missing_shard_and_duplicate_physical_key(tmp_path: Path) -> None:
    shard_a = tmp_path / "a.safetensors"
    shard_b = tmp_path / "b.safetensors"
    save_file({"weight": torch.tensor([1.0])}, shard_a)
    save_file({"weight": torch.tensor([2.0])}, shard_b)
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"weight": shard_a.name, "other": shard_b.name}}))
    with pytest.raises(ValueError, match="duplicate|weight_map"):
        tensor_entries(index)
    shard_b.unlink()
    with pytest.raises(ValueError, match="missing shard"):
        tensor_entries(index)
