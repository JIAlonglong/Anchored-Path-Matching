"""Bounded-memory safetensors materialization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file

from anchorstep.weights.safetensors_io import tensor_entries, tensor_state_digest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor(path: Path, key: str):
    entries = tensor_entries(path)
    entry = entries[key]
    with safe_open(entry.path, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def materialize(
    base: Path,
    artifact: Path,
    output_dir: Path,
    max_shard_bytes: int = 2_000_000_000,
) -> Path:
    """Apply exact replacement tensors and emit a standard sharded state."""
    if type(max_shard_bytes) is not int or max_shard_bytes <= 0:
        raise ValueError("max_shard_bytes must be a positive integer")
    artifact_dir = Path(artifact)
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    if tensor_state_digest(base) != manifest["base_digest"]:
        raise ValueError("base digest does not match artifact manifest")
    artifact_path = artifact_dir / manifest["artifact_file"]
    if _sha256(artifact_path) != manifest["artifact_sha256"]:
        raise ValueError("artifact checksum does not match manifest")
    base_entries = tensor_entries(base)
    artifact_entries = tensor_entries(artifact_path)
    if set(artifact_entries) != set(manifest["keys"]):
        raise ValueError("artifact tensor keys do not match manifest")
    logical_entries = dict(base_entries)
    logical_entries.update(artifact_entries)
    partitions: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for key in sorted(logical_entries):
        size = logical_entries[key].length
        if current and current_size + size > max_shard_bytes:
            partitions.append(current)
            current = []
            current_size = 0
        current.append(key)
        current_size += size
    if current:
        partitions.append(current)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    weight_map: dict[str, str] = {}
    total = len(partitions)
    for index, keys in enumerate(partitions, start=1):
        shard_name = f"model-{index:05d}-of-{total:05d}.safetensors"
        tensors = {
            key: _tensor(
                artifact_path if key in artifact_entries else Path(base), key
            )
            for key in keys
        }
        save_file(tensors, output / shard_name)
        weight_map.update({key: shard_name for key in keys})
    index_path = output / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "total_size": sum(entry.length for entry in logical_entries.values())
                },
                "weight_map": weight_map,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    digest = tensor_state_digest(index_path)
    if digest != manifest["source_digest"]:
        raise ValueError(
            f"materialized logical digest mismatch: {digest} != {manifest['source_digest']}"
        )
    return index_path
