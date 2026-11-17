"""Classify and export exact changed-tensor artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Set
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from safetensors import safe_open
from safetensors.torch import save_file

from anchorstep.weights.safetensors_io import (
    entries_equal,
    tensor_entries,
    tensor_state_digest,
)


@dataclass(frozen=True)
class ArtifactPlan:
    base: Path
    variant: Path
    kind: Literal["adapter", "replacement"]
    changed_keys: tuple[str, ...]
    base_digest: str
    source_digest: str


@dataclass(frozen=True)
class ArtifactManifest:
    variant_id: str
    kind: str
    artifact_file: str
    artifact_sha256: str
    base_digest: str
    source_digest: str
    keys: dict[str, dict[str, Any]]
    paper_role: str
    topology: str
    cfg_contract: dict[str, Any]
    license: str = "Apache-2.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_artifact(
    base: Path, variant: Path, adapter_keys: Set[str]
) -> ArtifactPlan:
    """Use adapter form only when every changed/new key is allowlisted."""
    base_entries = tensor_entries(base)
    variant_entries = tensor_entries(variant)
    removed = sorted(set(base_entries) - set(variant_entries))
    if removed:
        raise ValueError(f"variant removed base tensor keys: {removed}")
    changed: list[str] = []
    for key in sorted(variant_entries):
        if key not in base_entries:
            changed.append(key)
            continue
        left, right = base_entries[key], variant_entries[key]
        if (left.dtype, left.shape) != (right.dtype, right.shape):
            raise ValueError(f"variant changed shape or dtype for {key}")
        if not entries_equal(left, right):
            changed.append(key)
    if not changed:
        raise ValueError("artifact would be empty; variant equals base")
    kind = "adapter" if set(changed) <= set(adapter_keys) else "replacement"
    return ArtifactPlan(
        base=Path(base),
        variant=Path(variant),
        kind=kind,
        changed_keys=tuple(changed),
        base_digest=tensor_state_digest(base),
        source_digest=tensor_state_digest(variant),
    )


def _load_selected(path: Path, keys: tuple[str, ...]):
    if path.name.endswith(".index.json"):
        entries = tensor_entries(path)
        result = {}
        handles: dict[Path, Any] = {}
        try:
            for key in keys:
                entry = entries[key]
                handle = handles.get(entry.path)
                if handle is None:
                    handle = safe_open(entry.path, framework="pt", device="cpu")
                    handles[entry.path] = handle
                result[key] = handle.get_tensor(key)
            return result
        finally:
            for handle in handles.values():
                handle.__exit__(None, None, None)
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {key: handle.get_tensor(key) for key in keys}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_artifact(
    plan: ArtifactPlan,
    output_dir: Path,
    *,
    variant_id: str,
    paper_role: str,
    topology: str,
    cfg_contract: Mapping[str, Any],
) -> ArtifactManifest:
    """Write changed tensors as exact replacement values plus an audit manifest."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    file_name = "adapter_model.safetensors" if plan.kind == "adapter" else "delta.safetensors"
    artifact_path = output / file_name
    tensors = _load_selected(plan.variant, plan.changed_keys)
    save_file(tensors, artifact_path)
    entries = tensor_entries(artifact_path)
    manifest = ArtifactManifest(
        variant_id=variant_id,
        kind=plan.kind,
        artifact_file=file_name,
        artifact_sha256=_file_sha256(artifact_path),
        base_digest=plan.base_digest,
        source_digest=plan.source_digest,
        keys={
            key: {"dtype": entries[key].dtype, "shape": list(entries[key].shape)}
            for key in plan.changed_keys
        },
        paper_role=paper_role,
        topology=topology,
        cfg_contract=dict(cfg_contract),
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
