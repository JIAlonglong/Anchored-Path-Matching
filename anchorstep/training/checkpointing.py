"""Atomic adapter-aware online/EMA checkpoint contract."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from safetensors.torch import load_file, save_file
from torch import Tensor

from anchorstep.adapters.modality_routed import AdapterTopology
from anchorstep.config import GuidanceContract, redact_config


@dataclass(frozen=True)
class CheckpointBundle:
    """Validated adapter states and immutable run metadata."""

    online_state: dict[str, Tensor]
    ema_state: dict[str, Tensor]
    metadata: dict[str, Any]


def topology_record(topology: AdapterTopology) -> dict[str, Any]:
    """Convert adapter topology to a deterministic JSON record."""
    return {
        "config": asdict(topology.config),
        "video_parameter_names": sorted(topology.video_parameter_names),
        "action_parameter_names": sorted(topology.action_parameter_names),
        "adapted_module_names": list(topology.adapted_module_names),
    }


def _validate_state(
    state: dict[str, Tensor], topology: AdapterTopology, label: str
) -> dict[str, Tensor]:
    expected = topology.trainable_parameter_names
    if set(state) != expected:
        missing = sorted(expected - set(state))
        unexpected = sorted(set(state) - expected)
        raise ValueError(
            f"{label} adapter state does not match topology; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return {name: tensor.detach().cpu().contiguous() for name, tensor in state.items()}


def save_checkpoint(
    path: Path,
    *,
    online_state: dict[str, Tensor],
    ema_state: dict[str, Tensor],
    resolved_config: dict[str, Any],
    guidance: GuidanceContract,
    topology: AdapterTopology,
    stage: str,
    step: int,
    git_commit: str,
    source_checkpoint: dict[str, Any],
) -> None:
    """Write a complete checkpoint directory and publish it by atomic rename."""
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"checkpoint already exists: {destination}")
    if stage not in {"stage1", "stage2"}:
        raise ValueError("stage must be stage1 or stage2")
    if type(step) is not int or step < 0:
        raise ValueError("step must be a non-negative integer")
    online = _validate_state(online_state, topology, "online")
    ema = _validate_state(ema_state, topology, "EMA")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        save_file(online, temporary / "online_adapter.safetensors")
        save_file(ema, temporary / "ema_adapter.safetensors")
        metadata = {
            "format_version": 1,
            "state_scope": "adapter",
            "stage": stage,
            "step": step,
            "git_commit": git_commit,
            "source_checkpoint": redact_config(source_checkpoint),
            "resolved_config": redact_config(resolved_config),
            "guidance": asdict(guidance),
            "topology": topology_record(topology),
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / "COMPLETE").write_text("complete\n", encoding="utf-8")
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_checkpoint(
    path: Path,
    *,
    expected_stage: str | None = None,
    expected_topology: AdapterTopology | None = None,
) -> CheckpointBundle:
    """Load only a complete, correctly scoped, topology-compatible checkpoint."""
    directory = Path(path)
    required = {
        "online_adapter.safetensors",
        "ema_adapter.safetensors",
        "metadata.json",
        "COMPLETE",
    }
    missing = sorted(name for name in required if not (directory / name).is_file())
    if missing:
        raise ValueError(f"partial checkpoint; missing {missing}")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("state_scope") != "adapter":
        raise ValueError(
            "checkpoint state_scope must be adapter, not nested transformer"
        )
    if expected_stage is not None and metadata.get("stage") != expected_stage:
        raise ValueError(
            f"expected {expected_stage} checkpoint, found {metadata.get('stage')}"
        )
    if expected_topology is not None and metadata.get("topology") != topology_record(
        expected_topology
    ):
        raise ValueError("checkpoint topology does not match expected topology")
    online = load_file(directory / "online_adapter.safetensors", device="cpu")
    ema = load_file(directory / "ema_adapter.safetensors", device="cpu")
    topology = metadata.get("topology", {})
    expected_names = set(topology.get("video_parameter_names", ())) | set(
        topology.get("action_parameter_names", ())
    )
    if set(online) != expected_names or set(ema) != expected_names:
        raise ValueError("checkpoint tensor keys do not match recorded topology")
    return CheckpointBundle(dict(online), dict(ema), metadata)
