from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from anchorstep.adapters.modality_routed import AdapterTopology, RoutedAdapterConfig
from anchorstep.config import GuidanceContract
from anchorstep.training.checkpointing import load_checkpoint, save_checkpoint


def _topology(rank: int = 64) -> AdapterTopology:
    return AdapterTopology(
        config=RoutedAdapterConfig(
            ffn_video_rank=rank, ffn_action_rank=rank, temporal_rank=128
        ),
        video_parameter_names=frozenset({"block.video_ffn.adapter_up.weight"}),
        action_parameter_names=frozenset({"block.action_ffn.adapter_up.weight"}),
        adapted_module_names=("block.video_ffn", "block.action_ffn"),
    )


def _states() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    online = {
        "block.video_ffn.adapter_up.weight": torch.tensor([[1.0]]),
        "block.action_ffn.adapter_up.weight": torch.tensor([[2.0]]),
    }
    ema = {name: value + 0.5 for name, value in online.items()}
    return online, ema


def _guidance() -> GuidanceContract:
    return GuidanceContract(0.1, 5.0, 1.0, 5.0, 1.0, "inference_composed")


def _save(path: Path, *, topology: AdapterTopology | None = None) -> None:
    online, ema = _states()
    save_checkpoint(
        path,
        online_state=online,
        ema_state=ema,
        resolved_config={
            "paths": {"output_dir": "/private/run"},
            "train": {"stage": 2},
        },
        guidance=_guidance(),
        topology=topology or _topology(),
        stage="stage2",
        step=25,
        git_commit="a" * 40,
        source_checkpoint={"sha256": "b" * 64, "step": 4000},
    )


def test_atomic_checkpoint_round_trip_preserves_contract(tmp_path: Path) -> None:
    path = tmp_path / "step_25"
    _save(path)

    bundle = load_checkpoint(
        path, expected_stage="stage2", expected_topology=_topology()
    )

    online, ema = _states()
    assert set(bundle.online_state) == set(online)
    assert set(bundle.ema_state) == set(ema)
    for name in online:
        assert torch.equal(bundle.online_state[name], online[name])
        assert torch.equal(bundle.ema_state[name], ema[name])
    assert (
        bundle.metadata["resolved_config"]["paths"]["output_dir"] == "<redacted-path>"
    )
    assert bundle.metadata["guidance"]["teacher_video_cfg"] == 5.0
    assert bundle.metadata["stage"] == "stage2"
    assert bundle.metadata["step"] == 25
    assert bundle.metadata["git_commit"] == "a" * 40
    assert bundle.metadata["source_checkpoint"]["sha256"] == "b" * 64
    assert not list(tmp_path.glob(".step_25.tmp-*"))


def test_checkpoint_rejects_partial_and_wrong_stage(tmp_path: Path) -> None:
    path = tmp_path / "step_25"
    _save(path)
    (path / "COMPLETE").unlink()
    with pytest.raises(ValueError, match="partial"):
        load_checkpoint(path)

    _save(tmp_path / "step_26")
    with pytest.raises(ValueError, match="expected stage1"):
        load_checkpoint(tmp_path / "step_26", expected_stage="stage1")


def test_checkpoint_rejects_nested_scope_and_topology_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "step_25"
    _save(path)
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["state_scope"] = "nested_transformer"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="state_scope"):
        load_checkpoint(path)

    valid = tmp_path / "step_26"
    _save(valid)
    with pytest.raises(ValueError, match="topology"):
        load_checkpoint(valid, expected_topology=_topology(rank=32))
