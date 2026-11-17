from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path

import torch

from anchorstep.backends.mock import MockBackend
from anchorstep.config import GuidanceContract
from anchorstep.core.flow_map import stage1_objective
from anchorstep.robotwin.inference import rollout_student
from anchorstep.training.checkpointing import load_checkpoint, save_checkpoint
from anchorstep.training.trainer import Trainer


def _config() -> dict[str, object]:
    return {
        "train": {
            "teacher_reference_steps": 8,
            "ema_decay": 0.0,
            "gradient_clip_norm": 10.0,
        },
        "loss": {
            "video_weight": 1.0,
            "action_weight": 0.1,
            "anchor_video_weight": 1.0,
            "anchor_action_weight": 0.1,
            "field_video_weight": 1.0,
            "field_action_weight": 0.1,
        },
        "apm": {
            "rollout_steps": 2,
            "anchor_source": "restart",
            "field_target": "finite_local",
            "field_enabled": True,
        },
    }


def test_mock_stage1_checkpoint_stage2_and_multibudget_inference(
    tmp_path: Path,
) -> None:
    backend = MockBackend()
    batch = backend.encode_batch(
        {"video": torch.tensor([[1.0], [0.5]]), "action": torch.tensor([[0.25], [-0.5]])}
    )
    optimizer = torch.optim.SGD(backend.trainable_parameters(), lr=0.05)
    stage1 = stage1_objective(
        backend, batch, _config(), torch.Generator().manual_seed(2)
    )
    optimizer.zero_grad()
    stage1.total.backward()
    optimizer.step()
    backend.copy_online_to_target()

    checkpoint = tmp_path / "stage1"
    guidance = GuidanceContract(0.1, 5.0, 1.0, 5.0, 1.0, "inference_composed")
    save_checkpoint(
        checkpoint,
        online_state=backend.adapter_state(),
        ema_state=backend.adapter_state(target=True),
        resolved_config={"train": {"stage": 1}},
        guidance=guidance,
        topology=backend.topology,
        stage="stage1",
        step=1,
        git_commit="c" * 40,
        source_checkpoint={"sha256": "d" * 64},
    )

    resumed = MockBackend()
    bundle = load_checkpoint(
        checkpoint, expected_stage="stage1", expected_topology=resumed.topology
    )
    resumed.load_adapter_state(bundle.online_state)
    resumed.load_adapter_state(bundle.ema_state, target=True)
    trainer = Trainer(
        backend=resumed,
        optimizer=torch.optim.SGD(resumed.trainable_parameters(), lr=0.01),
        config=_config(),
        generator=torch.Generator().manual_seed(3),
    )
    trainer.train_accumulated_step(
        [{"video": batch.clean.video, "action": batch.clean.action}]
    )

    for steps in (1, 2, 4):
        resumed.reset_call_counts()
        prior = resumed.sample_prior(batch, torch.Generator().manual_seed(steps))
        rollout_student(resumed, prior, batch, steps)
        assert resumed.student_call_count == steps
        assert resumed.teacher_call_count == 0
