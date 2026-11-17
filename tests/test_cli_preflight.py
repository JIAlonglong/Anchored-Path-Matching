from __future__ import annotations

import sys
from pathlib import Path

from anchorstep.backends.mock import MockBackend
from anchorstep.cli.evaluate import main as evaluate_main
from anchorstep.cli.train import main as train_main
from anchorstep.cli.verify_checkpoint import main as verify_main
from anchorstep.config import GuidanceContract
from anchorstep.training.checkpointing import save_checkpoint


def _config(path: Path) -> None:
    path.write_text(
        """
data:
  text_dropout_probability: 0.1
teacher_target_cfg:
  video: 5.0
  action: 1.0
deployment_cfg:
  video: 5.0
  action: 1.0
guidance:
  target_guidance_mode: inference_composed
train:
  stage: 2
backend:
  name: mock
""".strip()
        + "\n"
    )


def test_train_and_eval_check_only_do_not_import_simulator_or_allocate_cuda(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    _config(config)

    assert train_main(["--config", str(config), "--check-only"]) == 0
    assert evaluate_main(["--config", str(config), "--check-only"]) == 0
    assert "sapien" not in sys.modules
    assert not any(name == "robotwin" or name.startswith("robotwin.") for name in sys.modules)


def test_checkpoint_check_only_validates_contract_without_model_allocation(
    tmp_path: Path,
) -> None:
    backend = MockBackend()
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(
        checkpoint,
        online_state=backend.adapter_state(),
        ema_state=backend.adapter_state(target=True),
        resolved_config={"train": {"stage": 1}},
        guidance=GuidanceContract(0.1, 5.0, 1.0, 5.0, 1.0, "inference_composed"),
        topology=backend.topology,
        stage="stage1",
        step=1,
        git_commit="e" * 40,
        source_checkpoint={"sha256": "f" * 64},
    )

    assert (
        verify_main(
            [
                "--checkpoint",
                str(checkpoint),
                "--expected-stage",
                "stage1",
                "--check-only",
            ]
        )
        == 0
    )
    assert "sapien" not in sys.modules
