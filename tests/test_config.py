from __future__ import annotations

from pathlib import Path

import pytest

from anchorstep.config import (
    GuidanceContract,
    load_config,
    redact_config,
    validate_config,
)


def _paper_config() -> dict[str, object]:
    return {
        "data": {"text_dropout_probability": 0.1},
        "teacher_target_cfg": {"video": 5.0, "action": 1.0},
        "deployment_cfg": {"video": 5.0, "action": 1.0},
        "guidance": {"target_guidance_mode": "inference_composed"},
        "train": {"stage": 2},
    }


def test_paper_guidance_contract_is_explicit_and_valid() -> None:
    config = _paper_config()
    validate_config(config, "stage2")

    contract = GuidanceContract.from_config(config)
    assert contract.text_dropout_probability == 0.1
    assert (contract.teacher_video_cfg, contract.teacher_action_cfg) == (5.0, 1.0)
    assert (contract.deployment_video_cfg, contract.deployment_action_cfg) == (
        5.0,
        1.0,
    )
    assert contract.target_guidance_mode == "inference_composed"


def test_cfg_probability_is_not_accepted_as_guidance_or_text_dropout() -> None:
    config = {
        "data": {"cfg_probability": 0.1},
        "train": {"stage": 2},
    }
    with pytest.raises(ValueError, match="text_dropout_probability"):
        validate_config(config, "stage2")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda config: config.pop("teacher_target_cfg"),
        lambda config: config["deployment_cfg"].update(video=0.0),
        lambda config: config["teacher_target_cfg"].update(action=-1.0),
    ],
)
def test_formal_validation_rejects_missing_or_nonpositive_cfg(mutation) -> None:
    config = _paper_config()
    mutation(config)
    with pytest.raises(ValueError, match="CFG|cfg"):
        validate_config(config, "stage2")


def test_distilled_target_cannot_be_conflated_with_guided_deployment() -> None:
    config = _paper_config()
    config["guidance"] = {"target_guidance_mode": "target_distilled"}
    with pytest.raises(ValueError, match="deployment CFG 1/1"):
        validate_config(config, "stage2")


def test_load_config_composes_files_and_typed_overrides(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    overlay = tmp_path / "overlay.yaml"
    base.write_text("train:\n  stage: 1\nloss:\n  video_weight: 1.0\n")
    overlay.write_text("train:\n  stage: 2\n")

    config = load_config(
        [base, overlay], ["loss.action_weight=0.1", "apm.field_enabled=true"]
    )

    assert config["train"]["stage"] == 2
    assert config["loss"] == {"video_weight": 1.0, "action_weight": 0.1}
    assert config["apm"]["field_enabled"] is True


def test_redact_config_removes_secrets_and_machine_paths() -> None:
    redacted = redact_config(
        {"paths": {"output_dir": "/private/machine/run"}, "hf_token": "secret"}
    )
    assert redacted == {
        "paths": {"output_dir": "<redacted-path>"},
        "hf_token": "<redacted>",
    }
