from __future__ import annotations

from pathlib import Path

import yaml

from anchorstep.config import GuidanceContract, load_config, validate_config

ROOT = Path(__file__).resolve().parents[1]


def test_formal_stage_and_eval_configs_keep_dropout_and_cfg_separate() -> None:
    model = ROOT / "configs/model/lingbot_robotwin.yaml"
    for relative, mode in (
        ("configs/train/stage1.yaml", "stage1"),
        ("configs/train/stage2.yaml", "stage2"),
        ("configs/eval/robotwin.yaml", "eval"),
    ):
        config = load_config([model, ROOT / relative])
        validate_config(config, mode)
        contract = GuidanceContract.from_config(config)
        assert contract.text_dropout_probability == 0.1
        assert (contract.teacher_video_cfg, contract.teacher_action_cfg) == (5.0, 1.0)
        assert (contract.deployment_video_cfg, contract.deployment_action_cfg) == (
            5.0,
            1.0,
        )
        assert contract.target_guidance_mode == "inference_composed"
        assert "cfg_probability" not in config["data"]


def test_no_public_yaml_uses_legacy_cfg_probability() -> None:
    for path in (ROOT / "configs").rglob("*.yaml"):
        payload = yaml.safe_load(path.read_text())
        assert "cfg_probability" not in str(payload)
