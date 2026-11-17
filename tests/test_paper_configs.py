from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from anchorstep.config import load_config, validate_config


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "configs/model/lingbot_robotwin.yaml"
STAGE2 = ROOT / "configs/train/stage2.yaml"
ABLATIONS = {
    "continued": ("none", False, False),
    "original_no_field": ("original", False, False),
    "original_field_projected": ("original", True, True),
    "restart_no_field": ("restart", False, False),
    "restart_field_no_projection": ("restart", True, False),
    "apm": ("restart", True, True),
}


def _strings(value: Any):
    if isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)
    elif isinstance(value, str):
        yield value


@pytest.mark.parametrize("name", sorted(ABLATIONS))
def test_paper_ablation_has_unambiguous_semantics_and_cfg(name: str) -> None:
    config = load_config(
        [MODEL, STAGE2, ROOT / f"configs/ablation/{name}.yaml"]
    )
    validate_config(config, "stage2")

    expected_anchor, expected_field, expected_projection = ABLATIONS[name]
    assert config["ablation"]["public_id"] == name.replace("_", "-")
    assert config["ablation"]["anchor_source"] == expected_anchor
    assert config["ablation"]["field_enabled"] is expected_field
    assert config["ablation"]["anchor_priority_projection"] is expected_projection
    assert config["deployment"]["budgets"] == [1, 2, 4]
    assert config["teacher_target_cfg"] == {"video": 5.0, "action": 1.0}
    assert config["deployment_cfg"] == {"video": 5.0, "action": 1.0}
    assert config["data"]["text_dropout_probability"] == 0.1
    assert all("/kpfs-intern/" not in value for value in _strings(config))
    assert all(
        not Path(value).is_absolute()
        for value in _strings(config)
        if not value.startswith(("http://", "https://"))
    )
