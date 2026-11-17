from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from anchorstep.backends.lingbot import (
    load_routed_adapter_state,
    verify_checkout_revision,
)
from anchorstep.backends.registry import create_backend


def test_importing_anchorstep_does_not_import_lingbot() -> None:
    importlib.import_module("anchorstep")
    assert not any(
        name == "wan_va" or name.startswith("wan_va.") for name in sys.modules
    )


def test_selecting_lingbot_without_checkout_names_required_config() -> None:
    with pytest.raises(RuntimeError, match=r"paths\.lingbot_repo"):
        create_backend("lingbot", {"paths": {}})


def test_revision_error_reports_expected_and_actual(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", tmp_path], check=True)
    subprocess.run(["git", "-C", tmp_path, "config", "user.name", "Test"], check=True)
    subprocess.run(
        ["git", "-C", tmp_path, "config", "user.email", "test@example.com"],
        check=True,
    )
    (tmp_path / "tracked.txt").write_text("fixture\n")
    subprocess.run(["git", "-C", tmp_path, "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", tmp_path, "commit", "-qm", "fixture"], check=True)
    actual = subprocess.check_output(
        ["git", "-C", tmp_path, "rev-parse", "HEAD"], text=True
    ).strip()

    with pytest.raises(RuntimeError) as error:
        verify_checkout_revision(tmp_path, "0" * 40)

    assert "0" * 40 in str(error.value)
    assert actual in str(error.value)


def test_routed_adapter_loader_rejects_missing_and_unexpected_keys() -> None:
    model = nn.Linear(1, 1, bias=False)
    expected = {"weight"}
    with pytest.raises(ValueError, match="missing"):
        load_routed_adapter_state(model, {}, expected)
    with pytest.raises(ValueError, match="unexpected"):
        load_routed_adapter_state(
            model,
            {"weight": torch.ones(1, 1), "extra": torch.ones(1)},
            expected,
        )
