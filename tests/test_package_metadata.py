from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_package_metadata_declares_public_identity_and_license() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = metadata["project"]

    assert project["name"] == "anchored-path-matching"
    assert project["requires-python"] == ">=3.10"
    assert project["license"] == "Apache-2.0"
    assert {"torch", "numpy", "PyYAML", "safetensors"} <= {
        dependency.split("[")[0].split(">=")[0].split("==")[0]
        for dependency in project["dependencies"]
    }
