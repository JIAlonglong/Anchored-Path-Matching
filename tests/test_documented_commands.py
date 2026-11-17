from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = (
    ROOT / "README.md",
    ROOT / "docs/training.md",
    ROOT / "docs/evaluation.md",
)


def _preflight_commands() -> list[str]:
    commands: list[str] = []
    for path in DOCUMENTS:
        text = path.read_text(encoding="utf-8")
        inside = False
        for line in text.splitlines():
            if line.strip() == "```bash test=preflight":
                inside = True
                continue
            if inside and line.strip() == "```":
                inside = False
                continue
            if inside and line.strip() and not line.lstrip().startswith("#"):
                commands.append(line.strip())
    return commands


def test_documented_preflight_commands_execute_without_external_assets() -> None:
    commands = _preflight_commands()
    assert len(commands) >= 3
    for command in commands:
        completed = subprocess.run(
            shlex.split(command),
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        assert completed.returncode == 0, (
            f"command failed: {command}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
