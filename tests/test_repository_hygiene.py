from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".astro",
    ".bib",
    ".css",
    ".html",
    ".js",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
}
FORBIDDEN_BINARY_SUFFIXES = {".ckpt", ".pt", ".pth", ".safetensors"}
SECRET_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----"),
)
INTERNAL_PATH_PREFIX = "/" + "kpfs-intern/"


def _tracked_files() -> list[Path]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    return [ROOT / item.decode() for item in output.split(b"\0") if item]


def test_repository_excludes_private_and_runtime_artifacts() -> None:
    violations: list[str] = []
    for path in _tracked_files():
        relative = path.relative_to(ROOT).as_posix()
        if path.suffix.lower() in FORBIDDEN_BINARY_SUFFIXES:
            violations.append(f"checkpoint binary: {relative}")
        if any(part in {"outputs", "logs", "checkpoints"} for part in path.parts):
            violations.append(f"runtime directory: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if INTERNAL_PATH_PREFIX in text:
            violations.append(f"internal path: {relative}")
        if re.search(r"experiments/robotwin-[^\s'\"]*-2026", text):
            violations.append(f"dated experiment import: {relative}")
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            violations.append(f"secret-like token: {relative}")

    assert violations == []
