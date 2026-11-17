from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from anchorstep.weights.manifest import (
    validate_release_manifest,
    validate_staging_tree,
)

INTERNAL_PATH_PREFIX = "/" + "kpfs-intern/"


def test_staging_manifest_may_report_missing_exact_roles() -> None:
    manifest = {
        "status": "staging",
        "required_roles": ["apm", "continued"],
        "artifacts": [
            {
                "id": "continued",
                "paper_status": "exact",
                "verified": True,
                "base": "bases/shared",
            }
        ],
    }
    report = validate_release_manifest(manifest)
    assert report.missing_exact_roles == ("apm",)
    assert not report.eligible_for_completion


def test_complete_manifest_requires_verified_exact_roles_and_safe_paths() -> None:
    manifest = {
        "status": "complete",
        "required_roles": ["apm", "continued"],
        "artifacts": [
            {
                "id": "apm",
                "paper_status": "exact",
                "verified": True,
                "base": "bases/shared",
            },
            {
                "id": "continued",
                "paper_status": "exact",
                "verified": True,
                "base": "bases/shared",
            },
        ],
    }
    assert validate_release_manifest(manifest).eligible_for_completion
    manifest["artifacts"][0]["verified"] = False
    with pytest.raises(ValueError, match="complete"):
        validate_release_manifest(manifest)
    manifest["status"] = "staging"
    manifest["artifacts"][0]["source"] = INTERNAL_PATH_PREFIX + "private"
    with pytest.raises(ValueError, match="private|internal"):
        validate_release_manifest(manifest)


def _staging_tree(root: Path) -> None:
    artifact = root / "bases/shared/model.safetensors"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"weights")
    manifest = {
        "status": "uploading",
        "required_roles": ["shared", "apm"],
        "artifacts": [
            {
                "id": "shared",
                "paper_status": "exact",
                "verified": True,
                "path": "bases/shared/model.safetensors",
            }
        ],
        "bases": {"shared": {"artifact": "bases/shared/model.safetensors"}},
        "variants": {},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    checksums = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            checksums.append(f"{digest}  {path.relative_to(root)}")
    (root / "checksums.sha256").write_text("\n".join(checksums) + "\n")


def test_staging_tree_checks_declared_files_hashes_and_private_paths(
    tmp_path: Path,
) -> None:
    _staging_tree(tmp_path)
    report = validate_staging_tree(tmp_path)
    assert report.missing_exact_roles == ("apm",)

    artifact = tmp_path / "bases/shared/model.safetensors"
    artifact.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        validate_staging_tree(tmp_path)

    _staging_tree(tmp_path)
    artifact.unlink()
    with pytest.raises(ValueError, match="missing"):
        validate_staging_tree(tmp_path)

    _staging_tree(tmp_path)
    (tmp_path / "README.md").write_text(INTERNAL_PATH_PREFIX + "private")
    with pytest.raises(ValueError, match="private|internal"):
        validate_staging_tree(tmp_path)
