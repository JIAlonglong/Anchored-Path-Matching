from __future__ import annotations

import pytest

from anchorstep.weights.manifest import validate_release_manifest

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
