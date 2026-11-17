"""Publication-state validation for staged model repositories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping


INTERNAL_PATH_PREFIX = "/" + "kpfs-intern/"


@dataclass(frozen=True)
class ReleaseManifestReport:
    missing_exact_roles: tuple[str, ...]
    unverified_artifacts: tuple[str, ...]
    eligible_for_completion: bool


def validate_release_manifest(
    manifest: Mapping[str, Any],
) -> ReleaseManifestReport:
    """Reject premature completion and private or ambiguous inventory data."""
    serialized = json.dumps(manifest, sort_keys=True)
    if INTERNAL_PATH_PREFIX in serialized or "<redacted-path>" in serialized:
        raise ValueError("manifest contains an internal/private path")
    status = manifest.get("status")
    if status not in {"staging", "uploading", "complete"}:
        raise ValueError("manifest status must be staging, uploading, or complete")
    required = tuple(str(item) for item in manifest.get("required_roles", ()))
    artifacts = manifest.get("artifacts", ())
    if not isinstance(artifacts, list):
        raise ValueError("manifest artifacts must be a list")
    exact = {
        str(item.get("id"))
        for item in artifacts
        if item.get("paper_status") == "exact"
    }
    missing = tuple(role for role in required if role not in exact)
    unverified = tuple(
        str(item.get("id")) for item in artifacts if not item.get("verified", False)
    )
    eligible = not missing and not unverified
    if status == "complete" and not eligible:
        raise ValueError(
            "complete manifest requires every exact role and verified artifact"
        )
    return ReleaseManifestReport(missing, unverified, eligible)
