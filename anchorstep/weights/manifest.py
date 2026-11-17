"""Publication-state validation for staged model repositories."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
        str(item.get("id")) for item in artifacts if item.get("paper_status") == "exact"
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_staging_tree(root: Path) -> ReleaseManifestReport:
    """Validate paths, redaction, and hashes for a staged Hub repository."""
    directory = Path(root)
    manifest_path = directory / "manifest.json"
    checksums_path = directory / "checksums.sha256"
    if not manifest_path.is_file() or not checksums_path.is_file():
        raise ValueError("staging tree is missing manifest.json or checksums.sha256")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must contain an object")
    report = validate_release_manifest(manifest)

    text_suffixes = {".json", ".md", ".py", ".txt", ".yaml", ".yml"}
    for path in directory.rglob("*"):
        if (
            path.is_file()
            and ".cache" not in path.relative_to(directory).parts
            and path.suffix.lower() in text_suffixes
        ):
            text = path.read_text(encoding="utf-8", errors="replace")
            if INTERNAL_PATH_PREFIX in text or "<redacted-path>" in text:
                raise ValueError(
                    "staging tree contains an internal/private path: "
                    f"{path.name}"
                )

    declared_paths = {
        str(item.get("path"))
        for item in manifest.get("artifacts", ())
        if isinstance(item, Mapping)
    }
    base_rows = manifest.get("bases", {})
    if not isinstance(base_rows, Mapping):
        raise ValueError("manifest bases must be a mapping")
    base_paths = [
        str(item.get("artifact"))
        for item in base_rows.values()
        if isinstance(item, Mapping)
    ]
    if len(base_paths) != len(set(base_paths)):
        raise ValueError("manifest stores a duplicate base artifact")
    declared_paths.update(base_paths)
    for relative in declared_paths:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe staged artifact path: {relative}")
        if not (directory / path).is_file():
            raise ValueError(f"staging tree is missing declared artifact: {relative}")

    expected: dict[str, str] = {}
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator or len(digest) != 64 or relative in expected:
            raise ValueError("invalid or duplicate checksums.sha256 entry")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe checksum path: {relative}")
        expected[relative] = digest
    actual_files = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
        and path != checksums_path
        and ".cache" not in path.relative_to(directory).parts
    }
    if actual_files != set(expected):
        raise ValueError("checksum inventory has missing or unexpected files")
    for relative, digest in expected.items():
        if _sha256(directory / relative) != digest:
            raise ValueError(f"checksum mismatch: {relative}")
    return report
