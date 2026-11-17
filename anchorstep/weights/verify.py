"""End-to-end logical state verification."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from anchorstep.weights.materialize import materialize
from anchorstep.weights.safetensors_io import tensor_state_digest


@dataclass(frozen=True)
class VerificationReport:
    ok: bool
    expected_digest: str
    materialized_digest: str


def verify_logical_state(
    base: Path, artifact: Path, expected: Path
) -> VerificationReport:
    """Materialize in a temporary directory and compare logical state digests."""
    expected_digest = tensor_state_digest(expected)
    with tempfile.TemporaryDirectory(prefix="apm-verify-") as temporary:
        index = materialize(base, artifact, Path(temporary) / "model")
        materialized_digest = tensor_state_digest(index)
    return VerificationReport(
        ok=materialized_digest == expected_digest,
        expected_digest=expected_digest,
        materialized_digest=materialized_digest,
    )
