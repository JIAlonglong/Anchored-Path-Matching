"""Materialize a shared base plus changed-tensor artifact."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from anchorstep.weights.materialize import materialize


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="apm-materialize-weights")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-shard-bytes", type=int, default=2_000_000_000)
    args = parser.parse_args(argv)
    materialize(args.base, args.artifact, args.output, args.max_shard_bytes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
