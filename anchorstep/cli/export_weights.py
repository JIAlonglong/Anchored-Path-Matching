"""Export one verified adapter/replacement artifact."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from anchorstep.weights.export import classify_artifact, export_artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="apm-export-weights")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--variant", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--paper-role", required=True)
    parser.add_argument("--adapter-key", action="append", default=[])
    args = parser.parse_args(argv)
    plan = classify_artifact(args.base, args.variant, set(args.adapter_key))
    export_artifact(
        plan,
        args.output,
        variant_id=args.id,
        paper_role=args.paper_role,
        topology="declared-by-caller",
        cfg_contract={},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
