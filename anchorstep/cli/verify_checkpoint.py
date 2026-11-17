"""Validate a public checkpoint contract without constructing the model."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from anchorstep.training.checkpointing import load_checkpoint


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="apm-verify-checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-stage", choices=("stage1", "stage2"))
    parser.add_argument("--check-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    bundle = load_checkpoint(
        args.checkpoint, expected_stage=args.expected_stage
    )
    if not args.check_only:
        summary = {
            "stage": bundle.metadata["stage"],
            "step": bundle.metadata["step"],
            "git_commit": bundle.metadata["git_commit"],
            "tensor_count": len(bundle.ema_state),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
