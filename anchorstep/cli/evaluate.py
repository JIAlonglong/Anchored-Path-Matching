"""RoboTwin evaluation entry point with simulator-free preflight."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict

from anchorstep.config import load_config, validate_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="apm-eval")
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--check-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config, args.overrides)
    validate_config(config, "eval")
    if args.check_only:
        return 0
    from anchorstep.robotwin.evaluate import evaluate

    print(json.dumps(asdict(evaluate(config)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
