"""Training entry point with dependency-free preflight."""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Mapping, Sequence
from typing import Any

from anchorstep.config import load_config, validate_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="apm-train")
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--check-only", action="store_true")
    return parser


def _run_factory(path: str, config: Mapping[str, Any]) -> None:
    if ":" not in path:
        raise ValueError("train.runner_factory must be 'module:function'")
    module_name, function_name = path.split(":", 1)
    try:
        runner = getattr(importlib.import_module(module_name), function_name)
    except (ImportError, AttributeError) as error:
        raise RuntimeError(f"cannot import training runner factory {path}") from error
    runner(config)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config, args.overrides)
    train = config.get("train", {})
    if not isinstance(train, Mapping):
        raise ValueError("train must be a mapping")
    stage = train.get("stage")
    mode = "stage1" if stage in {1, "stage1"} else "stage2"
    validate_config(config, mode)
    if args.check_only:
        return 0
    factory = train.get("runner_factory")
    if not isinstance(factory, str):
        raise ValueError("train.runner_factory is required outside --check-only")
    _run_factory(factory, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
