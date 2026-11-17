"""Lazy RoboTwin evaluation orchestration."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from anchorstep.robotwin.results import (
    EpisodeRecord,
    EvaluationSummary,
    aggregate_results,
)


def load_episode_runner(factory_path: str) -> Callable[..., Iterable[EpisodeRecord]]:
    """Load a user-provided simulator runner only after preflight succeeds."""
    if ":" not in factory_path:
        raise ValueError("evaluation.runner_factory must be 'module:function'")
    module_name, function_name = factory_path.split(":", 1)
    try:
        factory = getattr(importlib.import_module(module_name), function_name)
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            f"cannot import RoboTwin runner factory {factory_path}"
        ) from error
    return factory


def evaluate(config: Mapping[str, Any]) -> EvaluationSummary:
    """Run an external episode factory and aggregate its explicit records."""
    evaluation = config.get("evaluation", {})
    if not isinstance(evaluation, Mapping):
        raise ValueError("evaluation must be a mapping")
    factory_path = evaluation.get("runner_factory")
    if not isinstance(factory_path, str):
        raise ValueError("evaluation.runner_factory is required for live evaluation")
    records = load_episode_runner(factory_path)(config)
    return aggregate_results(records)
