"""Deterministic, resumable RoboTwin evaluation records."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeRecord:
    """One fully specified episode outcome."""

    task: str
    seed: int
    domain: str
    success: bool
    steps: int
    video_cfg: float
    action_cfg: float

    def __post_init__(self) -> None:
        if not self.task:
            raise ValueError("task must not be empty")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.domain not in {"Clean", "Random"}:
            raise ValueError("domain must be Clean or Random")
        if type(self.steps) is not int or self.steps not in {1, 2, 4}:
            raise ValueError("steps must be one of {1, 2, 4}")
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in (self.video_cfg, self.action_cfg)
        ):
            raise ValueError("video_cfg and action_cfg must be positive finite values")

    @property
    def episode_id(self) -> str:
        payload = f"{self.domain}\0{self.task}\0{self.seed}".encode()
        return hashlib.sha256(payload).hexdigest()[:20]


@dataclass(frozen=True)
class DomainSummary:
    episode_count: int
    per_task_success: dict[str, float]
    macro_success: float


@dataclass(frozen=True)
class EvaluationSummary:
    domains: dict[str, DomainSummary]


def pending_episode_ids(
    candidates: Sequence[EpisodeRecord], existing: Iterable[EpisodeRecord]
) -> list[str]:
    """Return candidate identities not already present in a resume file."""
    completed = {record.episode_id for record in existing}
    result: list[str] = []
    seen = set(completed)
    for candidate in candidates:
        if candidate.episode_id not in seen:
            result.append(candidate.episode_id)
            seen.add(candidate.episode_id)
    return result


def aggregate_results(records: Iterable[EpisodeRecord]) -> EvaluationSummary:
    """Aggregate per-task and task-macro success separately by domain."""
    by_domain: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    seen: set[str] = set()
    for record in records:
        if record.episode_id in seen:
            raise ValueError(f"duplicate episode record: {record.episode_id}")
        seen.add(record.episode_id)
        by_domain[record.domain][record.task].append(record.success)
    summaries: dict[str, DomainSummary] = {}
    for domain, tasks in sorted(by_domain.items()):
        per_task = {
            task: sum(outcomes) / len(outcomes)
            for task, outcomes in sorted(tasks.items())
        }
        summaries[domain] = DomainSummary(
            episode_count=sum(len(outcomes) for outcomes in tasks.values()),
            per_task_success=per_task,
            macro_success=sum(per_task.values()) / len(per_task),
        )
    return EvaluationSummary(summaries)
