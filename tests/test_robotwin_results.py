from __future__ import annotations

import pytest

from anchorstep.robotwin.data import RoboTwinContract
from anchorstep.robotwin.results import (
    EpisodeRecord,
    aggregate_results,
    pending_episode_ids,
)


def test_robotwin_contract_pins_checkpoint_camera_action_and_window_metadata() -> None:
    contract = RoboTwinContract()
    assert contract.camera_keys == (
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    )
    assert contract.used_action_channel_ids == (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        28,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        29,
    )
    assert contract.model_action_dimension == 30
    assert contract.raw_action_dimension == 16
    assert contract.chunk_size == 2
    assert contract.attention_window == 72
    assert contract.normalization == "quantiles-epsilon-1e-6-clamp-1.5-once"


def _record(
    task: str, seed: int, domain: str, success: bool, *, steps: int = 2
) -> EpisodeRecord:
    return EpisodeRecord(
        task=task,
        seed=seed,
        domain=domain,
        success=success,
        steps=steps,
        video_cfg=5.0,
        action_cfg=1.0,
    )


def test_episode_ids_are_deterministic_and_resume_skips_existing() -> None:
    first = _record("put-bowl", 3, "Clean", True)
    second = _record("put-bowl", 3, "Clean", False)
    assert first.episode_id == second.episode_id

    candidates = [
        _record("put-bowl", 3, "Clean", False),
        _record("put-bowl", 4, "Clean", False),
    ]
    assert pending_episode_ids(candidates, [first]) == [candidates[1].episode_id]


def test_aggregation_separates_domains_and_uses_task_macro_average() -> None:
    summary = aggregate_results(
        [
            _record("task-a", 0, "Clean", True),
            _record("task-a", 1, "Clean", False),
            _record("task-b", 0, "Clean", True),
            _record("task-a", 0, "Random", False),
        ]
    )

    assert summary.domains["Clean"].per_task_success == {
        "task-a": 0.5,
        "task-b": 1.0,
    }
    assert summary.domains["Clean"].macro_success == pytest.approx(0.75)
    assert summary.domains["Random"].per_task_success == {"task-a": 0.0}
    assert summary.domains["Random"].macro_success == 0.0


def test_records_require_explicit_budget_and_cfg_and_reject_duplicates() -> None:
    with pytest.raises(ValueError, match="steps"):
        _record("task", 0, "Clean", True, steps=3)
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_results(
            [_record("task", 0, "Clean", True), _record("task", 0, "Clean", False)]
        )
