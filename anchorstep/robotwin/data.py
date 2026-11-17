"""Pinned RoboTwin observation/action contract without simulator imports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

ROBOTWIN_REVISION = "30954692d06ba7e89f7a6b76064f4062c488fa81"


@dataclass(frozen=True)
class RoboTwinContract:
    """Serialized inputs needed to reproduce the released policy interface."""

    camera_keys: tuple[str, ...] = (
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    )
    used_action_channel_ids: tuple[int, ...] = (
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
    action_q01: tuple[float, ...] = (
        -0.06172713458538055,
        -3.6716461181640625e-05,
        -0.08783501386642456,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        0.0,
        -0.3547105032205582,
        -1.3113021850585938e-06,
        -0.11975435614585876,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        0.0,
    )
    action_q99: tuple[float, ...] = (
        0.3462600058317184,
        0.39966784834861746,
        0.14745532035827624,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.034201726913452024,
        0.39142737388610793,
        0.1792279863357542,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    )
    model_action_dimension: int = 30
    raw_action_dimension: int = 16
    normalization: str = "quantiles-epsilon-1e-6-clamp-1.5-once"
    chunk_size: int = 2
    attention_window: int = 72
    revision: str = ROBOTWIN_REVISION


def validate_observation(
    observation: Mapping[str, Any], contract: RoboTwinContract
) -> None:
    """Validate camera presence before an external simulator/runtime consumes it."""
    missing = [key for key in contract.camera_keys if key not in observation]
    if missing:
        raise ValueError(f"RoboTwin observation is missing camera keys: {missing}")
