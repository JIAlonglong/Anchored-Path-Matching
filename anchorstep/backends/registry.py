"""Lazy backend selection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def create_backend(name: str, config: Mapping[str, Any]):
    """Construct a named backend without importing optional dependencies early."""
    if name == "mock":
        from anchorstep.backends.mock import MockBackend

        return MockBackend()
    if name == "lingbot":
        from anchorstep.backends.lingbot import LingBotBackend

        return LingBotBackend.from_config(config)
    raise ValueError(f"unknown backend: {name}")
