from __future__ import annotations

import importlib
import pkgutil
import sys


def test_core_imports_do_not_load_optional_runtime_dependencies() -> None:
    core = importlib.import_module("anchorstep.core")
    for module in pkgutil.walk_packages(core.__path__, f"{core.__name__}."):
        importlib.import_module(module.name)

    forbidden = ("wan_va", "robotwin", "sapien", "lingbot")
    loaded = {name for name in sys.modules if name.lower().startswith(forbidden)}
    assert loaded == set()
