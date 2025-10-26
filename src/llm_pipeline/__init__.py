"""LLM pipeline helpers for neuroradiology batch generation.

The package keeps imports light so `python -m llm_pipeline.cli --help` works
even without optional dependencies installed.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["run_generation", "get_provider", "list_providers"]


def __getattr__(name: str) -> Any:
    if name == "run_generation":
        from .generation import run_generation

        return run_generation
    if name in {"get_provider", "list_providers"}:
        module = import_module("llm_pipeline.providers")
        return getattr(module, name)
    raise AttributeError(f"module 'llm_pipeline' has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - debug helper
    return sorted(__all__)
