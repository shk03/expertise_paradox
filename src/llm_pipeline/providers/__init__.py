"""Provider registry for neuroradiology LLM backends.

The heavy SDKs (notably google-generativeai + grpc) are imported lazily so that
basic CLI operations (e.g. `--help`) work even if optional dependencies are
missing from the runtime environment. Call :func:`get_provider` to access the
actual generator utilities.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, Iterable

ProviderCallable = Callable[..., object]


@dataclass(frozen=True)
class ProviderConfig:
    """Runtime configuration for a single provider."""

    client: ProviderCallable
    generator: ProviderCallable
    rename_map: Dict[str, str]


_PROVIDER_SPECS: Dict[str, Dict[str, object]] = {
    "openai": {
        "module": "llm_pipeline.providers.openai_provider",
        "rename_map": {
            "first_diagnosis": "openai_first_diagnosis",
            "second_diagnosis": "openai_second_diagnosis",
            "third_diagnosis": "openai_third_diagnosis",
            "rationale": "openai_rationale",
        },
    },
    "deepseek": {
        "module": "llm_pipeline.providers.deepseek_provider",
        "rename_map": {},
    },
    "gemini": {
        "module": "llm_pipeline.providers.gemini_provider",
        "rename_map": {},
    },
}


def list_providers() -> Iterable[str]:
    """Return the available provider names."""
    return _PROVIDER_SPECS.keys()


@lru_cache(maxsize=None)
def get_provider(name: str) -> ProviderConfig:
    """Resolve the given provider name into its callable configuration."""
    if name not in _PROVIDER_SPECS:
        raise KeyError(f"Unknown provider '{name}'.")

    spec = _PROVIDER_SPECS[name]
    module = importlib.import_module(spec["module"])
    return ProviderConfig(
        client=getattr(module, "get_client"),
        generator=getattr(module, "generate_diagnosis"),
        rename_map=spec["rename_map"],
    )


__all__ = ["get_provider", "list_providers", "ProviderConfig"]
