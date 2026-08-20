"""Provider registry.

A model is selected by name. Anything unprefixed is treated as an OpenAI
model; ``ollama:<model>`` and ``compatible:<model>`` route to an
OpenAI-compatible endpoint. Adding a vendor means adding one class plus one
entry here - nothing in the graph changes.
"""

from __future__ import annotations

from dataclasses import dataclass

from reposentinel.config import Settings, get_settings
from reposentinel.models.providers.base import (
    Message,
    ModelProvider,
    ModelResponse,
    ProviderError,
    ToolInvocation,
    Usage,
    price_for,
)
from reposentinel.models.providers.openai_provider import OpenAIProvider
from reposentinel.models.providers.scripted import ScriptedProvider, ScriptedTurn

__all__ = [
    "Message",
    "ModelProvider",
    "ModelResponse",
    "OpenAIProvider",
    "ProviderError",
    "ScriptedProvider",
    "ScriptedTurn",
    "ToolInvocation",
    "Usage",
    "available_models",
    "build_provider",
]


@dataclass(frozen=True)
class ModelOption:
    """A model the UI may offer."""

    id: str
    label: str
    provider: str
    notes: str = ""

    def to_dict(self, available: bool) -> dict[str, object]:
        price = price_for(self.id)
        return {
            "id": self.id,
            "label": self.label,
            "provider": self.provider,
            "available": available,
            "notes": self.notes,
            "input_per_mtok": price[0] if price else None,
            "output_per_mtok": price[1] if price else None,
        }


# Curated list for the model-comparison feature. Every entry is a real model
# id; availability is resolved from configured credentials at request time.
MODEL_CATALOG: tuple[ModelOption, ...] = (
    ModelOption("gpt-4.1-mini", "GPT-4.1 mini", "openai", "fast, cheap default"),
    ModelOption("gpt-4.1", "GPT-4.1", "openai", "stronger reasoning"),
    ModelOption("gpt-4.1-nano", "GPT-4.1 nano", "openai", "cheapest"),
    ModelOption("gpt-4o-mini", "GPT-4o mini", "openai", ""),
    ModelOption("gpt-5-mini", "GPT-5 mini", "openai", "reasoning model"),
    ModelOption("gpt-5", "GPT-5", "openai", "reasoning model"),
    ModelOption("o4-mini", "o4-mini", "openai", "reasoning model"),
)


def build_provider(
    model: str | None = None,
    settings: Settings | None = None,
) -> ModelProvider:
    settings = settings or get_settings()
    name = (model or settings.default_model).strip()

    if name.startswith("ollama:"):
        return OpenAIProvider(
            model=name.split(":", 1)[1],
            api_key="ollama",
            base_url=f"{settings.ollama_base_url.rstrip('/')}/v1",
            timeout=settings.llm_request_timeout,
        )
    if name.startswith("compatible:"):
        if not settings.openai_base_url:
            raise ProviderError(
                "compatible: models require REPOSENTINEL_OPENAI_BASE_URL to be set"
            )
        return OpenAIProvider(
            model=name.split(":", 1)[1],
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout=settings.llm_request_timeout,
        )
    if not settings.openai_api_key:
        raise ProviderError(
            "No model credentials found. Set OPENAI_API_KEY (or point "
            "REPOSENTINEL_OPENAI_BASE_URL at a compatible endpoint)."
        )
    return OpenAIProvider(
        model=name,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout=settings.llm_request_timeout,
    )


def available_models(settings: Settings | None = None) -> list[dict[str, object]]:
    settings = settings or get_settings()
    has_openai = bool(settings.openai_api_key)
    return [
        option.to_dict(available=has_openai if option.provider == "openai" else False)
        for option in MODEL_CATALOG
    ]
