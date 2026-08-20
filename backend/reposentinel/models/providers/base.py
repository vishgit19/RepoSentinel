"""Provider-agnostic model interface.

Nothing above this layer knows which vendor is answering. A provider must
support three things the agent depends on:

1. plain completion,
2. **structured output** validated against a Pydantic model,
3. **tool calling** with OpenAI-style function schemas.

Cost is computed from a price table so runs are comparable across models even
when a vendor does not return billing information.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class ProviderError(RuntimeError):
    """Raised when a provider cannot fulfil a request."""


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.cost_usd + other.cost_usd,
        )


@dataclass
class ToolInvocation:
    """A tool call requested by the model."""

    call_id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponse:
    text: str = ""
    parsed: BaseModel | None = None
    tool_calls: list[ToolInvocation] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    provider: str = ""
    duration_ms: int = 0
    finish_reason: str = ""
    raw_json: dict[str, Any] | None = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


@dataclass
class Message:
    role: str  # system | user | assistant | tool
    content: str
    tool_call_id: str | None = None
    tool_calls: list[ToolInvocation] | None = None
    name: str | None = None

    def to_openai(self) -> dict[str, Any]:
        if self.role == "tool":
            return {
                "role": "tool",
                "tool_call_id": self.tool_call_id or "",
                "content": self.content,
            }
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": _dumps(call.arguments)},
                }
                for call in self.tool_calls
            ]
            # The API requires content to be present but allows it to be empty.
            payload["content"] = self.content or ""
        return payload


def _dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)


# Price per million tokens (input, output), USD. Unlisted models are recorded
# with zero cost and flagged, rather than silently guessed.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-5": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
    "o3": (2.00, 8.00),
    "o3-mini": (1.10, 4.40),
    "o4-mini": (1.10, 4.40),
}


def price_for(model: str) -> tuple[float, float] | None:
    if model in PRICES:
        return PRICES[model]
    # Dated snapshots such as gpt-4.1-mini-2025-04-14 share their base price.
    for name, price in PRICES.items():
        if model.startswith(f"{name}-"):
            return price
    return None


def compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price = price_for(model)
    if price is None:
        return 0.0
    return prompt_tokens / 1_000_000 * price[0] + completion_tokens / 1_000_000 * price[1]


class ModelProvider(abc.ABC):
    """A source of model completions."""

    name: str = "abstract"

    def __init__(self, model: str) -> None:
        self.model = model

    @abc.abstractmethod
    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        response_model: type[T] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ModelResponse:
        """Produce one completion.

        When ``response_model`` is given the reply must be parsed into that
        model and returned in ``ModelResponse.parsed``.
        """

    @classmethod
    @abc.abstractmethod
    def available(cls, settings: Any) -> bool: ...

    def describe(self) -> dict[str, Any]:
        price = price_for(self.model)
        return {
            "provider": self.name,
            "model": self.model,
            "priced": price is not None,
            "input_per_mtok": price[0] if price else None,
            "output_per_mtok": price[1] if price else None,
        }
