"""Scripted provider used by the test suite.

This exists so the graph, the tool loop and the API can be tested
deterministically and offline. It is **not** offered in the UI's model list and
never masquerades as a real model: ``describe()`` reports it plainly, and every
run records the provider that produced it.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel

from reposentinel.models.providers.base import (
    Message,
    ModelProvider,
    ModelResponse,
    ProviderError,
    ToolInvocation,
    Usage,
)

T = TypeVar("T", bound=BaseModel)


@dataclass
class ScriptedTurn:
    """One canned reply.

    ``matcher`` lets a turn apply only when the request looks a certain way,
    which keeps tests readable when a graph makes many calls.
    """

    parsed: BaseModel | None = None
    text: str = ""
    tool_calls: list[ToolInvocation] = field(default_factory=list)
    matcher: Callable[[list[Message], type[BaseModel] | None], bool] | None = None
    prompt_tokens: int = 120
    completion_tokens: int = 40


class ScriptedProvider(ModelProvider):
    name = "scripted"

    def __init__(self, turns: list[ScriptedTurn], model: str = "scripted-test-model") -> None:
        super().__init__(model)
        self.turns = list(turns)
        self.calls: list[dict[str, Any]] = []

    @classmethod
    def available(cls, settings: Any) -> bool:  # noqa: ARG003
        return True

    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        response_model: type[T] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ModelResponse:
        self.calls.append(
            {
                "messages": messages,
                "response_model": response_model.__name__ if response_model else None,
                "tools": [t["function"]["name"] for t in (tools or [])],
            }
        )

        for position, turn in enumerate(self.turns):
            if turn.matcher is not None and not turn.matcher(messages, response_model):
                continue
            # An unmatched turn must at least produce the requested type.
            if (
                turn.matcher is None
                and response_model is not None
                and not isinstance(turn.parsed, response_model)
            ):
                continue
            self.turns.pop(position)
            return ModelResponse(
                text=turn.text or (turn.parsed.model_dump_json() if turn.parsed else ""),
                parsed=turn.parsed,
                tool_calls=list(turn.tool_calls),
                usage=Usage(turn.prompt_tokens, turn.completion_tokens, 0.0),
                model=self.model,
                provider=self.name,
                duration_ms=1,
                finish_reason="stop",
                raw_json=json.loads(turn.parsed.model_dump_json()) if turn.parsed else None,
            )

        raise ProviderError(
            f"scripted provider exhausted: no turn matches a request for "
            f"{response_model.__name__ if response_model else 'plain text'}"
        )

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.model,
            "priced": False,
            "note": "deterministic canned responses for tests only",
        }
