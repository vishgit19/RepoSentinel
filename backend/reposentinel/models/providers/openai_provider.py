"""OpenAI (and OpenAI-compatible) provider.

Covers the OpenAI API plus anything speaking the same protocol - Ollama,
vLLM, Together, OpenRouter, Azure - by overriding ``base_url``. Structured
output uses ``response_format: json_schema`` with ``strict: true`` where the
model supports it, falling back to ``json_object`` plus schema-in-prompt for
older or third-party endpoints.
"""

from __future__ import annotations

import json
import time
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from reposentinel.models.providers.base import (
    Message,
    ModelProvider,
    ModelResponse,
    ProviderError,
    ToolInvocation,
    Usage,
    compute_cost,
)

T = TypeVar("T", bound=BaseModel)

# Models that only accept the default temperature.
_FIXED_TEMPERATURE_PREFIXES = ("o1", "o3", "o4", "gpt-5")


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic model into a strict OpenAI JSON schema.

    The API requires every property to be listed in ``required`` and
    ``additionalProperties: false`` at each object level, which is stricter
    than Pydantic's default output.
    """
    schema = model.model_json_schema()
    definitions = schema.pop("$defs", {})

    def tighten(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref = node.pop("$ref")
                target = ref.rsplit("/", 1)[-1]
                node.update(tighten(dict(definitions.get(target, {}))))
            # Enums arrive as anyOf/allOf wrappers; flatten single-branch ones.
            for key in ("allOf", "anyOf", "oneOf"):
                if key in node and len(node[key]) == 1:
                    merged = tighten(node.pop(key)[0])
                    node.update(merged)
            if node.get("type") == "object":
                properties = node.get("properties", {})
                node["properties"] = {k: tighten(v) for k, v in properties.items()}
                node["required"] = list(node["properties"].keys())
                node["additionalProperties"] = False
            if "items" in node:
                node["items"] = tighten(node["items"])
            # Defaults are not permitted in strict schemas.
            node.pop("default", None)
            return {k: tighten(v) if k in {"properties", "items"} else v for k, v in node.items()}
        if isinstance(node, list):
            return [tighten(item) for item in node]
        return node

    tightened = tighten(schema)
    tightened.setdefault("type", "object")
    return tightened


class OpenAIProvider(ModelProvider):
    name = "openai"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 120,
        max_retries: int = 2,
    ) -> None:
        super().__init__(model)
        from openai import OpenAI

        if not api_key and not base_url:
            raise ProviderError("OpenAI provider needs an API key (set OPENAI_API_KEY)")
        self._client = OpenAI(
            api_key=api_key or "not-needed",
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.base_url = base_url

    @classmethod
    def available(cls, settings: Any) -> bool:
        return bool(getattr(settings, "openai_api_key", None))

    def _supports_temperature(self) -> bool:
        return not self.model.startswith(_FIXED_TEMPERATURE_PREFIXES)

    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        response_model: type[T] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_openai() for m in messages],
        }
        if self._supports_temperature():
            payload["temperature"] = temperature
        if max_tokens:
            # Reasoning models renamed this parameter.
            key = (
                "max_completion_tokens"
                if self.model.startswith(_FIXED_TEMPERATURE_PREFIXES)
                else "max_tokens"
            )
            payload[key] = max_tokens
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if response_model is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": strict_json_schema(response_model),
                    "strict": True,
                },
            }

        started = time.perf_counter()
        try:
            completion = self._client.chat.completions.create(**payload)
        except Exception as exc:  # noqa: BLE001 - surfaced as ProviderError
            message = str(exc)
            if response_model is not None and (
                "json_schema" in message or "response_format" in message
            ):
                completion = self._retry_without_strict_schema(payload, response_model)
            else:
                raise ProviderError(f"{type(exc).__name__}: {message}") from exc

        duration_ms = int((time.perf_counter() - started) * 1000)
        choice = completion.choices[0]
        text = choice.message.content or ""

        usage_data = getattr(completion, "usage", None)
        prompt_tokens = getattr(usage_data, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage_data, "completion_tokens", 0) or 0
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=compute_cost(self.model, prompt_tokens, completion_tokens),
        )

        invocations: list[ToolInvocation] = []
        for call in choice.message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {"__raw": call.function.arguments}
            invocations.append(
                ToolInvocation(call_id=call.id, name=call.function.name, arguments=arguments)
            )

        parsed: BaseModel | None = None
        raw_json: dict[str, Any] | None = None
        if response_model is not None and text.strip():
            parsed, raw_json = _parse_structured(text, response_model)

        return ModelResponse(
            text=text,
            parsed=parsed,
            tool_calls=invocations,
            usage=usage,
            model=self.model,
            provider=self.name,
            duration_ms=duration_ms,
            finish_reason=choice.finish_reason or "",
            raw_json=raw_json,
        )

    def _retry_without_strict_schema(
        self, payload: dict[str, Any], response_model: type[T]
    ) -> Any:
        """Fallback for endpoints without json_schema support."""
        retry = dict(payload)
        retry["response_format"] = {"type": "json_object"}
        schema = json.dumps(strict_json_schema(response_model), indent=2)
        retry["messages"] = [
            *payload["messages"],
            {
                "role": "system",
                "content": f"Reply with JSON only, matching this schema:\n{schema}",
            },
        ]
        return self._client.chat.completions.create(**retry)


def _parse_structured(
    text: str, response_model: type[T]
) -> tuple[BaseModel | None, dict[str, Any] | None]:
    """Parse a structured reply, tolerating code fences."""
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("```")[1]
        if candidate.startswith("json"):
            candidate = candidate[4:]
        candidate = candidate.strip()
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            raise ProviderError(f"model did not return JSON: {text[:200]}") from None
        try:
            data = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ProviderError(f"model returned invalid JSON: {exc}") from exc
    try:
        return response_model.model_validate(data), data
    except ValidationError as exc:
        raise ProviderError(f"structured output failed validation: {exc}") from exc
