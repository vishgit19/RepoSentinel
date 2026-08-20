"""Manual probe: exercise the provider layer against the real OpenAI API.

Run with:  python scripts/probe_provider.py
Requires OPENAI_API_KEY. Prints structured output, tool calling and cost.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from reposentinel.config import get_settings  # noqa: E402
from reposentinel.models.providers import Message, build_provider  # noqa: E402
from reposentinel.models.providers.openai_provider import strict_json_schema  # noqa: E402
from reposentinel.models.schemas import RepairPlan, TriageResult  # noqa: E402
from reposentinel.tools import registry  # noqa: E402


def main() -> int:
    settings = get_settings()
    if not settings.openai_api_key:
        print("FAIL: no OPENAI_API_KEY visible to the process")
        return 1
    print(f"key present (len={len(settings.openai_api_key)})")

    provider = build_provider("gpt-4.1-mini", settings)
    print(f"provider: {provider.describe()}")

    print("\n--- schema generation ---")
    schema = strict_json_schema(TriageResult)
    print(f"TriageResult required fields: {schema['required']}")
    print(f"additionalProperties: {schema['additionalProperties']}")

    print("\n--- 1. structured output (TriageResult) ---")
    response = provider.complete(
        [
            Message("system", "You are a software triage assistant. Be concise."),
            Message(
                "user",
                "Issue: Users with an expired session token are still being authenticated. "
                "Three tests in tests/test_auth.py fail. Classify this issue.",
            ),
        ],
        response_model=TriageResult,
    )
    triage = response.parsed
    print(f"kind={triage.issue_kind.value} confidence={triage.confidence}")
    print(f"summary={triage.summary}")
    print(f"search_terms={triage.search_terms}")
    print(
        f"tokens={response.usage.prompt_tokens}+{response.usage.completion_tokens} "
        f"cost=${response.usage.cost_usd:.6f} latency={response.duration_ms}ms"
    )

    print("\n--- 2. structured output with nested list (RepairPlan) ---")
    plan_response = provider.complete(
        [
            Message("system", "Plan an investigation in at most 5 steps."),
            Message("user", "Expired session tokens are still accepted by the auth middleware."),
        ],
        response_model=RepairPlan,
    )
    plan = plan_response.parsed
    print(f"goal={plan.goal}")
    for step in plan.steps:
        print(f"  {step.index}. {step.action}  [{step.tool_hint}]")

    print("\n--- 3. tool calling ---")
    tools = registry.openai_schemas(["search_symbols", "read_file", "run_targeted_tests"])
    print(f"offering {len(tools)} tools: {[t['function']['name'] for t in tools]}")
    tool_response = provider.complete(
        [
            Message(
                "system",
                "You are investigating a repository. Call a tool to locate the "
                "symbol responsible for token expiry.",
            ),
            Message("user", "Find where token expiry is decided. The symbol is is_expired."),
        ],
        tools=tools,
    )
    if tool_response.wants_tools:
        for call in tool_response.tool_calls:
            print(f"  -> {call.name}({call.arguments})")
    else:
        print(f"  no tool requested; text={tool_response.text[:200]}")

    total = response.usage + plan_response.usage + tool_response.usage
    print(
        f"\nTOTAL: {total.total_tokens} tokens, ${total.cost_usd:.6f}, "
        f"3 calls"
    )
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
