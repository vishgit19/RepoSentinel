"""RETRIEVE CONTEXT and TOOL EXECUTION nodes."""

from __future__ import annotations

from reposentinel.graph.nodes.common import (
    call_model,
    issue_block,
    repo_overview,
    system_message,
)
from reposentinel.graph.state import GraphState, LimitExceeded, RunContext
from reposentinel.models.providers.base import Message
from reposentinel.models.schemas import (
    NodeName,
    RetrievalDecision,
    SafetyEvent,
    StepStatus,
    ToolCallRecord,
)
from reposentinel.sandbox.guardrails import scan_for_injection
from reposentinel.strategies import get_strategy
from reposentinel.tools.base import execute_tool, registry

MAX_TOOL_ITERATIONS = 8
# Tools the investigation loop may use. Patching and verification happen in
# their own nodes so the timeline stays legible and the patch is auditable.
INVESTIGATION_TOOLS = (
    "hybrid_search",
    "semantic_search",
    "bm25_search",
    "expand_dependencies",
    "search_docs",
    "list_files",
    "read_file",
    "search_code",
    "search_symbols",
    "inspect_imports",
    "find_callers",
    "find_callees",
    "git_history",
    "run_targeted_tests",
    "run_security_scan",
    "run_lint",
)


def _record_tool(context: RunContext, state_updates: dict, record: ToolCallRecord) -> None:
    budget = context.budget
    budget.tool_calls += 1
    if record.blocked:
        budget.blocked_tool_calls += 1

    status = StepStatus.SUCCESS if record.ok else StepStatus.FAILURE
    if record.blocked:
        status = StepStatus.BLOCKED

    arguments = ", ".join(
        f"{key}={_short(value)}" for key, value in list(record.arguments.items())[:3]
    )
    context.emitter.emit(
        NodeName.TOOLS.value,
        f"{record.name}({arguments})",
        status=status,
        detail=record.summary or (record.error or ""),
        tool_call=record,
        duration_ms=record.duration_ms,
    )
    state_updates.setdefault("tool_history", []).append(record)


def _short(value: object, limit: int = 60) -> str:
    text = str(value)
    return text if len(text) <= limit else f"{text[:limit]}..."


def retrieve_node(context: RunContext, state: GraphState) -> GraphState:
    """Decide whether retrieval is needed, then retrieve only what was asked for."""
    emitter = context.emitter
    strategy = get_strategy(context.strategy)
    mode = strategy.retrieval

    if mode is None:
        seeds = []
        if context.benchmark is not None:
            seeds = list(context.benchmark.baseline_seed_files)
        chunks = []
        if seeds and context.retriever is not None:
            chunks = context.retriever.chunks_for_paths(seeds, query="seed")
            context.budget.retrieved_chunks += len(chunks)
        emitter.emit(
            NodeName.RETRIEVE.value,
            "Retrieval skipped",
            status=StepStatus.SKIPPED,
            detail=(
                f"Baseline {strategy.baseline} ('{strategy.label}') supplies seed "
                "files directly without retrieval."
                + (f" Seeded {len(chunks)} chunk(s) from {', '.join(seeds)}." if chunks else "")
            ),
            evidence=[chunk.location for chunk in chunks],
        )
        emitter.step_status(NodeName.RETRIEVE.value, StepStatus.SKIPPED)
        updates: dict = {
            "current_step": NodeName.RETRIEVE.value,
            "retrieved_context": chunks,
            "files_inspected": list(seeds),
            "evidence": [chunk.location for chunk in chunks],
        }
        safety = _injection_events(chunks)
        if safety:
            updates["safety_events"] = safety
            _emit_injection(emitter, safety)
        return updates

    if context.retriever is None:
        emitter.emit(
            NodeName.RETRIEVE.value,
            "Retrieval skipped",
            status=StepStatus.SKIPPED,
            detail="No retriever is available for this run.",
        )
        emitter.step_status(NodeName.RETRIEVE.value, StepStatus.SKIPPED)
        return {"current_step": NodeName.RETRIEVE.value}

    emitter.step_status(NodeName.RETRIEVE.value, StepStatus.RUNNING)
    triage = state.get("triage_result")
    plan = state.get("plan")

    decision_outcome = call_model(
        context,
        node=NodeName.RETRIEVE.value,
        purpose="retrieval_decision",
        messages=[
            system_message(
                "Decide whether repository retrieval is needed and what to search for. "
                "Write 1-3 focused queries. A good query names the behaviour and the "
                "likely identifiers, e.g. 'session token expiry comparison is_expired'. "
                "Do not request the whole repository."
            ),
            Message(
                "user",
                f"{issue_block(context, state['issue'], state.get('issue_id', ''))}\n\n"
                f"Triage: {triage.issue_kind.value if triage else 'unknown'}; "
                f"search terms suggested: {', '.join(triage.search_terms) if triage else 'none'}\n"
                f"Plan step 1: {plan.steps[0].action if plan and plan.steps else 'n/a'}\n\n"
                f"{repo_overview(context, limit=40)}",
            ),
        ],
        response_model=RetrievalDecision,
    )

    updates: dict = {"current_step": NodeName.RETRIEVE.value, "llm_calls": [decision_outcome.record]}

    decision = (
        decision_outcome.response.parsed
        if decision_outcome.ok and isinstance(decision_outcome.response.parsed, RetrievalDecision)
        else None
    )
    if decision is None:
        # Fall back to triage's search terms rather than skipping retrieval.
        queries = [state["issue"][:200]]
        if triage and triage.search_terms:
            queries = [" ".join(triage.search_terms[:5])] + queries
        decision = RetrievalDecision(
            needs_retrieval=True,
            queries=queries[:2],
            reason="Retrieval decision call failed; falling back to triage search terms.",
        )

    if not decision.needs_retrieval or not decision.queries:
        emitter.emit(
            NodeName.RETRIEVE.value,
            "Retrieval not required",
            status=StepStatus.SKIPPED,
            detail=decision.reason or "The agent judged retrieval unnecessary.",
        )
        emitter.step_status(NodeName.RETRIEVE.value, StepStatus.SKIPPED)
        return updates

    all_chunks = []
    all_paths: list[str] = []
    for query in decision.queries[:3]:
        try:
            context.budget.check()
        except LimitExceeded:
            break

        span = (
            context.tracer.span(
                f"retrieval:{mode}", kind="retrieval", attributes={"query": query, "mode": mode}
            )
            if context.tracer is not None
            else None
        )
        if span is not None:
            with span as active:
                result = context.retriever.retrieve(query, mode=mode)
                if active is not None:
                    active.attributes.update(
                        {
                            "bm25_candidates": result.stats.bm25_candidates,
                            "dense_candidates": result.stats.dense_candidates,
                            "merged": result.stats.merged_candidates,
                            "graph_expanded": result.stats.graph_expanded,
                            "returned": len(result.chunks),
                            "rerank_backend": result.stats.rerank_backend,
                            "total_tokens": result.stats.usage.total_tokens,
                            "cost_usd": result.stats.usage.cost_usd,
                        }
                    )
        else:  # pragma: no cover
            result = context.retriever.retrieve(query, mode=mode)

        budget = context.budget
        budget.retrieval_queries += 1
        budget.retrieved_chunks += len(result.chunks)
        # Reranking spends tokens; charge them to the run.
        budget.prompt_tokens += result.stats.usage.prompt_tokens
        budget.completion_tokens += result.stats.usage.completion_tokens
        budget.cost_usd += result.stats.usage.cost_usd
        budget.llm_calls += result.stats.llm_calls

        all_chunks.extend(result.chunks)
        for path in result.paths:
            if path not in all_paths:
                all_paths.append(path)

        emitter.emit(
            NodeName.RETRIEVE.value,
            "Retrieval",
            status=StepStatus.SUCCESS,
            detail=f'Query: "{query}"',
            lines=[
                f"Mode: {result.mode} (bm25={result.stats.bm25_candidates}, "
                f"dense={result.stats.dense_candidates}, merged={result.stats.merged_candidates}, "
                f"graph+{result.stats.graph_expanded})",
                f"Reranker: {result.stats.rerank_backend}",
                "Retrieved:",
                *[
                    f"  {chunk.path}::{chunk.symbol or '(module)'} "
                    f"L{chunk.start_line}-{chunk.end_line}"
                    f"{f' [{chunk.provenance.retriever}]' if chunk.provenance else ''}"
                    for chunk in result.chunks
                ],
            ],
            evidence=result.evidence(),
            duration_ms=result.stats.total_ms,
        )

    emitter.step_status(NodeName.RETRIEVE.value, StepStatus.SUCCESS)
    emitter.state_patch(
        {
            "retrieved_files": all_paths,
            "retrieved_chunks": [
                {
                    "path": chunk.path,
                    "symbol": chunk.symbol,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "content": chunk.content,
                    "provenance": chunk.provenance.model_dump() if chunk.provenance else None,
                }
                for chunk in all_chunks
            ],
        }
    )

    updates["retrieved_context"] = all_chunks
    updates["files_inspected"] = all_paths
    updates["evidence"] = [chunk.location for chunk in all_chunks]

    safety = _injection_events(all_chunks)
    if safety:
        updates["safety_events"] = safety
        _emit_injection(emitter, safety)
    return updates


def _injection_events(chunks) -> list[SafetyEvent]:
    safety: list[SafetyEvent] = []
    seen: set[tuple[str, str]] = set()
    for chunk in chunks:
        for match in scan_for_injection(chunk.content, source=chunk.path):
            key = (match.label, chunk.path)
            if key in seen:
                continue
            seen.add(key)
            safety.append(
                SafetyEvent(
                    kind="prompt_injection",
                    detail=f"{match.label} in {chunk.path}: \"{match.excerpt[:160]}\"",
                    source=chunk.path,
                    severity="critical",
                )
            )
    return safety


def _emit_injection(emitter, safety: list[SafetyEvent]) -> None:
    emitter.emit(
        NodeName.RETRIEVE.value,
        "Prompt injection detected",
        status=StepStatus.SUCCESS,
        detail=(
            f"{len(safety)} injection pattern(s) in retrieved files were treated "
            "as hostile data and ignored."
        ),
        lines=[event.detail for event in safety],
    )


def _context_digest(state: GraphState, max_chars: int = 9000) -> str:
    """Render already-retrieved chunks for the tool loop's prompt."""
    chunks = state.get("retrieved_context") or []
    if not chunks:
        return "No code has been retrieved yet."
    blocks: list[str] = []
    budget = max_chars
    for chunk in chunks:
        label = f"{chunk.path}::{chunk.symbol}" if chunk.symbol else chunk.path
        block = (
            f"--- {label} (lines {chunk.start_line}-{chunk.end_line}) ---\n{chunk.content}"
        )
        if len(block) > budget:
            break
        blocks.append(block)
        budget -= len(block)
    return "\n\n".join(blocks)


def tools_node(context: RunContext, state: GraphState) -> GraphState:
    """The agentic investigation loop: the model drives the tools."""
    emitter = context.emitter
    updates: dict = {"current_step": NodeName.TOOLS.value, "llm_calls": [], "tool_history": []}

    strategy = get_strategy(context.strategy)
    if not strategy.tools:
        detail = (
            f"Baseline {strategy.baseline} ('{strategy.label}') does not call tools"
            + (
                "; it works from retrieved context alone."
                if strategy.retrieval
                else "; it works from the seed files alone."
            )
        )
        emitter.emit(
            NodeName.TOOLS.value,
            "Investigation skipped",
            status=StepStatus.SKIPPED,
            detail=detail,
        )
        emitter.step_status(NodeName.TOOLS.value, StepStatus.SKIPPED)
        return updates

    emitter.step_status(NodeName.TOOLS.value, StepStatus.RUNNING)
    plan = state.get("plan")
    available = [name for name in INVESTIGATION_TOOLS if registry.get(name) is not None]
    if context.retriever is None:
        available = [
            name
            for name in available
            if registry.get(name).category != "retrieval"  # type: ignore[union-attr]
        ]
    tool_schemas = registry.openai_schemas(available)

    messages = [
        system_message(
            "Investigate with tools until you can name the exact defect and the exact "
            "lines responsible. Call one or more tools per turn. Reproduce the failure "
            "by running the relevant tests before proposing a fix. When you have enough "
            "evidence, stop calling tools and reply with a short factual summary of what "
            "you found: the file, the symbol, the line numbers and the observed failure."
        ),
        Message(
            "user",
            f"{issue_block(context, state['issue'], state.get('issue_id', ''))}\n\n"
            f"Plan:\n"
            + "\n".join(f"  {s.index}. {s.action}" for s in (plan.steps if plan else []))
            + f"\n\n{repo_overview(context, limit=40)}\n\n"
            f"Code already retrieved:\n{_context_digest(state)}\n\n"
            "Investigate now.",
        ),
    ]

    findings = ""
    for iteration in range(1, MAX_TOOL_ITERATIONS + 1):
        try:
            context.budget.check()
        except LimitExceeded as exc:
            emitter.emit(
                NodeName.TOOLS.value,
                "Investigation stopped by limit",
                status=StepStatus.BLOCKED,
                detail=str(exc),
            )
            break

        outcome = call_model(
            context,
            node=NodeName.TOOLS.value,
            purpose=f"tool_selection_{iteration}",
            messages=messages,
            tools=tool_schemas,
        )
        updates["llm_calls"].append(outcome.record)

        if not outcome.ok:
            emitter.emit(
                NodeName.TOOLS.value,
                "Investigation model call failed",
                status=StepStatus.FAILURE,
                detail=outcome.error or "",
            )
            break

        response = outcome.response
        if not response.wants_tools:
            findings = response.text.strip()
            if findings:
                emitter.emit(
                    NodeName.TOOLS.value,
                    "Investigation complete",
                    status=StepStatus.SUCCESS,
                    detail=findings[:600],
                    duration_ms=outcome.record.duration_ms,
                )
            break

        messages.append(
            Message("assistant", response.text or "", tool_calls=response.tool_calls)
        )

        for invocation in response.tool_calls:
            tool_span = (
                context.tracer.span(
                    f"tool:{invocation.name}",
                    kind="tool",
                    attributes={"arguments": invocation.arguments},
                )
                if context.tracer is not None
                else None
            )
            if tool_span is not None:
                with tool_span as active:
                    record = execute_tool(
                        invocation.name, invocation.arguments, context.tool_context
                    )
                    if active is not None:
                        # A tool that ran and reported a failing test is not a
                        # trace error; only a tool that could not run is.
                        active.ok = record.executed
                        active.error = record.error if not record.executed else None
                        active.attributes.update(
                            {
                                "summary": record.summary,
                                "blocked": record.blocked,
                                "finding_ok": record.ok,
                            }
                        )
            else:  # pragma: no cover
                record = execute_tool(invocation.name, invocation.arguments, context.tool_context)

            _record_tool(context, updates, record)
            messages.append(
                Message(
                    "tool",
                    record.output or record.summary or (record.error or "(no output)"),
                    tool_call_id=invocation.call_id,
                    name=invocation.name,
                )
            )

    emitter.step_status(NodeName.TOOLS.value, StepStatus.SUCCESS)

    inspected = sorted(context.tool_context.files_inspected)
    if inspected:
        emitter.state_patch({"files_inspected": inspected})
    updates["files_inspected"] = inspected
    if findings:
        updates["hypotheses"] = [f"investigation: {findings[:400]}"]
    safety = list(context.tool_context.safety_events)
    if safety:
        updates["safety_events"] = safety
        context.tool_context.safety_events = []
        for event in safety:
            emitter.emit(
                NodeName.TOOLS.value,
                f"Guardrail: {event.kind.replace('_', ' ')}",
                status=StepStatus.BLOCKED if event.severity == "critical" else StepStatus.SUCCESS,
                detail=event.detail[:400],
            )
    return updates
