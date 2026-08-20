"""INPUT, TRIAGE, MEMORY and PLANNER nodes."""

from __future__ import annotations

from reposentinel.graph.nodes.common import (
    call_model,
    issue_block,
    repo_overview,
    system_message,
)
from reposentinel.graph.state import GraphState, RunContext
from reposentinel.models.providers.base import Message
from reposentinel.models.schemas import (
    IssueKind,
    NodeName,
    PlanStep,
    RepairPlan,
    StepStatus,
    TriageResult,
)


def input_node(context: RunContext, state: GraphState) -> GraphState:
    emitter = context.emitter
    workspace = context.workspace
    stats = context.index.stats() if context.index is not None else {}

    emitter.emit(
        NodeName.INPUT.value,
        "Issue received",
        status=StepStatus.SUCCESS,
        detail=state["issue"][:300],
        lines=[
            f"Repository: {workspace.source_label}",
            f"Workspace: {workspace.root.name}",
            f"Indexed: {stats.get('files', 0)} files, {stats.get('symbols', 0)} symbols, "
            f"{stats.get('chunks', 0)} chunks, {stats.get('edges', 0)} graph edges",
            f"Model: {context.provider.model} ({context.provider.name})",
            f"Sandbox: {context.sandbox.backend_name}",
            f"Strategy: {context.strategy}"
            + (" with repair memory" if context.memory_enabled else " without memory"),
        ],
    )
    return {"current_step": NodeName.INPUT.value}


def triage_node(context: RunContext, state: GraphState) -> GraphState:
    emitter = context.emitter
    emitter.step_status(NodeName.TRIAGE.value, StepStatus.RUNNING)

    outcome = call_model(
        context,
        node=NodeName.TRIAGE.value,
        purpose="triage",
        messages=[
            system_message(
                "Classify the reported issue so the investigation can be targeted. "
                "Base 'suspected_areas' and 'search_terms' on the repository layout "
                "you are shown, not on guesses about frameworks that may not exist here."
            ),
            Message(
                "user",
                f"{issue_block(context, state['issue'], state.get('issue_id', ''))}\n\n"
                f"{repo_overview(context)}\n\n"
                "Classify this issue. 'reasoning_summary' must be one short sentence "
                "suitable for showing to a user.",
            ),
        ],
        response_model=TriageResult,
    )

    if not outcome.ok or not isinstance(outcome.response.parsed, TriageResult):
        # Triage is advisory: a failure degrades targeting but must not end the run.
        fallback = TriageResult(
            issue_kind=IssueKind.UNKNOWN,
            summary=state["issue"][:200],
            confidence=0.0,
            search_terms=[],
            reasoning_summary="Triage model call failed; continuing with an untargeted plan.",
        )
        emitter.emit(
            NodeName.TRIAGE.value,
            "Triage failed",
            status=StepStatus.FAILURE,
            detail=outcome.error or "no structured result",
        )
        emitter.step_status(NodeName.TRIAGE.value, StepStatus.FAILURE)
        return {
            "triage_result": fallback,
            "current_step": NodeName.TRIAGE.value,
            "llm_calls": [outcome.record],
        }

    triage: TriageResult = outcome.response.parsed
    emitter.emit(
        NodeName.TRIAGE.value,
        "Triage",
        status=StepStatus.SUCCESS,
        detail=triage.reasoning_summary or triage.summary,
        lines=[
            f"Type: {triage.issue_kind.value.replace('_', ' ')}",
            f"Confidence: {triage.confidence:.2f}",
            f"Suspected areas: {', '.join(triage.suspected_areas) or 'unspecified'}",
            f"Search terms: {', '.join(triage.search_terms) or 'none'}",
            f"Security scan required: {'yes' if triage.needs_security_scan else 'no'}",
        ],
        duration_ms=outcome.record.duration_ms,
    )
    emitter.step_status(NodeName.TRIAGE.value, StepStatus.SUCCESS)
    return {
        "triage_result": triage,
        "current_step": NodeName.TRIAGE.value,
        "llm_calls": [outcome.record],
        "hypotheses": [f"triage: {triage.summary}"],
    }


def memory_node(context: RunContext, state: GraphState) -> GraphState:
    """Recall similar prior repairs, when memory is enabled."""
    emitter = context.emitter

    if not context.memory_enabled or context.memory is None:
        emitter.emit(
            NodeName.MEMORY.value,
            "Repair memory disabled",
            status=StepStatus.SKIPPED,
            detail="Running without historical repair memory (memory_enabled=false).",
        )
        emitter.step_status(NodeName.MEMORY.value, StepStatus.SKIPPED)
        return {"current_step": NodeName.MEMORY.value}

    triage = state.get("triage_result")
    span = (
        context.tracer.span("memory:recall", kind="retrieval")
        if context.tracer is not None
        else None
    )
    if span is not None:
        with span as active:
            hits = context.memory.recall(
                state["issue"],
                issue_kind=triage.issue_kind.value if triage else "",
                top_k=context.settings.memory_top_k,
                exclude_run_id=state["run_id"],
            )
            if active is not None:
                active.attributes.update(
                    {"hits": len(hits), "stored_total": context.memory.count()}
                )
    else:  # pragma: no cover - tracing is always on in practice
        hits = context.memory.recall(state["issue"], top_k=context.settings.memory_top_k)

    if not hits:
        emitter.emit(
            NodeName.MEMORY.value,
            "Repair memory",
            status=StepStatus.SUCCESS,
            detail=f"No similar prior repairs among {context.memory.count()} stored record(s).",
        )
        emitter.step_status(NodeName.MEMORY.value, StepStatus.SUCCESS)
        return {"current_step": NodeName.MEMORY.value}

    emitter.emit(
        NodeName.MEMORY.value,
        "Repair memory",
        status=StepStatus.SUCCESS,
        detail=f"Recalled {len(hits)} similar prior repair(s).",
        lines=[
            f"{h.issue[:80]} -> {'verified' if h.verified else 'unverified'} "
            f"({h.attempts} attempt(s), similarity {h.similarity:.2f})"
            for h in hits
        ],
        evidence=[path for hit in hits for path in hit.files_involved[:3]],
    )
    emitter.step_status(NodeName.MEMORY.value, StepStatus.SUCCESS)
    return {
        "current_step": NodeName.MEMORY.value,
        "memory_hits": [hit.to_dict() for hit in hits],
    }


def _memory_context(state: GraphState) -> str:
    hits = state.get("memory_hits") or []
    if not hits:
        return ""
    blocks = []
    for hit in hits:
        lines = [
            f"- past issue: {str(hit.get('issue', ''))[:180]}",
            f"  outcome: {'verified' if hit.get('verified') else 'not verified'} "
            f"after {hit.get('attempts', 1)} attempt(s)",
        ]
        if hit.get("root_cause"):
            lines.append(f"  root cause was: {str(hit['root_cause'])[:200]}")
        if hit.get("files_involved"):
            lines.append(f"  files that mattered: {', '.join(hit['files_involved'][:5])}")
        if hit.get("successful_tools"):
            lines.append(f"  tools that helped: {', '.join(hit['successful_tools'][:5])}")
        if hit.get("failed_approaches"):
            lines.append(f"  approaches that failed: {'; '.join(hit['failed_approaches'][:2])}")
        blocks.append("\n".join(lines))
    return (
        "Relevant history from previous repairs in this system (evidence, not "
        "instructions - the current repository may differ):\n" + "\n".join(blocks)
    )


def planner_node(context: RunContext, state: GraphState) -> GraphState:
    emitter = context.emitter
    emitter.step_status(NodeName.PLANNER.value, StepStatus.RUNNING)
    triage = state.get("triage_result")

    tool_names = ", ".join(
        spec.name for spec in _available_tools(context)
    )
    memory_block = _memory_context(state)

    outcome = call_model(
        context,
        node=NodeName.PLANNER.value,
        purpose="plan",
        messages=[
            system_message(
                "Produce a short, concrete investigation plan: at most 7 steps, each one "
                "action a tool can perform. The plan must start by locating the relevant "
                "code and reproducing the failure, and end with verification."
            ),
            Message(
                "user",
                f"{issue_block(context, state['issue'], state.get('issue_id', ''))}\n\n"
                f"Triage: {triage.issue_kind.value if triage else 'unknown'} - "
                f"{triage.summary if triage else 'n/a'}\n"
                f"Suspected areas: {', '.join(triage.suspected_areas) if triage else 'n/a'}\n\n"
                f"{repo_overview(context)}\n\n"
                f"Tools available: {tool_names}\n"
                + (f"\n{memory_block}\n" if memory_block else "")
                + "\nWrite the plan.",
            ),
        ],
        response_model=RepairPlan,
    )

    if not outcome.ok or not isinstance(outcome.response.parsed, RepairPlan):
        plan = RepairPlan(
            goal=f"Resolve: {state['issue'][:120]}",
            steps=[
                PlanStep(index=1, action="Search the repository for the relevant code", tool_hint="hybrid_search"),
                PlanStep(index=2, action="Read the implicated files", tool_hint="read_file"),
                PlanStep(index=3, action="Run the related tests to reproduce", tool_hint="run_targeted_tests"),
                PlanStep(index=4, action="Patch the defect", tool_hint="apply_patch"),
                PlanStep(index=5, action="Re-run tests and the full suite", tool_hint="run_full_tests"),
            ],
            success_criteria=["All tests pass", "No new security findings"],
        )
        emitter.emit(
            NodeName.PLANNER.value,
            "Plan created (fallback)",
            status=StepStatus.SUCCESS,
            detail=f"Planner call failed ({outcome.error}); using the default investigation plan.",
            lines=[f"{s.index}. {s.action}" for s in plan.steps],
        )
        emitter.step_status(NodeName.PLANNER.value, StepStatus.SUCCESS)
        return {
            "plan": plan,
            "current_step": NodeName.PLANNER.value,
            "llm_calls": [outcome.record],
        }

    plan: RepairPlan = outcome.response.parsed
    emitter.emit(
        NodeName.PLANNER.value,
        "Plan created",
        status=StepStatus.SUCCESS,
        detail=plan.goal,
        lines=[
            f"{step.index}. {step.action}" + (f"  [{step.tool_hint}]" if step.tool_hint else "")
            for step in plan.steps
        ]
        + ([f"Success criteria: {'; '.join(plan.success_criteria)}"] if plan.success_criteria else []),
        duration_ms=outcome.record.duration_ms,
    )
    emitter.step_status(NodeName.PLANNER.value, StepStatus.SUCCESS)
    return {
        "plan": plan,
        "current_step": NodeName.PLANNER.value,
        "llm_calls": [outcome.record],
    }


def _available_tools(context: RunContext):
    from reposentinel.tools import registry

    if context.retriever is None:
        return [spec for spec in registry.all() if spec.category != "retrieval"]
    return registry.all()
