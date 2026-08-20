"""The LangGraph repair workflow.

    INPUT -> TRIAGE -> MEMORY -> PLANNER -> RETRIEVE -> TOOLS -> ROOT CAUSE
          -> PATCH -> CHECKS -+--(pass)--> VERIFIER -> APPROVAL -> REPORT
                              |
                              +--(fail)--> REFLECT --(retry)--> PATCH
                                                 \\--(give up)--> REPORT

The loop is the point: a failed check run is diagnosed and fed back into
PATCH, bounded by ``max_repair_attempts`` and by the run budget.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from reposentinel.graph.nodes.finalize import approval_node, report_node, verifier_node
from reposentinel.graph.nodes.investigate import retrieve_node, tools_node
from reposentinel.graph.nodes.repair import (
    checks_node,
    checks_passed,
    patch_node,
    reflect_node,
    root_cause_node,
    should_retry,
)
from reposentinel.graph.nodes.understand import (
    input_node,
    memory_node,
    planner_node,
    triage_node,
)
from reposentinel.graph.state import GraphState, LimitExceeded, RunContext
from reposentinel.models.schemas import NodeName, RunStatus, StepStatus

NODE_FUNCTIONS = {
    NodeName.INPUT.value: input_node,
    NodeName.TRIAGE.value: triage_node,
    NodeName.MEMORY.value: memory_node,
    NodeName.PLANNER.value: planner_node,
    NodeName.RETRIEVE.value: retrieve_node,
    NodeName.TOOLS.value: tools_node,
    NodeName.ROOT_CAUSE.value: root_cause_node,
    NodeName.PATCH.value: patch_node,
    NodeName.CHECKS.value: checks_node,
    NodeName.REFLECT.value: reflect_node,
    NodeName.VERIFIER.value: verifier_node,
    NodeName.APPROVAL.value: approval_node,
    NodeName.REPORT.value: report_node,
}


def _guarded(node_name: str, function, context: RunContext):
    """Wrap a node so a limit breach or crash ends the run cleanly.

    The wrapper deliberately does *not* use ``functools.wraps``: LangGraph
    inspects a node callable's signature to discover per-node input schemas,
    and copying the inner ``(context, state)`` annotations would make it adopt
    ``RunContext`` as a state schema and register its fields as channels.
    """

    def wrapper(state: GraphState) -> dict[str, Any]:
        if state.get("status") in {RunStatus.ABORTED.value, RunStatus.FAILED.value}:
            return {}
        try:
            return function(context, state)
        except LimitExceeded as exc:
            context.emitter.emit(
                node_name,
                f"Run stopped: {exc.which}",
                status=StepStatus.BLOCKED,
                detail=f"Limit reached ({exc.detail}). The run stops with a partial result.",
            )
            context.emitter.step_status(node_name, StepStatus.BLOCKED)
            return {
                "status": RunStatus.ABORTED.value,
                "failure_reason": f"limit exceeded: {exc.which} ({exc.detail})",
                "safety_events": [
                    _limit_event(exc),
                ],
            }
        except Exception as exc:  # noqa: BLE001 - a node crash must not hang the run
            context.emitter.emit(
                node_name,
                f"Node '{node_name}' failed",
                status=StepStatus.FAILURE,
                detail=f"{type(exc).__name__}: {exc}",
            )
            context.emitter.step_status(node_name, StepStatus.FAILURE)
            return {
                "status": RunStatus.FAILED.value,
                "failure_reason": f"{node_name}: {type(exc).__name__}: {exc}",
            }

    wrapper.__name__ = f"{node_name}_node"
    wrapper.__doc__ = getattr(function, "__doc__", None)
    return wrapper


def _limit_event(exc: LimitExceeded):
    from reposentinel.models.schemas import SafetyEvent

    return SafetyEvent(
        kind="limit_exceeded",
        detail=f"{exc.which}: {exc.detail}",
        source="budget",
        severity="warning",
    )


def _route_after_checks(context: RunContext):
    def router(state: GraphState) -> str:
        if state.get("status") in {RunStatus.ABORTED.value, RunStatus.FAILED.value}:
            return NodeName.REPORT.value
        return NodeName.VERIFIER.value if checks_passed(state) else NodeName.REFLECT.value

    return router


def _route_after_reflect(context: RunContext):
    def router(state: GraphState) -> str:
        if state.get("status") in {RunStatus.ABORTED.value, RunStatus.FAILED.value}:
            return NodeName.REPORT.value
        return (
            NodeName.PATCH.value
            if should_retry(context, state) == "retry"
            else NodeName.REPORT.value
        )

    return router


def _route_after_verifier(context: RunContext):
    def router(state: GraphState) -> str:
        if state.get("status") in {RunStatus.ABORTED.value, RunStatus.FAILED.value}:
            return NodeName.REPORT.value
        return NodeName.APPROVAL.value

    return router


def build_graph(context: RunContext):
    """Compile the repair graph with nodes bound to this run's context."""
    graph = StateGraph(GraphState)

    for name, function in NODE_FUNCTIONS.items():
        graph.add_node(name, _guarded(name, function, context))

    graph.add_edge(START, NodeName.INPUT.value)
    graph.add_edge(NodeName.INPUT.value, NodeName.TRIAGE.value)
    graph.add_edge(NodeName.TRIAGE.value, NodeName.MEMORY.value)
    graph.add_edge(NodeName.MEMORY.value, NodeName.PLANNER.value)
    graph.add_edge(NodeName.PLANNER.value, NodeName.RETRIEVE.value)
    graph.add_edge(NodeName.RETRIEVE.value, NodeName.TOOLS.value)
    graph.add_edge(NodeName.TOOLS.value, NodeName.ROOT_CAUSE.value)
    graph.add_edge(NodeName.ROOT_CAUSE.value, NodeName.PATCH.value)
    graph.add_edge(NodeName.PATCH.value, NodeName.CHECKS.value)

    graph.add_conditional_edges(
        NodeName.CHECKS.value,
        _route_after_checks(context),
        {
            NodeName.VERIFIER.value: NodeName.VERIFIER.value,
            NodeName.REFLECT.value: NodeName.REFLECT.value,
            NodeName.REPORT.value: NodeName.REPORT.value,
        },
    )
    graph.add_conditional_edges(
        NodeName.REFLECT.value,
        _route_after_reflect(context),
        {
            NodeName.PATCH.value: NodeName.PATCH.value,
            NodeName.REPORT.value: NodeName.REPORT.value,
        },
    )
    graph.add_conditional_edges(
        NodeName.VERIFIER.value,
        _route_after_verifier(context),
        {
            NodeName.APPROVAL.value: NodeName.APPROVAL.value,
            NodeName.REPORT.value: NodeName.REPORT.value,
        },
    )
    graph.add_edge(NodeName.APPROVAL.value, NodeName.REPORT.value)
    graph.add_edge(NodeName.REPORT.value, END)

    # The recursion limit must accommodate max_repair_attempts loops through
    # PATCH -> CHECKS -> REFLECT plus the linear prologue and epilogue.
    return graph.compile()


def graph_topology() -> dict[str, Any]:
    """Static description of the workflow, for the UI's graph view."""
    return {
        "nodes": [
            {"id": name, "label": name.replace("_", " ").title()}
            for name in NODE_FUNCTIONS
        ],
        "edges": [
            {"from": "input", "to": "triage"},
            {"from": "triage", "to": "memory"},
            {"from": "memory", "to": "planner"},
            {"from": "planner", "to": "retrieve"},
            {"from": "retrieve", "to": "tools"},
            {"from": "tools", "to": "root_cause"},
            {"from": "root_cause", "to": "patch"},
            {"from": "patch", "to": "checks"},
            {"from": "checks", "to": "verifier", "label": "checks pass"},
            {"from": "checks", "to": "reflect", "label": "checks fail"},
            {"from": "reflect", "to": "patch", "label": "retry"},
            {"from": "reflect", "to": "report", "label": "give up"},
            {"from": "verifier", "to": "approval"},
            {"from": "approval", "to": "report"},
        ],
    }
