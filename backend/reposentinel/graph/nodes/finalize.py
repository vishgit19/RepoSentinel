"""VERIFIER, HUMAN APPROVAL and FINAL REPORT nodes.

Verification is deliberately two-part: a set of deterministic gates that a
model cannot talk its way past, and then a model judgement about behaviour
preservation and residual risk. If the gates fail, the run is not verified
regardless of what the model says.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from reposentinel.github import maybe_open_pull_request
from reposentinel.graph.nodes.common import call_model, issue_block, system_message, untrusted_block
from reposentinel.graph.state import GraphState, RunContext
from reposentinel.memory.store import MemoryRecord
from reposentinel.models.providers.base import Message
from reposentinel.models.schemas import (
    FinalReport,
    NodeName,
    RunStatus,
    StepStatus,
    VerificationVerdict,
    new_id,
)


@dataclass
class ApprovalGate:
    """A real human-in-the-loop barrier.

    The graph thread blocks here until the API records a decision. Nothing that
    touches a remote repository happens before this returns approved. Human
    review is not charged against the run's wall-clock budget.
    """

    event: threading.Event = field(default_factory=threading.Event)
    approved: bool | None = None
    note: str = ""
    decided_at: float | None = None

    def decide(self, approved: bool, note: str = "") -> None:
        self.approved = approved
        self.note = note
        self.decided_at = time.time()
        self.event.set()

    def wait(self, timeout: float | None = None) -> bool | None:
        signaled = self.event.wait(timeout=timeout)
        if not signaled:
            return None
        return self.approved


def _deterministic_gates(state: GraphState) -> tuple[bool, list[str], list[str]]:
    """Machine-checkable evidence. Returns (passed, satisfied, failures)."""
    satisfied: list[str] = []
    failures: list[str] = []

    patches = state.get("patches") or []
    if patches and patches[-1].applied:
        satisfied.append(
            f"Patch applied to {len(patches[-1].files_changed)} file(s) "
            f"(+{patches[-1].lines_added} -{patches[-1].lines_removed})"
        )
    else:
        failures.append("No patch was successfully applied")

    reports = state.get("test_results") or []
    targeted = [r for r in reports if r.scope == "targeted"]
    full = [r for r in reports if r.scope == "full"]

    if targeted:
        latest = targeted[-1]
        if latest.ok:
            satisfied.append(f"Targeted tests: {latest.passed} passed")
        else:
            failures.append(f"Targeted tests: {latest.failed} failed, {latest.errors} error(s)")
    if full:
        latest = full[-1]
        if latest.ok:
            satisfied.append(f"Full suite: {latest.passed}/{latest.total} passed (no regressions)")
        else:
            failures.append(f"Full suite: {latest.failed} failed, {latest.errors} error(s)")
    else:
        failures.append("The full test suite was never run")

    lint = state.get("lint_results") or []
    if lint:
        if lint[-1].ok:
            satisfied.append("Ruff: passed")
        else:
            satisfied.append(f"Ruff: {lint[-1].issue_count} issue(s) (non-blocking)")

    security = state.get("security_results") or []
    if security:
        latest = security[-1]
        if latest.ok:
            satisfied.append(
                f"Security scan ({latest.backend}): "
                f"{'no findings' if not latest.findings else f'{len(latest.findings)} non-blocking'}"
            )
        else:
            failures.append(
                f"Security scan ({latest.backend}): {latest.blocking_count} blocking finding(s)"
            )

    injections = [e for e in (state.get("safety_events") or []) if e.kind == "prompt_injection"]
    if injections:
        satisfied.append(
            f"Prompt-injection attempts detected and ignored: {len(injections)}"
        )

    return not failures, satisfied, failures


def verifier_node(context: RunContext, state: GraphState) -> GraphState:
    emitter = context.emitter
    emitter.step_status(NodeName.VERIFIER.value, StepStatus.RUNNING)

    gates_passed, satisfied, failures = _deterministic_gates(state)
    patches = state.get("patches") or []
    diff = patches[-1].diff if patches else ""
    root_cause = state.get("root_cause_analysis")

    outcome = call_model(
        context,
        node=NodeName.VERIFIER.value,
        purpose="verify",
        messages=[
            system_message(
                "Review the final patch as a sceptical reviewer. Judge whether it fixes "
                "the reported root cause without changing unrelated behaviour, and list "
                "any residual risk. The deterministic evidence below is authoritative: "
                "do not claim tests passed if they did not."
            ),
            Message(
                "user",
                f"{issue_block(context, state['issue'], state.get('issue_id', ''))}\n\n"
                f"Root cause: {root_cause.statement if root_cause else 'not established'}\n\n"
                f"Final diff:\n{untrusted_block(diff or '(no diff)', 'final patch')}\n\n"
                f"Evidence collected:\n"
                + "\n".join(f"  PASS {item}" for item in satisfied)
                + "\n"
                + "\n".join(f"  FAIL {item}" for item in failures),
            ),
        ],
        response_model=VerificationVerdict,
    )

    updates: dict = {
        "current_step": NodeName.VERIFIER.value,
        "llm_calls": [outcome.record],
    }

    if outcome.ok and isinstance(outcome.response.parsed, VerificationVerdict):
        verdict: VerificationVerdict = outcome.response.parsed
    else:
        verdict = VerificationVerdict(
            verified=gates_passed,
            explanation=(
                "The reviewer model was unavailable; the verdict reflects the "
                "deterministic checks only."
            ),
            remaining_risks=["Model review of the patch did not complete."],
        )

    # The gates have the final say.
    final_verified = bool(gates_passed and verdict.verified)
    verdict = verdict.model_copy(update={"verified": final_verified})

    emitter.emit(
        NodeName.VERIFIER.value,
        "Solution verified" if final_verified else "Verification failed",
        status=StepStatus.SUCCESS if final_verified else StepStatus.FAILURE,
        detail=verdict.explanation,
        lines=[
            *[f"PASS {item}" for item in satisfied],
            *[f"FAIL {item}" for item in failures],
            *[f"Risk: {risk}" for risk in verdict.remaining_risks[:4]],
            f"Behaviour preserved: {'yes' if verdict.behaviour_preserved else 'unclear'}",
        ],
        evidence=satisfied,
        duration_ms=outcome.record.duration_ms,
    )
    emitter.step_status(
        NodeName.VERIFIER.value, StepStatus.SUCCESS if final_verified else StepStatus.FAILURE
    )

    updates["verification"] = verdict
    updates["verification_status"] = "verified" if final_verified else "failed"
    updates["metrics"] = {"gates_satisfied": satisfied, "gates_failed": failures}
    return updates


def approval_node(context: RunContext, state: GraphState) -> GraphState:
    """Block for a human decision before anything leaves the sandbox."""
    emitter = context.emitter
    verification = state.get("verification")
    verified = bool(verification and verification.verified)

    if context.auto_approve or not context.settings.require_human_approval:
        emitter.emit(
            NodeName.APPROVAL.value,
            "Approval auto-granted",
            status=StepStatus.SUCCESS,
            detail=(
                "auto_approve was requested for this run (used by the evaluation "
                "harness); no remote repository is modified either way."
            ),
        )
        emitter.step_status(NodeName.APPROVAL.value, StepStatus.SUCCESS)
        return {"current_step": NodeName.APPROVAL.value, "status": RunStatus.APPROVED.value}

    gate = context.approval_gate
    if gate is None:
        emitter.emit(
            NodeName.APPROVAL.value,
            "Approval skipped",
            status=StepStatus.SKIPPED,
            detail="No approval gate was attached to this run.",
        )
        return {"current_step": NodeName.APPROVAL.value}

    emitter.emit(
        NodeName.APPROVAL.value,
        "Awaiting human approval",
        status=StepStatus.RUNNING,
        detail=(
            "The patch is ready for review. Approve to accept it, or reject to "
            "record it as declined. Pushing to a remote repository requires this "
            "approval and is never automatic."
        ),
        lines=[
            f"Verified: {'yes' if verified else 'no'}",
            f"Files changed: {', '.join((state.get('patches') or [])[-1].files_changed) if state.get('patches') else 'none'}",
        ],
    )
    emitter.step_status(NodeName.APPROVAL.value, StepStatus.RUNNING)
    emitter.status(RunStatus.AWAITING_APPROVAL.value, "waiting for human decision")

    # Human review is not part of the run's compute budget. A remaining-wall-clock
    # timeout made Approve a no-op: wait() returned immediately once the
    # investigation had already consumed the limit, the graph wrote the report
    # while the UI still showed the approval bar, and later clicks woke nobody.
    decision = gate.wait(timeout=None)

    if decision is None:
        emitter.emit(
            NodeName.APPROVAL.value,
            "Approval timed out",
            status=StepStatus.BLOCKED,
            detail="No decision was recorded; the patch stays in the sandbox.",
        )
        emitter.step_status(NodeName.APPROVAL.value, StepStatus.BLOCKED)
        return {
            "current_step": NodeName.APPROVAL.value,
            "status": RunStatus.AWAITING_APPROVAL.value,
        }

    emitter.emit(
        NodeName.APPROVAL.value,
        "Patch approved" if decision else "Patch rejected",
        status=StepStatus.SUCCESS if decision else StepStatus.FAILURE,
        detail=gate.note or ("Approved by a human reviewer." if decision else "Rejected by a human reviewer."),
    )
    emitter.step_status(
        NodeName.APPROVAL.value, StepStatus.SUCCESS if decision else StepStatus.FAILURE
    )
    emitter.status(
        (RunStatus.APPROVED if decision else RunStatus.REJECTED).value,
        "approved by reviewer" if decision else "rejected by reviewer",
    )

    if decision:
        _maybe_open_pr(context, state)

    return {
        "current_step": NodeName.APPROVAL.value,
        "status": (RunStatus.APPROVED if decision else RunStatus.REJECTED).value,
    }


def _maybe_open_pr(context: RunContext, state: GraphState) -> None:
    """Attempt a GitHub PR only after a human approved. Usually a documented no-op."""
    patches = state.get("patches") or []
    latest = patches[-1] if patches else None
    result = maybe_open_pull_request(
        workspace_root=context.workspace.root,
        settings=context.settings,
        title=(latest.summary if latest else "RepoSentinel repair")[:80],
        body=(state.get("issue") or "")[:1500],
        head=f"reposentinel/{context.run_id}",
    )
    context.emitter.emit(
        NodeName.APPROVAL.value,
        "Pull request opened" if result.opened else "No pull request opened",
        status=StepStatus.SUCCESS if result.opened else StepStatus.SKIPPED,
        detail=result.url or result.reason,
        lines=[f"url: {result.url}"] if result.url else [result.reason],
    )


def report_node(context: RunContext, state: GraphState) -> GraphState:
    emitter = context.emitter
    emitter.step_status(NodeName.REPORT.value, StepStatus.RUNNING)

    patches = state.get("patches") or []
    latest = patches[-1] if patches else None
    verification = state.get("verification")
    root_cause = state.get("root_cause_analysis")
    reports = state.get("test_results") or []
    security = state.get("security_results") or []
    gates = state.get("metrics") or {}

    validation: list[str] = list(gates.get("gates_satisfied", []))
    if not validation:
        for report in reports:
            validation.append(
                f"{report.scope} tests: {report.passed} passed, {report.failed} failed"
            )
        if security:
            validation.append(
                f"security scan ({security[-1].backend}): {len(security[-1].findings)} finding(s)"
            )

    evidence = list(dict.fromkeys(state.get("evidence") or []))[:20]
    injections = [e for e in (state.get("safety_events") or []) if e.kind == "prompt_injection"]

    report = FinalReport(
        problem=state["issue"],
        root_cause=root_cause.statement if root_cause else "Not established.",
        changed_files=latest.files_changed if latest else [],
        explanation=(
            verification.explanation
            if verification
            else (latest.summary if latest else "No patch was produced.")
        ),
        evidence=evidence,
        validation_performed=validation,
        remaining_risks=(
            list(verification.remaining_risks) if verification else ["Run did not reach verification."]
        )
        + (
            [
                f"{len(injections)} prompt-injection attempt(s) were found in repository "
                f"content and ignored; the affected files may warrant review."
            ]
            if injections
            else []
        ),
        diff=latest.diff if latest else "",
        verified=bool(verification and verification.verified),
    )

    status = state.get("status", RunStatus.RUNNING.value)
    if status == RunStatus.RUNNING.value:
        status = RunStatus.SUCCEEDED.value if report.verified else RunStatus.FAILED.value

    emitter.emit(
        NodeName.REPORT.value,
        "Final report",
        status=StepStatus.SUCCESS if report.verified else StepStatus.FAILURE,
        detail=report.explanation[:400],
        lines=[
            f"Changed: {', '.join(report.changed_files) or 'nothing'}",
            *[f"Validated: {item}" for item in report.validation_performed[:6]],
            *[f"Risk: {risk}" for risk in report.remaining_risks[:3]],
        ],
        evidence=report.evidence[:10],
    )
    emitter.step_status(NodeName.REPORT.value, StepStatus.SUCCESS)

    _write_memory(context, state, report)

    metrics = context.budget.snapshot()
    emitter.metrics(metrics)
    emitter.state_patch({"final_report": report.model_dump()})

    return {
        "current_step": NodeName.REPORT.value,
        "final_report": report,
        "status": status,
        "metrics": {**gates, **metrics},
    }


def _write_memory(context: RunContext, state: GraphState, report: FinalReport) -> None:
    """Record the outcome so future runs can learn from it."""
    if context.memory is None or not context.memory_enabled:
        return

    triage = state.get("triage_result")
    patches = state.get("patches") or []
    failed_approaches = [
        f"attempt {p.attempt}: {p.summary or p.apply_error or 'failed'}"
        for p in patches[:-1]
    ]
    if patches and not patches[-1].applied:
        failed_approaches.append(f"final attempt failed to apply: {patches[-1].apply_error}")

    successful_tools = sorted(
        {record.name for record in (state.get("tool_history") or []) if record.ok}
    )
    diagnosis = state.get("diagnosis")
    lesson = ""
    if diagnosis is not None and report.verified:
        lesson = (
            f"First fix failed because {diagnosis.what_was_wrong_with_last_patch[:200]}; "
            f"the working fix followed {diagnosis.revised_hypothesis[:150]}"
        )

    try:
        context.memory.remember(
            MemoryRecord(
                memory_id=new_id("mem"),
                run_id=state["run_id"],
                repo=state.get("repo", ""),
                benchmark_id=context.benchmark.id if context.benchmark else "",
                issue=state["issue"],
                issue_kind=triage.issue_kind.value if triage else "unknown",
                root_cause=report.root_cause,
                files_involved=report.changed_files,
                successful_tools=successful_tools,
                failed_approaches=failed_approaches,
                patch_summary=patches[-1].summary if patches else "",
                patch_diff=report.diff,
                verified=report.verified,
                attempts=state.get("retry_count", 1) or 1,
                lesson=lesson,
            )
        )
    except Exception as exc:  # noqa: BLE001 - memory must never fail a run
        context.emitter.emit(
            NodeName.REPORT.value,
            "Memory write skipped",
            status=StepStatus.SKIPPED,
            detail=f"{type(exc).__name__}: {exc}",
        )
