"""ROOT CAUSE, PATCH, CHECKS and REFLECT nodes.

CHECKS and REFLECT form the self-correction loop that the demo is built
around: a failed check run produces a structured diagnosis, the previous patch
is rolled back to the baseline, and PATCH runs again with the failure evidence
in hand.
"""

from __future__ import annotations

from reposentinel.graph.nodes.common import (
    call_model,
    issue_block,
    system_message,
    untrusted_block,
)
from reposentinel.graph.nodes.investigate import _context_digest, _record_tool
from reposentinel.graph.state import GraphState, LimitExceeded, RunContext
from reposentinel.models.providers.base import Message
from reposentinel.models.schemas import (
    Diagnosis,
    LintReport,
    NodeName,
    PatchProposal,
    PatchRecord,
    RootCause,
    StepStatus,
    TestReport,
)
from reposentinel.security_scan import run_security_scan
from reposentinel.strategies import get_strategy
from reposentinel.tools.base import execute_tool
from reposentinel.tools.exec_tools import run_pytest


def _tool_digest(state: GraphState, limit: int = 12) -> str:
    history = state.get("tool_history") or []
    if not history:
        return "No tools have been run yet."
    lines = []
    for record in history[-limit:]:
        status = "ok" if record.ok else "FAILED"
        lines.append(f"- {record.name}: {status} - {record.summary[:200]}")
    return "\n".join(lines)


def _failure_digest(state: GraphState) -> str:
    reports = state.get("test_results") or []
    if not reports:
        return ""
    latest = reports[-1]
    if latest.ok:
        return ""
    lines = [
        f"Last test run: {latest.command}",
        f"{latest.passed} passed, {latest.failed} failed, {latest.errors} error(s)",
    ]
    for failure in latest.failures[:6]:
        lines.append(f"\nFAILED {failure.node_id}\n{failure.message[:700]}")
    if latest.stdout_tail:
        lines.append(f"\npytest output tail:\n{latest.stdout_tail[-1500:]}")
    return "\n".join(lines)


def root_cause_node(context: RunContext, state: GraphState) -> GraphState:
    emitter = context.emitter
    emitter.step_status(NodeName.ROOT_CAUSE.value, StepStatus.RUNNING)

    outcome = call_model(
        context,
        node=NodeName.ROOT_CAUSE.value,
        purpose="root_cause",
        messages=[
            system_message(
                "State the root cause precisely: which file, which symbol, and what is "
                "wrong with it. Cite the evidence you actually have. If the evidence is "
                "insufficient, say so and give your best-supported hypothesis with a low "
                "confidence value."
            ),
            Message(
                "user",
                f"{issue_block(context, state['issue'], state.get('issue_id', ''))}\n\n"
                f"Investigation so far:\n{_tool_digest(state)}\n\n"
                f"Hypotheses recorded:\n"
                + "\n".join(f"- {h}" for h in (state.get('hypotheses') or [])[-4:])
                + f"\n\nCode retrieved:\n{untrusted_block(_context_digest(state, 7000), 'retrieved code')}"
                + (f"\n\nTest failures:\n{_failure_digest(state)}" if _failure_digest(state) else ""),
            ),
        ],
        response_model=RootCause,
    )

    updates: dict = {
        "current_step": NodeName.ROOT_CAUSE.value,
        "llm_calls": [outcome.record],
    }

    if not outcome.ok or not isinstance(outcome.response.parsed, RootCause):
        emitter.emit(
            NodeName.ROOT_CAUSE.value,
            "Root-cause analysis failed",
            status=StepStatus.FAILURE,
            detail=outcome.error or "no structured result",
        )
        emitter.step_status(NodeName.ROOT_CAUSE.value, StepStatus.FAILURE)
        return updates

    root_cause: RootCause = outcome.response.parsed
    emitter.emit(
        NodeName.ROOT_CAUSE.value,
        "Root cause identified",
        status=StepStatus.SUCCESS,
        detail=root_cause.statement,
        lines=[
            f"Location: {root_cause.file_path}"
            + (f"::{root_cause.symbol}" if root_cause.symbol else ""),
            f"Confidence: {root_cause.confidence:.2f}",
            *[f"Evidence: {item}" for item in root_cause.evidence[:5]],
        ],
        evidence=root_cause.evidence,
        duration_ms=outcome.record.duration_ms,
    )
    emitter.step_status(NodeName.ROOT_CAUSE.value, StepStatus.SUCCESS)
    updates["root_cause_analysis"] = root_cause
    updates["hypotheses"] = [f"root cause: {root_cause.statement}"]
    return updates


def patch_node(context: RunContext, state: GraphState) -> GraphState:
    emitter = context.emitter
    emitter.step_status(NodeName.PATCH.value, StepStatus.RUNNING)

    attempt = state.get("retry_count", 0) + 1
    root_cause = state.get("root_cause_analysis")
    diagnosis = state.get("diagnosis")
    updates: dict = {"current_step": NodeName.PATCH.value, "llm_calls": [], "tool_history": []}

    # A retry starts from the baseline, so attempts never stack on each other.
    if attempt > 1:
        context.workspace.restore_baseline()
        emitter.emit(
            NodeName.PATCH.value,
            f"Reverted to baseline for attempt {attempt}",
            status=StepStatus.SUCCESS,
            detail="The previous patch was rolled back so the new attempt is clean.",
        )

    retry_block = ""
    if diagnosis is not None:
        retry_block = (
            f"\nThis is attempt {attempt}. The previous patch did NOT work.\n"
            f"What went wrong: {diagnosis.what_was_wrong_with_last_patch}\n"
            f"Failure summary: {diagnosis.failure_summary}\n"
            f"Revised hypothesis: {diagnosis.revised_hypothesis}\n"
            f"Do something materially different from the previous attempt.\n"
        )
        previous = state.get("patches") or []
        if previous:
            retry_block += (
                f"\nThe previous attempt's diff (do not simply repeat it):\n"
                f"{previous[-1].diff[:2000]}\n"
            )

    outcome = call_model(
        context,
        node=NodeName.PATCH.value,
        purpose=f"patch_attempt_{attempt}",
        messages=[
            system_message(
                "Produce a minimal patch as exact search/replace edits.\n"
                "Rules for each edit:\n"
                "- 'search' must be copied VERBATIM from the current file, including "
                "indentation, and must appear EXACTLY ONCE in that file.\n"
                "- Include enough surrounding lines to be unique.\n"
                "- Change only what the root cause requires.\n"
                "- Never modify or delete a test in order to make it pass.\n"
                "Also list the tests that should be run first to prove the fix."
            ),
            Message(
                "user",
                f"{issue_block(context, state['issue'], state.get('issue_id', ''))}\n\n"
                f"Root cause: {root_cause.statement if root_cause else 'not established'}\n"
                f"Location: {root_cause.file_path if root_cause else '?'}"
                f"{f'::{root_cause.symbol}' if root_cause and root_cause.symbol else ''}\n"
                f"{retry_block}\n"
                f"Current code (copy 'search' text from here, or re-read the file with a tool):\n"
                f"{untrusted_block(_context_digest(state, 9000), 'retrieved code')}\n\n"
                f"Tool findings:\n{_tool_digest(state)}",
            ),
        ],
        response_model=PatchProposal,
    )
    updates["llm_calls"].append(outcome.record)

    if not outcome.ok or not isinstance(outcome.response.parsed, PatchProposal):
        emitter.emit(
            NodeName.PATCH.value,
            "Patch generation failed",
            status=StepStatus.FAILURE,
            detail=outcome.error or "no structured result",
        )
        emitter.step_status(NodeName.PATCH.value, StepStatus.FAILURE)
        updates["patches"] = [
            PatchRecord(
                attempt=attempt,
                summary="patch generation failed",
                applied=False,
                apply_error=outcome.error or "no structured result",
            )
        ]
        updates["retry_count"] = attempt
        return updates

    proposal: PatchProposal = outcome.response.parsed
    context.budget.patches_attempted += 1

    span = (
        context.tracer.span(
            "patch:apply", kind="patch", attributes={"attempt": attempt, "edits": len(proposal.edits)}
        )
        if context.tracer is not None
        else None
    )
    edits = [
        {"path": edit.path, "search": edit.search, "replace": edit.replace}
        for edit in proposal.edits
    ]
    if span is not None:
        with span as active:
            record = execute_tool("apply_patch", {"edits": edits}, context.tool_context)
            if active is not None:
                active.ok = record.ok
                active.error = record.error
    else:  # pragma: no cover
        record = execute_tool("apply_patch", {"edits": edits}, context.tool_context)

    _record_tool(context, updates, record)

    if not record.ok:
        emitter.emit(
            NodeName.PATCH.value,
            "Patch rejected",
            status=StepStatus.FAILURE,
            detail=record.error or "the edits could not be applied",
            lines=[f"{edit.path}: {edit.rationale[:120]}" for edit in proposal.edits],
        )
        emitter.step_status(NodeName.PATCH.value, StepStatus.FAILURE)
        updates["patches"] = [
            PatchRecord(
                attempt=attempt,
                summary=proposal.summary,
                applied=False,
                apply_error=record.error,
                files_changed=[edit.path for edit in proposal.edits],
            )
        ]
        updates["retry_count"] = attempt
        return updates

    diff = str(record.output or "")
    files, added, removed = context.workspace.diff_stat()
    patch_record = PatchRecord(
        attempt=attempt,
        summary=proposal.summary,
        diff=diff,
        files_changed=files,
        lines_added=added,
        lines_removed=removed,
        applied=True,
    )

    emitter.emit(
        NodeName.PATCH.value,
        f"Patch generated (attempt {attempt})",
        status=StepStatus.SUCCESS,
        detail=proposal.summary,
        lines=[
            *[f"{path}  +{added} -{removed}" for path in files],
            *[f"Risk: {note}" for note in proposal.risk_notes[:3]],
        ],
        evidence=files,
        duration_ms=outcome.record.duration_ms,
    )
    emitter.step_status(NodeName.PATCH.value, StepStatus.SUCCESS)
    emitter.state_patch(
        {
            "diff": diff,
            "patch": {
                "attempt": attempt,
                "summary": proposal.summary,
                "files_changed": files,
                "lines_added": added,
                "lines_removed": removed,
            },
            "tests_to_run": proposal.tests_to_run,
        }
    )

    updates["patches"] = [patch_record]
    updates["retry_count"] = attempt
    updates["metrics"] = {"tests_to_run": proposal.tests_to_run}
    return updates


def _targets_for(context: RunContext, state: GraphState) -> list[str]:
    """Choose targeted test paths: the model's suggestion, else the manifest, else all."""
    suggested = (state.get("metrics") or {}).get("tests_to_run") or []
    valid = [
        target
        for target in suggested
        if (context.workspace.root / target.split("::")[0]).exists()
    ]
    if valid:
        return valid
    if context.benchmark is not None and context.benchmark.targeted_tests:
        return list(context.benchmark.targeted_tests)
    return []


def checks_node(context: RunContext, state: GraphState) -> GraphState:
    """Run targeted tests, then the full suite, then lint and security."""
    emitter = context.emitter
    emitter.step_status(NodeName.CHECKS.value, StepStatus.RUNNING)
    updates: dict = {
        "current_step": NodeName.CHECKS.value,
        "test_results": [],
        "lint_results": [],
        "security_results": [],
        "tool_history": [],
    }

    latest_patch = (state.get("patches") or [None])[-1]
    if latest_patch is None or not latest_patch.applied:
        emitter.emit(
            NodeName.CHECKS.value,
            "Checks skipped",
            status=StepStatus.SKIPPED,
            detail="No patch was applied, so there is nothing to verify.",
        )
        emitter.step_status(NodeName.CHECKS.value, StepStatus.FAILURE)
        return updates

    # 1. Targeted tests - cheapest signal first.
    targets = _targets_for(context, state)
    targeted_report: TestReport | None = None
    if targets:
        span = (
            context.tracer.span("test:targeted", kind="test", attributes={"targets": targets})
            if context.tracer is not None
            else None
        )
        if span is not None:
            with span as active:
                targeted_report, blocked = run_pytest(context.tool_context, targets, "targeted")
                if active is not None:
                    active.ok = targeted_report.ok
                    active.attributes.update(
                        {
                            "passed": targeted_report.passed,
                            "failed": targeted_report.failed,
                            "errors": targeted_report.errors,
                        }
                    )
        else:  # pragma: no cover
            targeted_report, blocked = run_pytest(context.tool_context, targets, "targeted")

        context.budget.tool_calls += 1
        updates["test_results"].append(targeted_report)
        emitter.emit(
            NodeName.CHECKS.value,
            f"pytest {' '.join(targets)}",
            status=StepStatus.SUCCESS if targeted_report.ok else StepStatus.FAILURE,
            detail=(
                f"{targeted_report.passed} passed, {targeted_report.failed} failed, "
                f"{targeted_report.errors} error(s)"
            ),
            lines=[f"FAILED {f.node_id}" for f in targeted_report.failures[:8]],
            duration_ms=targeted_report.duration_ms,
        )
        emitter.state_patch({"targeted_tests": targeted_report.model_dump()})

        if not targeted_report.ok:
            # No point running the full suite while the target still fails.
            emitter.step_status(NodeName.CHECKS.value, StepStatus.FAILURE)
            return updates

    # 2. Full suite - regression check.
    span = (
        context.tracer.span("test:full", kind="test") if context.tracer is not None else None
    )
    if span is not None:
        with span as active:
            full_report, _ = run_pytest(context.tool_context, [], "full")
            if active is not None:
                active.ok = full_report.ok
                active.attributes.update(
                    {"passed": full_report.passed, "failed": full_report.failed}
                )
    else:  # pragma: no cover
        full_report, _ = run_pytest(context.tool_context, [], "full")

    context.budget.tool_calls += 1
    updates["test_results"].append(full_report)
    emitter.emit(
        NodeName.CHECKS.value,
        "pytest (full suite)",
        status=StepStatus.SUCCESS if full_report.ok else StepStatus.FAILURE,
        detail=f"{full_report.passed}/{full_report.total} passed"
        + (f", {full_report.failed} failed" if full_report.failed else ""),
        lines=[f"FAILED {f.node_id}" for f in full_report.failures[:8]],
        duration_ms=full_report.duration_ms,
    )
    emitter.state_patch({"full_tests": full_report.model_dump()})

    if not full_report.ok:
        emitter.step_status(NodeName.CHECKS.value, StepStatus.FAILURE)
        return updates

    # 3. Lint.
    lint_record = execute_tool("run_lint", {}, context.tool_context)
    context.budget.tool_calls += 1
    updates["tool_history"].append(lint_record)
    lint_payload = (lint_record.data or {}).get("report", {}) if hasattr(lint_record, "data") else {}
    lint_report = LintReport(**lint_payload) if lint_payload else LintReport(
        ok=lint_record.ok, issue_count=0 if lint_record.ok else 1
    )
    updates["lint_results"].append(lint_report)
    emitter.emit(
        NodeName.CHECKS.value,
        "ruff check",
        status=StepStatus.SUCCESS if lint_report.ok else StepStatus.FAILURE,
        detail=lint_record.summary,
        lines=lint_report.issues[:6],
        duration_ms=lint_record.duration_ms,
    )

    # 4. Security scan.
    span = (
        context.tracer.span("security:scan", kind="security")
        if context.tracer is not None
        else None
    )
    relative_paths = context.workspace.relative_files(extensions=(".py",))
    if span is not None:
        with span as active:
            security_report = run_security_scan(
                context.workspace.root, relative_paths, settings=context.settings
            )
            if active is not None:
                active.ok = security_report.ok
                active.attributes.update(
                    {
                        "backend": security_report.backend,
                        "findings": len(security_report.findings),
                        "blocking": security_report.blocking_count,
                    }
                )
    else:  # pragma: no cover
        security_report = run_security_scan(
            context.workspace.root, relative_paths, settings=context.settings
        )

    context.budget.tool_calls += 1
    updates["security_results"].append(security_report)
    emitter.emit(
        NodeName.CHECKS.value,
        f"security scan ({security_report.backend})",
        status=StepStatus.SUCCESS if security_report.ok else StepStatus.FAILURE,
        detail=(
            "no findings"
            if not security_report.findings
            else f"{len(security_report.findings)} finding(s), "
            f"{security_report.blocking_count} blocking"
        ),
        lines=[
            f"[{f.severity}] {f.rule_id} {f.path}:{f.line}"
            for f in security_report.findings[:8]
        ],
        duration_ms=security_report.duration_ms,
    )
    emitter.state_patch({"security": security_report.model_dump()})

    all_ok = full_report.ok and security_report.ok
    emitter.step_status(
        NodeName.CHECKS.value, StepStatus.SUCCESS if all_ok else StepStatus.FAILURE
    )
    return updates


def checks_passed(state: GraphState) -> bool:
    """Deterministic gate: the latest test and security runs must both be clean."""
    reports = state.get("test_results") or []
    if not reports:
        return False
    patches = state.get("patches") or []
    if not patches or not patches[-1].applied:
        return False
    if not all(report.ok for report in reports[-2:]):
        return False
    if not any(report.scope == "full" for report in reports):
        return False
    security = state.get("security_results") or []
    return not security or security[-1].ok


def reflect_node(context: RunContext, state: GraphState) -> GraphState:
    """Diagnose a failed check run and decide whether to retry."""
    emitter = context.emitter
    attempt = state.get("retry_count", 0)
    max_attempts = context.settings.limits.max_repair_attempts
    updates: dict = {"current_step": NodeName.REFLECT.value, "llm_calls": []}

    # Baselines A-D stop at their first failed patch. Self-correction is the
    # capability under test, so lending it to a baseline would erase the very
    # difference the evaluation is meant to measure.
    strategy = get_strategy(context.strategy)
    if not strategy.reflection:
        emitter.emit(
            NodeName.REFLECT.value,
            "Self-correction not available",
            status=StepStatus.SKIPPED,
            detail=(
                f"Baseline {strategy.baseline} ('{strategy.label}') stops after its "
                "first failed attempt; it has no reflect-and-retry loop."
            ),
        )
        emitter.step_status(NodeName.REFLECT.value, StepStatus.SKIPPED)
        return updates

    emitter.step_status(NodeName.REFLECT.value, StepStatus.RUNNING)

    security = state.get("security_results") or []
    patches = state.get("patches") or []
    failure_text = _failure_digest(state)
    if not failure_text and security and not security[-1].ok:
        failure_text = "Security findings:\n" + "\n".join(
            f"[{f.severity}] {f.rule_id} {f.path}:{f.line} {f.message}"
            for f in security[-1].findings[:8]
        )
    if not failure_text and patches and not patches[-1].applied:
        failure_text = f"The patch could not be applied: {patches[-1].apply_error}"

    outcome = call_model(
        context,
        node=NodeName.REFLECT.value,
        purpose=f"diagnose_attempt_{attempt}",
        messages=[
            system_message(
                "A repair attempt failed. Diagnose why using only the evidence below, "
                "then decide whether another attempt is worthwhile. Be concrete about "
                "what was wrong with the last patch. If the failing test reveals a "
                "second requirement that the patch broke, say exactly which one."
            ),
            Message(
                "user",
                f"{issue_block(context, state['issue'], state.get('issue_id', ''))}\n\n"
                f"Attempt {attempt} of at most {max_attempts}.\n"
                f"Last patch summary: {patches[-1].summary if patches else 'n/a'}\n"
                f"Last patch diff:\n{patches[-1].diff[:2500] if patches else '(none)'}\n\n"
                f"Failure evidence:\n{untrusted_block(failure_text or '(no output)', 'test output')}\n\n"
                f"Code context:\n{untrusted_block(_context_digest(state, 5000), 'retrieved code')}",
            ),
        ],
        response_model=Diagnosis,
    )
    updates["llm_calls"].append(outcome.record)

    if not outcome.ok or not isinstance(outcome.response.parsed, Diagnosis):
        emitter.emit(
            NodeName.REFLECT.value,
            "Diagnosis failed",
            status=StepStatus.FAILURE,
            detail=outcome.error or "no structured result",
        )
        emitter.step_status(NodeName.REFLECT.value, StepStatus.FAILURE)
        updates["diagnosis"] = Diagnosis(
            failure_summary=failure_text[:300] or "checks failed",
            revised_hypothesis="unavailable",
            what_was_wrong_with_last_patch="unavailable",
            should_retry=attempt < max_attempts,
        )
        return updates

    diagnosis: Diagnosis = outcome.response.parsed
    context.budget.retries += 1

    will_retry = diagnosis.should_retry and attempt < max_attempts
    emitter.emit(
        NodeName.REFLECT.value,
        "Diagnosis",
        status=StepStatus.SUCCESS,
        detail=diagnosis.failure_summary,
        lines=[
            f"What was wrong: {diagnosis.what_was_wrong_with_last_patch}",
            f"Revised hypothesis: {diagnosis.revised_hypothesis}",
            *[f"Next: {action}" for action in diagnosis.next_actions[:4]],
            *(
                [f"Wants to re-read: {', '.join(diagnosis.files_to_retrieve[:4])}"]
                if diagnosis.files_to_retrieve
                else []
            ),
            f"Decision: {'retry' if will_retry else 'stop'} "
            f"(attempt {attempt} of {max_attempts})",
        ],
        duration_ms=outcome.record.duration_ms,
    )
    emitter.step_status(NodeName.REFLECT.value, StepStatus.SUCCESS)

    updates["diagnosis"] = diagnosis
    updates["hypotheses"] = [f"revised: {diagnosis.revised_hypothesis}"]

    # Pull in any extra files the diagnosis asked for, so the next PATCH sees them.
    if will_retry and diagnosis.files_to_retrieve and context.retriever is not None:
        extra = []
        for path in diagnosis.files_to_retrieve[:3]:
            if (context.workspace.root / path).is_file():
                extra.extend(context.retriever.chunks_for_paths([path], query="reflection"))
        if extra:
            updates["retrieved_context"] = extra
            context.budget.retrieved_chunks += len(extra)
            emitter.emit(
                NodeName.REFLECT.value,
                "Retrieved additional context",
                status=StepStatus.SUCCESS,
                detail=f"Pulled {len(extra)} chunk(s) from "
                f"{', '.join(sorted({c.path for c in extra}))}",
                evidence=[c.location for c in extra],
            )
    return updates


def should_retry(context: RunContext, state: GraphState) -> str:
    """Router used after REFLECT."""
    if not get_strategy(context.strategy).reflection:
        return "give_up"
    diagnosis = state.get("diagnosis")
    attempt = state.get("retry_count", 0)
    if attempt >= context.settings.limits.max_repair_attempts:
        return "give_up"
    if diagnosis is not None and not diagnosis.should_retry:
        return "give_up"
    try:
        context.budget.check()
    except LimitExceeded:
        return "give_up"
    return "retry"
