"""Execution tools: tests, linting, type checking and security analysis.

Every command in this module goes through the sandbox, so all of them inherit
the allow-list, the workspace confinement, the scrubbed environment and the
timeout. Test results are parsed from pytest's JUnit XML rather than scraped
from terminal output, which gives exact per-test outcomes and messages.
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from reposentinel.models.schemas import (
    LintReport,
    SafetyEvent,
    TestCaseResult,
    TestReport,
)
from reposentinel.security_scan import resolve_backend, run_security_scan
from reposentinel.tools.base import ToolContext, ToolResult, registry, schema
from reposentinel.workspace import ARTIFACTS_DIRNAME

JUNIT_RELATIVE = f"{ARTIFACTS_DIRNAME}/junit.xml"


def _blocked_result(reason: str, tool: str) -> ToolResult:
    return ToolResult(
        ok=False,
        executed=False,
        error=f"blocked by guardrail: {reason}",
        summary=f"command blocked: {reason}",
        output=f"The sandbox refused this command: {reason}",
        safety_events=[
            SafetyEvent(kind="blocked_command", detail=reason, source=tool, severity="critical")
        ],
    )


def _parse_junit(xml_path: Path, command: str, scope: str, exit_code: int, duration_ms: int) -> TestReport:
    """Turn pytest's JUnit XML into a structured report."""
    report = TestReport(command=command, scope=scope, exit_code=exit_code, duration_ms=duration_ms)  # type: ignore[arg-type]
    if not xml_path.is_file():
        return report
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return report

    for case in tree.getroot().iter("testcase"):
        classname = (case.get("classname") or "").replace(".", "/")
        name = case.get("name") or "?"
        node_id = f"{classname}::{name}" if classname else name
        case_duration = int(float(case.get("time") or 0.0) * 1000)

        failure = case.find("failure")
        error = case.find("error")
        skipped = case.find("skipped")

        if failure is not None:
            report.failed += 1
            report.failures.append(
                TestCaseResult(
                    node_id=node_id,
                    outcome="failed",
                    message=(failure.get("message") or failure.text or "").strip()[:600],
                    duration_ms=case_duration,
                )
            )
        elif error is not None:
            report.errors += 1
            report.failures.append(
                TestCaseResult(
                    node_id=node_id,
                    outcome="error",
                    message=(error.get("message") or error.text or "").strip()[:600],
                    duration_ms=case_duration,
                )
            )
        elif skipped is not None:
            report.skipped += 1
        else:
            report.passed += 1
    return report


def run_pytest(
    context: ToolContext,
    targets: list[str],
    scope: str = "targeted",
) -> tuple[TestReport, ToolResult | None]:
    """Run pytest in the sandbox and parse the result.

    Returns ``(report, blocked_result)``; ``blocked_result`` is non-None only
    when a guardrail refused the command.
    """
    junit_path = context.workspace.artifacts_dir / "junit.xml"
    if junit_path.exists():
        junit_path.unlink()

    argv = [
        "pytest",
        "-q",
        "--tb=short",
        "-p",
        "no:cacheprovider",
        f"--junitxml={JUNIT_RELATIVE}",
        *[t for t in targets if t.strip()],
    ]
    result = context.sandbox.run(argv, timeout=context.settings.limits.sandbox_command_seconds)
    if result.blocked:
        return TestReport(command=" ".join(argv), scope=scope), _blocked_result(  # type: ignore[arg-type]
            result.block_reason, "run_tests"
        )

    report = _parse_junit(
        junit_path,
        command=" ".join(argv),
        scope=scope,
        exit_code=result.exit_code,
        duration_ms=result.duration_ms,
    )
    report.timed_out = result.timed_out
    report.stdout_tail = result.sanitised(limit=6000)[-6000:]
    return report, None


def _format_test_report(report: TestReport) -> str:
    header = (
        f"$ {report.command}\n"
        f"{report.passed} passed, {report.failed} failed, {report.errors} error(s), "
        f"{report.skipped} skipped (exit {report.exit_code}, {report.duration_ms} ms)"
    )
    if report.timed_out:
        header += "\nTIMED OUT"
    if not report.failures:
        return header
    details = "\n\n".join(
        f"FAILED {f.node_id}\n{f.message}" for f in report.failures[:10]
    )
    tail = f"\n\n--- pytest output (tail) ---\n{report.stdout_tail[-2500:]}"
    return f"{header}\n\n{details}{tail}"


@registry.register(
    name="run_targeted_tests",
    description=(
        "Run specific test files or node ids. Use this for fast feedback on the "
        "tests related to the issue, e.g. ['tests/test_auth.py'] or "
        "['tests/test_auth.py::test_expired_token']."
    ),
    parameters=schema(
        {
            "targets": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Test files or pytest node ids.",
            }
        },
        required=["targets"],
    ),
    category="execution",
    expose_via_mcp=True,
)
def run_targeted_tests(context: ToolContext, targets: list[str]) -> ToolResult:
    if not targets:
        return ToolResult.failure("at least one test target is required")
    report, blocked = run_pytest(context, targets, scope="targeted")
    if blocked is not None:
        return blocked
    return ToolResult(
        ok=report.ok,
        summary=(
            f"targeted tests: {report.passed} passed, {report.failed} failed, "
            f"{report.errors} error(s)"
        ),
        output=_format_test_report(report),
        data={"report": report.model_dump(), "scope": "targeted"},
        evidence=[f.node_id for f in report.failures],
    )


@registry.register(
    name="run_full_tests",
    description="Run the repository's entire test suite to check for regressions.",
    parameters=schema({}),
    category="execution",
    expose_via_mcp=True,
)
def run_full_tests(context: ToolContext) -> ToolResult:
    report, blocked = run_pytest(context, [], scope="full")
    if blocked is not None:
        return blocked
    return ToolResult(
        ok=report.ok,
        summary=(
            f"full suite: {report.passed}/{report.total} passed"
            + (f", {report.failed} failed" if report.failed else "")
        ),
        output=_format_test_report(report),
        data={"report": report.model_dump(), "scope": "full"},
        evidence=[f.node_id for f in report.failures],
    )


@registry.register(
    name="run_lint",
    description="Run the Ruff linter over the repository.",
    parameters=schema(
        {
            "path": {
                "type": "string",
                "description": "Path to lint, relative to the repository root.",
                "default": ".",
            }
        }
    ),
    category="execution",
    expose_via_mcp=True,
)
def run_lint(context: ToolContext, path: str = ".") -> ToolResult:
    argv = ["ruff", "check", "--output-format=concise", "--no-cache", path]
    result = context.sandbox.run(argv)
    if result.blocked:
        return _blocked_result(result.block_reason, "run_lint")

    issues = [
        line.strip()
        for line in result.combined_output.splitlines()
        if line.strip() and not line.startswith("Found") and "All checks passed" not in line
    ]
    report = LintReport(
        tool="ruff",
        ok=result.exit_code == 0,
        issue_count=len(issues) if result.exit_code != 0 else 0,
        issues=issues[:50],
        exit_code=result.exit_code,
    )
    return ToolResult(
        ok=report.ok,
        summary=f"ruff: {'passed' if report.ok else f'{report.issue_count} issue(s)'}",
        output=f"$ {' '.join(argv)}\n{result.sanitised(limit=6000) or 'All checks passed!'}",
        data={"report": report.model_dump()},
    )


@registry.register(
    name="run_type_check",
    description=(
        "Run mypy if it is installed. Reports 'unavailable' when the tool is "
        "not present rather than failing the run."
    ),
    parameters=schema(
        {"path": {"type": "string", "description": "Path to check.", "default": "."}}
    ),
    category="execution",
)
def run_type_check(context: ToolContext, path: str = ".") -> ToolResult:
    try:
        import mypy  # noqa: F401
    except ImportError:
        if shutil.which("mypy") is None:
            return ToolResult(
                ok=True,
                summary="type check skipped: mypy is not installed",
                output=(
                    "mypy is not installed in this environment, so type checking was "
                    "skipped. Install it with `pip install mypy` to enable this tool."
                ),
                data={"available": False, "skipped": True},
            )

    argv = ["mypy", "--ignore-missing-imports", "--no-error-summary", path]
    result = context.sandbox.run(argv)
    if result.blocked:
        return _blocked_result(result.block_reason, "run_type_check")
    issues = [line for line in result.combined_output.splitlines() if ": error:" in line]
    return ToolResult(
        ok=result.exit_code == 0,
        summary=f"mypy: {'passed' if result.exit_code == 0 else f'{len(issues)} error(s)'}",
        output=f"$ {' '.join(argv)}\n{result.sanitised(limit=6000)}",
        data={"available": True, "errors": len(issues)},
    )


@registry.register(
    name="run_security_scan",
    description=(
        "Run static security analysis (Semgrep when available, otherwise the "
        "built-in AST analyser) and report findings with file and line numbers."
    ),
    parameters=schema(
        {
            "path_filter": {
                "type": "string",
                "description": "Only report findings in paths containing this substring.",
                "default": "",
            }
        }
    ),
    category="execution",
    expose_via_mcp=True,
)
def run_security_scan_tool(context: ToolContext, path_filter: str = "") -> ToolResult:
    relative_paths = context.workspace.relative_files(extensions=(".py",))
    report = run_security_scan(context.workspace.root, relative_paths, settings=context.settings)
    findings = [f for f in report.findings if not path_filter or path_filter in f.path]

    if not findings:
        return ToolResult(
            ok=True,
            summary=f"security scan ({report.backend}): no findings",
            output=(
                f"Scanned {report.files_scanned} file(s) with the '{report.backend}' backend. "
                f"No security findings."
            ),
            data={"report": report.model_dump(), "backend": report.backend},
        )

    lines = [
        f"[{f.severity}] {f.rule_id} ({f.cwe})\n  {f.path}:{f.line}\n  {f.message}\n  > {f.snippet}"
        for f in findings
    ]
    blocking = sum(1 for f in findings if f.severity == "ERROR")
    return ToolResult(
        ok=blocking == 0,
        summary=(
            f"security scan ({report.backend}): {len(findings)} finding(s), "
            f"{blocking} blocking"
        ),
        output=f"Backend: {report.backend}\n\n" + "\n\n".join(lines),
        data={"report": report.model_dump(), "backend": report.backend, "blocking": blocking},
        evidence=[f"{f.path}:{f.line}" for f in findings],
    )


def describe_check_backends(context: ToolContext) -> dict[str, object]:
    return {
        "security_backend": resolve_backend(context.settings),
        "sandbox_backend": context.sandbox.backend_name,
    }
