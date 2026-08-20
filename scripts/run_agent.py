"""Run the agent on a benchmark from the command line.

    python scripts/run_agent.py logic_bug
    python scripts/run_agent.py logic_bug --model gpt-4.1 --strategy hybrid_rag
    python scripts/run_agent.py logic_bug --no-memory

Prints the live timeline as it happens, then the diff, metrics and report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from reposentinel.benchmarks import get_benchmark, list_benchmarks  # noqa: E402
from reposentinel.config import get_settings  # noqa: E402
from reposentinel.models.schemas import RunRequest  # noqa: E402
from reposentinel.observability.events import bus  # noqa: E402
from reposentinel.orchestrator import Orchestrator  # noqa: E402

STATUS_GLYPH = {
    "success": "[ok]",
    "failure": "[XX]",
    "running": "[..]",
    "pending": "[  ]",
    "skipped": "[--]",
    "blocked": "[!!]",
}


def print_event(event: dict) -> None:
    if event.get("type") != "timeline":
        return
    payload = event["event"]
    glyph = STATUS_GLYPH.get(payload["status"], "[??]")
    duration = payload.get("duration_ms", 0) / 1000.0
    at = payload.get("run_elapsed_ms", 0) / 1000.0
    print(f"{glyph} {payload['node']:<11} {payload['title']}   (+{duration:.1f}s @ {at:.1f}s)")
    if payload.get("detail"):
        for line in str(payload["detail"]).splitlines()[:4]:
            print(f"        {line[:150]}")
    for line in payload.get("lines", [])[:12]:
        print(f"        {line[:150]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RepoSentinel on a benchmark.")
    parser.add_argument("benchmark", nargs="?", default="logic_bug")
    parser.add_argument("--model", default="")
    parser.add_argument("--strategy", default="agentic")
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--keep-workspace", action="store_true")
    args = parser.parse_args()

    if args.list:
        for manifest in list_benchmarks():
            print(f"{manifest.id:20s} {manifest.category:24s} {manifest.title}")
        return 0

    manifest = get_benchmark(args.benchmark)
    if manifest is None:
        print(f"unknown benchmark '{args.benchmark}'")
        return 2

    settings = get_settings()
    print(f"benchmark : {manifest.id} - {manifest.title}")
    print(f"issue     : {manifest.issue[:160]}")
    print(f"model     : {args.model or settings.default_model}")
    print(f"strategy  : {args.strategy}   memory={'off' if args.no_memory else 'on'}")
    print("-" * 100)

    orchestrator = Orchestrator(settings=settings)
    request = RunRequest(
        issue=manifest.issue,
        repo=manifest.id,
        benchmark_id=manifest.id,
        issue_id=manifest.issue_id,
        model=args.model,
        strategy=args.strategy,
        memory_enabled=not args.no_memory,
        auto_approve=True,
    )

    handle = orchestrator.create_run(request)
    # Print events as the run progresses by draining the bus replay buffer.
    printed = 0
    orchestrator.start_background(handle)
    while handle.thread is not None and handle.thread.is_alive():
        events = bus.replay(handle.run_id)
        for event in events[printed:]:
            print_event(event)
        printed = len(events)
        handle.thread.join(timeout=0.4)
    for event in bus.replay(handle.run_id)[printed:]:
        print_event(event)

    print("-" * 100)
    state = handle.state
    print(f"status    : {handle.status}")
    if handle.error:
        print(f"error     : {handle.error}")

    patches = state.get("patches") or []
    if patches and patches[-1].get("diff"):
        print("\n=== FINAL DIFF ===")
        print(patches[-1]["diff"])

    report = state.get("final_report")
    if report:
        print("=== FINAL REPORT ===")
        print(f"verified      : {report['verified']}")
        print(f"root cause    : {report['root_cause']}")
        print(f"changed files : {', '.join(report['changed_files'])}")
        print(f"explanation   : {report['explanation']}")
        for item in report["validation_performed"]:
            print(f"  validated: {item}")
        for risk in report["remaining_risks"]:
            print(f"  risk     : {risk}")

    context = handle.context
    if context is not None:
        metrics = context.budget.snapshot()
        print("\n=== METRICS ===")
        for key, value in metrics.items():
            print(f"{key:22s} {value}")
        print(f"trace                  {context.tracer.totals()}")

        gold = set(manifest.gold_files)
        changed = set(patches[-1]["files_changed"] if patches else [])
        print(f"\ngold files             {sorted(gold)}")
        print(f"changed files          {sorted(changed)}")
        print(f"correct file targeted  {bool(gold & changed)}")

        if not args.keep_workspace:
            context.workspace.cleanup()

    return 0 if handle.status in {"succeeded", "approved"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
