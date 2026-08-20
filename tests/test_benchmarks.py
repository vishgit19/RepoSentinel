"""Each benchmark is a real defective repository, not a fixture description."""

from __future__ import annotations

import subprocess
import sys

import pytest
from reposentinel.benchmarks import get_benchmark, list_benchmarks, reload_benchmarks
from reposentinel.sandbox.guardrails import scan_for_injection
from reposentinel.strategies import BY_ID, describe_strategies

reload_benchmarks()

EXPECTED_IDS = ("logic_bug", "cache_bug", "sql_injection", "retry_bug", "injection")


class TestRegistry:
    def test_all_five_benchmarks_are_registered(self):
        ids = {manifest.id for manifest in list_benchmarks()}
        assert ids == set(EXPECTED_IDS)

    def test_each_manifest_has_ground_truth(self):
        for manifest in list_benchmarks():
            assert manifest.issue.strip()
            assert manifest.gold_files
            assert manifest.expected_failing_tests
            assert manifest.repo_path.is_dir()
            for relative in [*manifest.gold_files, *manifest.supporting_files]:
                assert (manifest.repo_path / relative).is_file(), relative

    def test_injection_benchmark_actually_contains_injection_patterns(self):
        manifest = get_benchmark("injection")
        assert manifest is not None
        text = (manifest.repo_path / manifest.gold_files[0]).read_text(encoding="utf-8")
        labels = {match.label for match in scan_for_injection(text, source="gold")}
        assert "override_instructions" in labels
        assert "sabotage_verification" in labels
        assert manifest.expects_injection is True

    def test_sql_injection_benchmark_requires_a_security_scan(self):
        manifest = get_benchmark("sql_injection")
        assert manifest is not None
        assert manifest.security_scan_required is True

    def test_retry_benchmark_expects_self_correction(self):
        manifest = get_benchmark("retry_bug")
        assert manifest is not None
        assert manifest.expected_retry is True


class TestSeededFailures:
    """The tests named in each manifest must actually fail in the seeded repo."""

    @pytest.mark.parametrize("benchmark_id", EXPECTED_IDS)
    def test_expected_failing_tests_fail(self, benchmark_id: str):
        manifest = get_benchmark(benchmark_id)
        assert manifest is not None
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--tb=no", *manifest.expected_failing_tests],
            cwd=manifest.repo_path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert completed.returncode != 0, completed.stdout + completed.stderr
        # pytest -q prints a line like "3 failed, 0 passed" for a focused run.
        assert "failed" in completed.stdout.lower() or "failed" in completed.stderr.lower()


class TestStrategies:
    def test_only_the_full_agent_has_a_retry_loop(self):
        for strategy in describe_strategies():
            if strategy["id"] == "agentic":
                assert strategy["tools"] is True
                assert strategy["reflection"] is True
            else:
                assert strategy["reflection"] is False
        assert [s.baseline for s in BY_ID.values()] == ["A", "B", "C", "D", "E"]
