"""Benchmark registry.

Each benchmark directory holds a ``manifest.json`` describing the seeded defect
plus a ``repo/`` tree containing the defective code. The manifest is the ground
truth used by the evaluation harness (gold files for retrieval metrics,
expected failing tests for repair metrics).
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

from pydantic import BaseModel, Field

from reposentinel.config import get_settings


class BenchmarkManifest(BaseModel):
    id: str
    title: str
    category: str
    issue: str
    issue_id: str = ""
    difficulty: str = "medium"
    repo_dir: str = "repo"
    language: str = "python"

    # Ground truth for retrieval metrics.
    gold_files: list[str] = Field(default_factory=list)
    gold_symbols: list[str] = Field(default_factory=list)
    supporting_files: list[str] = Field(default_factory=list)

    # Ground truth for repair metrics.
    targeted_tests: list[str] = Field(default_factory=list)
    regression_tests: list[str] = Field(default_factory=list)
    expected_failing_tests: list[str] = Field(default_factory=list)

    # Baseline A ("LLM only") is handed these files directly.
    baseline_seed_files: list[str] = Field(default_factory=list)

    security_scan_required: bool = False
    expects_injection: bool = False
    expected_retry: bool = False
    gold_patch_hint: str = ""
    expected_behaviour: list[str] = Field(default_factory=list)

    # Populated at load time.
    root: Path | None = None

    @property
    def repo_path(self) -> Path:
        if self.root is None:  # pragma: no cover - defensive
            raise RuntimeError("manifest was not loaded from disk")
        return (self.root / self.repo_dir).resolve()

    @property
    def relevant_files(self) -> list[str]:
        """Gold + supporting files, used for relevant-file recall."""
        seen: list[str] = []
        for path in [*self.gold_files, *self.supporting_files]:
            if path not in seen:
                seen.append(path)
        return seen


@functools.lru_cache(maxsize=1)
def _load_all() -> dict[str, BenchmarkManifest]:
    settings = get_settings()
    registry: dict[str, BenchmarkManifest] = {}
    if not settings.benchmarks_dir.is_dir():
        return registry
    for manifest_path in sorted(settings.benchmarks_dir.glob("*/manifest.json")):
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = BenchmarkManifest(**data, root=manifest_path.parent)
        registry[manifest.id] = manifest
    return registry


def list_benchmarks() -> list[BenchmarkManifest]:
    return list(_load_all().values())


def get_benchmark(benchmark_id: str) -> BenchmarkManifest | None:
    return _load_all().get(benchmark_id)


def reload_benchmarks() -> None:
    """Drop the cache (used by tests and by the API's reload endpoint)."""
    _load_all.cache_clear()
