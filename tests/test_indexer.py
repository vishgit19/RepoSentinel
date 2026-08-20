"""Symbol indexing, symbol-boundary chunking and graph construction."""

from __future__ import annotations

import pytest
from reposentinel.benchmarks import get_benchmark
from reposentinel.retrieval.indexer import CodeIndexer, is_test_path
from reposentinel.workspace import Workspace


@pytest.fixture(scope="module")
def indexed():
    workspace = Workspace.prepare("logic_bug", "test_index_ws")
    try:
        indexer = CodeIndexer("logic_bug", workspace.root)
        yield indexer.index(workspace.relative_files())
    finally:
        workspace.cleanup()


class TestSymbolExtraction:
    def test_files_are_indexed(self, indexed):
        assert "app/auth/token.py" in indexed.files
        assert "tests/test_auth.py" in indexed.files
        assert all(f.parse_error is None for f in indexed.files.values())

    def test_class_and_method_are_found(self, indexed):
        matches = indexed.find_symbols("SessionToken")
        assert len(matches) == 1
        assert matches[0].kind == "class"
        assert matches[0].path == "app/auth/token.py"

        methods = indexed.find_symbols("is_expired")
        assert len(methods) == 1
        method = methods[0]
        assert method.kind == "method"
        assert method.dotted == "SessionToken.is_expired"
        assert method.parent == "app/auth/token.py::SessionToken"
        assert method.docstring.startswith("Return True when")

    def test_dotted_lookup(self, indexed):
        assert indexed.find_symbols("SessionToken.is_expired")[0].name == "is_expired"

    def test_line_spans_are_exact(self, indexed):
        symbol = indexed.find_symbols("decode_token")[0]
        source = (indexed.root / symbol.path).read_text(encoding="utf-8").splitlines()
        assert source[symbol.start_line - 1].startswith("def decode_token")
        # The span must cover the whole body, not spill into the next symbol.
        body = "\n".join(source[symbol.start_line - 1 : symbol.end_line])
        assert "return SessionToken(" in body
        assert "def issue_token" not in body

    def test_imports_are_recorded(self, indexed):
        middleware = indexed.files["app/auth/middleware.py"]
        assert "app.auth.token" in middleware.imports
        assert "decode_token" in middleware.imported_names

    def test_calls_are_recorded(self, indexed):
        validate = indexed.find_symbols("validate_token")[0]
        assert "decode_token" in validate.calls
        assert "is_expired" in validate.calls


class TestChunking:
    def test_every_symbol_has_a_chunk(self, indexed):
        chunked = {(c.path, c.symbol) for c in indexed.chunks}
        for symbol in indexed.symbols.values():
            assert (symbol.path, symbol.dotted) in chunked, symbol.qualname

    def test_chunk_content_matches_source_lines(self, indexed):
        for chunk in indexed.chunks:
            source = (indexed.root / chunk.path).read_text(encoding="utf-8").splitlines()
            expected = "\n".join(source[chunk.start_line - 1 : chunk.end_line]).strip()
            assert chunk.content.strip() == expected, chunk.location

    def test_module_header_chunk_holds_imports(self, indexed):
        headers = [c for c in indexed.chunks_for_file("app/auth/token.py") if c.symbol_kind == "file"]
        assert len(headers) == 1
        assert "import hmac" in headers[0].content
        assert "SESSION_TTL_SECONDS = 3600" in headers[0].content

    def test_class_chunk_does_not_swallow_methods(self, indexed):
        class_chunks = [
            c for c in indexed.chunks if c.symbol == "SessionToken" and c.symbol_kind == "class"
        ]
        assert len(class_chunks) == 1
        assert "def is_expired" not in class_chunks[0].content

    def test_target_symbol_chunk_contains_the_bug(self, indexed):
        chunk = next(c for c in indexed.chunks if c.symbol == "SessionToken.is_expired")
        assert "SESSION_TTL_SECONDS" in chunk.content
        assert chunk.symbol_kind == "method"

    def test_chunk_ids_are_unique_and_stable(self, indexed):
        ids = [c.chunk_id for c in indexed.chunks]
        assert len(ids) == len(set(ids))
        reindexed = CodeIndexer("logic_bug", indexed.root).index(sorted(indexed.files))
        assert {c.chunk_id for c in reindexed.chunks} == set(ids)


class TestGraph:
    def test_contains_edges(self, indexed):
        edges = [
            e
            for e in indexed.edges
            if e.relation == "contains" and e.source.endswith("::SessionToken")
        ]
        targets = {e.target.rsplit("::", 1)[-1] for e in edges}
        assert {"SessionToken.is_expired", "SessionToken.seconds_remaining"} <= targets

    def test_calls_edges(self, indexed):
        callers = {s.name for s in indexed.callers_of("decode_token")}
        assert "validate_token" in callers
        assert "describe" in callers

    def test_callees(self, indexed):
        callees = {s.name for s in indexed.callees_of("validate_token")}
        assert "decode_token" in callees
        assert "is_expired" in callees

    def test_imports_edges(self, indexed):
        edges = [
            e
            for e in indexed.edges
            if e.relation == "imports" and e.source == "app/auth/middleware.py"
        ]
        assert any(e.target == "app.auth.token" for e in edges)

    def test_tests_edges_point_at_production_code(self, indexed):
        edges = [e for e in indexed.edges if e.relation == "tests"]
        assert edges, "no test->function edges were derived"
        targets = {e.target for e in edges}
        assert any(t == "app/auth/token.py::SessionToken.is_expired" for t in targets)
        assert all(is_test_path(e.path) for e in edges)

    def test_neighbour_expansion_reaches_related_symbols(self, indexed):
        chunk = next(c for c in indexed.chunks if c.symbol == "SessionToken.is_expired")
        neighbours = indexed.neighbours(chunk, hops=1)
        assert neighbours
        related = {target for _, target in neighbours}
        # Reached from the containing class and from its callers/tests.
        assert any("SessionToken" in r for r in related)

    def test_stats_are_reported(self, indexed):
        stats = indexed.stats()
        assert stats["files"] >= 8
        assert stats["symbols"] >= 20
        assert stats["chunks"] >= 25
        assert stats["rel_calls"] > 0
        assert stats["rel_contains"] > 0
        assert stats["rel_imports"] > 0


class TestSyntaxErrorTolerance:
    def test_broken_file_is_still_indexed(self, tmp_path):
        (tmp_path / "broken.py").write_text("def oops(:\n    pass\n", encoding="utf-8")
        index = CodeIndexer("tmp", tmp_path).index(["broken.py"])
        assert index.files["broken.py"].parse_error is not None
        assert index.chunks_for_file("broken.py")


class TestTestPathDetection:
    @pytest.mark.parametrize(
        "path", ["tests/test_auth.py", "test_x.py", "app/foo_test.py", "conftest.py"]
    )
    def test_positive(self, path):
        assert is_test_path(path) is True

    @pytest.mark.parametrize("path", ["app/auth/token.py", "app/api.py", "latest/thing.py"])
    def test_negative(self, path):
        assert is_test_path(path) is False


def test_gold_symbol_is_discoverable():
    """The manifest's gold symbol must resolve, or retrieval metrics are wrong."""
    manifest = get_benchmark("logic_bug")
    workspace = Workspace.prepare("logic_bug", "test_index_gold")
    try:
        index = CodeIndexer("logic_bug", workspace.root).index(workspace.relative_files())
        for dotted in manifest.gold_symbols:
            assert index.find_symbols(dotted), dotted
    finally:
        workspace.cleanup()
