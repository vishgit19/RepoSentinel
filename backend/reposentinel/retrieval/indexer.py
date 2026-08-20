"""Symbol-aware repository indexing.

Chunking happens along **symbol boundaries** rather than fixed token windows:
one chunk per function, method and class, plus a synthetic "module header"
chunk carrying the module docstring, imports and module-level constants. A
retrieved chunk is therefore always a self-contained, quotable unit of code
with an exact line range.

Parsing uses Python's built-in :mod:`ast`. It is the correct tool for the
Python repositories in this project: no native build step, exact
``end_lineno`` spans, and full access to call/import structure. The
:class:`RepoIndex` interface is parser-agnostic, so a Tree-sitter backend for
other languages can be added without touching retrieval.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from reposentinel.models.schemas import CodeChunk, SymbolEdge

TEST_FILE_MARKERS = ("test_", "_test.py", "tests/", "conftest.py")


def is_test_path(path: str) -> bool:
    normalised = path.replace("\\", "/").lower()
    name = normalised.rsplit("/", 1)[-1]
    return (
        name.startswith("test_")
        or normalised.endswith("_test.py")
        or "/tests/" in f"/{normalised}"
        or name == "conftest.py"
    )


@dataclass
class Symbol:
    """One indexed code symbol."""

    qualname: str  # "app/auth/token.py::SessionToken.is_expired"
    name: str  # "is_expired"
    dotted: str  # "SessionToken.is_expired"
    kind: str  # function | method | class | module
    path: str
    start_line: int
    end_line: int
    signature: str = ""
    docstring: str = ""
    parent: str | None = None
    calls: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)

    @property
    def line_count(self) -> int:
        return max(1, self.end_line - self.start_line + 1)


@dataclass
class FileIndex:
    path: str
    module: str
    imports: list[str] = field(default_factory=list)
    imported_names: list[str] = field(default_factory=list)
    symbols: list[Symbol] = field(default_factory=list)
    line_count: int = 0
    parse_error: str | None = None


class _ModuleVisitor(ast.NodeVisitor):
    """Collects symbols, calls and imports from one module."""

    def __init__(self, path: str, source_lines: list[str]) -> None:
        self.path = path
        self.source_lines = source_lines
        self.symbols: list[Symbol] = []
        self.imports: list[str] = []
        self.imported_names: list[str] = []
        self._scope: list[str] = []

    # -- imports ---------------------------------------------------------
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(alias.name)
            self.imported_names.append((alias.asname or alias.name).split(".")[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if module:
            self.imports.append(module)
        for alias in node.names:
            self.imported_names.append(alias.asname or alias.name)
            if module:
                self.imports.append(f"{module}.{alias.name}")
        self.generic_visit(node)

    # -- definitions -----------------------------------------------------
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._add_symbol(node, kind="class")
        self._scope.append(node.name)
        for child in node.body:
            self.visit(child)
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._add_symbol(node, kind="method" if self._scope else "function")
        self._scope.append(node.name)
        for child in node.body:
            self.visit(child)
        self._scope.pop()

    # -- helpers ---------------------------------------------------------
    def _add_symbol(self, node: ast.AST, kind: str) -> None:
        name = getattr(node, "name", "<anonymous>")
        dotted = ".".join([*self._scope, name])
        start = getattr(node, "lineno", 1)
        end = getattr(node, "end_lineno", start) or start
        symbol = Symbol(
            qualname=f"{self.path}::{dotted}",
            name=name,
            dotted=dotted,
            kind=kind,
            path=self.path,
            start_line=start,
            end_line=end,
            signature=self._signature(node),
            docstring=(ast.get_docstring(node) or "").strip()[:400]
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
            else "",
            parent=f"{self.path}::{'.'.join(self._scope)}" if self._scope else None,
            calls=sorted(_extract_calls(node)),
            decorators=[_expr_name(d) for d in getattr(node, "decorator_list", [])],
        )
        self.symbols.append(symbol)

    def _signature(self, node: ast.AST) -> str:
        line = getattr(node, "lineno", 1) - 1
        if 0 <= line < len(self.source_lines):
            return self.source_lines[line].strip()
        return ""


def _expr_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _expr_name(node.func)
    return ""


def _extract_calls(node: ast.AST) -> set[str]:
    """Names invoked directly inside a definition (excluding nested defs)."""
    calls: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            name = _expr_name(child.func)
            if name:
                calls.add(name)
    return calls


@dataclass
class RepoIndex:
    """The complete index for one workspace."""

    repo_id: str
    root: Path
    files: dict[str, FileIndex] = field(default_factory=dict)
    symbols: dict[str, Symbol] = field(default_factory=dict)
    chunks: list[CodeChunk] = field(default_factory=list)
    edges: list[SymbolEdge] = field(default_factory=list)

    # -- lookups ---------------------------------------------------------
    def chunk_by_id(self, chunk_id: str) -> CodeChunk | None:
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None

    def chunks_for_file(self, path: str) -> list[CodeChunk]:
        return [c for c in self.chunks if c.path == path]

    def find_symbols(self, name: str) -> list[Symbol]:
        """Match on bare name, dotted name or fully-qualified name."""
        needle = name.strip()
        if not needle:
            return []
        exact = [s for s in self.symbols.values() if s.qualname == needle]
        if exact:
            return exact
        lowered = needle.lower()
        return sorted(
            (
                s
                for s in self.symbols.values()
                if s.name.lower() == lowered or s.dotted.lower() == lowered
            ),
            key=lambda s: (s.path, s.start_line),
        )

    def callers_of(self, name: str) -> list[Symbol]:
        targets = {s.name for s in self.find_symbols(name)} or {name}
        return sorted(
            (s for s in self.symbols.values() if targets & set(s.calls)),
            key=lambda s: (s.path, s.start_line),
        )

    def callees_of(self, name: str) -> list[Symbol]:
        callees: list[Symbol] = []
        seen: set[str] = set()
        for symbol in self.find_symbols(name):
            for called in symbol.calls:
                for match in self.find_symbols(called):
                    if match.qualname not in seen:
                        seen.add(match.qualname)
                        callees.append(match)
        return callees

    def neighbours(self, chunk: CodeChunk, hops: int = 1) -> list[tuple[SymbolEdge, str]]:
        """Graph edges reachable from a chunk's symbol, out to ``hops``."""
        if not chunk.symbol:
            frontier = {f"{chunk.path}::__module__"}
        else:
            frontier = {f"{chunk.path}::{chunk.symbol}"}
        seen = set(frontier)
        found: list[tuple[SymbolEdge, str]] = []
        for _ in range(max(1, hops)):
            next_frontier: set[str] = set()
            for edge in self.edges:
                if edge.source in frontier and edge.target not in seen:
                    found.append((edge, edge.target))
                    next_frontier.add(edge.target)
                elif edge.target in frontier and edge.source not in seen:
                    found.append((edge, edge.source))
                    next_frontier.add(edge.source)
            seen |= next_frontier
            frontier = next_frontier
            if not frontier:
                break
        return found

    def stats(self) -> dict[str, int]:
        kinds: dict[str, int] = {}
        for symbol in self.symbols.values():
            kinds[symbol.kind] = kinds.get(symbol.kind, 0) + 1
        relations: dict[str, int] = {}
        for edge in self.edges:
            relations[edge.relation] = relations.get(edge.relation, 0) + 1
        return {
            "files": len(self.files),
            "symbols": len(self.symbols),
            "chunks": len(self.chunks),
            "edges": len(self.edges),
            **{f"kind_{k}": v for k, v in sorted(kinds.items())},
            **{f"rel_{k}": v for k, v in sorted(relations.items())},
        }


def _chunk_id(repo_id: str, path: str, dotted: str, start: int) -> str:
    digest = hashlib.sha1(f"{repo_id}|{path}|{dotted}|{start}".encode()).hexdigest()
    return f"ch_{digest[:16]}"


class CodeIndexer:
    """Builds a :class:`RepoIndex` from a workspace directory."""

    def __init__(self, repo_id: str, root: Path) -> None:
        self.repo_id = repo_id
        self.root = root.resolve()

    def index(self, relative_paths: list[str]) -> RepoIndex:
        index = RepoIndex(repo_id=self.repo_id, root=self.root)
        for relative in relative_paths:
            self._index_file(index, relative)
        self._build_edges(index)
        return index

    # -- per-file --------------------------------------------------------
    def _index_file(self, index: RepoIndex, relative: str) -> None:
        absolute = self.root / relative
        try:
            source = absolute.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            index.files[relative] = FileIndex(
                path=relative, module=_module_name(relative), parse_error=str(exc)
            )
            return

        lines = source.splitlines()
        file_index = FileIndex(
            path=relative, module=_module_name(relative), line_count=len(lines)
        )

        try:
            tree = ast.parse(source, filename=relative)
        except SyntaxError as exc:
            file_index.parse_error = f"SyntaxError: {exc.msg} (line {exc.lineno})"
            index.files[relative] = file_index
            # A file that does not parse is still searchable as one chunk.
            index.chunks.append(
                CodeChunk(
                    chunk_id=_chunk_id(self.repo_id, relative, "__module__", 1),
                    repo_id=self.repo_id,
                    path=relative,
                    symbol="",
                    symbol_kind="file",
                    start_line=1,
                    end_line=max(1, len(lines)),
                    content=source[:8000],
                )
            )
            return

        visitor = _ModuleVisitor(relative, lines)
        visitor.visit(tree)

        file_index.imports = sorted(set(visitor.imports))
        file_index.imported_names = sorted(set(visitor.imported_names))
        file_index.symbols = visitor.symbols
        index.files[relative] = file_index

        for symbol in visitor.symbols:
            index.symbols[symbol.qualname] = symbol

        index.chunks.extend(self._build_chunks(relative, lines, visitor, tree))

    def _build_chunks(
        self,
        relative: str,
        lines: list[str],
        visitor: _ModuleVisitor,
        tree: ast.Module,
    ) -> list[CodeChunk]:
        chunks: list[CodeChunk] = []

        # Module header: everything before the first definition, which is where
        # docstrings, imports and module constants live.
        first_def = min(
            (s.start_line for s in visitor.symbols),
            default=len(lines) + 1,
        )
        header_end = max(1, first_def - 1)
        header_text = "\n".join(lines[:header_end]).strip()
        if header_text:
            chunks.append(
                CodeChunk(
                    chunk_id=_chunk_id(self.repo_id, relative, "__module__", 1),
                    repo_id=self.repo_id,
                    path=relative,
                    symbol="",
                    symbol_kind="file",
                    start_line=1,
                    end_line=header_end,
                    content=header_text,
                )
            )

        for symbol in visitor.symbols:
            # A class body is represented by its own chunk only down to the
            # first method, so class-level and method-level chunks do not
            # duplicate each other's content.
            end_line = symbol.end_line
            if symbol.kind == "class":
                child_starts = [
                    s.start_line
                    for s in visitor.symbols
                    if s.parent == symbol.qualname and s.start_line > symbol.start_line
                ]
                if child_starts:
                    end_line = min(child_starts) - 1
            body = "\n".join(lines[symbol.start_line - 1 : end_line])
            if not body.strip():
                continue
            chunks.append(
                CodeChunk(
                    chunk_id=_chunk_id(self.repo_id, relative, symbol.dotted, symbol.start_line),
                    repo_id=self.repo_id,
                    path=relative,
                    symbol=symbol.dotted,
                    symbol_kind=symbol.kind,
                    start_line=symbol.start_line,
                    end_line=end_line,
                    content=body,
                )
            )
        return chunks

    # -- graph -----------------------------------------------------------
    def _build_edges(self, index: RepoIndex) -> None:
        """Materialise the relations the spec calls for.

        ``class -> contains -> method``, ``function -> calls -> function``,
        ``file -> imports -> module``, ``test -> tests -> function``.
        ``function -> modified_by -> commit`` is added separately by
        :func:`attach_commit_edges` because it needs git.
        """
        by_name: dict[str, list[Symbol]] = {}
        for symbol in index.symbols.values():
            by_name.setdefault(symbol.name, []).append(symbol)

        edges: list[SymbolEdge] = []

        for file_index in index.files.values():
            for module in file_index.imports:
                edges.append(
                    SymbolEdge(
                        source=file_index.path,
                        relation="imports",
                        target=module,
                        path=file_index.path,
                    )
                )

        for symbol in index.symbols.values():
            if symbol.parent:
                edges.append(
                    SymbolEdge(
                        source=symbol.parent,
                        relation="contains",
                        target=symbol.qualname,
                        path=symbol.path,
                        line=symbol.start_line,
                    )
                )

            is_test = is_test_path(symbol.path) and symbol.name.startswith("test")
            for called in symbol.calls:
                for target in by_name.get(called, []):
                    if target.qualname == symbol.qualname:
                        continue
                    edges.append(
                        SymbolEdge(
                            source=symbol.qualname,
                            relation="tests" if is_test else "calls",
                            target=target.qualname,
                            path=symbol.path,
                            line=symbol.start_line,
                        )
                    )

        # Deduplicate while preserving order.
        seen: set[tuple[str, str, str]] = set()
        for edge in edges:
            key = (edge.source, edge.relation, edge.target)
            if key in seen:
                continue
            seen.add(key)
            index.edges.append(edge)


def attach_commit_edges(
    index: RepoIndex, history: dict[str, list[dict[str, str]]]
) -> None:
    """Add ``symbol -> modified_by -> commit`` edges from git history."""
    for path, commits in history.items():
        for commit in commits:
            sha = commit.get("commit", "")
            if not sha:
                continue
            index.edges.append(
                SymbolEdge(
                    source=path,
                    relation="modified_by",
                    target=f"commit:{sha}",
                    path=path,
                )
            )


def _module_name(relative: str) -> str:
    stem = relative.replace("\\", "/")
    if stem.endswith(".py"):
        stem = stem[: -len(".py")]
    if stem.endswith("/__init__"):
        stem = stem[: -len("/__init__")]
    return stem.replace("/", ".")
