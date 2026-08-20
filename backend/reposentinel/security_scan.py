"""Static security analysis with two interchangeable backends.

``semgrep``
    Used when the ``semgrep`` executable is present (Linux/macOS/Docker).
    Rules live in ``backend/reposentinel/resources/semgrep-rules.yml``.

``builtin``
    A real AST-based analyser used when Semgrep is unavailable - notably on
    Windows, where Semgrep publishes no wheel. It is not a Semgrep clone; it
    implements the specific rule set below by walking Python ASTs, so findings
    carry exact file/line/snippet information.

Both backends emit :class:`~reposentinel.models.schemas.SecurityFinding`, so
nothing downstream depends on which one ran.
"""

from __future__ import annotations

import ast
import functools
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from reposentinel.config import Settings, get_settings
from reposentinel.models.schemas import SecurityFinding, SecurityReport

RULES_FILE = Path(__file__).with_name("resources") / "semgrep-rules.yml"

# Method names whose first argument is executed as SQL.
_SQL_SINKS = {"execute", "executemany", "executescript", "execute_many", "raw"}
# Names that indicate a value is a hard-coded credential.
_SECRET_NAMES = ("password", "passwd", "secret", "api_key", "apikey", "token", "private_key")
_WEAK_HASHES = {"md5", "sha1"}


@dataclass
class Rule:
    rule_id: str
    severity: str
    message: str
    cwe: str


RULES: dict[str, Rule] = {
    "python.sql-injection": Rule(
        "python.sql-injection",
        "ERROR",
        "SQL query is built with string interpolation; use parameterised queries instead.",
        "CWE-89",
    ),
    "python.command-injection": Rule(
        "python.command-injection",
        "ERROR",
        "Shell command built from a dynamic value; avoid shell=True and pass an argv list.",
        "CWE-78",
    ),
    "python.dangerous-eval": Rule(
        "python.dangerous-eval",
        "ERROR",
        "Dynamic code execution via eval/exec on a non-literal value.",
        "CWE-95",
    ),
    "python.insecure-deserialization": Rule(
        "python.insecure-deserialization",
        "ERROR",
        "Deserialising untrusted data with pickle or yaml.load is unsafe.",
        "CWE-502",
    ),
    "python.hardcoded-secret": Rule(
        "python.hardcoded-secret",
        "WARNING",
        "Hard-coded credential assigned in source.",
        "CWE-798",
    ),
    "python.weak-hash": Rule(
        "python.weak-hash",
        "WARNING",
        "Weak hash function used; prefer sha256 or a password KDF.",
        "CWE-327",
    ),
    "python.tls-verify-disabled": Rule(
        "python.tls-verify-disabled",
        "ERROR",
        "TLS certificate verification is disabled.",
        "CWE-295",
    ),
    "python.insecure-random": Rule(
        "python.insecure-random",
        "WARNING",
        "Non-cryptographic randomness used for a security value; use the secrets module.",
        "CWE-338",
    ),
}


def _snippet(lines: list[str], line_number: int) -> str:
    if 1 <= line_number <= len(lines):
        return lines[line_number - 1].strip()[:200]
    return ""


def _is_dynamic_string(node: ast.AST) -> bool:
    """True when a string expression mixes in non-literal values."""
    if isinstance(node, ast.JoinedStr):  # f"..."
        return any(isinstance(v, ast.FormattedValue) for v in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add | ast.Mod):
        return True
    if isinstance(node, ast.Call):
        func = node.func
        # "...".format(x) / " ".join(parts)
        if isinstance(func, ast.Attribute) and func.attr in {"format", "join"}:
            return True
    return False


def _attr_chain(node: ast.AST) -> str:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


class _SecurityVisitor(ast.NodeVisitor):
    def __init__(self, path: str, lines: list[str]) -> None:
        self.path = path
        self.lines = lines
        self.findings: list[SecurityFinding] = []

    def _add(self, rule_id: str, line: int, extra: str = "") -> None:
        rule = RULES[rule_id]
        self.findings.append(
            SecurityFinding(
                rule_id=rule.rule_id,
                severity=rule.severity,  # type: ignore[arg-type]
                message=f"{rule.message}{f' ({extra})' if extra else ''}",
                path=self.path,
                line=line,
                snippet=_snippet(self.lines, line),
                cwe=rule.cwe,
            )
        )

    def visit_Call(self, node: ast.Call) -> None:
        chain = _attr_chain(node.func)
        tail = chain.rsplit(".", 1)[-1]

        # SQL sinks: cursor.execute(f"... {x} ...")
        if tail in _SQL_SINKS and node.args and _is_dynamic_string(node.args[0]):
            self._add("python.sql-injection", node.lineno, f"{chain}(...)")

        # Command execution
        if (
            (chain in {"os.system", "os.popen"} or tail in {"system", "popen"})
            and node.args
            and _is_dynamic_string(node.args[0])
        ):
            self._add("python.command-injection", node.lineno, chain)
        if chain.startswith("subprocess.") or tail in {"run", "Popen", "call", "check_output"}:
            for keyword in node.keywords:
                if (
                    keyword.arg == "shell"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                ):
                    self._add("python.command-injection", node.lineno, "shell=True")

        # eval / exec
        if chain in {"eval", "exec"} and node.args and not isinstance(node.args[0], ast.Constant):
            self._add("python.dangerous-eval", node.lineno, chain)

        # Deserialisation
        if chain in {"pickle.loads", "pickle.load", "cPickle.loads", "dill.loads"}:
            self._add("python.insecure-deserialization", node.lineno, chain)
        if chain == "yaml.load" and not any(k.arg == "Loader" for k in node.keywords):
            self._add("python.insecure-deserialization", node.lineno, "yaml.load without Loader")

        # Weak hashes
        if chain in {f"hashlib.{name}" for name in _WEAK_HASHES}:
            self._add("python.weak-hash", node.lineno, chain)

        # TLS verification
        for keyword in node.keywords:
            if (
                keyword.arg == "verify"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is False
            ):
                self._add("python.tls-verify-disabled", node.lineno, f"{chain}(verify=False)")

        # Insecure randomness for security material
        if chain.startswith("random."):
            self._add("python.insecure-random", node.lineno, chain)

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            value = node.value.value
            for target in node.targets:
                name = ""
                if isinstance(target, ast.Name):
                    name = target.id.lower()
                elif isinstance(target, ast.Attribute):
                    name = target.attr.lower()
                if name and any(marker in name for marker in _SECRET_NAMES) and len(value) >= 8:
                    self._add("python.hardcoded-secret", node.lineno, name)
        self.generic_visit(node)


def scan_source(path: str, source: str) -> list[SecurityFinding]:
    """Run the built-in rule set over one Python source file."""
    lines = source.splitlines()
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError:
        return []
    visitor = _SecurityVisitor(path, lines)
    visitor.visit(tree)
    return visitor.findings


def _builtin_scan(root: Path, relative_paths: list[str]) -> SecurityReport:
    started = time.perf_counter()
    findings: list[SecurityFinding] = []
    scanned = 0
    for relative in relative_paths:
        absolute = root / relative
        if absolute.suffix != ".py" or not absolute.is_file():
            continue
        scanned += 1
        findings.extend(
            scan_source(relative, absolute.read_text(encoding="utf-8", errors="replace"))
        )
    findings.sort(key=lambda f: (f.path, f.line))
    return SecurityReport(
        backend="builtin",
        ok=not any(f.severity == "ERROR" for f in findings),
        findings=findings,
        files_scanned=scanned,
        duration_ms=int((time.perf_counter() - started) * 1000),
    )


@functools.lru_cache(maxsize=1)
def semgrep_available() -> bool:
    return shutil.which("semgrep") is not None


def _semgrep_scan(root: Path, timeout: int = 180) -> SecurityReport:
    started = time.perf_counter()
    completed = subprocess.run(  # noqa: S603
        [
            "semgrep",
            "scan",
            "--config",
            str(RULES_FILE),
            "--json",
            "--quiet",
            "--no-git-ignore",
            str(root),
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        shell=False,
        check=False,
    )
    findings: list[SecurityFinding] = []
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        payload = {}
    for result in payload.get("results", []):
        extra = result.get("extra", {})
        metadata = extra.get("metadata", {})
        raw_severity = str(extra.get("severity", "WARNING")).upper()
        severity = {"ERROR": "ERROR", "WARNING": "WARNING", "INFO": "INFO"}.get(
            raw_severity, "WARNING"
        )
        try:
            relative = Path(result.get("path", "")).resolve().relative_to(root).as_posix()
        except (ValueError, OSError):
            relative = result.get("path", "")
        findings.append(
            SecurityFinding(
                rule_id=str(result.get("check_id", "semgrep.rule")),
                severity=severity,  # type: ignore[arg-type]
                message=str(extra.get("message", "")).strip(),
                path=relative,
                line=int(result.get("start", {}).get("line", 0) or 0),
                snippet=str(extra.get("lines", "")).strip()[:200],
                cwe=str((metadata.get("cwe") or [""])[0] if isinstance(metadata.get("cwe"), list) else metadata.get("cwe", "")),
            )
        )
    findings.sort(key=lambda f: (f.path, f.line))
    return SecurityReport(
        backend="semgrep",
        ok=not any(f.severity == "ERROR" for f in findings),
        findings=findings,
        files_scanned=len(payload.get("paths", {}).get("scanned", [])) or 0,
        duration_ms=int((time.perf_counter() - started) * 1000),
    )


def resolve_backend(settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    if settings.security_backend == "semgrep":
        return "semgrep" if semgrep_available() else "builtin"
    if settings.security_backend == "builtin":
        return "builtin"
    return "semgrep" if semgrep_available() else "builtin"


def run_security_scan(
    root: Path,
    relative_paths: list[str],
    settings: Settings | None = None,
) -> SecurityReport:
    settings = settings or get_settings()
    backend = resolve_backend(settings)
    if backend == "semgrep":
        try:
            return _semgrep_scan(root, timeout=settings.limits.sandbox_command_seconds)
        except (OSError, subprocess.SubprocessError):
            # Fall back rather than failing the run outright.
            pass
    return _builtin_scan(root, relative_paths)
