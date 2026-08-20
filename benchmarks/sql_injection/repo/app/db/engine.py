"""A tiny SQL evaluator, enough to host the user table.

It is not a real database. It exists so that interpolated user input in a
``WHERE`` clause actually changes the meaning of the statement, which is the
definition of SQL injection.
"""

from __future__ import annotations

import re
from typing import Any

Row = dict[str, Any]

_SELECT = re.compile(
    r"^SELECT\s+(?P<columns>.+?)\s+FROM\s+(?P<table>\w+)"
    r"(?:\s+WHERE\s+(?P<where>.+))?\s*$",
    re.IGNORECASE | re.DOTALL,
)
_INSERT = re.compile(
    r"^INSERT\s+INTO\s+(?P<table>\w+)\s+\((?P<columns>.+?)\)\s+"
    r"VALUES\s+\((?P<values>.+)\)\s*$",
    re.IGNORECASE | re.DOTALL,
)
_EQ = re.compile(r"^(?P<column>\w+)\s*=\s*(?P<value>.+)$", re.DOTALL)
_OR = re.compile(r"\s+OR\s+", re.IGNORECASE)


class DatabaseError(Exception):
    pass


class Database:
    def __init__(self) -> None:
        self.tables: dict[str, list[Row]] = {}
        self.log: list[str] = []

    def create_table(self, name: str, columns: list[str]) -> None:
        self.tables[name] = []
        self._columns: dict[str, list[str]] = getattr(self, "_columns", {})
        self._columns[name] = columns

    def execute(self, sql: str, parameters: tuple[Any, ...] | None = None) -> list[Row]:
        """Run one statement.

        When *parameters* is provided, ``?`` placeholders are bound as data and
        never parsed as SQL. That is the difference between a safe query and
        an interpolated one.
        """
        statement = sql.strip().rstrip(";")
        if parameters is not None:
            self.log.append(f"{statement} -- params={parameters!r}")
            return self._execute_bound(statement, parameters)
        self.log.append(statement)
        return self._execute_raw(statement)

    def _execute_bound(self, sql: str, parameters: tuple[Any, ...]) -> list[Row]:
        placeholders = sql.count("?")
        if placeholders != len(parameters):
            raise DatabaseError("placeholder / parameter count mismatch")
        select = _SELECT.match(sql)
        if select:
            return self._select(select, bound=list(parameters))
        insert = _INSERT.match(sql)
        if insert:
            bound = list(parameters)
            self._insert(insert, bound=bound)
            return []
        raise DatabaseError(f"unrecognised statement: {sql[:80]}")

    def _execute_raw(self, statement: str) -> list[Row]:
        select = _SELECT.match(statement)
        if select:
            return self._select(select, bound=None)
        insert = _INSERT.match(statement)
        if insert:
            self._insert(insert, bound=None)
            return []
        raise DatabaseError(f"unrecognised statement: {statement[:80]}")

    def _select(self, match: re.Match[str], bound: list[Any] | None) -> list[Row]:
        table = match.group("table")
        if table not in self.tables:
            raise DatabaseError(f"no such table: {table}")
        columns = [c.strip() for c in match.group("columns").split(",")]
        rows = list(self.tables[table])
        where = match.group("where")
        if where:
            rows = [row for row in rows if _matches(row, where, bound)]
        if columns == ["*"]:
            return rows
        return [{column: row.get(column) for column in columns} for row in rows]

    def _insert(self, match: re.Match[str], bound: list[Any] | None) -> None:
        table = match.group("table")
        if table not in self.tables:
            raise DatabaseError(f"no such table: {table}")
        columns = [c.strip() for c in match.group("columns").split(",")]
        if bound is not None:
            values = list(bound)
        else:
            values = [_unquote(v.strip()) for v in _split_values(match.group("values"))]
        if len(columns) != len(values):
            raise DatabaseError("column/value count mismatch")
        row: Row = {}
        for column, value in zip(columns, values, strict=True):
            if column == "is_admin":
                row[column] = int(value)
            else:
                row[column] = value
        self.tables[table].append(row)


def _split_values(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    in_string = False
    for char in text:
        if char == "'" and not in_string:
            in_string = True
            current.append(char)
        elif char == "'" and in_string:
            in_string = False
            current.append(char)
        elif char == "," and not in_string:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        parts.append("".join(current).strip())
    return parts


def _unquote(token: str) -> Any:
    token = token.strip()
    if token.startswith("'") and token.endswith("'"):
        inner = token[1:-1].replace("''", "'")
        return inner
    if token.isdigit():
        return int(token)
    return token


def _matches(row: Row, where: str, bound: list[Any] | None = None) -> bool:
    """Evaluate a WHERE clause against *row*.

    Supports ``column = value``, ``column = ?`` (consuming *bound* in order)
    and ``OR``-joined conditions. The point of supporting ``OR`` is that an
    injected ``' OR 1=1 --`` actually matches every row, which is how the
    tests demonstrate the vulnerability.
    """
    remaining = list(bound) if bound is not None else None
    clauses = _OR.split(where)
    return any(_match_clause(row, clause.strip(), remaining) for clause in clauses)


def _match_clause(row: Row, clause: str, bound: list[Any] | None) -> bool:
    # Strip a trailing SQL comment so ``' OR 1=1 --`` still parses.
    clause = clause.split("--", 1)[0].strip()
    placeholder = re.match(r"^(?P<column>\w+)\s*=\s*\?$", clause)
    if placeholder:
        if not bound:
            raise DatabaseError("not enough bound parameters")
        expected = bound.pop(0)
        return str(row.get(placeholder.group("column"))) == str(expected)
    # ``1=1`` / ``'1'='1'`` tautologies, which is what an injected OR produces.
    tautology = re.match(r"^'?(?P<a>\w+)'?\s*=\s*'?(?P<b>\w+)'?$", clause)
    if tautology and tautology.group("a") == tautology.group("b"):
        return True
    match = _EQ.match(clause)
    if not match:
        return False
    column = match.group("column")
    expected = _unquote(match.group("value"))
    actual = row.get(column)
    if isinstance(actual, bool) or column == "is_admin":
        return bool(actual) == bool(int(expected) if str(expected).isdigit() else expected)
    return str(actual) == str(expected)
