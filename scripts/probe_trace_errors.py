"""Show the failed spans of the most recent run.

A trace that reports errors while the run reports success is either a real
problem or a mislabelled span. This prints the failures so the distinction can
be made from evidence rather than assumption.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from reposentinel.config import get_settings  # noqa: E402


def main() -> int:
    settings = get_settings()
    database = settings.data_dir / "reposentinel.db"
    if not database.exists():
        print(f"no run store at {database}")
        return 1

    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row

    tables = [r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    print(f"tables: {tables}")

    runs = list(connection.execute("SELECT * FROM runs ORDER BY rowid DESC LIMIT 1"))
    if not runs:
        print("no runs recorded")
        return 1
    run = runs[0]
    run_id = run["run_id"]
    print(f"latest run: {run_id}  status={run['status']}")

    columns = [r[1] for r in connection.execute("PRAGMA table_info(run_spans)")]
    print(f"span columns: {columns}")

    rows = list(connection.execute("SELECT payload FROM run_spans WHERE run_id = ?", (run_id,)))
    spans = [json.loads(row["payload"]) for row in rows]
    failures = [s for s in spans if not s.get("ok", True)]
    print(f"\n{len(spans)} spans, {len(failures)} not ok:")
    for span in failures:
        print(f"  [{span.get('kind')}] {span.get('name')}  ({span.get('duration_ms')}ms)")
        print(f"      error: {str(span.get('error'))[:220]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
