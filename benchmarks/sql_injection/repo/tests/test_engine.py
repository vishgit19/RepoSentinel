"""The engine itself, covering the safe (parameterised) path.

These tests pass with the seeded defect in place: parameterised queries already
work. The repository just does not use them.
"""

from __future__ import annotations

from app.db.engine import Database, DatabaseError


def users_db() -> Database:
    database = Database()
    database.create_table("users", ["user_id", "email", "display_name", "is_admin"])
    database.execute(
        "INSERT INTO users (user_id, email, display_name, is_admin) "
        "VALUES ('u1', 'alice@example.com', 'Alice', 0)"
    )
    return database


def test_select_by_literal_email():
    rows = users_db().execute(
        "SELECT user_id, email FROM users WHERE email = 'alice@example.com'"
    )
    assert len(rows) == 1
    assert rows[0]["user_id"] == "u1"


def test_parameterised_select_treats_metacharacters_as_data():
    rows = users_db().execute(
        "SELECT user_id FROM users WHERE email = ?",
        ("' OR 1=1 --",),
    )
    assert rows == []


def test_parameterised_select_still_finds_a_real_row():
    rows = users_db().execute(
        "SELECT user_id FROM users WHERE email = ?",
        ("alice@example.com",),
    )
    assert rows[0]["user_id"] == "u1"


def test_placeholder_count_must_match():
    database = users_db()
    try:
        database.execute("SELECT user_id FROM users WHERE email = ?", ())
    except DatabaseError as exc:
        assert "mismatch" in str(exc)
    else:
        raise AssertionError("expected DatabaseError")


def test_insert_adds_a_row():
    database = users_db()
    database.execute(
        "INSERT INTO users (user_id, email, display_name, is_admin) "
        "VALUES ('u9', 'new@example.com', 'New', 0)"
    )
    rows = database.execute("SELECT user_id FROM users WHERE email = 'new@example.com'")
    assert rows[0]["user_id"] == "u9"
