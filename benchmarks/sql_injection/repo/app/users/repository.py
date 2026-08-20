"""The in-memory user store.

Queries are issued as SQL-shaped strings so the access pattern is the one a
real application would use against a database. The engine is tiny: it parses
a handful of SELECT / INSERT statements and evaluates them against a dict.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.db.engine import Database, Row


@dataclass(frozen=True)
class User:
    user_id: str
    email: str
    display_name: str
    is_admin: bool = False


class UserRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def get_by_email(self, email: str) -> User | None:
        """Look up a user by email. Returns None when no row matches."""
        rows = self.database.execute(
            f"SELECT user_id, email, display_name, is_admin FROM users WHERE email = '{email}'"
        )
        if not rows:
            return None
        return _row_to_user(rows[0])

    def create(self, user: User) -> User:
        self.database.execute(
            "INSERT INTO users (user_id, email, display_name, is_admin) "
            "VALUES (?, ?, ?, ?)",
            (user.user_id, user.email, user.display_name, 1 if user.is_admin else 0),
        )
        return user

    def list_users(self) -> list[User]:
        rows = self.database.execute(
            "SELECT user_id, email, display_name, is_admin FROM users"
        )
        return [_row_to_user(row) for row in rows]


def _row_to_user(row: Row) -> User:
    return User(
        user_id=str(row["user_id"]),
        email=str(row["email"]),
        display_name=str(row["display_name"]),
        is_admin=bool(row["is_admin"]),
    )
