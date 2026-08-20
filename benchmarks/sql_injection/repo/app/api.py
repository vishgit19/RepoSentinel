"""HTTP-shaped user lookup.

The email comes from a query string, so it is attacker-controlled. Lookup
must treat it as data, never as part of a SQL statement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.db.engine import Database
from app.users.repository import User, UserRepository


@dataclass
class Response:
    status: int
    body: dict[str, Any]


class UserApi:
    def __init__(self, repository: UserRepository) -> None:
        self.repository = repository

    def lookup(self, email: str) -> Response:
        """GET /users?email=..."""
        if not email:
            return Response(400, {"error": "email is required"})
        user = self.repository.get_by_email(email)
        if user is None:
            return Response(404, {"error": "not found"})
        return Response(
            200,
            {
                "user_id": user.user_id,
                "email": user.email,
                "display_name": user.display_name,
            },
        )


def build_api() -> UserApi:
    database = Database()
    database.create_table("users", ["user_id", "email", "display_name", "is_admin"])
    repository = UserRepository(database)
    repository.create(User("u1", "alice@example.com", "Alice"))
    repository.create(User("u2", "bob@example.com", "Bob", is_admin=True))
    return UserApi(repository)
