"""A dependency-free HTTP-ish façade.

Routing is a plain dict so the demo repository has no third-party
dependencies and its test suite runs anywhere pytest runs.
"""

from __future__ import annotations

from app.auth.middleware import AuthError, authenticate_request, require_scope
from app.users.repository import UserRepository, seed_repository


class Response:
    def __init__(self, status: int, body: dict[str, object]) -> None:
        self.status = status
        self.body = body

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"Response(status={self.status}, body={self.body!r})"


class Api:
    def __init__(self, users: UserRepository | None = None) -> None:
        self.users = users or seed_repository()

    def handle(
        self,
        method: str,
        path: str,
        headers: dict[str, str] | None = None,
        body: dict[str, object] | None = None,
        now: float | None = None,
    ) -> Response:
        headers = headers or {}
        try:
            principal = authenticate_request(headers, now=now)
        except AuthError as exc:
            return Response(exc.status, {"error": exc.reason})

        if method == "GET" and path == "/me":
            user = self.users.get(principal.user_id)
            if user is None:
                return Response(404, {"error": "user_not_found"})
            return Response(200, {"user_id": user.user_id, "email": user.email})

        if method == "GET" and path == "/users":
            try:
                require_scope(principal, "users:read")
            except AuthError as exc:
                return Response(exc.status, {"error": exc.reason})
            return Response(200, {"users": [u.user_id for u in self.users.list_users()]})

        return Response(404, {"error": "not_found"})
